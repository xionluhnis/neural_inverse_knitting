from __future__ import division
import os
import math
import time
import tensorflow as tf
import numpy as np
import scipy
import re
import pdb
from .nnlib import *
from .parameters import arch_para, hparams, Parameters
from util import read_image, comp_confusionmat
import tensorflow.contrib.slim as slim

from .tensorflow_vgg import custom_vgg19
from .layer_modules import prog_ch, tf_MILloss_xentropy, tf_loss_xentropy, tf_MILloss_accuracy, tf_background, syntax_loss, tf_accuracy, create_canonical_coordinates, oper_random_geo_perturb, oper_img2img, style_layer_loss, tf_frequency_weight, oper_img2prog_final, oper_img2img_bottleneck, oper_img2prog_final_complex


def conv2d(input_, output_dim, ks=3, s=2, stddev=0.02, padding='VALID', name="conv2d"):
    
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        padsz = math.ceil((ks - s) * 0.5)
        if padsz != 0:
            input_ = tf.pad(input_,
                        tf.constant([[0, 0], [padsz, padsz], [padsz, padsz], [0, 0]]),
        mode='SYMMETRIC')
        return slim.conv2d(input_, output_dim, ks, s, padding=padding,
                            activation_fn=None,
                            weights_initializer=tf.truncated_normal_initializer(stddev=stddev)
                            # biases_initializer=None
                            )
def deconv2d(input_, output_dim, ks=4, s=2, stddev=0.02, name="deconv2d"):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        return slim.conv2d_transpose(input_, output_dim, ks, s, padding='SAME',                            activation_fn=None,
                            weights_initializer=tf.truncated_normal_initializer(stddev=stddev)
                            # biases_initializer=None
                            )

def discriminator(image, params = dict(), name="discriminator"):

    feat_ch = int(params.get('feat_ch', 64))
    noise_sigma = params.get('noise_sigma', 3./255.)
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        # image is 256 x 256 x input_c_dim
        image = image + tf.random_normal(tf.shape(image), stddev = noise_sigma)

        h0 = lrelu(conv2d(image, feat_ch, name='d_h0_conv'))
        # h0 is (128 x 128 x self.df_dim) 80
        h1 = lrelu(conv2d(h0, feat_ch*2, name='d_h1_conv'))
        # h1 is (64 x 64 x self.df_dim*2) 40
        h2 = lrelu(conv2d(h1, feat_ch*4, name='d_h2_conv'))
        # h2 is (32x 32 x self.df_dim*4) 20
        h3 = lrelu(conv2d(h2, feat_ch*8, name='d_h3_conv'))
        # h3 is (32 x 32 x self.df_dim*8) 10
        h4 = lrelu(conv2d(h3, feat_ch*8, s=1, name='d_h4_conv'))
        h5 = conv2d(h4, 1, s=1, name='d_h4_pred')
        # h4 is (32 x 32 x 1)

        return h5


def discriminator_cond(image, instruction, params = dict(), name="discriminator"):

    feat_ch = int(params.get('feat_ch', 64))
    noise_sigma = params.get('noise_sigma', 3./255.)
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        # image is 256 x 256 x input_c_dim
        image = image + tf.random_normal(tf.shape(image), stddev = noise_sigma)

        t_onehot = tf.one_hot(tf.squeeze(instruction),depth=prog_ch,dtype=tf.float32)
        t_embed_inst = conv2d(t_onehot, 4, ks=1, s=1)
        t_embed_inst = tf.image.resize_images(t_embed_inst, 
                                            [image.get_shape()[1], image.get_shape()[2]],
                                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR )

        h0 = lrelu(conv2d(tf.concat([image, t_embed_inst], axis=-1), feat_ch, name='d_h0_conv'))
        # h0 is (128 x 128 x self.df_dim) 80
        h1 = lrelu(conv2d(h0, feat_ch*2, name='d_h1_conv'))
        # h1 is (64 x 64 x self.df_dim*2) 40
        h2 = lrelu(conv2d(h1, feat_ch*4, name='d_h2_conv'))
        # h2 is (32x 32 x self.df_dim*4) 20
        h3 = lrelu(conv2d(h2, feat_ch*8, s=1, name='d_h3_conv'))
        # h3 is (32 x 32 x self.df_dim*8) 10
        # h4 = lrelu(conv2d(h3, feat_ch*8, s=1, name='d_h4_conv'))
        h4 = conv2d(h3, 1, s=1, name='d_h4_pred')
        # h4 is (32 x 32 x 1)

        return h4


#https://github.com/xhujoy/CycleGAN-tensorflow
def generator_unet(image, out_dim, params = dict(), is_training = True, name="generator"):

    feat_ch = int(params.get('feat_ch', 64))
    dropout_rate = 0.5 if is_training else 1.0
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        # image is 256 x 256 x input_c_dim
        # if reuse:
        #     tf.get_variable_scope().reuse_variables()
        # else:
        #     assert tf.get_variable_scope().reuse is False

        # image is (256 x 256 x input_c_dim)
        # image = tf.image.resize_bilinear(image, [160, 160])
        e1 = inst_norm(conv2d(image, feat_ch, name='g_e1_conv'))
        # e1 is (128 x 128 x self.gf_dim) 80
        e2 = inst_norm(conv2d(lrelu(e1), feat_ch*2, name='g_e2_conv'), 'g_bn_e2')
        # e2 is (64 x 64 x self.gf_dim*2) 40
        e3 = inst_norm(conv2d(lrelu(e2), feat_ch*4, name='g_e3_conv'), 'g_bn_e3')
        # e3 is (32 x 32 x self.gf_dim*4) 20
        e4 = inst_norm(conv2d(lrelu(e3), feat_ch*8, name='g_e4_conv'), 'g_bn_e4')
        # e4 is (16 x 16 x self.gf_dim*8) 10
        e5 = inst_norm(conv2d(lrelu(e4), feat_ch*8, name='g_e5_conv'), 'g_bn_e5')
        # e5 is (8 x 8 x self.gf_dim*8) 5
        # e6 = inst_norm(conv2d(lrelu(e5), feat_ch*8, name='g_e6_conv'), 'g_bn_e6')
        # e6 is (4 x 4 x self.gf_dim*8) 
        # e7 = inst_norm(conv2d(lrelu(e6), feat_ch*8, name='g_e7_conv'), 'g_bn_e7')
        # e7 is (2 x 2 x self.gf_dim*8)
        # e8 = inst_norm(conv2d(lrelu(e7), feat_ch*8, name='g_e8_conv'), 'g_bn_e8')
        # e8 is (1 x 1 x self.gf_dim*8)

        # d1 = deconv2d(tf.nn.relu(e8), feat_ch*8, name='g_d1')
        # d1 = tf.nn.dropout(d1, dropout_rate)
        # d1 = tf.concat([inst_norm(d1, 'g_bn_d1'), e7], 3)
        # d1 is (2 x 2 x self.gf_dim*8*2)

        # d2 = deconv2d(tf.nn.relu(d1), feat_ch*8, name='g_d2')
        # d2 = tf.nn.dropout(d2, dropout_rate)
        # d2 = tf.concat([inst_norm(d2, 'g_bn_d2'), e6], 3)
        # d2 is (4 x 4 x self.gf_dim*8*2)
        
        # d3 = conv2d(upsample(tf.nn.relu(d2), blinear=False), feat_ch*8, ks=3, s=1, name='g_d3')
        # d3 = tf.nn.dropout(d3, dropout_rate)
        # d3 = tf.concat([inst_norm(d3, 'g_bn_d3'), e5], 3)
        # d3 is (8 x 8 x self.gf_dim*8*2)

        d3 = e5

        d4 = conv2d(upsample(tf.nn.relu(d3), blinear=False), feat_ch*8, ks=3, s=1, name='g_d4')
        d4 = tf.concat([inst_norm(d4, 'g_bn_d4'), e4], 3)
        # d4 is (16 x 16 x self.gf_dim*8*2)

        d5 = conv2d(upsample(tf.nn.relu(d4), blinear=False), feat_ch*4, ks=3, s=1, name='g_d5')
        d5 = tf.concat([inst_norm(d5, 'g_bn_d5'), e3], 3)
        # d5 is (32 x 32 x self.gf_dim*4*2)

        d6 = conv2d(upsample(tf.nn.relu(d5), blinear=False), feat_ch*2, ks=3, s=1, name='g_d6')
        d6 = tf.concat([inst_norm(d6, 'g_bn_d6'), e2], 3)
        # d6 is (64 x 64 x self.gf_dim*2*2)

        d7 = conv2d(upsample(tf.nn.relu(d6), blinear=False), feat_ch, ks=3, s=1, name='g_d7')
        d7 = tf.concat([inst_norm(d7, 'g_bn_d7'), e1], 3)
        # d7 is (128 x 128 x self.gf_dim*1*2)

        d8 = conv2d(upsample(tf.nn.relu(d7), blinear=False), out_dim, ks=3, s=1, name='g_d8')

        # d3 = deconv2d(tf.nn.relu(d2), feat_ch*8, name='g_d3')
        # d3 = tf.nn.dropout(d3, dropout_rate)
        # d3 = tf.concat([inst_norm(d3, 'g_bn_d3'), e5], 3)
        # # d3 is (8 x 8 x self.gf_dim*8*2)

        # d4 = deconv2d(tf.nn.relu(d3), feat_ch*8, name='g_d4')
        # d4 = tf.concat([inst_norm(d4, 'g_bn_d4'), e4], 3)
        # # d4 is (16 x 16 x self.gf_dim*8*2)

        # d5 = deconv2d(tf.nn.relu(d4), feat_ch*4, name='g_d5')
        # d5 = tf.concat([inst_norm(d5, 'g_bn_d5'), e3], 3)
        # # d5 is (32 x 32 x self.gf_dim*4*2)

        # d6 = deconv2d(tf.nn.relu(d5), feat_ch*2, name='g_d6')
        # d6 = tf.concat([inst_norm(d6, 'g_bn_d6'), e2], 3)
        # # d6 is (64 x 64 x self.gf_dim*2*2)

        # d7 = deconv2d(tf.nn.relu(d6), feat_ch, name='g_d7')
        # d7 = tf.concat([inst_norm(d7, 'g_bn_d7'), e1], 3)
        # # d7 is (128 x 128 x self.gf_dim*1*2)

        # d8 = deconv2d(tf.nn.relu(d7), out_dim, name='g_d8')
        # d8 is (256 x 256 x output_c_dim)
        # print(d8.get_shape())

        return d8


def generator_resnet(image, out_dim, params = dict(), is_training=True, name="generator"):

    feat_ch = int(params.get('feat_ch', 64))
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        # image is 256 x 256 x input_c_dim
        # if reuse:
        #     tf.get_variable_scope().reuse_variables()
        # else:
        #     assert tf.get_variable_scope().reuse is False

        def residule_block(x, dim, ks=3, s=1, name='res'):
            p = int((ks - 1) / 2)
            y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = inst_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c1'), name+'_bn1')
            y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = inst_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c2'), name+'_bn2')
            return y + x

        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
        c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c1 = tf.nn.relu(inst_norm(conv2d(c0, feat_ch, 7, 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))
        c2 = tf.nn.relu(inst_norm(conv2d(c1, feat_ch*2, 3, 2, name='g_e2_c'), 'g_e2_bn'))
        c3 = tf.nn.relu(inst_norm(conv2d(c2, feat_ch*4, 3, 2, name='g_e3_c'), 'g_e3_bn'))
        # define G network with 9 resnet blocks
        r1 = residule_block(c3, feat_ch*4, name='g_r1')
        r2 = residule_block(r1, feat_ch*4, name='g_r2')
        r3 = residule_block(r2, feat_ch*4, name='g_r3')
        r4 = residule_block(r3, feat_ch*4, name='g_r4')
        r5 = residule_block(r4, feat_ch*4, name='g_r5')
        r6 = residule_block(r5, feat_ch*4, name='g_r6')
        r7 = residule_block(r6, feat_ch*4, name='g_r7')
        r8 = residule_block(r7, feat_ch*4, name='g_r8')
        r9 = residule_block(r8, feat_ch*4, name='g_r9')

        d1 = deconv2d(r9, feat_ch*2, 3, 2, name='g_d1_dc')
        d1 = tf.nn.relu(inst_norm(d1, 'g_d1_bn'))
        d2 = deconv2d(d1, feat_ch, 3, 2, name='g_d2_dc')
        d2 = tf.nn.relu(inst_norm(d2, 'g_d2_bn'))
        d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        pred = tf.nn.tanh(conv2d(d2, out_dim, 7, 1, padding='VALID', name='g_pred_c'))

        return pred



def model_composited_RFI_complexnet(t_imgs_dict, t_labels_dict, params = dict()):
    '''
    Compose the full network model
    '''
    net = Parameters()
    net.inputs = t_imgs_dict
    net.imgs = dict()
    net.resi_imgs = dict()  # rend | tran | real
    net.resi_imgs_noaug = dict()  # rend | tran | real
    net.latent = dict()     # rend | tran | real
    net.logits = dict()     # rend | tran | real
    net.instr  = dict()     # rend | tran | real
    net.resi_outs = dict()  # rend | tran | real
    net.activations = dict()
    is_train = params['is_train']
    
    

    # activations
    def store_act(name, target, activations):
        if name not in net.activations:
            net.activations[name] = dict()
        net.activations[name][target] = activations

    # input augmentation
    coords_res = int(params.get('coords_res', 20))
    batch_size = net.inputs['real'].get_shape()[0]
    t_canonical_coords, blk_size = create_canonical_coordinates(batch_size, 160, coords_res)
    coords_sigma = params.get('coords_sigma', 1.0) * blk_size * 0.2
    for key, t_img in net.inputs.items():
        net.resi_imgs_noaug[key] = t_img
        
        # for the RFI net
        if is_train and key.startswith('real') and params.get('local_warping', 0):
            # local warp augmentation
            with tf.variable_scope("input"):
                net.imgs[key], _, __ = oper_random_geo_perturb(t_img, t_canonical_coords, coords_sigma)
        else:
            net.imgs[key] = t_img # no augmentation

    # mean inputs and residuals
    net.mean_imgs = dict()
    for key, t_img in net.imgs.items():
        value = params.get('mean_' + key, 0.5)
        if isinstance(value, str):
            value = read_image(value)
            value = np.expand_dims(value, axis=0)
            print('mean image', value.shape)
        net.mean_imgs[key] = value
        net.resi_imgs[key] = t_img - value

        if is_train: # if training
            noise_sigma = params.get('noise_sigma', 3./255.)
            t_noise = tf.random_normal(tf.shape(t_img), stddev = noise_sigma)
            net.resi_imgs[key] = net.resi_imgs[key] + t_noise
        net.resi_imgs_noaug[key] = net.resi_imgs_noaug[key] - value

    # fake targets
    fakes = filter(lambda name: name != 'real' and name != 'unsup', t_imgs_dict.keys())

    # create generator
    with tf.variable_scope("generator"):

        def transformer(t_input, name):
            with runits('in_relu') as activations:
                t_gene_img = oper_img2img_bottleneck(t_input, 1, params=params, name='transformer_r2s')
                t_gene_img = tf.nn.tanh(t_gene_img)*0.5
                # 160x160
                net.resi_outs[name] = t_gene_img
                store_act(name, 'real2syn', activations)
            return t_gene_img

        def encoder(t_input, name):
            with runits('relu') as activations:
                t_logits = oper_img2prog_final_complex(t_input, params=params, name='img2prog')
                t_instr = tf.argmax(t_logits, axis=3, name="prediction")
                net.latent[name] = t_logits
                net.logits[name] = t_logits
                net.instr[name]  = tf.expand_dims(t_instr, axis=3)
                store_act(name, 'img2prog', activations)
            return t_logits

        # program synthesis (encoding)
        curdataname = 'real'
        fakesyn_img = transformer(net.resi_imgs[curdataname], curdataname)
        _ = encoder(fakesyn_img, curdataname)
        
        if is_train:
            # program synthesis (encoding)
            curdataname = 'rend'
            _ = encoder(net.resi_imgs[curdataname], curdataname)

            # CH0118_14
            curdataname = 'tran'
            if curdataname in net.resi_imgs.keys():
                fakesyn_img = transformer(net.resi_imgs[curdataname], curdataname)
                _ = encoder(fakesyn_img, curdataname)

    return net

def oper_img2prog_final(t_img, params = dict(), name='img2prog'):
    '''
    Translate image domain with resnet 160x160->20x20
    '''
    feat_ch = int(params.get('feat_ch', 64))
    rblk_num = int(params.get('rblk_num', 6))
    conv_type = params.get('conv_type', 'conv_pad')
    if conv_type == 'conv_pad':
        conv_fn = conv_pad
    elif conv_type == 'conv':
        conv_fn = conv
    else:
        raise ValueError('Unsupported convolution type %s' % conv_type)

    rblk = [resi, [[conv_fn, feat_ch], [runit], [conv_fn, feat_ch]]]
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        t_act1 = NN('resnet1',
            [t_img, 
            [conv_fn, feat_ch, 2], [runit, 'in_relu'],
            [conv_fn, feat_ch, 2], [runit, 'in_relu'],
            [conv_fn, feat_ch, 2], [runit, 'in_relu'],
            ])
        t_act2 = NN('resnet2',
            [t_act1, *[rblk for _ in range(rblk_num)]])
        t_out  = NN('resnet3',
            [tf.concat([t_act1, t_act2], -1),
                [conv_fn, feat_ch], [runit],
                [conv_fn, prog_ch, 1, 1]
            ])
    return t_out

def oper_img2img_bottleneck(t_img, out_ch, params, name='img2img'):
    '''
    Translate image domain with resnet 160x160->160x160
    '''
    feat_ch = int(params.get('feat_ch', 64))
    rblk_num = int(params.get('rblk_num', 6))
    conv_type = params.get('conv_type', 'conv_pad')
    if conv_type == 'conv_pad':
        conv_fn = conv_pad
    elif conv_type == 'conv':
        conv_fn = conv
    else:
        raise ValueError('Unsupported convolution type %s' % conv_type)

    rblk = [resi, [[conv_fn, feat_ch], [runit], [conv_fn, feat_ch]]]
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        t_act1 = NN('resnet1',
            [t_img, 
            [conv_fn, feat_ch, 2], [runit, 'in_relu'],
            [conv_fn, feat_ch, 2], [runit, 'in_relu'],
            ])
        t_act2 = NN('resnet2',
            [t_act1, *[rblk for _ in range(rblk_num)]])
        t_out  = NN('resnet3',
            [tf.concat([t_act1, t_act2], -1),
                [upsample],
                [conv_fn, feat_ch], [runit],
                [upsample],
                [conv_fn, feat_ch], [runit],
                [conv_fn, out_ch, 1, 1]
            ])
    return t_out


def model_composited_RFI_2(t_imgs_dict, t_labels_dict, params = dict()):
    '''
    Compose the full network model
    '''
    net = Parameters()
    net.inputs = t_imgs_dict
    net.imgs = dict()
    net.resi_imgs = dict()  # rend | tran | real
    net.resi_imgs_noaug = dict()  # rend | tran | real
    net.latent = dict()     # rend | tran | real
    net.logits = dict()     # rend | tran | real
    net.instr  = dict()     # rend | tran | real
    net.resi_outs = dict()  # rend | tran | real
    net.activations = dict()
    is_train = params['is_train']
    
    # activations
    def store_act(name, target, activations):
        if name not in net.activations:
            net.activations[name] = dict()
        net.activations[name][target] = activations

    # input augmentation
    coords_res = int(params.get('coords_res', 20))
    batch_size = net.inputs['real'].get_shape()[0]
    t_canonical_coords, blk_size = create_canonical_coordinates(batch_size, 160, coords_res)
    coords_sigma = params.get('coords_sigma', 1.0) * blk_size * 0.2
    for key, t_img in net.inputs.items():
        net.resi_imgs_noaug[key] = t_img
        
        # for the RFI net
        if is_train and key.startswith('real') and params.get('local_warping', 0):
            # local warp augmentation
            with tf.variable_scope("input"):
                net.imgs[key], _, __ = oper_random_geo_perturb(t_img, t_canonical_coords, coords_sigma)
        else:
            net.imgs[key] = t_img # no augmentation

    # mean inputs and residuals
    net.mean_imgs = dict()
    for key, t_img in net.imgs.items():
        value = params.get('mean_' + key, 0.5)
        if isinstance(value, str):
            value = read_image(value)
            value = np.expand_dims(value, axis=0)
            print('mean image', value.shape)
        net.mean_imgs[key] = value
        net.resi_imgs[key] = t_img - value

        if is_train: # if training
            noise_sigma = params.get('noise_sigma', 3./255.)
            t_noise = tf.random_normal(tf.shape(t_img), stddev = noise_sigma)
            net.resi_imgs[key] = net.resi_imgs[key] + t_noise
        net.resi_imgs_noaug[key] = net.resi_imgs_noaug[key] - value

    # fake targets
    fakes = filter(lambda name: name != 'real' and name != 'unsup', t_imgs_dict.keys())

    # create generator
    with tf.variable_scope("generator"):

        def transformer(t_input, name):
            with runits('in_relu') as activations:
                t_gene_img = oper_img2img_bottleneck(t_input, 1, params=params, name='transformer_r2s')
                t_gene_img = tf.nn.tanh(t_gene_img)*0.5
                # 160x160
                net.resi_outs[name] = t_gene_img
                store_act(name, 'real2syn', activations)
            return t_gene_img

        def encoder(t_input, name):
            with runits('relu') as activations:
                t_logits = oper_img2prog_final(t_input, params=params, name='img2prog')
                t_instr = tf.argmax(t_logits, axis=3, name="prediction")
                net.latent[name] = t_logits
                net.logits[name] = t_logits
                net.instr[name]  = tf.expand_dims(t_instr, axis=3)
                store_act(name, 'img2prog', activations)
            return t_logits

        # program synthesis (encoding)
        curdataname = 'real'
        fakesyn_img = transformer(net.resi_imgs[curdataname], curdataname)
        _ = encoder(fakesyn_img, curdataname)
        
        if is_train:
            # program synthesis (encoding)
            curdataname = 'rend'
            _ = encoder(net.resi_imgs[curdataname], curdataname)

            # CH0118_14
            curdataname = 'tran'
            if curdataname in net.resi_imgs.keys():
                fakesyn_img = transformer(net.resi_imgs[curdataname], curdataname)
                _ = encoder(fakesyn_img, curdataname)

    return net


def model_composited_RFI(t_imgs_dict, t_labels_dict, params = dict()):
    '''
    Compose the full network model
    '''
    net = Parameters()
    net.inputs = t_imgs_dict
    net.imgs = dict()
    net.resi_imgs = dict()  # rend | tran | real
    net.resi_imgs_noaug = dict()  # rend | tran | real
    net.latent = dict()     # rend | tran | real
    net.logits = dict()     # rend | tran | real
    net.instr  = dict()     # rend | tran | real
    net.resi_outs = dict()  # rend | tran | real
    net.activations = dict()
    is_train = params['is_train']
    
    # activations
    def store_act(name, target, activations):
        if name not in net.activations:
            net.activations[name] = dict()
        net.activations[name][target] = activations

    # input augmentation
    coords_res = int(params.get('coords_res', 20))
    batch_size = net.inputs['real'].get_shape()[0]
    t_canonical_coords, blk_size = create_canonical_coordinates(batch_size, 160, coords_res)
    coords_sigma = params.get('coords_sigma', 1.0) * blk_size * 0.2
    for key, t_img in net.inputs.items():
        net.resi_imgs_noaug[key] = t_img
        
        # for the RFI net
        if is_train and key.startswith('real') and params.get('local_warping', 0):
            # local warp augmentation
            with tf.variable_scope("input"):
                net.imgs[key], _, __ = oper_random_geo_perturb(t_img, t_canonical_coords, coords_sigma)
        else:
            net.imgs[key] = t_img # no augmentation

    # mean inputs and residuals
    net.mean_imgs = dict()
    for key, t_img in net.imgs.items():
        value = params.get('mean_' + key, 0.5)
        if isinstance(value, str):
            value = read_image(value)
            value = np.expand_dims(value, axis=0)
            print('mean image', value.shape)
        net.mean_imgs[key] = value
        net.resi_imgs[key] = t_img - value

        if is_train: # if training
            noise_sigma = params.get('noise_sigma', 3./255.)
            t_noise = tf.random_normal(tf.shape(t_img), stddev = noise_sigma)
            net.resi_imgs[key] = net.resi_imgs[key] + t_noise
        net.resi_imgs_noaug[key] = net.resi_imgs_noaug[key] - value

    # fake targets
    fakes = filter(lambda name: name != 'real' and name != 'unsup', t_imgs_dict.keys())

    # create generator
    with tf.variable_scope("generator"):

        def transformer(t_input, name):
            with runits('in_relu') as activations:
                t_gene_img = oper_img2img(t_input, 1, params=params, name='transformer_r2s')
                t_gene_img = tf.nn.tanh(t_gene_img)*0.5
                # 160x160
                net.resi_outs[name] = t_gene_img
                store_act(name, 'real2syn', activations)
            return t_gene_img

        def encoder(t_input, name):
            with runits('relu') as activations:
                t_logits = oper_img2img(t_input, prog_ch, params=params, name='img2prog')
                t_logits = tf.contrib.layers.avg_pool2d(t_logits, [8,8], 8)
                t_instr = tf.argmax(t_logits, axis=3, name="prediction")
                net.latent[name] = t_logits
                net.logits[name] = t_logits
                net.instr[name]  = tf.expand_dims(t_instr, axis=3)
                store_act(name, 'img2prog', activations)
            return t_logits

        # program synthesis (encoding)
        curdataname = 'real'
        fakesyn_img = transformer(net.resi_imgs[curdataname], curdataname)
        _ = encoder(fakesyn_img, curdataname)
        
        if is_train:
            # program synthesis (encoding)
            curdataname = 'rend'
            _ = encoder(net.resi_imgs[curdataname], curdataname)

            # CH0118_14
            curdataname = 'tran'
            fakesyn_img = transformer(net.resi_imgs[curdataname], curdataname)
            _ = encoder(fakesyn_img, curdataname)

    return net



def total_loss_RFI(net, t_inst_dict, params = dict()):
    loss_dict_Disc = dict()
    loss_dict_Gene = dict()
    metrics = dict()

    # replay switch
    replay_worst = params.get('replay_worst', 0)

    # extract instructions
    t_inst_real = t_inst_dict['instr_real']
    if 'instr_synt' in t_inst_dict.keys():
        t_inst_synt = t_inst_dict['instr_synt']
    if replay_worst:
        t_inst_wors = t_inst_dict['worst']

    # get dimensions
    batch_size, h, w, _ = net.imgs['real'].get_shape()

    # fake targets
    fakes = filter(lambda name: name != 'real' and name != 'unsup' and name != 'worst', net.imgs.keys())


    # replay switch
    replay_worst = params.get('replay_worst', 0)

    # instruction masking and weighting
    ones = tf.ones_like(t_inst_synt, tf.float32)
    zeros = tf.zeros_like(t_inst_synt, tf.float32)
    bg_type = params.get('bg_type', 'global')
    t_bg_synt = tf_background(t_inst_synt, bg_type)
    t_bg_real = tf_background(t_inst_real, bg_type)
    if replay_worst:
        t_bg_wors = tf_background(t_inst_wors, bg_type)
    t_synt_mask = tf.where(t_bg_synt, zeros, ones)
    t_real_mask = tf.where(t_bg_real, zeros, ones)
    if replay_worst:
        t_wors_mask = tf.where(t_bg_wors, zeros, ones)
    bg_weight = params.get('bg_weight', 0.1)
    if isinstance(bg_weight, str):
        masked = bg_weight.startswith('mask_')
        if masked:
            bg_weight = bg_weight[5:]
        t_synt_weight = tf_frequency_weight(t_inst_synt, bg_weight)
        t_real_weight = tf_frequency_weight(t_inst_real, bg_weight)
        if replay_worst:
            t_wors_weight = tf_frequency_weight(t_inst_wors, bg_weight)
        if masked:
            t_synt_weight = tf.where(t_bg_synt, 0.1 * t_synt_weight, t_synt_weight)
            t_real_weight = tf.where(t_bg_real, 0.1 * t_real_weight, t_real_weight)
            if replay_worst:
                t_wors_weight = tf.where(t_bg_wors, 0.1 * t_wors_weight, t_wors_weight)
    else:
        t_synt_weight = tf.where(t_bg_synt, bg_weight * ones, ones)
        t_real_weight = tf.where(t_bg_real, bg_weight * ones, ones)
        if replay_worst:
            t_wors_weight = tf.where(t_bg_wors, bg_weight * ones, ones)
    t_simg_weight = tf.image.resize_bilinear(t_synt_weight, [h, w])
    t_rimg_weight = tf.image.resize_bilinear(t_real_weight, [h, w])
    if replay_worst:
        t_wimg_weight = tf.image.resize_bilinear(t_wors_weight, [h, w])

    # store background for debugging
    net.bg = dict()
    net.bg['synt'] = t_bg_synt
    net.bg['real'] = t_bg_real
    if replay_worst:
        net.bg['worst'] = t_bg_wors
    
    # create discriminator networks if needed for loss
    net.discr = {
        # 'instr': dict(), 
        # 'latent': dict(), 
        'image': dict(),
        #'wgan_grad': dict()
    }

     # summon VGG19
    if params.get('bvggloss', 0):
        if params.get('vgg16or19', '16') == '16':
            net.vggobj = custom_vgg19.Vgg16()
        else:
            net.vggobj = custom_vgg19.Vgg19()
        net.vgg = dict()

        # GT synthetic
        curdataname = 'rend'
        net.vgg['gt_' + curdataname] = net.vggobj.build(net.resi_imgs[curdataname])
        curdataname = 'real'
        net.vgg['gt_' + curdataname] = net.vggobj.build(net.resi_imgs[curdataname])
        # generated data
        curdataname = 'real'
        net.vgg[curdataname] = net.vggobj.build(net.resi_outs[curdataname])
        curdataname = 'tran'
        net.vgg[curdataname] = net.vggobj.build(net.resi_outs[curdataname])

    if params.get('discr_img', 0):
        with tf.variable_scope("discriminator"):
            # GT synthetic
            curdataname = 'rend'
            t_domain = discriminator_cond(net.resi_imgs[curdataname], 
                                            t_inst_synt,
                                            params, name="image_domain")
            net.discr['image']['gt_' + curdataname] = t_domain

            curdataname = 'real'
            t_domain = discriminator_cond(net.resi_imgs[curdataname], 
                                            t_inst_real,
                                            params, name="image_domain")
            net.discr['image']['gt_' + curdataname] = t_domain

            # generated data
            curdataname = 'tran'
            t_domain = discriminator_cond(net.resi_outs[curdataname], 
                                            t_inst_synt,
                                            params, name="image_domain")
            net.discr['image'][curdataname] = t_domain

            curdataname = 'real'
            t_domain = discriminator_cond(net.resi_outs[curdataname], 
                                            t_inst_real,
                                            params, name="image_domain")
            net.discr['image'][curdataname] = t_domain


    # generator and discriminator losses
    with tf.variable_scope("loss"):

        # adversarial loss for image
        discr_type = params.get('discr_type', 'l2')
        name = 'gt_rend'
        t_discr = net.discr['image'][name]
        loss_dis = tf_loss_with_select(t_discr, tf.ones_like(t_discr), which_loss = discr_type)
        loss_dict_Disc['loss_D_image/' + name] = loss_dis
        name = 'gt_real'
        t_discr = net.discr['image'][name]
        loss_dis = tf_loss_with_select(t_discr, -tf.ones_like(t_discr), which_loss = discr_type)
        loss_dict_Disc['loss_D_image/' + name] = loss_dis/3.
        name = 'tran'
        t_discr = net.discr['image'][name]
        loss_dis = tf_loss_with_select(t_discr, -tf.ones_like(t_discr), which_loss = discr_type)
        loss_dict_Disc['loss_D_image/' + name] = loss_dis/3.
        name = 'real'
        t_discr = net.discr['image'][name]
        loss_dis = tf_loss_with_select(t_discr, -tf.ones_like(t_discr), which_loss = discr_type)
        loss_dict_Disc['loss_D_image/' + name] = loss_dis/3.

        name = 'tran'
        t_discr = net.discr['image'][name]
        loss_gen = tf_loss_with_select(t_discr, tf.ones_like(t_discr), which_loss = discr_type)
        loss_dict_Gene['loss_G_image/' + name] = loss_gen 

        name = 'real'
        t_discr = net.discr['image'][name]
        loss_gen = tf_loss_with_select(t_discr, tf.ones_like(t_discr), which_loss = discr_type)
        loss_dict_Gene['loss_G_image/' + name] = loss_gen 
        

        def fn_downsize(images):
            smoother = Smoother({'data':images}, 11, 2.)
            images = smoother.get_output()
            return tf.image.resize_bilinear(images, [20,20])
        
        if params.get('discr_img', 1):
            curdataname = 'real'
            ae_loss_type = params.get('ae_loss_type', 'smooth_l1')
            loss_unsup = tf_loss_with_select(
                                fn_downsize(
                                    net.resi_imgs[curdataname]
                                ), 
                                fn_downsize(
                                    net.resi_outs[curdataname]
                                ),
                            which_loss = ae_loss_type)
            # loss_dict_Gene['loss_unsup/rough_real_vs_freal'] = loss_unsup

            loss_unsup = tf_loss_with_select(
                                    net.resi_imgs['rend'], 
                                    net.resi_outs['tran'],
                            which_loss = ae_loss_type)
            # loss_unsup = tf_loss_with_select(net.activations['unsup']['img2prog'][2],
            #                                  net.activations['unsup_feedback']['img2prog'][2],
            #                                  which_loss='smooth_l1')
            loss_dict_Gene['loss_unsup/rough_gtrend_vs_ftrans'] = 100.*loss_unsup

        # VGG perceptual loss
        if params.get('bvggloss', 0):
            curlayer = 'pool3'
            loss_perc_pool2 = 1.*tf_loss_with_select(
                                (1./128.)*net.vgg['gt_real'][curlayer], 
                                (1./128.)*net.vgg['real'][curlayer], 
                                which_loss = 'l2')
            loss_dict_Gene['loss_vgg_percept/' + curlayer] = loss_perc_pool2*0.25 
            # normalize by the number of combinations (real, unsuper, conv2_2, pool3)

            curlayer = 'pool3'
            loss_perc_pool2 = 1.*tf_loss_with_select(
                                (1./128.)*net.vgg['gt_rend'][curlayer], 
                                (1./128.)*net.vgg['tran'][curlayer], 
                                which_loss = 'l2')
            loss_dict_Gene['loss_vgg_percept/' + curlayer] = loss_perc_pool2*0.25 

            # VGG style losses 
            # applied to {synt, real} except {unsup}
            
            lst_lweight = [0.3, 0.5, 1.]
            lst_layers = ['conv1_2', 'conv2_2', 'conv3_3']
            no_gram_layers = float(len(lst_layers))
            for gram_layer, gram_weight in zip(lst_layers, lst_lweight):
                loss_prefix = 'loss_vgg_percept/' + 'gram_' + gram_layer + 'real'

                lst_gts = ['rend',] #['real', 'unsup']
                no_gts = float(len(lst_gts))
                for gts in lst_gts:
                    t_real = net.vgg['gt_rend'][gram_layer]/128.
                    t_synt = net.vgg['real'][gram_layer]/128.
                    t_loss = style_layer_loss(t_real, t_synt, params.get('gram_power', 2))
                    loss_dict_Gene[loss_prefix + 'real2rend'] = gram_weight*t_loss/(no_gram_layers*no_gts)

        # instruction x-entropy
        # applied to {*real*} including {rend_real, tran_real, real_real, real, real_feedback...}
        nsynthetic = len(net.logits.keys()) - 1.
        for name, t_logits in net.logits.items():
            if name.startswith('unsup'): # name.endswith('_real') or 
                continue # adapter network doesn't use entropy

            if name.startswith('real'):
                # name == 'real' or name == 'real_feedback' or name == 'real_gen' or name == 'real_gen_feedback':
                t_instr  = t_inst_real
                t_weight = t_real_weight
            else:
                t_instr  = t_inst_synt
                t_weight = t_synt_weight

            if params.get('bMILloss', 1) and name.startswith('real'):
                loss_xentropy = tf_MILloss_xentropy(labels = tf.squeeze(t_instr),
                                                    logits = t_logits,
                                                    weight = t_weight)
                                    #                      + \
                                    # tf_loss_xentropy(
                                    #                 labels = tf.squeeze(t_instr),
                                    #                 logits = t_logits,
                                    #                 weight = t_weight)[:,1:-1,1:-1,tf.newaxis]*0.5
                loss_xentropy *= (1.-params.get('mix_alpha', 0.5))*3.
            else:
                loss_xentropy = tf_loss_xentropy(
                                            labels = tf.squeeze(t_instr),
                                            logits = t_logits,
                                            weight = t_weight)
                loss_xentropy *= params.get('mix_alpha', 0.5)*3./float(nsynthetic)

            if 'feedback' in name:
                loss_prefix = 'loss_feedback/'
            else:
                loss_prefix = 'loss_xentropy/'
            loss_dict_Gene[loss_prefix + name] = loss_xentropy

        # syntax loss
        # applied to {all} except {unsup}
        syntax_binary = params.get('syntax_binary', 0)
        for name in net.instr.keys():
            if name.startswith('unsup') and not params.get('unsup_syntax', 0): # skip
                continue

            if syntax_binary:
                t_instr = net.instr[name]
            else:
                t_instr = net.logits[name]
            loss_syn = syntax_loss(t_instr, params, syntax_binary)
            loss_dict_Gene['loss_syntax/' + name] = loss_syn

        # accuracy measurements
        net.acc = { 'full' : dict(), 'fg': dict() }
        # applied to {all} except {unsup}
        for name, t_instr in net.instr.items():
            if name.startswith('unsup'):
                continue

            if name == 'real':
                t_label = t_inst_real
                t_mask  = t_real_mask
            else:
                t_label = t_inst_synt
                t_mask  = t_synt_mask

            if params.get('bMILloss', 1):
                # full accuracy (includes bg)
                metrics['accuracy/' + name], acc_batch    = tf_MILloss_accuracy(t_label, t_instr)
                # fg accuracy
                metrics['accuracy_fg/' + name], fac_batch = tf_MILloss_accuracy(t_label, t_instr, t_mask)

                # storing batch information for worst sample mining
                net.acc['full'][name] = acc_batch
                net.acc['fg'][name]   = fac_batch
            else:
                # full accuracy (includes bg)
                metrics['accuracy/' + name] = tf_accuracy(t_label, t_instr)
                # fg accuracy
                metrics['accuracy_fg/' + name] = tf_accuracy(t_label, t_instr, t_mask)

            metrics['confusionmat/' + name] = comp_confusionmat(t_instr, 
                                                                t_label, 
                                                                num_classes = prog_ch, 
                                                                normalized_row = True,
                                                                name=name)
    return loss_dict_Disc, loss_dict_Gene, metrics



def model_composited(t_imgs_dict, t_labels_dict, params = dict()):
    '''
    Compose the full network model
    '''

    net = Parameters()
    net.inputs = t_imgs_dict
    net.imgs = dict()
    net.resi_imgs = dict()  # rend | tran | real
    net.resi_imgs_noaug = dict()  # rend | tran | real
    net.latent = dict()     # rend | tran | real
    net.logits = dict()     # rend | tran | real
    net.instr  = dict()     # rend | tran | real
    net.resi_outs = dict()  # rend | tran | real
    net.activations = dict()
    is_train = params['is_train']

    # activations
    def store_act(name, target, activations):
        if name not in net.activations:
            net.activations[name] = dict()
        net.activations[name][target] = activations

    # input augmentation
    coords_res = int(params.get('coords_res', 20))
    batch_size = net.inputs['real'].get_shape()[0]
    t_canonical_coords, blk_size = create_canonical_coordinates(batch_size, 160, coords_res)
    coords_sigma = params.get('coords_sigma', 1.0) * blk_size * 0.2
    for key, t_img in net.inputs.items():
        net.resi_imgs_noaug[key] = t_img
        if is_train and params.get('local_warping', 0):
            # local warp augmentation
            with tf.variable_scope("input"):
                net.imgs[key], _, __ = oper_random_geo_perturb(t_img, t_canonical_coords, coords_sigma)
        else:
            net.imgs[key] = t_img # no augmentation

    # mean inputs and residuals
    net.mean_imgs = dict()
    for key, t_img in net.imgs.items():
        value = params.get('mean_' + key, 0.5)
        if isinstance(value, str):
            value = read_image(value)
            value = np.expand_dims(value, axis=0)
            print('mean image', value.shape)
        net.mean_imgs[key] = value
        net.resi_imgs[key] = t_img - value

        if is_train: # if training
            noise_sigma = params.get('noise_sigma', 3./255.)
            t_noise = tf.random_normal(tf.shape(t_img), stddev = noise_sigma)
            net.resi_imgs[key] = net.resi_imgs[key] + t_noise
        
        net.resi_imgs_noaug[key] = net.resi_imgs_noaug[key] - value

    # fake targets
    fakes = filter(lambda name: name != 'real' and name != 'unsup', t_imgs_dict.keys())

    # create generator
    with tf.variable_scope("generator"):

        def encoder(t_input, name):
            with runits('relu') as activations:
                t_logits = oper_img2img(t_input, prog_ch, params=params, name='img2prog')
                t_logits = tf.contrib.layers.avg_pool2d(t_logits, [8,8], 8)
                t_instr = tf.argmax(t_logits, axis=3, name="prediction")
                net.latent[name] = t_logits
                net.logits[name] = t_logits
                net.instr[name]  = tf.expand_dims(t_instr, axis=3)
                store_act(name, 'img2prog', activations)
            return t_logits
        
        feedback = params.get('feedback', 0) if is_train else 0
        decoding = params.get('decoder', 0)
        for name, t_resi_inp in net.resi_imgs.items():
            
            # program synthesis (encoding)
            t_latent = encoder(t_resi_inp, name)

            # check we want the decoder
            if not decoding:
                continue


    return net


def total_loss(net, t_inst_synt, t_inst_real, params = dict()):
    loss_dict_Disc = dict()
    loss_dict_Gene = dict()
    metrics = dict()

    # get dimensions
    batch_size, h, w, _ = net.imgs['real'].get_shape()

    # fake targets
    fakes = filter(lambda name: name != 'real' and name != 'unsup', net.imgs.keys())

    # instruction masking and weighting
    ones = tf.ones_like(t_inst_synt, tf.float32)
    zeros = tf.zeros_like(t_inst_synt, tf.float32)
    bg_type = params.get('bg_type', 'global')
    t_bg_synt = tf_background(t_inst_synt, bg_type)
    t_bg_real = tf_background(t_inst_real, bg_type)
    t_synt_mask = tf.where(t_bg_synt, zeros, ones)
    t_real_mask = tf.where(t_bg_real, zeros, ones)
    bg_weight = params.get('bg_weight', 0.1)
    if isinstance(bg_weight, str):
        masked = bg_weight.startswith('mask_')
        if masked:
            bg_weight = bg_weight[5:]
        t_synt_weight = tf_frequency_weight(t_inst_synt, bg_weight)
        t_real_weight = tf_frequency_weight(t_inst_real, bg_weight)
        if masked:
            t_synt_weight = tf.where(t_bg_synt, 0.1 * t_synt_weight, t_synt_weight)
            t_real_weight = tf.where(t_bg_real, 0.1 * t_real_weight, t_real_weight)
    else:
        t_synt_weight = tf.where(t_bg_synt, bg_weight * ones, ones)
        t_real_weight = tf.where(t_bg_real, bg_weight * ones, ones)
    t_simg_weight = tf.image.resize_bilinear(t_synt_weight, [h, w])
    t_rimg_weight = tf.image.resize_bilinear(t_real_weight, [h, w])


    # store background for debugging
    net.bg = dict()
    net.bg['synt'] = t_bg_synt
    net.bg['real'] = t_bg_real

    # create discriminator networks if needed for loss
    net.discr = {
        'instr': dict(), 'latent': dict(), 'image': dict(),
        #'wgan_grad': dict()
    }
    net.resi_aug_wgan = dict()

    # generator and discriminator losses
    with tf.variable_scope("loss"):
        
        # instruction x-entropy
        # applied to {*real*} including {rend_real, tran_real, real_real, real, real_feedback...}
        for name, t_logits in net.logits.items():
            if name.startswith('unsup'): # name.endswith('_real') or 
                continue # adapter network doesn't use entropy

            if (re.search('real', name) is not None) or (not params.get('adapter',0)):
                if name.startswith('real'):
                    # name == 'real' or name == 'real_feedback' or name == 'real_gen' or name == 'real_gen_feedback':
                    t_instr  = t_inst_real
                    t_weight = t_real_weight
                else:
                    t_instr  = t_inst_synt
                    t_weight = t_synt_weight

                if params.get('bMILloss', 1) and name.startswith('real'):
                    loss_xentropy = tf_MILloss_xentropy(labels = tf.squeeze(t_instr),
                                                        logits = t_logits,
                                                        weight = t_weight)
                                    #                      + \
                                    # tf_loss_xentropy(
                                    #                 labels = tf.squeeze(t_instr),
                                    #                 logits = t_logits,
                                    #                 weight = t_weight)[:,1:-1,1:-1,tf.newaxis]*0.5
                else:
                    loss_xentropy = tf_loss_xentropy(
                                                labels = tf.squeeze(t_instr),
                                                logits = t_logits,
                                                weight = t_weight)

                if 'feedback' in name:
                    loss_prefix = 'loss_feedback/'
                else:
                    loss_prefix = 'loss_xentropy/'
                loss_dict_Gene[loss_prefix + name] = loss_xentropy

        # syntax loss
        # applied to {all} except {unsup}
        syntax_binary = params.get('syntax_binary', 0)
        for name in net.instr.keys():
            if name.startswith('unsup') and not params.get('unsup_syntax', 0): # skip
                continue

            if syntax_binary:
                t_instr = net.instr[name]
            else:
                t_instr = net.logits[name]
            loss_syn = syntax_loss(t_instr, params, syntax_binary)
            loss_dict_Gene['loss_syntax/' + name] = loss_syn

        # accuracy measurements
        net.acc = { 'full' : dict(), 'fg': dict() }
        # applied to {all} except {unsup}
        for name, t_instr in net.instr.items():
            if name.startswith('unsup'):
                continue
                
            if name == 'real':
                t_label = t_inst_real
                t_mask  = t_real_mask
            else:
                t_label = t_inst_synt
                t_mask  = t_synt_mask

            if params.get('bMILloss', 1):
                # full accuracy (includes bg)
                metrics['accuracy/' + name], acc_batch    = tf_MILloss_accuracy(t_label, t_instr)
                # fg accuracy
                metrics['accuracy_fg/' + name], fac_batch = tf_MILloss_accuracy(t_label, t_instr, t_mask)

                # storing batch information for worst sample mining
                net.acc['full'][name] = acc_batch
                net.acc['fg'][name]   = fac_batch
            else:
                # full accuracy (includes bg)
                metrics['accuracy/' + name] = tf_accuracy(t_label, t_instr)
                # fg accuracy
                metrics['accuracy_fg/' + name] = tf_accuracy(t_label, t_instr, t_mask)
            
            metrics['confusionmat/' + name] = comp_confusionmat(t_instr, 
                                                                t_label, 
                                                                num_classes = prog_ch, 
                                                                normalized_row = True,
                                                                name=name)


    return loss_dict_Disc, loss_dict_Gene, metrics
