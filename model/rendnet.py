from __future__ import division
import os
import math
import time
import tensorflow as tf
import numpy as np
import scipy
import re
import pdb
import tensorflow.contrib.slim as slim
from .nnlib import *
from .parameters import arch_para, hparams, Parameters
from util import read_image, comp_confusionmat

from .tensorflow_vgg import custom_vgg19
from .layer_modules import prog_ch, tf_background, oper_prog2img

def dense_prog2img(t_input, params, name = 'prog2img'):
    feat_ch = int(params.get('feat_ch', 64))
    convfn = lambda x,n: conv_pad(x, n)
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        features = []
        with tf.variable_scope('decoder1', reuse = False):
            # pdb.set_trace()
            t_feat = runit(convfn(t_input, feat_ch))
            for i in range(3):
                features.append(upsample(t_feat, True, 8 // (2 ** i)))
                t_feat = runit(convfn(upsample(t_feat), feat_ch))
            features.append(t_feat)
        t_img = NN('decoder2',
            [tf.concat(features, axis = 3),
                [convfn,feat_ch], [runit],
                [convfn,1]
            ])
        return tf.tanh(t_img) * 0.5 + 0.5

def resnet_prog2img(t_input, params, name = 'prog2img'):
    feat_ch = int(params.get('feat_ch', 64))
    convfn = lambda x,n: conv2d(x, n, 3, 1)
    rblk = [resi, [[convfn, feat_ch], [runit], [convfn, feat_ch]]]
    rblk_num = int(params.get('rblk_num', 3))
    with tf.name_scope('upsampling'):
        t_input = upsample(t_input, False, 8)
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        t_feat1 = NN('decoder1',
            [t_input, [convfn, feat_ch], [runit]])
        t_feat2 = NN('decoder2',
            [t_feat1, *[rblk for i in range(rblk_num)]])
        t_img = NN('decoder3',
            [tf.concat([t_feat1, t_feat2], axis = 3),
                [convfn, feat_ch], [runit],
                [convfn, 1]])
        return tf.tanh(t_img) * 0.5 + 0.5

def network(t_label, params = dict(), output_layers = False, input_is_softmax = False, scope_name = 'renderer'):
    noise_level = params.get('noise_level', 0.45)
    render_type = params.get('render_type', 'dense')
    render_layers = []

    with tf.variable_scope(scope_name):
        # convert to probabilistic one-hot input
        if input_is_softmax:
            t_input = t_label
        else:
          # must transform into one-hot probability distribution
            with tf.name_scope('input_probability'):
                t_onehot = tf.one_hot(tf.squeeze(t_label, axis = [3]), 17)
                t_noise  = tf.random_uniform(t_onehot.shape,
                    minval = -noise_level, maxval = noise_level, dtype = tf.float32) if noise_level > 0 else 0
                t_input  = tf.nn.softmax(t_onehot + t_noise)

        # create network
        with runits('lrelu') as activations:
            if render_type == 'resnet':
                t_img = resnet_prog2img(t_input, params)
            elif render_type == 'dense':
                t_img = dense_prog2img(t_input, params)
            elif render_type == 'simple':
                t_img = oper_prog2img(t_input, params)
                t_img = tf.tanh(t_img) * 0.5 + 0.5 # part of other prog2img, but not base one
            else:
                raise ValueError('Invalid render type %s' % render_type)
            render_layers = activations
    if output_layers:
        return t_img, render_layers
    else:
        return t_img

def load_weights(sess, render_type):
    render_weights = render_type
    if '_' in render_type:
        render_type = render_type.split('_')[0]

    # load network weights
    model_file = os.path.join(
        os.path.dirname(__file__), 'renderer',
        render_weights + '.npy'
    )
    var_data = np.load(model_file).item()

    # remap generator name to renderer
    var_data = {
        key.replace('generator', 'renderer'): value
        for key, value in var_data.items()
    }

    # load network variables into session
    assignments = []
    for var in tf.get_collection('trainable_variables'):
        if var.name not in var_data:
            continue
        value  = var_data[var.name]
        assign = var.assign(value, read_value = False)
        assignments.append(assign)
    _ = sess.run(assignments)
    print('Loaded rendnet/%s with %d variables' % (render_type, len(assignments)))


def model_composited(t_labels_dict, t_imgs_dict, params = dict()):
    '''
    Compose the rendering network model
    '''

    # clear elements we don't need
    del t_labels_dict['instr_real']
    del t_imgs_dict['real']

    net = Parameters()
    net.inputs = t_imgs_dict
    net.imgs = t_imgs_dict
    net.resi_imgs = dict()
    net.resi_imgs_noaug = dict()
    net.latent = dict()
    net.logits = dict()
    net.instr  = t_labels_dict
    net.resi_outs = dict()
    net.activations = dict()
    net.mean_imgs = dict()

    noise_level = params.get('noise_level', 0.45)

    # activations
    def store_act(name, target, activations):
        if name not in net.activations:
            net.activations[name] = dict()
        net.activations[name][target] = activations

    # create generator
    with tf.variable_scope("generator"):

        # store label as instruction (with extra singleton dimension)
        t_label = net.instr['instr_synt']

        # convert to probabilistic one-hot input
        with tf.name_scope('input_probability'):
            t_onehot = tf.one_hot(tf.squeeze(t_label, axis = [3]), 17)
            t_noise  =  tf.random_uniform(t_onehot.shape,
                minval = -noise_level, maxval = noise_level, dtype = tf.float32) if noise_level > 0 else 0
            t_input  = tf.nn.softmax(t_onehot + t_noise)

        print('**************oper_prog2img')
        with runits('lrelu') as activations:

            render_type = params.get('render_type', 'simple')
            if render_type == 'resnet':
                t_img = resnet_prog2img(t_input, params)
            elif render_type == 'dense':
                t_img = dense_prog2img(t_input, params)
            elif render_type == 'simple':
                t_img = oper_prog2img(t_input, params)
                t_img = tf.tanh(t_img) * 0.5 + 0.5 # part of other prog2img, but not base one
            else:
                raise ValueError('Invalid render type %s' % render_type)

            net.resi_outs['rend'] = t_img
            store_act('rend', 'prog2img', activations)
    return net

def conv2d(input_, output_dim, ks=3, s=2, stddev=0.02, padding='VALID', name="conv2d"):

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        padsz = math.ceil((ks - s) * 0.5)
        input_ = tf.pad(input_,
            tf.constant([[0, 0], [padsz, padsz], [padsz, padsz], [0, 0]]),
            mode='SYMMETRIC')
        return slim.conv2d(input_, output_dim, ks, s, padding=padding,
            activation_fn=None,
            weights_initializer=tf.truncated_normal_initializer(stddev=stddev)
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
        h5 = conv2d(h3, 1, s=1, name='d_h4_pred')
        # h4 is (32 x 32 x 1)

        return h5

def total_loss(net, t_imgs_dict, params = dict()):
    loss_dict_Disc = dict()
    loss_dict_Gene = dict()
    metrics = dict()

    # extract instructions
    t_inst_synt = net.instr['instr_synt']
    # extract images
    t_gt  = net.imgs['rend']
    t_out = net.resi_outs['rend']

    # get dimensions
    batch_size, h, w, _ = t_gt.get_shape()

    # instruction masking and weighting
    ones = tf.ones_like(t_inst_synt, tf.float32)
    zeros = tf.zeros_like(t_inst_synt, tf.float32)
    bg_type = params.get('bg_type', 'global')
    t_bg_synt = tf_background(t_inst_synt, bg_type)
    t_synt_mask = tf.where(t_bg_synt, zeros, ones)
    bg_weight = params.get('bg_weight', 0.1)
    if isinstance(bg_weight, str):
        masked = bg_weight.startswith('mask_')
        if masked:
            bg_weight = bg_weight[5:]
        t_synt_weight = tf_frequency_weight(t_inst_synt, bg_weight)
        if masked:
            t_synt_weight = tf.where(t_bg_synt, 0.1 * t_synt_weight, t_synt_weight)
    else:
        t_synt_weight = tf.where(t_bg_synt, bg_weight * ones, ones)
    t_simg_weight = tf.image.resize_bilinear(t_synt_weight, [h, w])

    # store background for debugging
    net.bg = dict()
    net.bg['synt'] = t_bg_synt

    # create discriminator networks if needed for loss
    net.discr = {
        'image': dict(),
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
        net.vgg['gt_' + curdataname] = net.vggobj.build(t_gt)
        # generated synthetic
        net.vgg[curdataname] = net.vggobj.build(t_out)

    if params.get('discr_img', 0):
        with tf.variable_scope("discriminator"):
            # GT synthetic
            curdataname = 'rend'
            t_domain = discriminator(t_gt, params, name="image_domain")
            net.discr['image']['gt_' + curdataname] = t_domain
            # generated synthetic
            t_domain = discriminator(t_out, params, name="image_domain")
            net.discr['image'][curdataname] = t_domain

    # generator and discriminator losses
    with tf.variable_scope("loss"):

        # adversarial loss for image
        discr_type = params.get('discr_type', 'l2')
        for name, t_discr in net.discr['image'].items():
            if name.startswith('gt_'): #name == 'gt_rend':
                loss_dis = tf_loss_with_select(t_discr, tf.ones_like(t_discr), which_loss = discr_type)
                loss_dict_Disc['loss_D_image/' + name] = loss_dis
            else: # 'rend' (prog2img)
                loss_dis = tf_loss_with_select(t_discr, -tf.ones_like(t_discr), which_loss = discr_type)
                loss_dict_Disc['loss_D_image/' + name] = loss_dis
                loss_gen = tf_loss_with_select(t_discr, tf.ones_like(t_discr), which_loss = discr_type)
                loss_dict_Gene['loss_G_image/' + name] = loss_gen 
            print(name)

        def fn_downsize(images):
            smoother = Smoother({'data':images}, 11, 2.)
            images = smoother.get_output()
            return tf.image.resize_bilinear(images, [20,20])
        
        # VGG perceptual loss
        # TODO: style loss (Gram) needs to be added
        if params.get('bvggloss', 0):

            curdataname = 'rend'
            net.vgg['gt_' + curdataname]
            net.vgg[curdataname]
            
            curlayer = 'conv2_2'
            loss_perc_pool2 = 0.1*tf_loss_with_select(
                                (1./128.)*net.vgg['gt_rend'][curlayer], 
                                (1./128.)*net.vgg['rend'][curlayer], 
                                which_loss = 'l2')
            loss_dict_Gene['loss_vgg_percept/' + curlayer] = loss_perc_pool2*0.25 
            # normalize by the number of combinations (real, unsuper, conv2_2, pool3)

            curlayer = 'pool3'
            loss_perc_pool5 = 1.*tf_loss_with_select(
                                (1./128.)*net.vgg['gt_rend'][curlayer], 
                                (1./128.)*net.vgg['rend'][curlayer], 
                                which_loss = 'l2')
            loss_dict_Gene['loss_vgg_percept/' + curlayer] = loss_perc_pool5*0.25

        # Image loss
        render_loss_type = params.get('render_loss_type', 'smooth_l1')
        loss_dict_Gene['loss_rendering'] = tf_loss_with_select(
            t_gt, t_out, which_loss = render_loss_type, weight = t_simg_weight)

        # accuracy measurements
        net.acc = { 'full' : dict(), 'fg': dict() }

    return loss_dict_Disc, loss_dict_Gene, metrics
