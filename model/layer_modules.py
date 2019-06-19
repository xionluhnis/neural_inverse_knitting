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
from .spatial_transformer import transformer
from util import read_image, comp_confusionmat

from .tensorflow_vgg import custom_vgg19

prog_ch = 17
nch_res = arch_para.resblk_DimCh
resblk = [resi, [[conv, nch_res], [runit], [conv, nch_res]]]
fn_clipping01 = lambda tensor: tf.fake_quant_with_min_max_args(tensor, min=0., max=1., num_bits=8)
fn_clipping005 = lambda tensor: tf.fake_quant_with_min_max_args(tensor, min=-0.5, max=0.5, num_bits=8)
fn_smooth_l1 = lambda tensor: tf.sqrt(tf.square(tensor + 1e-8))

def oper_random_geo_perturb(t_imgs, t_canonical_coords, sigma=1., name='geopert_aug_layer'):
    '''
    Applies random warping to image;
    outputs warped image, perturbed coordinates and flow field
    '''
    t_pert_coord = t_canonical_coords + tf.random_normal(t_canonical_coords.get_shape(), stddev=sigma)
    t_geo_pert_imgs, t_flow = oper_warping(t_imgs, t_canonical_coords, t_pert_coord) # also use parameters?
    return t_geo_pert_imgs, t_pert_coord, t_flow

def noise(t_img, sigma):
    return t_img + tf.random_normal(t_img.get_shape(), stddev=sigma)

def noise_channel(t_img, sigma, num_ch = 1):
    B, h, w, _ = t_img.get_shape()
    t_noise = tf.random_normal([B.value, h.value, w.value, num_ch], stddev=sigma)
    return tf.concat([t_img, t_noise], axis = -1)

def oper_syn2real(t_img, name='syn2real', sigma = 1, dropout_prob = 0.7):
    '''
    Translate a synthetic 160x160x1 image into a realistic one of same dimension.
    First encodes into 40x40xC features, which are then decoded with oper_syn2real_decoder
    '''
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        ndim_ch = arch_para.syn2real_DimCh
        t_feat = NN('encoder',
            [t_img,
                [conv,ndim_ch,1,5], [runit],    # 160 x 160 x ndim_ch
                [conv,ndim_ch,2],   [runit],    # 80 x 80 x ndim_ch
                [noise_channel, sigma],         # 80 x 80 x ndim_ch+1
                [conv,nch_res-1,2], [runit],    # 40 x 40 x nch_res-1
                [noise_channel, sigma],         # 40 x 40 x nch_res
                *[resblk for i in range(arch_para.syn2real_Nresblk)],
                [tf.nn.dropout, dropout_prob],
                [conv,nch_res-1],   [runit],    # 40 x 40 x nch_res-1
                [noise_channel, sigma],         # 40 x 40 x nch_res
                *[resblk for i in range(arch_para.syn2real_Nresblk)]
                # [tf.nn.dropout, dropout_prob],
            ])
    return oper_syn2real_decoder(t_feat, name)

def oper_syn2real_decoder(t_feat, name='syn2real'):
    '''
    Decodes 40x40xC features into a fake image;
    C=nch_dim from last residual block of feature encoder
    '''
    print('decoding features:', t_feat.get_shape())
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        ndim_ch = arch_para.syn2real_DimCh
        t_fake = NN('decoder',
            [t_feat,                        # 40 x 40 x nch_res
                [upsample],                 # 80 x 80 x ndim_ch
                [conv,ndim_ch], [runit],
                [upsample],                 # 160 x 160 x ndim_ch
                [conv,ndim_ch], [runit],
                [conv,1]                    # grayscale!
            ])                              # 160 x 160 x 1
        t_fake = t_fake
    print('fake:', t_fake.get_shape())
    return t_fake

def oper_programsyn(t_img, name='real2prog'):
    '''
    Encodes 160x160x1 image into 20x20xP logits without softmax,
    outputs those as well as the corresponding 20x20 argmax label prediction
    '''
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        ndim_ch = arch_para.programsyn_DimCh
        logit_act = NN('encoder',
            [t_img,
                [conv,ndim_ch,1,3], [runit], # 160 x 160
                [conv,ndim_ch,2], [runit],   # 80 x 80
                [conv,nch_res,2], [runit],   # 40 x 40
                resblk,
                [conv,nch_res,2], [runit],   # 20 x 20
                resblk,
                [conv,ndim_ch], [runit],     # 20 x 20
                [conv,prog_ch,1,1]           # 20 x 20 x P
            ])
        label_map = tf.argmax(logit_act, axis=3, name="prediction")

    return logit_act, tf.expand_dims(label_map, axis=3)

def oper_renderprog(t_logit, name='prog2real'):
    '''
    Converts 20x20xP logits into feature for shared decoder of syn2real
    and outputs the result of going through that shared decoder
    '''
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        t_feat = NN('translator',
            [t_logit,
                [conv,nch_res], [runit],        # 20 x 20 x nch_res
                *[resblk for i in range(arch_para.syn2real_Nresblk)],
                [upsample],                     # 40 x 40 x nch_res
                resblk                          # 40 x 40 x nch_res
            ])
    # finish with shared decoder
    return oper_syn2real_decoder(t_feat)

def oper_warping(t_texture,
                 t_src_landmark_yx,
                 t_dst_landmark_yx,
                 order = 1,
                 regularization  = 0.001,
                 boundary_pts = 2,
                 name='local_warp'):
    '''
    Apply a differentiable local warping to an image
    and outputs the warped image
    '''
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        warped_image, flow_field = tf.contrib.image.sparse_image_warp(t_texture,
            t_src_landmark_yx, t_dst_landmark_yx,
            interpolation_order = order,
            regularization_weight = regularization,
            num_boundary_points = boundary_pts)
    return warped_image, flow_field

def oper_keypoint_sampler(t_img, t_landmark_anchor, name='keypoint_sampler'):
    '''
    Find best keypoints for regularizing an image through warping
    and outputs those Nx2 keypoints
    '''
    batch_size, num_keypoints, _ = t_landmark_anchor.get_shape()
    K = int(math.sqrt(num_keypoints.value))
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        ndim_ch = arch_para.featext_for_warp_DimCh
        print('keypoints: ', num_keypoints.value)

        # extract features
        t_img_feat = NN('image_encoder',
            [t_img,
                [conv,ndim_ch,1,3], [runit], # 160 x 160
                [conv,ndim_ch,2], [runit],   # 80 x 80 x ndim_ch
                [conv,nch_res,2], [runit],   # 40 x 40 x nch_res
                *[resblk for i in range(arch_para.featext_for_warp_Nresblk)],
                [conv,nch_res,2], [runit]    # 20 x 20 x nch_res
            ])
        print('img features:', t_img_feat.get_shape())
        # compute global features
        t_glo_feat = NN('global_encoder',
            [t_img_feat,
                [conv,ndim_ch,2], [runit],  # 10 x 10 x ndim_ch
                [conv,ndim_ch,2], [runit],  # 5  x 5  x ndim_ch
                [conv,nch_res,2], [runit],  # 3  x 3  x nch_res
                [global_pool],
                [tf.tile,[1,K,K,1]]         # K  x K  x nch_res
            ])
        print('global features:', t_glo_feat.get_shape())
        # compute local features
        t_loc_feat = NN('local_encoder',
            [t_img_feat,                    # 20 x 20 x nch_res
                *[[seq, [[conv,nch_res,2], [runit]]] # reduction until K x K
                    for i in filter(lambda k: k > K, [20, 10, 5])],
                *[resblk for i in range(arch_para.featext_for_warp_Nresblk)],
                [conv], [runit]             # K  x K  x nch_res
            ])
        print('local features:', t_loc_feat.get_shape())
        # concatenate global + local
        Fglo = t_glo_feat.get_shape()[-1]
        Floc = t_loc_feat.get_shape()[-1]
        t_features = tf.concat([
            t_loc_feat[:, :, :, :Floc//2], t_glo_feat[:, :, :, :Fglo//2],
            t_loc_feat[:, :, :, Floc//2:], t_glo_feat[:, :, :, Fglo//2:]
        ], axis = -1)
        print('features: ', t_features.get_shape())

        # infer keypoints
        batch_size = t_features.get_shape()[0]
        weight_initializer = tf.variance_scaling_initializer(
            scale=1.0, mode='fan_in')
        # group F features -> Bx2Nx1x(F/2)  = reshape(, [B, 2*N, 1, -1])
        # infer coordinate -> Bx2Nx1x1      = [conv, 1, 1, 1, weight_initializer]
        # split coordinate -> BxNx2         = reshape(, [B, N, 2])
        t_coords = NN('keypoint_decoder',
            [tf.reshape(t_features, [batch_size, num_keypoints * 2, 1, -1]), 
                [conv, ndim_ch, 1, 1],              # B x 2N x 1 x ndim_ch
                [conv, 1, 1, 1, weight_initializer] # B x 2N x 1 x 1
            ])
        t_coords = tf.reshape(t_coords, [batch_size, num_keypoints, 2])

        return t_landmark_anchor + t_coords

def oper_global_warping(t_img, name='global_warp'):
    '''
    Find and apply a global 6x2 warping to an image
    and output the warped image
    '''
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        # learnable delta transformation
        ndim_ch = arch_para.global_warp_DimCh
        xform_diff = NN('encoder',
            [t_img-arch_para.mean_img_values,
                #[conv,ndim_ch,1,3], [runit], # 160 x 160
                [conv,ndim_ch,2], [runit],   # 80 x 80
                [conv,ndim_ch,2], [runit],   # 40 x 40
                [conv,ndim_ch,2], [runit]    # 20 x 20
            ])
        print('xform: ', xform_diff.get_shape())
        xform_diff = tf.reshape(xform_diff, [xform_diff.get_shape()[0], -1])
        xform_diff = tf.contrib.layers.fully_connected(xform_diff, 6, scope='fc-1d')

        # identity transformation
        batch_size = t_img.get_shape()[0]
        img_size   = t_img.get_shape()[1:]
        xform_iden = np.array([[1., 0., 0.], [0., 1., 0.]], dtype=np.float32).flatten()
        xform_iden = tf.tile(tf.reshape(xform_iden, [1, 6]), [batch_size, 1])
        print('ident: ', xform_iden.get_shape(), ', batch: ', batch_size, ', img_size: ', img_size)

        return transformer(t_img, xform_iden + xform_diff, img_size)

def create_canonical_coordinates(batch_size, img_size, anchor_res):
        block_size = img_size // anchor_res
        anchor_offset = block_size // 2
        # block center locations
        xx = np.linspace(anchor_offset, img_size - anchor_offset, anchor_res, dtype = np.float32)
        # yy = np.linspace(anchor_offset, h - anchor_offset, anchor_res, dtype = np.float32)
        print('coordinates: ', xx)
        xv, yv = np.meshgrid(xx, xx)
        canonical_coords = np.concatenate([np.reshape(yv, (-1, 1)), np.reshape(xv, (-1, 1))], 1)
        t_canonical_coords = tf.convert_to_tensor(
            np.expand_dims(canonical_coords, axis=0).astype(np.float32),
            dtype=tf.float32,
            name='canonical_coord')
        return tf.tile(t_canonical_coords, [batch_size, 1, 1]), block_size

###################################################################################################

def oper_collapse(t_resi, params = dict(), name='collapse'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        t_resi = global_pool(t_resi) # collapse into global information
        t_resi = tf.tile(t_resi, [1, 20, 20, 1]) # copy to each location
    return t_resi

def oper_img2prog(t_img, params = dict(), name='img2prog'):
    '''
    Encodes 160x160x1 image into 20x20xF latent features (0:P for instructions, P:-1 for residual),
    outputs those as well as the corresponding 20x20 argmax label prediction
    '''
    def dilated_conv(h, n=64, r=2, k=3):
        h = tf.contrib.layers.convolution2d(h, n, 
                                            kernel_size=k, 
                                            rate=2, 
                                            stride=1, 
                                            padding='SAME', 
                                            activation_fn=None)
        return h
    
    feat_ch = int(params.get('feat_ch', 64))
    enc_runit = params.get('enc_runit', params.get('runit', 'relu'))
    if enc_runit == 'glu':
        feat_ch2 = feat_ch * 2
    else:
        feat_ch2 = feat_ch
    resi_ch = int(params.get('resi_ch', 16)) # default => latent_ch = 100
    rblk = [resi, [[conv, feat_ch2], [runit, enc_runit], [conv, feat_ch]]]
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        latent_ch = prog_ch + resi_ch
        t_features, enc_layers = NN('encoder',
            [t_img,
                [conv,feat_ch,1,3], [runit, 'in_relu'], # 160 x 160 x D
                [conv,feat_ch,2], [runit, 'in_relu'],   # 80 x 80 x D
                [conv,feat_ch], [runit],
                [conv,feat_ch,2], [runit],   # 40 x 40 x D
                [conv,feat_ch], [runit],
                [conv,feat_ch,2], [runit],   # 20 x 20 x D
            ], output_layers = True)

        use_skip = params.get('use_skip', 0)
        use_rskip = params.get('use_rskip', 0)
        use_sskip = params.get('use_sskip', 0)
        use_dilated = params.get('use_dilated', 0)

        # dilated features
        if use_dilated:
            t_dil1 = dilated_conv(t_features, feat_ch, 2)
            t_dil2 = dilated_conv(t_features, feat_ch, 4)
            t_features = tf.concat([t_features, t_dil1, t_dil2], axis = 3)

        # skip features
        if use_skip or use_rskip or use_sskip:
            feat_list = [t_features]
            for i in [1, 4, 8]:
                skip_layer = enc_layers[i]
                blk_size = skip_layer.shape[1].value // 20
                print('Skip connection from %s | %s (blk=%d)' % (skip_layer.name, skip_layer.shape, blk_size))
                feat_blk = tf.space_to_depth(skip_layer, blk_size)
                if use_rskip: # reduce
                    feat_blk = conv(feat_blk, feat_ch)
                elif use_sskip: # reduce with 1x1 convolution
                    feat_blk = conv(feat_blk, feat_ch, 1, 1)
                feat_list.append(feat_blk)
            t_features = tf.concat(feat_list, axis = 3)

        t_latent = NN('encoder2', 
            [t_features,
                [runit],
                [conv,feat_ch2], [runit, enc_runit],     # 20 x 20 x D
                rblk,rblk,
                [conv,feat_ch2], [runit, enc_runit],     # 20 x 20 x D
                [conv,latent_ch,1,1]         # 20 x 20 x R = (34 + f)
            ])
        logit_act = t_latent[:, :, :, 0:prog_ch]
        label_map = tf.argmax(logit_act, axis=3, name="prediction")

        # global residual
        if params.get('resi_global', 0):
            t_resi = oper_collapse(t_latent[:, :, :, prog_ch:], params)
            t_latent = tf.concat([logit_act, t_resi], axis=-1)

    return t_latent, tf.expand_dims(label_map, axis=3)

def oper_prog2img(t_latent, params = dict(), name='prog2img'):
    '''
    Decodes 20x20xF features into an image;
    F = prog_ch + resi_ch is the latent space dimension (e.g. 17 + 16)
    '''
    
    print('decoding latent vector:', t_latent.get_shape())
    feat_ch = int(params.get('feat_ch', 64))
    dec_runit = params.get('dec_runit', params.get('runit', 'relu'))
    if dec_runit == 'glu':
        feat_ch2 = feat_ch * 2
    else:
        feat_ch2 = feat_ch
    rblk = [resi, [[conv, feat_ch2], [runit, dec_runit], [conv, feat_ch]]]
    rblk_num = int(params.get('rblk_num', arch_para.syn2real_Nresblk)) # def = 3
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        t_img = NN('decoder',
            [t_latent,                          # 20 x 20 x F = (34 + f)
                [upsample],
                [conv,feat_ch2], [runit, dec_runit],        # 20 x 20 x D
                # *[rblk for i in range(rblk_num)],
                # ,                     # 40 x 40 x D
                [upsample],
                [conv,feat_ch2], [runit, dec_runit],        # 40 x 40 x D
                # rblk,                           # 40 x 40 x D
                # ,                     # 80 x 80 x D
                [upsample],
                [conv,feat_ch2], [runit, dec_runit],
                # ,                     # 160 x 160 x D
                [conv,feat_ch2], [runit, dec_runit],
                [conv,1],                        # grayscale!
            ])                                  # 160 x 160 x 1
    #smoother = Smoother({'data':t_img}, 11, 2.)
    #t_img = smoother.get_output()
    print('output:', t_img.get_shape())
    return t_img

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
        t_act1 = NN('img2feat',
            [t_img, 
            [conv_fn, feat_ch, 2], [runit, 'in_relu'],
            [conv_fn, feat_ch, 2], [runit, 'in_relu'],
            [conv_fn, feat_ch, 2], [runit, 'in_relu'],
            ])
        t_act2 = NN('feat2feat',
            [t_act1, *[rblk for _ in range(rblk_num)]])
        t_out  = NN('feat2prog',
            [tf.concat([t_act1, t_act2], -1),
                [conv_fn, feat_ch], [runit],
                [conv_fn, prog_ch, 1, 1]
            ])
    return t_out


def oper_img2prog_final_complex(t_img, params = dict(), name='img2prog'):
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
        t_act0, enc_feat = NN('img2feat',
            [t_img, 
            [conv_fn, feat_ch, 2], [runit, 'in_relu'],
            [conv_fn, feat_ch, 2], [runit, 'in_relu'],
            [conv_fn, feat_ch, 2], [runit, 'in_relu']
            ], output_layers = True)
        feat_list = [t_act0]
        for i in [1, 3]: # skip connections
            skip_layer = enc_feat[i]
            blk_size = skip_layer.shape[1].value // 20
            print('Skip connection from %s | %s (blk=%d)' % (skip_layer.name, skip_layer.shape, blk_size))
            with tf.variable_scope('skip_reduce/' + str(skip_layer.shape[1].value)):
                feat_blk = tf.space_to_depth(skip_layer, blk_size)
                feat_blk = conv_fn(feat_blk, feat_ch, 1, 1)
            feat_list.append(feat_blk)
        t_act1 = NN('skip2feat',
            [tf.concat(feat_list, axis = 3),
                [conv_fn, feat_ch], [runit]])
        t_act2 = NN('feat2feat',
            [t_act1, *[rblk for _ in range(rblk_num)]])
        t_out  = NN('feat2prog',
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

def oper_img2img(t_img, out_ch, params, name):
    '''
    Translate image domain with resnet
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
            [t_img, [conv_fn, feat_ch], [runit, 'in_relu']])
        t_act2 = NN('resnet2',
            [t_act1, *[rblk for _ in range(rblk_num)]])
        t_out  = NN('resnet3',
            [tf.concat([t_act1, t_act2], -1),
                [conv_fn, feat_ch], [runit],
                [conv_fn, out_ch, 1, 1]
            ])
    return t_out

def oper_img2img(t_img, out_ch, params, name):
    '''
    Translate image domain with resnet
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
            [t_img, [conv_fn, feat_ch], [runit, 'in_relu']])
        t_act2 = NN('resnet2',
            [t_act1, *[rblk for _ in range(rblk_num)]])
        t_out  = NN('resnet3',
            [tf.concat([t_act1, t_act2], -1),
                [conv_fn, feat_ch], [runit],
                [conv_fn, out_ch, 1, 1]
            ])
    return t_out

def oper_img2prog_simple(t_img, params = dict(), name='img2prog'):
    '''
    Encodes 160x160x1 image into 20x20xP instruction outputs 
    as well as the corresponding 20x20 argmax label prediction
    '''
    feat_ch = int(params.get('feat_ch', 64))
    rblk = [resi, [[conv, feat_ch], [runit], [conv, feat_ch]]]
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        t_logit_act = NN('encoder',
            [t_img,                         # 160 x 160 x D
                [conv,feat_ch*2], [runit], [maxpool],  # 80 x 80 x D*2
                [conv,feat_ch*2], [runit], [maxpool],  # 40 x 40 x D*2
                [conv,feat_ch], [runit], [maxpool],  # 20 x 20 x D
                rblk,rblk,rblk,rblk,
                [conv,feat_ch], [runit], # 20 x 20 x D
                [conv,prog_ch,1,1],         # 20 x 20 x P
            ])
        label_map = tf.argmax(t_logit_act, axis=3, name="prediction")
        
    return t_logit_act, tf.expand_dims(label_map, axis=3)

def oper_adapter(t_latent, params = dict(), name='adapter'):
    '''
    Translate latent data for a specific domain (i.e. real looking)
    '''
    print('adapting latent vector:', t_latent.get_shape())
    feat_ch = int(params.get('feat_ch', 64))
    latent_ch = t_latent.get_shape()[3].value
    rblk = [resi, [[conv, -1], [runit], [conv, -1]]]
    rblk_num = int(params.get('rblk_num', 2)) # def = 3
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        # global residual
        if params.get('resi_global', 0):
            t_inst = t_latent[:, :, :, 0:prog_ch]
            t_resi = tf.expand_dims(tf.expand_dims(t_latent[:, 0, 0, prog_ch:], 1), 2)
            t_adapt_resi = NN('adapter_resi', [t_resi, [conv,-1,1,1], [runit], [conv,-1,1,1]])
            t_adapt_inst = NN('adapter_inst', [t_inst, rblk])
            t_latent = tf.concat([t_adapt_inst, tf.tile(t_adapt_resi, [1, 20, 20, 1])], axis = -1)
        else:
            t_latent = NN('adapter', [t_latent,
                [conv,-1,1,1],
                [runit],
                *[rblk for i in range(rblk_num)],
                [conv,-1,1,1]
                ])
    print('output:', t_latent.get_shape())
    return t_latent

def oper_generator(t_noise, t_label, params = dict(), name='resi_gen'):
    '''
    Generate 20x20xF latent data from noise and an instruction
    F = ndim_instr + ndim_resi is the latent space dimension (e.g. 17 + 16)
    '''
    print('generator latent vector from noise:', t_noise.get_shape())
    # feat_ch = params.get('feat_ch', 64)
    resi_ch = int(params.get('resi_ch', 16))
    latent_ch = prog_ch + resi_ch
    rblk = [resi, [[conv, latent_ch], [runit], [conv, latent_ch]]]
    rblk_num = int(params.get('rblk_num', 2)) # def = 3
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        # label to logits reshaping
        t_init_label = tf.one_hot(tf.squeeze(t_label, axis = [3]), prog_ch)
        # noise reshaping
        if params.get('resi_global', 0):
            resi_num = 1
        else:
            resi_num = 20
        weight_init = tf.variance_scaling_initializer(
            scale=1.0, mode='fan_in')
        t_init_noise = NN('noise',
            [t_noise,
                [fc, resi_num * resi_num * resi_ch, weight_init], [relu],
                [tf.reshape, [-1, resi_num, resi_num, resi_ch]],
            ])
        if params.get('resi_global', 0):
            t_init_noise = tf.tile(t_init_noise, [1, 20, 20, 1])
        t_init = tf.concat([t_init_label, t_init_noise], axis = -1)
        # scale and style
        t_latent = NN('generator', [t_init, rblk, [conv,latent_ch,1,1]])
        if params.get('resi_global', 0):
            t_logits = t_latent[:, :, :, 0:prog_ch]
            t_resi = oper_collapse(t_latent[:, :, :, prog_ch:], params)
            t_latent = tf.concat([t_logits, t_resi], axis = -1)
    print('output:', t_latent.get_shape())
    return t_latent


def wgan_loss(d_logit_real, d_logit_fake):
    d_loss_real = tf.reduce_mean(d_logit_real)
    d_loss_fake = tf.reduce_mean(d_logit_fake)
    d_loss = d_loss_fake - d_loss_real
    g_loss = -d_loss_fake
    return d_loss, g_loss

        
    
     
    
def oper_discriminator(t_input, reductions, num_domains, params = dict(), name='discriminator', arg_disc_layers = 2):
    '''
    Discriminate an input's domain by using a sequence of convolution and reductions until reaching
    a specified resolution.
    The inputs include the number of reductions as well as the number of domains to discriminate.
    '''
    disc_ch = int(params.get('disc_ch', params.get('feat_ch', 64)))
    disc_layers = arg_disc_layers if arg_disc_layers != 2 else int(params.get('disc_layers', 2))
    disc_ch_factor = params.get('disc_ch_factor', 2.0)
    disc_bottleneck = params.get('disc_bottleneck', 1)
    noise_sigma = params.get('noise_sigma', 3./255.)

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        layers = [t_input + tf.random_normal(tf.shape(t_input), stddev = noise_sigma)]
        for i in range(reductions):
            # learn filters
            for j in range(disc_layers):
                layers += [[conv_valid, disc_ch * (disc_ch_factor ** i)], 
                           [runit]
                        #    [runit, 'lrelu_in' if i < 2 else 'lrelu']
                          ]
            # decrease resolution
            if disc_bottleneck:
                power = i # reduce activation count
            else:
                power = i + 1 # keep number of activations (with factor=2)
            layers += [[conv_valid, disc_ch * (disc_ch_factor ** power), 2], 
                       [runit]
                    #    [runit, 'lrelu_in' if i < 2 else 'lrelu']
                      ]
        # output domain
        layers.append([conv_valid, 1, 1, 1])
        
        # instantiate subgraph
        t_output = NN('discriminator', layers)
    return t_output

###################################################################################################


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
    fakes = filter(lambda name: name != 'real' and name != 'unsup' and name != 'worst', t_imgs_dict.keys())

    # create generator
    with tf.variable_scope("generator"):

        def encoder(t_input, name):
            print('**************oper_img2prog')
            with runits('relu') as activations:
                t_latent, t_instr = oper_img2prog(t_input, params)
                net.latent[name] = t_latent
                net.logits[name] = t_latent[:, :, :, 0:prog_ch]
                net.instr[name]  = t_instr
                store_act(name, 'img2prog', activations)
            return t_latent

        def decoder(t_latent, name):
            with runits('relu') as activations:
                t_resi_out = oper_prog2img(t_latent, params)
                net.resi_outs[name] = t_resi_out
                store_act(name, 'prog2img', activations)
            return t_resi_out

        feedback = params.get('feedback', 0) if is_train else 0
        decoding = params.get('decoder', 1)
        for name, t_resi_inp in net.resi_imgs.items():
            
            # program synthesis (encoding)
            t_latent = encoder(t_resi_inp, name)

            # check we want the decoder
            if not decoding:
                continue

            # rendering from program (decoding)
            t_resi_out = decoder(t_latent, name)
            
            # feedback flow
            if name == 'unsup' or feedback:
                # program re-synthesis (re-encoding)
                 t_latent_fb = encoder(t_resi_out, name + '_feedback')

        # if no decoder => no adapter or generator either
        if not decoding:
            return net

        # adapter network paths
        if params.get('adapter', 0):
            for name in ['real'] + list(fakes):
                t_latent = net.latent[name]

                # adapter
                with runits() as activations:
                    t_latent = oper_adapter(t_latent, params)
                    t_logits = t_latent[:, :, :, 0:prog_ch]
                    net.latent[name + '_real'] = t_latent
                    net.logits[name + '_real'] = t_logits
                    net.instr[name + '_real'] = tf.expand_dims(
                        tf.argmax(t_logits, axis=3, name="prediction"),
                        axis = 3)
                    store_act(name, 'adapter', activations)

                # no decoding / feedback for the real path
                if name == 'real':
                    continue # real path is only for semantic identity

                # rendering
                t_resi_out = decoder(t_latent, name + '_real')

                # feedback flow
                if feedback:
                    # program re-synthesis
                    t_latent_fb = encoder(t_resi_out, name + '_real_feedback')

        # generator network path
        if is_train and params.get('generator', 0):
            noise_depth = int(params.get('noise_depth', 4))
            noise_sigma = params.get('noise_sigma', 1)
            noise_uniform = params.get('noise_uniform', 0)
            noise_shape = [net.imgs['real'].get_shape()[0].value, 5 * 5 * noise_depth]
            for name, t_label in t_labels_dict.items():
                # generator
                with runits() as activations:
                    if noise_uniform:
                        t_noise = tf.random_uniform(noise_shape, -noise_uniform, noise_uniform)
                    else:
                        t_noise = tf.random_normal(noise_shape, stddev = noise_sigma)
                    t_latent = oper_generator(t_noise, t_label, params)
                    t_logits = t_latent[:, :, :, 0:prog_ch]
                    short_name = name.replace('instr_', '')
                    name_gen = short_name + '_gen'
                    net.latent[name_gen] = t_latent
                    net.logits[name_gen] = t_logits
                    net.instr[name_gen] = tf.expand_dims(
                        tf.argmax(t_logits, axis=3, name="prediction"),
                        axis = 3)
                    store_act(short_name, 'generator', activations)

                # rendering
                t_resi_out = decoder(t_latent, name_gen)

                # feedback flow
                if feedback:
                    t_latent_fb = encoder(t_resi_out, name_gen + '_feedback')

    return net

###################################################################################################


def translate_tensor(h, dx, dy):
    h = tf.pad(
        h,
        tf.constant([[0, 0], [1, 1], [1, 1]]),
        mode='SYMMETRIC')
    imsz = tf.shape(h)
    return h[:,(1+dy):(imsz[1]+dy-1),(1+dx):(imsz[2]+dx-1)]

def tf_MILloss_xentropy(labels, logits, weight=None):
    # INCORRECT implementation, it should be maximum over the logits, not the minimum over the instruction
    # Note: tf.contrib.image.translate() should not be used for this purpose!!
    dx = [0, -1, 0, 1, -1, 1, -1, 0, 1]
    dy = [0, -1, -1, -1, 0, 0, 1, 1, 1]

    # def my_func(x):
    #     print('!!!DBG] {}'.format(x.shape))
    #     return x

    lst_entropy = []
    for i in range(len(dx)):    # 8 directions
        shifted_lbl = translate_tensor(labels, dx[i], dy[i])
        # shifted_lbl = tf.contrib.image.translate(labels, 
        #                                          translations = [dx[i], dy[i]],
        #                                          interpolation='NEAREST')
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = shifted_lbl, 
                                                                  logits = logits)
        # xentropy = tf.Print(xentropy, [tf.reduce_mean(xentropy)])

        if weight is not None:
            shifted_weight = translate_tensor(tf.squeeze(weight, [3]), dx[i], dy[i])
            # shifted_weight = tf.contrib.image.translate(weight, 
            #                                      translations = [dx[i], dy[i]],
            #                                      interpolation='NEAREST')
            xentropy = tf.multiply(shifted_weight, xentropy)
            # xentropy = tf.Print(xentropy, [tf.reduce_mean(xentropy)])

        # check whether xentropy has channel dim 1
        # print('xentropy dim: {}'.format(xentropy.get_shape()))
        if i == 0: # make it prefer the first element
            xentropy -= 1e-6

        lst_entropy.append(xentropy)
    
    # y = tf.py_func(my_func, [lst_entropy[0]], tf.float32)
    # with tf.control_dependencies(y):

    entropy_tensor = tf.stack(lst_entropy, axis=3)[:,1:-1,1:-1,:] # BxHxWxD
    print('xentropy dim: {}'.format(entropy_tensor.get_shape()))
    entropy_disp = tf.reduce_mean(entropy_tensor,axis=[1,2], keepdims=True) # spatial collapsing
    # entropy_disp = tf.Print(entropy_disp, [entropy_disp, tf.reduce_min(entropy_disp)])
    print('xentropy dim: {}'.format(entropy_disp.get_shape()))
    total_entropy = tf.reduce_min(entropy_disp, axis=3, keepdims=True) # take minimum loss over displacement
    print('xentropy dim: {}'.format(total_entropy.get_shape()))
    # total_entropy = tf.Print(total_entropy, [tf.reduce_mean(total_entropy), tf.reduce_max(total_entropy)])
    return total_entropy


def tf_MILloss_xentropy2(labels, logits, weight=None):
    pooled_logits = tf.layers.max_pooling2d(logits,
                                            pool_size=3,
                                            strides=1,
                                            padding='same')
    
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels, 
                                                            logits = pooled_logits)
    if weight is not None:
        xentropy = tf.multiply(tf.squeeze(weight, [3]), xentropy)

    total_entropy = xentropy[:,1:-1,1:-1]
    # print('xentropy dim: {}'.format(total_entropy.get_shape()))
    return total_entropy
    
# classification loss (instr cross-entropy)
def tf_loss_xentropy(labels, logits, weight = None):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels, logits = logits)
    if weight is None:
        return xentropy
    else:
        return tf.multiply(tf.squeeze(weight, [3]), xentropy)

    


def tf_accuracy(labels, instrs, mask = None):
    # regularize arguments
    if mask is None:
        mask = tf.ones_like(labels, tf.float32)
    else:
        mask = tf.cast(mask, tf.float32)
    
    instrs = tf.cast(instrs, tf.int32)
    # print('accuracy: ', labels.get_shape(), instrs.get_shape(), mask.get_shape())
    # compute accuracy
    total = tf.reduce_sum(mask)
    matches = tf.where(tf.equal(labels, instrs), mask, tf.zeros_like(labels, tf.float32))
    # matches = debug(matches, 'matches')
    count = tf.reduce_sum(matches)
    return count / tf.maximum(total, tf.ones_like(total, tf.float32))


def tf_2daccuracy(labels, instrs, mask):
    # regularize arguments
    matches = tf.where(tf.equal(labels, instrs), mask, tf.zeros_like(labels, tf.float32))
    matches = matches[:,1:-1,1:-1]
    total = tf.reduce_sum(mask[:,1:-1,1:-1], axis=[1,2], keepdims=True) # per image
    return matches/tf.maximum(total, tf.ones_like(total, tf.float32)) # BxHxW


def tf_MILloss_accuracy(labels, instrs, mask=None):
    dx = [0, -1, 0, 1, -1, 1, -1, 0, 1]
    dy = [0, -1, -1, -1, 0, 0, 1, 1, 1]
    lst_accuracy = []
    instrs = tf.cast(instrs, tf.int32)
    instrs = tf.squeeze(instrs)
    labels = tf.squeeze(labels)

    if mask is None:
        mask = tf.ones_like(labels, tf.float32)
    else:
        mask = tf.cast(mask, tf.float32)
        mask = tf.squeeze(mask)

    for i in range(len(dx)):    # 8 directions
        shifted_lbl = translate_tensor(labels, dx[i], dy[i])
        shifted_mask = translate_tensor(mask, dx[i], dy[i])

        matches = tf_2daccuracy(shifted_lbl, instrs, shifted_mask) # BxHxW
        lst_accuracy.append(matches)
    
    accuracy_tensor = tf.stack(lst_accuracy, axis=3) # BxHxWxD
    accuracy = tf.reduce_sum(accuracy_tensor, axis=[1,2], keepdims=True) #Bx1x1xD
    accuracy = tf.reduce_max(accuracy, axis=3, keepdims=True) #Bx1x1x1
    return tf.reduce_mean(accuracy), accuracy

def tf_background(t_inst, btype = 'global'):
    """
    Find background of instruction
    """
    if btype == 'global':
        return tf_global_background(t_inst)
    elif btype == 'local':
        return tf_local_background(t_inst)
    else:
        raise ValueError('Invalid background type %s' % btype)

def tf_global_background(t_inst):
    """
    Background = most frequent instruction per image
    """
    batch_size = t_inst.get_shape()[0].value
    bgs = []
    for i in range(batch_size): # done per image (not per batch)
        t_label = tf.slice(t_inst, [i, 0, 0, 0], [1, 20, 20, 1]) # t_inst[i, :, :]
        t_samples = tf.reshape(t_label, [-1])
        t_unique, _, t_count = tf.unique_with_counts(t_samples)
        t_max_occ = tf.reduce_max(t_count)
        t_bg_inst = tf.gather(t_unique, tf.where(tf.equal(t_count, t_max_occ)))
        t_bg_inst = t_bg_inst[0] # tf.squeeze only works if there's no tie between max occurrences
        # t_bg_inst = debug(t_bg_inst, 'BG_Inst', tf.int32)
        # t_bg_inst = t_bg_inst[0]
        t_bg_inst.set_shape((1))
        t_bg_mask = tf.equal(t_label, t_bg_inst)
        # single-instruction mask
        # - all 1 when only one instruction in label (min == max)
        # - all 0 otherwise
        t_single = tf.logical_and(
            tf.equal(t_label, tf.reduce_min(t_samples)),
            tf.equal(t_label, tf.reduce_max(t_samples))
        )
        t_bg_mask = tf.logical_and(t_bg_mask, tf.logical_not(t_single)) # if single => no background
        bgs.append(t_bg_mask)
    return tf.concat(bgs, axis = 0)

def tf_local_background(t_inst):
    """
    Background = when neighbors are all the same as instruction
    """
    # compute local background
    max_inst = tf.nn.max_pool(t_inst, [1, 3, 3, 1], [1, 1, 1, 1], 'SAME')
    min_inst = -tf.nn.max_pool(-t_inst, [1, 3, 3, 1], [1, 1, 1, 1], 'SAME')
    t_local_bg = tf.logical_and(
        tf.equal(max_inst, t_inst),
        tf.equal(min_inst, t_inst)
    )
    # check that the instruction isn't fully background
    t_samples = tf.reshape(t_inst, [t_inst.shape[0], -1])
    t_not_single = tf.logical_or(
        tf.not_equal(t_inst, tf.reshape(tf.reduce_min(t_samples, axis = 1), [-1, 1, 1, 1])),
        tf.not_equal(t_inst, tf.reshape(tf.reduce_max(t_samples, axis = 1), [-1, 1, 1, 1]))
    )
    return tf.logical_and(t_local_bg, t_not_single)

def tf_frequency_weight(t_inst, weight_type = 'linear', factor = 1.0):
    """
    Compute a per-instruction weighting based on inverse frequency
    """
    freqs = np.array([0.456536, 0.456536, 0.001150, 0.026873, 0.012083, 0.000129, 0.012083, 0.000129, 0.012083, 0.000129, 0.012083, 0.000129, 0.002053, 0.002053, 0.002053, 0.002053, 0.001846]).astype(np.float32)
    if weight_type == 'linear':
        weights = 1 / freqs
    elif weight_type == 'sqrt':
        weights = np.sqrt(1 / freqs)
    elif weight_type == 'log':
        weights = -np.log(freqs)
    elif weight_type == 'local':
        return tf_per_batch_frequency_weight(t_inst, factor=factor)
    else:
        raise ValueError('Unsupported weight type %s' % weight_type)
    weights /= np.mean(weights) # not sum
    return tf.gather(weights * factor, t_inst)

def tf_per_batch_frequency_weight(t_inst, weight_type = 'linear', factor = 1.0):
    """
    Compute a per-instruction weighting based on inverse frequency per a batch
    t_inst: BxHxWx1
    """
    t_onehot = tf.one_hot(tf.squeeze(t_inst),depth=prog_ch,dtype=tf.float32)
    t_hist = tf.reduce_sum(t_onehot, axis = [0,1,2]) # D dim. vec
#     t_hist = t_hist/tf.reduce_sum(t_hist) # range [0,1]
    weight = tf.maximum(factor, tf.minimum(2.5, 20./(t_hist+1)))
    return tf.expand_dims(tf.gather_nd(weight, t_inst), -1)


def total_loss(net, t_inst_dict, params = dict()):
    loss_dict_Disc = dict()
    loss_dict_Gene = dict()
    metrics = dict()

    # replay switch
    replay_worst = params.get('replay_worst', 0)

    # extract instructions
    t_inst_real = t_inst_dict['instr_real']
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

    # selection functions
    def get_gt(name):
        if name.startswith('real'):
            return t_inst_real
        elif name.startswith('worst'):
            return t_inst_wors
        else:
            return t_inst_synt
    def get_mask(name):
        if name.startswith('real'):
            return t_real_mask
        elif name.startswith('worst'):
            return t_wors_mask
        else:
            return t_synt_mask
    def get_inst_weight(name):
        if name.startswith('real'):
            return t_real_weight
        elif name.startswith('worst'):
            return t_wors_weight
        else:
            return t_synt_weight
    def get_img_weight(name):
        if name.startswith('real'):
            return t_rimg_weight
        elif name.startswith('worst'):
            return t_wimg_weight
        else:
            return t_simg_weight
    def is_real(name):
        return name == 'real' or name == 'unsup' or name == 'worst'
    def looks_real(name):
        return name.endswith('real') or name == 'unsup' or name == 'worst'

    # create discriminator networks if needed for loss
    net.discr = {
        'instr': dict(), 'latent': dict(), 'image': dict(),
        #'wgan_grad': dict()
    }
    net.resi_aug_wgan = dict()

    # summon VGG19
    if params.get('bvggloss', 0):
        if params.get('vgg16or19', '16') == '16':
            net.vggobj = custom_vgg19.Vgg16()
        else:
            net.vggobj = custom_vgg19.Vgg19()
        net.vgg = dict()

    def wgan_interp_input(net, nbatch):
        net.resi_aug_wgan = dict()
        alpha = tf.random_uniform([nbatch, 1, 1, 1], 0.0, 1.0)
        net.resi_imgs_noaug['real']
        interpolated = alpha * net.resi_imgs_noaug['real'] + \
                       (1 - alpha) * net.resi_outs['real']
        net.resi_aug_wgan['real'] = interpolated

    if params.get('discr_img', 0) or params.get('discr_instr', 0) or params.get('discr_latent', 0):
        with tf.variable_scope("discriminator"):
            # default = 1 = binary <real | fake>
            # other = CatGAN
            # - 3: real | rend | tran
            # - 4: real | rend | tran | generator
            domains = params.get('domains', 1)
            # error discriminator discriminates domain through the error instead of the direct output
            error_discr = params.get('error_discr', 0)

            # instruction discriminator
            # applied to {all} except {'unsup'}
            if params.get('discr_instr', 0):
                for name, t_logits in net.logits.items():
                    if name.startswith('unsup'):
                        continue

                    # 20x20xP to 5x5xD (20->10->5)
                    if error_discr:
                        # discriminate instruction
                        t_inst_gt = get_gt(name)
                        t_input = tf.nn.softmax(t_logits) - tf.one_hot(tf.squeeze(t_inst_gt, axis = [-1]), prog_ch)
                    else:
                        # discriminate decoded instruction
                        t_input = t_logits
                    with runits('lrelu'):
                        t_domain = oper_discriminator(t_input, 2, domains, params, 'instr_domain')
                        # t_domain = tf.sigmoid(t_domain)
                    net.discr['instr'][name] = t_domain

            # TODO: it has not been fed any 'real' or 'unsup' data
            # latent residual discriminator for adapter and generator networks)
            # applied to {*_real}
            if params.get('discr_latent', 0):
                for name, t_latent in net.latent.items():
                    if name.endswith('real') or name.endswith('_gen'):
                        with runits('lrelu'):
                            t_domain = oper_discriminator(t_latent, 1, domains, params, 'latent_domain', arg_disc_layers = 6)
                            # t_domain = tf.sigmoid(t_domain)
                        net.discr['latent'][name] = t_domain

            # Generated image discriminator (for adapter and generator networks)
            if params.get('discr_img', 0):
                for name, t_resi_out in net.resi_outs.items():
    #                 if name.endswith('real') or name.endswith('_gen'):
                    with runits('lrelu'):
                        t_domain = oper_discriminator(t_resi_out, 3, domains, params, 'image_domain')
                        # if not 'wgan_grad' in net.discr.keys():
                            # t_domain = tf.sigmoid(t_domain)
                    net.discr['image'][name] = t_domain
                
                # GT_REAL data path for Discriminator
                for name, t_resi_img in net.resi_imgs.items():
                    if name.endswith('real') or name == 'unsup' or name == 'worst':
                        with runits('lrelu'):
                            t_domain = oper_discriminator(t_resi_img, 3, domains, params, 'image_domain')
                            # if not 'wgan_grad' in net.discr.keys():
                                # t_domain = tf.sigmoid(t_domain)
                        net.discr['image']['gt_' + name] = t_domain
                
            
            # gradient penality
            if 'wgan_grad' in net.discr.keys():
                wgan_interp_input(net, batch_size.value) # return net.resi_aug_wgan
                for name, t_resi in net.resi_aug_wgan.items():
                    if name.endswith('real'):
                    #  or name.endswith('_gen'):
                        with runits('lrelu'):
                            t_domain = oper_discriminator(t_resi, 3, domains, params, 'image_domain')
                        # tf.gradients returns a list of sum(dy/dx) for each x in xs.
                        gradients = tf.gradients(t_domain, [t_resi, ], name="WGAN_grad")[0]
                        net.discr['wgan_grad'][name] = gradients


            
    # generator and discriminator losses
    with tf.variable_scope("loss"):

        # VGG perceptual loss
        # TODO: style loss (Gram) needs to be added
        if params.get('bvggloss', 0):
            # applied to {real, unsup}
            for name, t_resi_img in net.resi_imgs.items():
                if looks_real(name):
                    net.vgg['gt_' + name] = net.vggobj.build(t_resi_img)
            
            for name, t_resi_out in net.resi_outs.items():
                if looks_real(name):
                    net.vgg[name] = net.vggobj.build(t_resi_out)

                    curlayer = 'conv2_2'
                    loss_perc_pool2 = 0.1*tf_loss_with_select(
                                        (1./128.)*net.vgg['gt_' + name][curlayer], 
                                        (1./128.)*net.vgg[name][curlayer], 
                                        which_loss = 'l2')
                    loss_dict_Gene['loss_vgg_percept/' + curlayer + '/' + name] = loss_perc_pool2*0.25 
                    # normalize by the number of combinations (real, unsuper, conv2_2, pool3)

                    curlayer = 'pool3'
                    loss_perc_pool5 = 1.*tf_loss_with_select(
                                        (1./128.)*net.vgg['gt_' + name][curlayer], 
                                        (1./128.)*net.vgg[name][curlayer], 
                                        which_loss = 'l2')
                    loss_dict_Gene['loss_vgg_percept/' + curlayer + '/' + name] = loss_perc_pool5*0.25
                    
            # VGG style losses 
            # applied to {synt, real} except {unsup}
            for name, t_resi_out in net.resi_outs.items():
                if name not in net.vgg.keys(): # synthetic
                    net.vgg[name] = net.vggobj.build(t_resi_out)
                elif name != 'real':    
                    continue

                lst_lweight = [0.3, 1., 1.]
                lst_layers = ['conv1_2', 'conv2_2', 'conv3_3']
                no_gram_layers = float(len(lst_layers))
                for gram_layer, gram_weight in zip(lst_layers, lst_lweight):
                    loss_prefix = 'loss_vgg_percept/' + 'gram_' + gram_layer + name

                    lst_gts = ['unsup',] #['real', 'unsup']
                    no_gts = float(len(lst_gts))
                    for gts in lst_gts:
                        t_real = net.vgg['gt_' + gts][gram_layer]/128.
                        t_synt = net.vgg[name][gram_layer]/128.
                        t_loss = style_layer_loss(t_real, t_synt, params.get('gram_power', 2))
                        loss_dict_Gene[loss_prefix + '2' + gts] = gram_weight*t_loss/(no_gram_layers*no_gts)


        if params.get('discr_img', 0) or params.get('discr_instr', 0) or params.get('discr_latent', 0):
            # binary discrimination game
            discr_type = params.get('discr_type', 'l2')

            # DISC_INST
            # applied to {real, unsup} and {all the others}
            for name, t_discr in net.discr['instr'].items():
                if is_real(name):
                    t_domain = tf.ones_like(t_discr)
                else:
                    t_domain = tf.zeros_like(t_discr)
                # discriminator part
                loss_dis = tf_loss_with_select(t_discr, t_domain, which_loss = discr_type)
                loss_dict_Disc['loss_D_instr/' + name] = loss_dis
                # generator part
                loss_gen = tf_loss_with_select(t_discr, 1 - t_domain, which_loss = discr_type)
                loss_dict_Gene['loss_G_instr/' + name] = loss_gen

            # DISC_LATENT
            # applied to {real, unsup} and {all the others}
            for name, t_discr in net.discr['latent'].items():
                if is_real(name):
                    t_domain = tf.ones_like(t_discr)
                    # discriminator part
                    loss_dis = tf_loss_with_select(t_discr, t_domain, which_loss = discr_type)
                    loss_dict_Disc['loss_D_latent/' + name] = loss_dis
                else:
                    t_domain = tf.zeros_like(t_discr)
                    # generator part
                    loss_dis = tf_loss_with_select(t_discr, t_domain, which_loss = discr_type)
                    loss_dict_Disc['loss_D_latent/' + name] = loss_dis
                    loss_gen = tf_loss_with_select(t_discr, 1. - t_domain, which_loss = discr_type)
                    loss_dict_Gene['loss_G_latent/' + name] = loss_gen
                
            # DISC_IMAGE
            # applied to {gt_real, gt_unsup, rend, tran, real, unsup} and {rend_real, tran_real, real_real}
            disc_weight = 1./float(len(net.discr['image'].items())-1)
            for name, t_discr in net.discr['image'].items():
                if not 'wgan_grad' in net.discr.keys(): # normal lsq-gan
                    if name.startswith('gt_'): #name == 'gt_real':
                        loss_dis = tf_loss_with_select(t_discr, tf.ones_like(t_discr), which_loss = discr_type)
                        loss_dict_Disc['loss_D_image/' + name] = loss_dis
                    # elif name == 'real':
                    else:
                        loss_dis = tf_loss_with_select(t_discr, tf.zeros_like(t_discr), which_loss = discr_type)
                        loss_dict_Disc['loss_D_image/' + name] = loss_dis*disc_weight
                        loss_gen = tf_loss_with_select(t_discr, tf.ones_like(t_discr), which_loss = discr_type)
                        loss_dict_Gene['loss_G_image/' + name] = loss_gen if name != 'real' else loss_gen*5.
                else: # wgan
                    loss_dis = tf.reduce_mean(t_discr)
                    if name == 'gt_real':
                        loss_dict_Disc['loss_D_image/' + name] = -loss_dis
                    else:
                        loss_dict_Disc['loss_D_image/' + name] = loss_dis*disc_weight
                        loss_dict_Gene['loss_G_image/' + name] = -loss_dis

            if 'wgan_grad' in net.discr.keys(): # wgan
                for name, t_discr in net.discr['wgan_grad'].items():
                    grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(t_discr), axis=[1, 2, 3]))
                    grad_penalty = tf.square(grad_l2 - 1.0)
                    loss_dict_Disc['loss_D_image/wgan_grad'] = params.get('gp_lambda', 10.)*grad_penalty
        
        # instruction x-entropy
        # applied to {*real*} including {rend_real, tran_real, real_real, real, real_feedback...}
        for name, t_logits in net.logits.items():
            if name.startswith('unsup'): # name.endswith('_real') or 
                continue # adapter network doesn't use entropy

            if (re.search('real', name) is not None) or (not params.get('adapter',0)):
                t_instr  = get_gt(name)
                t_weight = get_inst_weight(name)

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

        # adapter losses
        # applied to {rend, tran, real} and {rend_real, tran_real, real_real}
        if params.get('adapter', 0):

            # applied to {rend, tran}
            for name in fakes:
                t_logits_from = net.logits[name]
                t_logits_to   = net.logits[name + '_real']
                # instruction maps should be the same
                loss_adapt = tf_loss_with_select(
                    t_logits_from, t_logits_to,
#                     tf.nn.softmax(t_logits_from), tf.nn.softmax(t_logits_to),
                    which_loss='smooth_l1')
                loss_dict_Gene['loss_adapt/' + name] = loss_adapt

            # applied to {real}
            # identity for real data
            loss_adapt = tf_loss_with_select(
                net.latent['real'], net.latent['real_real'],
                which_loss='smooth_l1')
            loss_dict_Gene['loss_adapt/real'] = loss_adapt

        
        def GramSchmidtLoss(latent):
            # latent: BxHxWxC (0-(P-1): instruction, P-: residual latent)
            t_sz = tf.shape(latent)
            # print(latent.get_shape())
            # print(t_sz)
            
            mat = tf.reshape(latent, [t_sz[0], -1, t_sz[-1]]) # tf.concat
            _, R = tf.qr(mat, full_matrices=False) # BxHWxC => BxCxC
            abscorr = tf.reduce_mean(tf.abs(R[:, 0:prog_ch, prog_ch:]))
            return abscorr

        # disentanglement loss
        # applied to {all}
        if params.get('bloss_disentangle', 0):
            for name, t_latent in net.latent.items():
                loss_disentangle = GramSchmidtLoss(t_latent)
                loss_dict_Gene['loss_disentgl/' + name] = loss_disentangle

        # auto-encoding with instructions as latent vector space
        # applied to {real} and {tran*}
        fn_normalize = tf.identity # tf.contrib.layers.instance_norm
        if params.get('bloss_ae', 0):
            ae_loss_type = params.get('ae_loss_type', 'smooth_l1')
            for name, t_res_inp in net.resi_imgs_noaug.items():
                if name == 'real':  # apply AE loss only for real data
                    # if name == 'real':
                    #     t_weight = t_rimg_weight
                    # else:
                    #     t_weight = t_simg_weight
                    t_res_out = net.resi_outs[name]
                    loss_ae = tf_loss_with_select(
                                    fn_normalize(t_res_inp), 
                                    fn_normalize(t_res_out),
                                    which_loss = ae_loss_type)
                    # loss_ae = tf_loss_with_select(
                    #                             tf.contrib.layers.instance_norm(t_res_inp), 
                    #                             tf.contrib.layers.instance_norm(t_res_out),
                    #     which_loss='smooth_l1', weight = t_weight)
                        # scale difference compensation by IN
                    loss_dict_Gene['loss_AE/' + name] = loss_ae
            # recon loss with trans
            for name, t_resi_imgs in net.resi_imgs.items():        
                if name.startswith('tran'):
                    loss_dict_Gene['loss_AE/' + name] = 0.5*tf_loss_with_select(
                                                            fn_normalize(t_resi_imgs),
                                                            fn_normalize(net.resi_outs['tran']),
                                                            which_loss = ae_loss_type)


        # unsupervised loss
        # applied to {unsup}
        def fn_downsize(images):
            smoother = Smoother({'data':images}, 11, 2.)
            images = smoother.get_output()
            return tf.image.resize_bilinear(images, [40,40])
        fn_normalize = tf.identity

        if params.get('bloss_unsup', 0):
            ae_loss_type = params.get('ae_loss_type', 'smooth_l1')
            loss_unsup = tf_loss_with_select(
                                fn_downsize(
                                    fn_normalize(net.resi_imgs['unsup'])
                                ), 
                                fn_downsize(
                                    fn_normalize(net.resi_outs['unsup'])
                                ),
                            which_loss = ae_loss_type)
            # loss_unsup = tf_loss_with_select(net.activations['unsup']['img2prog'][2],
            #                                  net.activations['unsup_feedback']['img2prog'][2],
            #                                  which_loss='smooth_l1')
            loss_dict_Gene['loss_unsup'] = loss_unsup

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

        # style losses on instruction decoder
        for gram_layer in params.get('gram_layers', []):
            path = gram_layer.split('/')
            if len(path) == 1:
                path = ['img2prog'].concat(path)
            for module_name, activations in net.activations['real'].items():
                if module_name != path[0]:
                    continue
                if path[1] == '*':
                    act_list = range(len(activations))
                else:
                    act_list = [ int(path[1]) ]
                synt_list = filter(
                    lambda name: name != 'real' and module_name in net.activations[name],
                    net.activations.keys())
                for i in act_list:
                    loss_prefix = 'loss_gram_' + module_name + '_' + str(i)
                    for synt in synt_list:
                        loss_name = loss_prefix + '/' + synt
                        t_real = activations[i]
                        t_synt = net.activations[synt][module_name][i]
                        t_loss = style_layer_loss(t_real, t_synt, params.get('gram_power', 2))
                        loss_dict_Gene[loss_name] = t_loss

        # accuracy measurements
        net.acc = { 'full' : dict(), 'fg': dict() }
        # applied to {all} except {unsup}
        for name, t_instr in net.instr.items():
            if name.startswith('unsup'):
                continue

            t_label = get_gt(name)
            t_mask  = get_mask(name)

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

def syntax_loss(instr, params, binary = False):
    '''
    Syntactic loss using valid pairwise transitions
    The transition number corresponds to the following neighborhood matrix

    1 2 3
    4 0 5
    6 7 8

    Instruction parameter depends on binary mode:
    - if binary == True, it must be BxHxWx1 (the label map)
    - if binary == False, it must be BxHxWxP (the logits)
    '''
    dataset = params['dataset']
    dx = [ -1, 0, 1, -1, 1, -1, 0, 1]
    dy = [ -1, -1, -1, 0, 0, 1, 1, 1]
    src_from = { -1: 1, 0: 0, 1: 0 }
    trg_from = { -1: 0, 0: 0, 1: 1 }
    rng_size = { -1: 19, 0: 20, 1: 19 }
    B, h, w, channels = instr.get_shape()
    syntax_weight = params.get('syntax_weight', 10) # used for binary case
    syntax_softmax = params.get('syntax_softmax', 1)
    # transform logits with softmax
    if not(binary) and syntax_softmax:
        instr = tf.nn.softmax(instr)
    losses = []
    for i in range(8):
        matname = os.path.join(dataset, 'syntax', 'T' + str(i+1) + '.txt')
        T = np.loadtxt(matname, delimiter = ',')
        # select target slice of instructions
        t_src_slice = tf.slice(instr,
            [0, src_from[dy[i]], src_from[dx[i]], 0],
            [B.value, rng_size[dy[i]], rng_size[dx[i]], channels.value])
        t_trg_slice = tf.slice(instr,
            [0, trg_from[dy[i]], trg_from[dx[i]], 0],
            [B.value, rng_size[dy[i]], rng_size[dx[i]], channels.value])
        # type of loss
        if binary:
            # binary => use label map
            t_src = tf.reshape(t_src_slice, [-1])
            t_trg = tf.reshape(t_trg_slice, [-1]) # instr[:, trg_y, trg_x, :], [-1, 1])
            t_indices = tf.stack([t_src, t_trg], axis = 1)
            # t_indices = debug(t_indices, 't_indices', dtype = tf.int64)
            t_count = tf.gather_nd(T, t_indices)
            # t_count = debug(t_count, 't_count')
            t_bad = tf.ones_like(t_count, tf.float32) * syntax_weight
            t_good = tf.zeros_like(t_count, tf.float32)
            t_loss = tf.where(t_count < 1.0, t_bad, t_good)
        else:
            # use logits
            #   loss = ||src x P x trg||_F
            # where src, trg are BxH'xW'x17 and P is 1x1x1x17x17 based on T
            #
            T[T >= 1] = 1
            P = tf.reshape(1 - T.astype(np.float32), [1,1,1,prog_ch,prog_ch]) # penalty matrix whose one entries denote invalid pairs
            P = tf.tile(P, [B.value, rng_size[dy[i]], rng_size[dx[i]], 1, 1])
            # compute loss using Einstein summation notation
            # note: we do not reduce as this is done automatically in the total loss computation
            t_loss = tf.einsum('bhwi,bhwij,bhwj->bhw', t_src_slice, P, t_trg_slice)
            t_loss = tf.reshape(fn_smooth_l1(t_loss), [-1, 1]) # to be able to use concat below
        # t_loss = debug(t_loss, 't_loss')
        losses.append(t_loss)
    t_loss = tf.concat(losses, axis = 0)
    if params.get('syntax_sum', 0):
        t_loss = tf.reduce_sum(t_loss)
    return t_loss

def debug(x, name = 'debug', dtype = tf.float32):
    return tf.py_func(lambda t: check_tensor(t, name), [x], dtype)

def check_tensor(x, name):
    print(name)
    print(x)
    pdb.set_trace()
    return x

'''
Losses from neural style transfer from cysmith
@see https://github.com/cysmith/neural-style-tf/
'''

def content_layer_loss(p, x, content_loss_function = 1):
    _, h, w, d = p.get_shape()
    M = h.value * w.value
    N = d.value
    if content_loss_function   == 1:
        K = 1. / (2. * N**0.5 * M**0.5)
    elif content_loss_function == 2:
        K = 1. / (N * M)
    elif content_loss_function == 3:
        K = 1. / 2.
    loss = K * tf.reduce_sum(tf.pow((x - p), 2))
    return loss


# https://stackoverflow.com/questions/41564321/split-image-tensor-into-small-patches
def style_layer_loss(X, Y, power = 2):
    B, h, w, d = X.get_shape()
    M = h.value * w.value
    N = d.value
    def gram_loss(pair):
        x, y = pair
        Gx = gram_matrix(x, M, N)
        Gy = gram_matrix(y, M, N)
        if power == 2:
            loss = (1. / (4 * N**2 * M**2)) * tf.reduce_sum(tf.pow((Gx - Gy), 2))
        elif power == 1:
            loss = tf.reduce_mean(tf.abs(Gx - Gy)) / 2.
        else:
            raise ValueError('Style loss does not support power %d' % power)
        return loss

    # map gram loss per batch pair
    return tf.map_fn(gram_loss, (X, Y), dtype=tf.float32)

def gram_matrix(x, area, depth):
    F = tf.reshape(x, (area, depth))
    G = tf.matmul(tf.transpose(F), F)
    return G
