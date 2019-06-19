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

from .layer_modules import prog_ch, tf_MILloss_xentropy, tf_loss_xentropy, tf_MILloss_accuracy, tf_background, syntax_loss, tf_accuracy, create_canonical_coordinates, oper_random_geo_perturb, oper_img2img, oper_img2prog_final, oper_img2prog_final_complex
from . import rendnet
from .danet import generator_unet

def model_composited(t_imgs_dict, t_labels_dict, params = dict()):
    '''
    Compose the base network model
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
    net.discr = dict()
    net.render = dict()
    net.render_layers = dict()

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
        if params.get('local_warping', 0):
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

        if not params.get('istest', False): # if training
            noise_sigma = params.get('noise_sigma', 3./255.)
            t_noise = tf.random_normal(tf.shape(t_img), stddev = noise_sigma)
            net.resi_imgs[key] = net.resi_imgs[key] + t_noise
        
        net.resi_imgs_noaug[key] = net.resi_imgs_noaug[key] - value

    # create generator
    with tf.variable_scope("generator"):

        def encoder(t_input, name):
            return t_latent

        for name, t_resi_inp in net.resi_imgs.items():
            print('**************oper_img2img')
            with runits('relu') as activations:

                reduce_type = params.get('reduce_type', 'conv')
                if reduce_type == 'conv':
                    t_logits = oper_img2prog_final(t_resi_inp, params, 'img2prog')
                elif reduce_type == 'conv_skip':
                    t_logits = oper_img2prog_final_complex(t_resi_inp, params, 'img2prog')
                elif reduce_type == 'tiling':
                    with tf.name_scope('tiling'):
                        t_input  = tf.space_to_depth(t_resi_inp, 8)
                    t_logits = oper_img2img(t_input, prog_ch, params, 'img2prog')
                elif reduce_type == 'avg_pooling':
                    t_logits = oper_img2img(t_resi_inp, prog_ch, params, 'img2prog')
                    t_logits = tf.contrib.layers.avg_pool2d(t_logits, [8,8], 8)
                elif reduce_type == 'max_pooling':
                    t_logits = oper_img2img(t_resi_inp, prog_ch, params, 'img2prog')
                    t_logits = tf.contrib.layers.max_pool2d(t_logits, [8,8], 8)
                elif reduce_type == 'unet':
                    t_logits = generator_unet(t_resi_inp, prog_ch, params, 'img2prog')
                    t_logits = tf.contrib.layers.avg_pool2d(t_logits, [8,8], 8)
                else:
                    raise ValueError('Invalid reduction type %s' % reduce_type)

                t_instr = tf.argmax(t_logits, axis = 3, name = 'prediction')
                net.latent[name] = t_logits
                net.logits[name] = t_logits
                net.instr[name]  = tf.expand_dims(t_instr, axis = 3)
                store_act(name, 'img2prog', activations)

    return net

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

    # rendering for ho-syntax loss
    if params.get('use_hosyntax', 0):
        net.render = dict()
        net.render_layers = dict()
        # render inputs
        for name, t_logits in net.logits.items():
            # t_label = tf.nn.softmax(t_logits)
            # @see https://arxiv.org/abs/1503.02531
            t_label = tf.nn.softmax((t_logits - tf.reduce_mean(t_logits, axis = -1, keep_dims = True)) / 2)
            print('**************rendnet.network ' + name)
            net.render[name], net.render_layers[name] = rendnet.network(
                t_label, params, output_layers = True, input_is_softmax = True
            )
            net.resi_outs[name] = net.render[name]
        # render gt
        for name, t_instr in t_inst_dict.items():
            print('**************rendnet.network ' + name)
            net.render[name], net.render_layers[name] = rendnet.network(
                t_instr, params, output_layers = True, input_is_softmax = False
            )
            net.resi_outs[name] = net.render[name]

    # generator and discriminator losses
    with tf.variable_scope("loss"):

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

        # syntax loss
        # applied to {all} except {unsup}
        if params.get('use_syntax', 1):
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

        # higher-order syntax loss
        if params.get('use_hosyntax', 0):
            # different types
            render_layers = params.get('hos_layers', -1)
            if isinstance(render_layers, str):
                if render_layers == '':
                    render_layers = []
                else:
                    layer_list = render_layers.split(':')
                    render_layers = [int(id) for id in layer_list]
            elif not isinstance(render_layers, list):
                render_layers = [int(render_layers)]
            hos_loss_type = params.get('hos_loss_type', 'l2')

            # compute hos loss for each type of data
            for name in net.instr.keys():
                if name.startswith('unsup'):
                    continue

                if is_real(name):
                    gt_img = net.render['instr_real']
                    gt_layers = net.render_layers['instr_real']
                else:
                    gt_img = net.render['instr_synt']
                    gt_layers = net.render_layers['instr_synt']
                out_img = net.render[name]
                out_layers = net.render_layers[name]
                for layer_id in render_layers:
                    # pdb.set_trace()
                    loss_hos = tf_loss_with_select(
                        gt_layers[layer_id], 
                        out_layers[layer_id], 
                        which_loss = hos_loss_type)
                    loss_dict_Gene['loss_HOS/' + str(layer_id) + '/' + name] = loss_hos
                if params.get('hos_img', 0):
                    loss_hos = tf_loss_with_select(gt_img, out_img, which_loss = hos_loss_type)
                    loss_dict_Gene['loss_HOS/img/' + name] = loss_hos

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
