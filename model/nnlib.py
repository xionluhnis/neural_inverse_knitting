#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import division
import os
import scipy.misc
import numpy as np
from math import floor
from PIL import Image
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from contextlib import contextmanager
import re
""" helper functions """


def loadimg(fn, scale=4):
    try:
        img = Image.open(fn).convert('RGB')
    except IOError:
        return None
    w, h = img.size
    img.crop((0, 0, floor(w / scale), floor(h / scale)))
    img = img.resize((w // scale, h // scale), Image.ANTIALIAS)
    return np.array(img) / 255


def saveimg(img, filename):
    img = 255 * np.copy(img)
    if len(np.shape(img)) > 2 and np.shape(img)[2] == 1:
        img = np.reshape(img, (np.shape(img)[0], np.shape(img)[1]))
    img = scipy.misc.toimage(img, cmin=0, cmax=255)
    scipy.misc.imsave(filename, img)


""" loss layers (spatial preserving) """


def tf_loss_with_select(tensor1, tensor2, which_loss="l2", b_normalize=False, weight = None):
    if which_loss == "l1":
        loss = loss_l1(tensor1, tensor2, b_normalize)
    elif which_loss == "smooth_l1":
        loss = loss_smoothl1(tensor1, tensor2, b_normalize)
    elif which_loss == "ncos":
        loss = loss_neg_cos(tensor1, tensor2)
    elif which_loss == "l1+l2":
        loss = loss_l1(tensor1, tensor2, b_normalize) + loss_l2(tensor1, tensor2, b_normalize)
    elif which_loss == "l2":
        loss = loss_l2(tensor1, tensor2, b_normalize)
    elif which_loss == "ssim":
        loss = 1 - tf.image.ssim(tensor1, tensor2, max_val=1.0)
    elif which_loss == "msssim":
        loss = 1 - tf.image.ssim_multiscale(tensor1, tensor2, max_val=1.0)
    else:
        raise ValueError('Unsupported loss %s' % which_loss)
    if weight is None:
        return loss
    else:
        return tf.multiply(weight, loss)

def tf_normalize_tensor(tensor):
    tensor1d = tf.reshape(tensor, [tf.shape(tensor)[0], -1])
    tensor = tf.nn.l2_normalize(tensor1d, axis=1)  # norm(each row) == 1
    return tensor


def loss_neg_cos(tensor1, tensor2):
    tensor1 = tf_normalize_tensor(tensor1)
    tensor2 = tf_normalize_tensor(tensor2)
    # To Do: reduce_sum except the 1st dim
    return 1. - tf.reduce_sum(tf.multiply(tensor1, tensor2), axis=1)


def loss_l1(tensor1, tensor2, b_normalize=False):
    if b_normalize:
        tensor1 = tf_normalize_tensor(tensor1)
        tensor2 = tf_normalize_tensor(tensor2)
    return tf.abs(tensor1 - tensor2)


def loss_smoothl1(tensor1, tensor2, b_normalize=False):
    if b_normalize:
        tensor1 = tf_normalize_tensor(tensor1)
        tensor2 = tf_normalize_tensor(tensor2)
    return tf.sqrt(tf.square(tensor1 - tensor2) + 1e-8)
    # fn_smoothl1 =  lambda tensor:


def loss_l2(tensor1, tensor2, b_normalize=False):
    if b_normalize:
        tensor1 = tf_normalize_tensor(tensor1)
        tensor2 = tf_normalize_tensor(tensor2)
    return tf.square(tensor1 - tensor2)


def loss_huber(labels, predictions, delta=1.0):
    residual = tf.abs(predictions - labels)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    return tf.where(condition, small_res, large_res)


""" neural network layers """


def avgpool(h, s=2, k=2):
    h = tf.contrib.layers.avg_pool2d(
        h, kernel_size=k, stride=s, padding='VALID')
    return h


def maxpool(h, s=2, k=2):
    h = tf.contrib.layers.max_pool2d(
        h, kernel_size=k, stride=s, padding='SAME')
    return h


def global_pool(t_feat, op = 'max'):
    b, h, w, _ = t_feat.get_shape()
    if op == 'max':
        return tf.nn.max_pool(t_feat, [1, h.value, w.value, 1], [1, 1, 1, 1], 'VALID')
    elif op == 'maxred':
        return tf.reduce_max(t_feat, [1, 2], True)
    elif op == 'avg':
        return tf.nn.avg_pool(t_feat, [1, h.value, w.value, 1], [1, 1, 1, 1], 'VALID')
    else:
        raise ValueError('Unsupported operation %s' % op)


def conv_pad(h, n=64, s=1, k=3):
    padsz = k // 2
    h = tf.pad(
        h,
        tf.constant([[0, 0], [padsz, padsz], [padsz, padsz], [0, 0]]),
        mode='SYMMETRIC')
    h = tf.contrib.layers.convolution2d(
        h, n, kernel_size=k, stride=s, padding='VALID', activation_fn=None)
    return h

def conv_valid(h, n=64, s=1, k=3):
    h = tf.contrib.layers.convolution2d(
        h, n, kernel_size=k, stride=s, padding='VALID', activation_fn=None)
    return h


def conv2d_weightnorm(h, n, k, s, gain=np.sqrt(2), use_wscale=False):
    w = get_weight([k, k, h.shape[-1].value, n], gain=gain, use_wscale=use_wscale)
    w = tf.cast(w, h.dtype)
    return tf.nn.conv2d(h, w, strides=[1,s,s,1], padding='SAME')

def conv(h, n=64, s=1, k=3, w_initializer=None, w_normalization=True):
    if n == -1:
        n = h.get_shape()[-1] # use input's shape
    
    if w_initializer is None:
        h = tf.contrib.layers.convolution2d(
            h, n, kernel_size=k, stride=s, padding='SAME', activation_fn=None)
    else:
        h = tf.contrib.layers.convolution2d(
            h,
            n,
            kernel_size=k,
            stride=s,
            padding='SAME',
            activation_fn=None,
            weights_initializer=w_initializer)

    return h


def fc(h, n=1024, w_initializer=None):
    if w_initializer is None:
        h = tf.contrib.layers.fully_connected(
            tf.contrib.layers.flatten(h), num_outputs=n, activation_fn=None)
    else:
        h = tf.contrib.layers.fully_connected(
            tf.contrib.layers.flatten(h),
            num_outputs=n,
            activation_fn=None,
            weights_initializer=w_initializer)
    return h

def_runit = 'relu'
def set_runit(rtype):
    global def_runit
    def_runit = rtype

runit_list = []
def search_runit(name):
    for runit in runit_list:
        if re.search(name, runit.name) != None:
            yield runit

runit_ctx = []
@contextmanager
def runits(rtype = None):
    global def_runit

    rlist = []
    runit_ctx.append(rlist)

    if rtype is not None:   # temporarily change activation func.
        tmp_runit = def_runit
        set_runit(rtype)

    yield rlist
    runit_ctx.pop()

    if rtype is not None:
        set_runit(tmp_runit)

def runit(h, rtype = None):
    global def_runit
    if rtype is None:
        rtype = def_runit
    
    print(rtype)
    if rtype == 'relu':
        out = relu(h)
    elif rtype == 'elu':
        out = elu(h)
    elif rtype == 'lrelu':
        out = lrelu(h)
    elif rtype == 'selu':
        out = selu(h)
    elif rtype == 'relu_bn':
        out = relu(h)
        out = batch_norm(out, True, False)
    elif rtype == 'relu_bns':
        out = relu(h)
        out = batch_norm(out, True, True)
    elif rtype == 'relu_bn_test':
        out = relu(h)
        out = batch_norm(out, False, False)
    elif rtype == 'relu_bns_test':
        out = relu(h)
        out = batch_norm(out, False, True)
    elif rtype == 'lrelu_pn':
        out = lrelu(h)
        out = pixel_norm(out)
    elif rtype == 'relu_pn':
        out = relu(h)
        out = pixel_norm(out)
    elif rtype == 'relu_in':
        out = relu(h)
        out = tf.contrib.layers.instance_norm(out)
    elif rtype == 'in_relu':
        out = tf.contrib.layers.instance_norm(h,
            scale = False, activation_fn = relu)
    elif rtype == 'lrelu_in':
        out = lrelu(h)
        out = tf.contrib.layers.instance_norm(out)
    elif rtype == 'in_lrelu':
        out = tf.contrib.layers.instance_norm(h,
            scale = False, activation_fn = lrelu)
    elif rtype == 'bn_relu':
        out = tf.contrib.layers.batch_norm(h,
            decay = 0.9, scale = False, activation_fn = relu, is_training = True)
    elif rtype == 'bn_relu_test':
        out = tf.contrib.layers.batch_norm(h,
            decay = 0.9, scale = False, activation_fn = relu, is_training = False)
    elif rtype == 'glu':
        # gated linear unit
        # @see https://arxiv.org/abs/1612.08083
        # => split in half for gate and activations
        F = h.get_shape()[-1].value
        acti = h[:, :, :, :F//2]
        gate = h[:, :, :, F//2:]
        mask = tf.sigmoid(gate)
        out = tf.multiply(acti, mask)
    else:
        raise ValueError('Invalid rectifier type %s' % rtype)
    # store activation output
    runit_list.append(out) # global list
    if len(runit_ctx) > 0:
        runit_ctx[-1].append(out) # local context
    # return it
    return out

def in_relu(h):
    print('in_relu')
    h = tf.contrib.layers.instance_norm(h,
            scale = False, activation_fn = relu)
    return h

def relu(h):
    h = tf.nn.relu(h)
    return h


def elu(h):
    h = tf.nn.elu(h)
    return h


def lrelu(h, a=0.2):
    h = tf.nn.leaky_relu(h, alpha=a)
    return h


def selu(h):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * tf.where(h > 0.0, h, alpha * tf.exp(h) - alpha)


def contribut_group_norm(x, groups=32):
    # normalize
    # tranpose: [bs, h, w, c] to [bs, c, h, w] following the paper
    with tf.variable_scope('GroupNorm', reuse=False) as sc:
        # values=[x],
        eps = 1e-5
        G = groups
        x = tf.transpose(x, [0, 3, 1, 2])
        N, C, H, W = x.get_shape().as_list()
        G = min(G, C)
        x = tf.reshape(x, [-1, G, C // G, H, W])
        # counts, means_ss, variance_ss, _ = nn.sufficient_statistics(inputs, moments_axes, keep_dims=True)
        # mean, variance = nn.normalize_moments(counts, means_ss, variance_ss, shift=None)
        mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
        x = (x - mean) * tf.rsqrt(var + eps)
        # per channel gamma and beta
        gamma = tf.Variable(
            tf.constant(1.0, shape=[C]), dtype=tf.float32, name='gamma')
        beta = tf.Variable(
            tf.constant(0.0, shape=[C]), dtype=tf.float32, name='beta')
        gamma = tf.reshape(gamma, [1, C, 1, 1])
        beta = tf.reshape(beta, [1, C, 1, 1])

        output = tf.reshape(x, [-1, C, H, W]) * gamma + beta
        # tranpose: [bs, c, h, w, c] to [bs, h, w, c] following the paper
        output = tf.transpose(output, [0, 2, 3, 1])
    return output


def grp_norm(h, ngroup=32, invaxis=(-3, -2)):
    h = contribut_group_norm(h, groups=ngroup)
    # h = tf.contrib.layers.group_norm(h, groups = ngroup, reduction_axes=invaxis)
    return h


def batch_norm(h, is_training=True, scale=True):
    # h = contribut_group_norm(h, groups = ngroup)
    h = tf.contrib.layers.batch_norm(
        h, decay=0.9, scale=scale, is_training=is_training)
    return h


def inst_norm(h, name='inst_norm'):
    with tf.variable_scope(name):
        h = tf.contrib.layers.instance_norm(h)
    return h

def pixel_norm(x, epsilon=1e-8):
    with tf.variable_scope('PixelNorm'):
        return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=3, keepdims=True) + epsilon)

# def upsample(h):
#     # dynamic shape version
#     h = tf.image.resize_images(h, [2*tf.shape(h)[1], 2*tf.shape(h)[2]],
#                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#     return h


# NEAREST_NEIGHBOR, BILINEAR
def upsample(h, blinear=True, scale = 2):
    # static shape version
    h = tf.image.resize_images(
        h, [scale * h.get_shape()[1], scale * h.get_shape()[2]],
        method=tf.image.ResizeMethod.BILINEAR if blinear else tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # h = tf.image.resize_images(h, [2*tf.shape(h)[1], 2*tf.shape(h)[2]],
    #                            method=tf.image.ResizeMethod.BILINEAR)
    return h


def resi(h, sublayers):
    htemp = h
    for layer in sublayers:
        h = layer[0](h, *layer[1:])
    h += htemp
    return h

def NN(name, layers, output_layers = False):
    ilayer = 0
    h = layers[0]
    intermediate = []
    with tf.variable_scope(name, reuse=False) as scope:
        for layer in layers[1:]:
            with tf.variable_scope('{}'.format(ilayer)):
                h = layer[0](h, *layer[1:])
                intermediate.append(h)
            ilayer += 1
    if output_layers:
        return h, intermediate
    else:
        return h

def seq(h, layers):
    for layer in layers:
        h = layer[0](h, *layer[1:])
    return h

def branch(h, sublayers1, sublayers2, concat_axis = -1):
    h1 = seq(h, sublayers1)
    h2 = seq(h, sublayers2)
    h = tf.concat([h1, h2], axis=concat_axis)
    return h




# Written by Antonio Loquercio
import scipy.stats as st

def layer(op):
    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated


class Smoother(object):
    def __init__(self, inputs, filter_size, sigma):
        self.inputs = inputs
        self.terminals = []
        self.layers = dict(inputs)
        self.filter_size = filter_size
        self.sigma = sigma
        self.setup()

    def setup(self):
        (self.feed('data')
             .conv(name = 'smoothing'))

    def get_unique_name(self, prefix):
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def feed(self, *args):
        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            if isinstance(fed_layer, str):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

    def gauss_kernel(self, kernlen=21, nsig=3, channels=1):
        interval = (2*nsig+1.)/(kernlen)
        x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw/kernel_raw.sum()
        out_filter = np.array(kernel, dtype = np.float32)
        out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
        out_filter = np.repeat(out_filter, channels, axis = 2)
        return out_filter

    def make_gauss_var(self, name, size, sigma, c_i):
        with tf.device("/cpu:0"):
            kernel = self.gauss_kernel(size, sigma, c_i)
            var = tf.Variable(tf.convert_to_tensor(kernel), name = name)
        return var

    def get_output(self):
        '''Returns the smoother output.'''
        return self.terminals[-1]

    @layer
    def conv(self,
             input,
             name,
             padding='SAME'):
        # Get the number of channels in the input
        c_i = input.get_shape().as_list()[3]
        # Convolution for a given input and kernel
        convolve = lambda i, k: tf.nn.depthwise_conv2d(i, k, [1, 1, 1, 1],
                                                             padding=padding)
        with tf.variable_scope(name) as scope:
            kernel = self.make_gauss_var('gauss_weight', self.filter_size,
                                                         self.sigma, c_i)
            output = convolve(input, kernel)
            return output
