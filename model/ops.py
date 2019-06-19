import math
import numpy as np
import tensorflow as tf

TF_EPS = tf.constant(1e-12, tf.float32)
tf_fn_expdim2to4 = lambda tensor,dim0,dim1: tf.expand_dims(tf.expand_dims(tensor, dim0), dim1)
tf_fn_expdim1to4 = lambda tensor,dim0,dim1,dim2: tf.expand_dims(tf.expand_dims(tf.expand_dims(tensor, dim0), dim1), dim2)


def tf_lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)


########################################
## Laplacian pyramid functions
##


def thoo_resz_img(t_img, scaling, target_img=None):
    if target_img != None:
        imsz = tf.shape(target_img)[1:3]
    else:
        imsz = tf.shape(t_img)
        imsz = tf.to_int32(tf.to_float(imsz[1:3]) * scaling)

    # ResizeMethod.BILINEAR,
    t_output = tf.image.resize_images(
        t_img, imsz, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return t_output


def laplacian_decomp_onestep(t_img, scale=2):
    dw_img = thoo_resz_img(t_img, 1. / scale)
    up_img = thoo_resz_img(dw_img, -1, t_img)
    return t_img - up_img, dw_img


def laplacian_recon_onestep(t_up_lapimg, t_dw_img, scale=2):
    up_img = thoo_resz_img(t_dw_img, -1, t_up_lapimg)
    return t_up_lapimg + up_img


def laplacian_pyr_decomposition(t_img, nscale, scale=2):
    # return: List (size: nscale + 1) of tensor images
    # t_lst_Lpyr[-1]: smallest scale blurred image
    t_lst_Lpyr = []

    t_L_img, t_D_img = laplacian_decomp_onestep(t_img, scale)
    t_lst_Lpyr.append(t_L_img)
    for iscale in range(nscale - 1):
        t_L_img, t_D_img = laplacian_decomp_onestep(t_D_img, scale)
        t_lst_Lpyr.append(t_L_img)
    t_lst_Lpyr.append(t_D_img)
    return t_lst_Lpyr


def multiscale_pyr_decomposition(t_img, nscale, scale=2.):
    # return: List (size: nscale + 1) of tensor images
    # t_lst_Lpyr[-1]: smallest scale blurred image
    t_lst_Lpyr = []

    t_L_img = thoo_resz_img(t_img, 1. / scale)
    t_lst_Lpyr.append(t_L_img)
    for iscale in range(nscale - 1):
        t_L_img = thoo_resz_img(t_L_img, 1. / scale)
        t_lst_Lpyr.append(t_L_img)
    t_lst_Lpyr.append(t_L_img)
    return t_lst_Lpyr


def laplacian_pyr_reconstruction(t_lst_Lpyr):
    t_lst_Lreconpyr = []
    t_dw_img = t_lst_Lpyr[-1]
    t_lst_Lreconpyr.append(t_dw_img)
    for iscale in range(len(t_lst_Lpyr) - 1):
        t_dw_img = laplacian_recon_onestep(t_lst_Lpyr[-iscale - 2], t_dw_img)
        t_lst_Lreconpyr.append(t_dw_img)
    t_ori_img = t_dw_img
    return t_ori_img, t_lst_Lreconpyr


def testcode_laplacian():
    cur_img = np.concatenate([
        np.zeros([48, 48], dtype=np.float32),
        np.ones([48, 48], dtype=np.float32)
    ], 1)

    t_input = tf.placeholder(tf.float32, shape=[None, None])

    t_input4d = tf_fn_expdim2to4(t_input, 0, -1)

    t_lst_Lpyr = laplacian_pyr_decomposition(t_input4d, nscale=3)

    t_ori_img = laplacian_pyr_reconstruction(t_lst_Lpyr)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        # exc_output = sess.run([t_lst_Lpyr[iscale] for iscale in range(len(t_lst_Lpyr))],
        #              feed_dict={t_input: cur_img})

        exec_output = sess.run(t_ori_img, feed_dict={t_input: cur_img})
        print(exec_output.shape)
        print(np.squeeze(exec_output))
        print(np.sum(np.square(cur_img - np.squeeze(exec_output))))


def instance_norm(input, name="instance_norm"):
    with tf.variable_scope(name):
        depth = input.get_shape()[3]
        scale = tf.get_variable(
            "scale", [depth],
            initializer=tf.random_normal_initializer(
                1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable(
            "offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1, 2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input - mean) * inv
        return scale * normalized + offset


class batch_norm(object):
    # h1 = lrelu(tf.contrib.layers.batch_norm(conv2d(h0, self.df_dim*2, name='d_h1_conv'),decay=0.9,updates_collections=None,epsilon=0.00001,scale=True,scope="d_h1_conv"))
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(
            x,
            decay=self.momentum,
            updates_collections=None,
            epsilon=self.epsilon,
            scale=True,
            scope=self.name)


def disc_conv(batch_input, out_channels, stride):
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable(
            "filter", [4, 4, in_channels, out_channels],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]
        padded_input = tf.pad(
            batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        conv = tf.nn.conv2d(
            padded_input, filter, [1, stride, stride, 1], padding="VALID")
        return conv


########################################
## Simple NN operations
##


def fn_amp_aware_filtering(t_input, amp, name='Amplitude_aware_filtering'):
    with tf.variable_scope(name):
        blurkernel = tf.constant(
            np.array([[0.002969, 0.013306, 0.021938, 0.013306, 0.002969],
                      [0.013306, 0.059634, 0.09832, 0.059634, 0.013306],
                      [0.021938, 0.09832, 0.162103, 0.09832, 0.021938],
                      [0.013306, 0.059634, 0.09832, 0.059634, 0.013306],
                      [0.002969, 0.013306, 0.021938, 0.013306, 0.002969]],
                     np.float32), tf.float32)
        blurkernel3D = tf.tile(tf.expand_dims(blurkernel, -1), (1, 1, 3))
        blurkernel4D = tf.expand_dims(blurkernel3D, -1)
        t_denorm = thoo_depthwise_conv2d(amp, blurkernel4D)
        t_numer = thoo_depthwise_conv2d(t_input * amp, blurkernel4D)
    return t_numer / (t_denorm + TF_EPS)


# The clockwise shift-1 rotation permutation.
permutation = [[1, 0], [0, 0], [0, 1], [2, 0], [1, 1], [0, 2], [2, 1], [2, 2],
               [1, 2]]


def shift_rotate(w, shift=1):
    shape = w.get_shape()
    for i in range(shift):
        w = tf.reshape(tf.gather_nd(w, permutation), shape)
    return w


def thoo_steer_conv_ch(input_,
                       output_dim,
                       k_h=3,
                       k_w=3,
                       d_h=1,
                       d_w=1,
                       name="steer_conv2d"):
    p_stride = [1, d_h, d_w, 1]
    with tf.variable_scope(name):

        Rch, Gch, Bch = tf.split(input_, num_or_size_splits=3, axis=3)

        w = tf.get_variable(
            'weight', [k_h, k_w, 1, output_dim],
            initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.get_variable(
            'biases', [output_dim], initializer=tf.constant_initializer(0.0))

        w_rot = [w]
        for i in range(3):
            w = shift_rotate(w, 2)
            w_rot.append(w)
        # w_rot.append(tf.transpose(w, [1,0,2,3]))

        convR = tf.concat([tf.nn.bias_add(\
             tf.nn.conv2d(Rch, w_i, strides=p_stride, padding='SAME') \
             , biases) \
                for w_i in w_rot], 3)

        convG = tf.concat([tf.nn.bias_add(\
             tf.nn.conv2d(Gch, w_i, strides=p_stride, padding='SAME') \
             , biases) \
                for w_i in w_rot], 3)
        convB = tf.concat([tf.nn.bias_add(\
             tf.nn.conv2d(Bch, w_i, strides=p_stride, padding='SAME') \
             , biases) \
                for w_i in w_rot], 3)

        conv = tf.concat([convR, convG, convB], 3)
        # conv = tf.concat([tf.nn.conv2d(input_, w_i, strides=p_stride, padding='SAME') \
        # for w_i in w_rot], 3)
        # conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        return conv


def thoo_steer_conv(input_,
                    output_dim,
                    k_h=3,
                    k_w=3,
                    d_h=1,
                    d_w=1,
                    name="steer_conv2d"):
    p_stride = [1, d_h, d_w, 1]
    with tf.variable_scope(name):
        w = tf.get_variable(
            'weight', [k_h, k_w, input_.get_shape()[-1], output_dim],
            initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.get_variable(
            'biases', [output_dim], initializer=tf.constant_initializer(0.0))

        w_rot = [w]
        for i in range(3):
            w = shift_rotate(w, 2)
            w_rot.append(w)
        # w_rot.append(tf.transpose(w, [1,0,2,3]))

        conv = tf.concat([tf.nn.bias_add(\
             tf.nn.conv2d(input_, w_i, strides=p_stride, padding='SAME') \
             , biases) \
                for w_i in w_rot], 3)
        # conv = tf.concat([tf.nn.conv2d(input_, w_i, strides=p_stride, padding='SAME') \
        # for w_i in w_rot], 3)
        # conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        return conv


def thoo_depthwise_conv2d(t_input, filter_init):
    #
    # filter_init: [filter_height, filter_width, in_channels, channel_multiplier]

    t_output = tf.nn.depthwise_conv2d(
        t_input,
        filter_init,
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='dw_conv2d')
    return t_output


def testcode_thoo_depthwise_conv2d():
    tmp = np.concatenate([
        np.zeros([5, 3], dtype=np.float32),
        np.ones([5, 3], dtype=np.float32)
    ], 1)
    t_input = tf.placeholder(tf.float32, shape=[None, None])

    t_input4d = tf_fn_expdim2to4(t_input, 0, -1)

    t_weight_dx = tf.constant([[0., 0., 0.], [0.5, 0., -0.5], [0., 0., 0.]],
                              tf.float32)
    t_weight_dy = tf.transpose(
        tf.constant([[0., 0., 0.], [0.5, 0., -0.5], [0., 0., 0.]], tf.float32))
    t_weight_dx = tf_fn_expdim2to4(t_weight_dx, -1, -1)
    t_weight_dy = tf_fn_expdim2to4(t_weight_dy, -1, -1)

    t_output_dx = thoo_depthwise_conv2d(t_input4d, t_weight_dx)
    t_output_dy = thoo_depthwise_conv2d(t_input4d, t_weight_dy)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        exc_output_dx, exc_output_dy = sess.run([t_output_dx, t_output_dy],
                                                feed_dict={t_input: tmp})

        print(exc_output_dx.shape)
        print(exc_output_dy.shape)

    # print(tmp)
    # print(np.squeeze(exc_output_dx))
    # print(np.squeeze(exc_output_dy))


def conv2d(input_,
           output_dim,
           k_h=5,
           k_w=5,
           d_h=2,
           d_w=2,
           stddev=0.02,
           t_padding='SAME',
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable(
            'weight', [k_h, k_w, input_.get_shape()[-1], output_dim],
            initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(
            input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable(
            'biases', [output_dim], initializer=tf.constant_initializer(0.0))
        # conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        conv = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

        return conv


def conv2d_nobias(input_,
                  output_dim,
                  k_h=5,
                  k_w=5,
                  d_h=2,
                  d_w=2,
                  stddev=0.02,
                  t_padding='SAME',
                  name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable(
            'weight', [k_h, k_w, input_.get_shape()[-1], output_dim],
            initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(
            input_, w, strides=[1, d_h, d_w, 1], padding=t_padding)
        return conv


def convt2d(input_,
            output_dim,
            k_h=5,
            k_w=5,
            d_h=2,
            d_w=2,
            stddev=0.02,
            name="convt2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('weight', [k_h, k_w, input_.get_shape()[-1], output_dim], \
         initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d_transpose(
            input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable(
            'biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv


def rgb2ntsc(rgb):
    srgb_pixels = tf.reshape(rgb, [-1, 3])
    rgb_to_yiq = tf.constant([
        #    X        Y          Z
        [0.299, 0.587, 0.114],  # R
        [0.596, -0.274, -0.322],  # G
        [0.211, -0.523, 0.312],  # B
    ])
    yiq_pixels = tf.matmul(srgb_pixels, tf.transpose(rgb_to_yiq))
    return tf.reshape(yiq_pixels, tf.shape(rgb))


def ntsc2rgb(yiq):
    yiq_pixels = tf.reshape(yiq, [-1, 3])
    yiq_to_rgb = tf.constant([
        #    X        Y          Z
        [1., 0.956, 0.621],  # R
        [1., -0.272, -0.647],  # G
        [1., -1.106, 1.703],  # B
    ])
    rgb_pixels = tf.matmul(yiq_pixels, tf.transpose(yiq_to_rgb))
    return tf.reshape(rgb_pixels, tf.shape(yiq))


def preprocess_lab(lab):
    with tf.name_scope("preprocess_lab"):
        L_chan, a_chan, b_chan = tf.unstack(lab, axis=3)
        # L_chan: black and white with input range [0, 100]
        # a_chan/b_chan: color channels with input range ~[-110, 110], not exact
        # [0, 100] => [-1, 1],  ~[-110, 110] => [-1, 1]
        return tf.stack([L_chan / 50. - 1, a_chan / 110., b_chan / 110.],
                        axis=3)
        # return tf.stack([L_chan*2 - 100, a_chan, b_chan], axis=3)


def deprocess_lab(lab):
    with tf.name_scope("deprocess_lab"):
        L_chan, a_chan, b_chan = tf.unstack(lab, axis=3)
        # this is axis=3 instead of axis=2 sbecause we process individual images but deprocess batches
        return tf.stack(
            [(L_chan + 1.) / 2. * 100., a_chan * 110., b_chan * 110.], axis=3)


# based on https://github.com/torch/image/blob/9f65c30167b2048ecbe8b7befdc6b2d6d12baee9/generic/image.c
def rgb_to_lab(srgb):
    with tf.name_scope("rgb_to_lab"):
        srgb = check_image(srgb)
        srgb_pixels = tf.reshape(srgb, [-1, 3])

        with tf.name_scope("srgb_to_xyz"):
            linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
            exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
            rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + ((
                (srgb_pixels + 0.055) / 1.055)**2.4) * exponential_mask
            rgb_to_xyz = tf.constant([
                #    X        Y          Z
                [0.412453, 0.212671, 0.019334],  # R
                [0.357580, 0.715160, 0.119193],  # G
                [0.180423, 0.072169, 0.950227],  # B
            ])
            xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("xyz_to_cielab"):
            # convert to fx = f(X/Xn), fy = f(Y/Yn), fz = f(Z/Zn)

            # normalize for D65 white point
            xyz_normalized_pixels = tf.multiply(
                xyz_pixels, [1 / 0.950456, 1.0, 1 / 1.088754])

            epsilon = 6 / 29
            linear_mask = tf.cast(
                xyz_normalized_pixels <= (epsilon**3), dtype=tf.float32)
            exponential_mask = tf.cast(
                xyz_normalized_pixels > (epsilon**3), dtype=tf.float32)
            fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4 / 29
                             ) * linear_mask + (xyz_normalized_pixels**
                                                (1 / 3)) * exponential_mask

            # convert to lab
            fxfyfz_to_lab = tf.constant([
                #  l       a       b
                [0.0, 500.0, 0.0],  # fx
                [116.0, -500.0, 200.0],  # fy
                [0.0, 0.0, -200.0],  # fz
            ])
            lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant(
                [-16.0, 0.0, 0.0])

        return preprocess_lab(tf.reshape(lab_pixels, tf.shape(srgb)))


def lab_to_rgb(lab):
    with tf.name_scope("lab_to_rgb"):
        lab = check_image(lab)
        lab_pixels = tf.reshape(deprocess_lab(lab), [-1, 3])

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("cielab_to_xyz"):
            # convert to fxfyfz
            lab_to_fxfyfz = tf.constant([
                #   fx      fy        fz
                [1 / 116.0, 1 / 116.0, 1 / 116.0],  # l
                [1 / 500.0, 0.0, 0.0],  # a
                [0.0, 0.0, -1 / 200.0],  # b
            ])
            fxfyfz_pixels = tf.matmul(
                lab_pixels + tf.constant([16.0, 0.0, 0.0]), lab_to_fxfyfz)

            # convert to xyz
            epsilon = 6 / 29
            linear_mask = tf.cast(fxfyfz_pixels <= epsilon, dtype=tf.float32)
            exponential_mask = tf.cast(
                fxfyfz_pixels > epsilon, dtype=tf.float32)
            xyz_pixels = (3 * epsilon**2 *
                          (fxfyfz_pixels - 4 / 29)) * linear_mask + (
                              fxfyfz_pixels**3) * exponential_mask

            # denormalize for D65 white point
            xyz_pixels = tf.multiply(xyz_pixels, [0.950456, 1.0, 1.088754])

        with tf.name_scope("xyz_to_srgb"):
            xyz_to_rgb = tf.constant([
                #     r           g          b
                [3.2404542, -0.9692660, 0.0556434],  # x
                [-1.5371385, 1.8760108, -0.2040259],  # y
                [-0.4985314, 0.0415560, 1.0572252],  # z
            ])
            rgb_pixels = tf.matmul(xyz_pixels, xyz_to_rgb)
            # avoid a slightly negative number messing up the conversion
            rgb_pixels = tf.clip_by_value(rgb_pixels, 0.0, 1.0)
            linear_mask = tf.cast(rgb_pixels <= 0.0031308, dtype=tf.float32)
            exponential_mask = tf.cast(
                rgb_pixels > 0.0031308, dtype=tf.float32)
            srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + (
                (rgb_pixels**(1 / 2.4) * 1.055) - 0.055) * exponential_mask

        return tf.reshape(srgb_pixels, tf.shape(lab))


def check_image(image):
    assertion = tf.assert_equal(
        tf.shape(image)[-1], 3, message="image must have 3 color channels")
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    if image.get_shape().ndims not in (3, 4):
        raise ValueError("image must be either 3 or 4 dimensions")

    # make the last dimension 3 so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image


def put_kernels_on_grid(kernel, grid_Y, grid_X, pad=1):
    '''Visualize conv. features as an image (mostly for the 1st layer).
	Place kernel into a grid, with some paddings between adjacent filters.

	Args:
		kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
		(grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
												 User is responsible of how to break into two multiples.
		pad:               number of black pixels around each filter (between them)

	Return:
		Tensor of shape [(Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels, 1].
	'''

    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)

    kernel1 = (kernel - x_min) / (x_max - x_min)

    # pad X and Y
    x1 = tf.pad(
        kernel1,
        tf.constant([[pad, pad], [pad, pad], [0, 0], [0, 0]]),
        mode='CONSTANT')

    # X and Y dimensions, w.r.t. padding
    Y = kernel1.get_shape()[0] + 2 * pad
    X = kernel1.get_shape()[1] + 2 * pad

    channels = kernel1.get_shape()[2]

    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    # organize grid on Y axis
    x3 = tf.reshape(x2, [grid_X, Y * grid_Y, X, channels])  #3

    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4, [1, X * grid_X, Y * grid_Y, channels])  #3

    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))

    # scale to [0, 255] and convert to uint8
    return tf.image.convert_image_dtype(x7, dtype=tf.uint8)


def put_activation_on_grid(activ, grid_Y, grid_X, name, pad=1):
    '''Visualize conv. features as an image (mostly for the 1st layer).
	Place kernel into a grid, with some paddings between adjacent filters.

	Args:
		kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
		(grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
												 User is responsible of how to break into two multiples.
		pad:               number of black pixels around each filter (between them)

	Return:
		Tensor of shape [(Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels, 1].
	'''
    activation_tensor = tf.slice(activ, [0, 0, 0, 0], [1, -1, -1, -1])

    x_min = tf.reduce_min(activation_tensor)
    x_max = tf.reduce_max(activation_tensor)

    activ1 = (activation_tensor - x_min) / (x_max - x_min)

    # pad X and Y
    x1 = tf.pad(
        activ1,
        tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]]),
        mode='CONSTANT')

    # X and Y dimensions, w.r.t. padding
    Y = tf.shape(activ1)[1] + 2 * pad
    X = tf.shape(activ1)[2] + 2 * pad
    channels = 1

    x2 = tf.reshape(x1, tf.stack([Y, X, grid_Y, grid_X]))
    x3 = tf.transpose(x2, (2, 0, 3, 1))
    V = tf.reshape(x3, (1, grid_Y * Y, grid_X * X, 1))

    # scale to [0, 255] and convert to uint8
    Vq = tf.image.convert_image_dtype(V, dtype=tf.uint8)

    return tf.summary.image(name, Vq, max_outputs=1)
