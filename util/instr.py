import tensorflow as tf
import numpy as np
import scipy
from skimage.util import crop as imcrop
from PIL import Image
import pdb

palette = np.array([
    [ 255,   0,  16 ],
    [  43, 206,  72 ],
    [ 255, 255, 128 ],
    [  94, 241, 242 ],
    [   0, 129,  69 ],
    [   0,  92,  49 ],
    [ 255,   0, 190 ],
    [ 194,   0, 136 ],
    [ 126,   0, 149 ],
    [  96,   0, 112 ],
    [ 179, 179, 179 ],
    [ 128, 128, 128 ],
    [ 255, 230,   6 ],
    [ 255, 164,   4 ],
    [   0, 164, 255 ],
    [   0, 117, 220 ],
    [ 117,  59,  59 ]
])

mirror_mapping = np.array([
    0, 1, 2, 3, # KPTM
    6, 7,   # FR -> FL
    4, 5,   # FL -> FR
    10, 11, # BR -> BL
    8, 9,   # BL -> BR
    14, 15, # XR -> XL
    12, 13, # XL -> XR
    16      # S
]).astype(np.int32)

def tf_ind_to_rgb(t_ind):
    # HSV palette
    # t_rgb = tf.image.hsv_to_rgb(
    #     tf.concat(
    #         axis=3,
    #         values=[
    #             tf.cast(t_ind, dtype=tf.float32) / 34.,
    #             tf.ones(tf.shape(t_ind)),
    #             tf.ones(tf.shape(t_ind))
    #         ]))
    t_rgb = tf.cast(tf.concat(axis = 3,
        values = [
            tf.gather(palette[:, 0], t_ind),
            tf.gather(palette[:, 1], t_ind),
            tf.gather(palette[:, 2], t_ind)
        ]), dtype = tf.uint8)
    return t_rgb

def tf_mirror_image(t_img):
    return tf.image.flip_left_right(t_img)

def tf_mirror_instr(t_inst):
    return tf_mirror_image(tf.gather(mirror_mapping, t_inst))

def tf_mirror(t_img, t_inst):
    t_img = tf_mirror_image(t_img)
    t_inst = tf_mirror_instr(t_inst)
    return t_img, t_inst

def save_instr(fname, img):
    img = img[:,:,0].astype(np.uint8)
    img = Image.fromarray(img, mode = 'P')
    img.putpalette([
        255, 0, 16,
        43, 206, 72,
        255, 255, 128,
        94, 241, 242,
        0, 129, 69,
        0, 92, 49,
        255, 0, 190,
        194, 0, 136,
        126, 0, 149,
        96, 0, 112,
        179, 179, 179,
        128, 128, 128,
        255, 230, 6,
        255, 164, 4,
        0, 164, 255,
        0, 117, 220,
        117, 59, 59
    ])
    img.save(fname)
