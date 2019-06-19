#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import os
import sys
from PIL import Image
from scipy.misc import imsave
import pdb

from model import rendnet

flags = tf.app.flags
flags.DEFINE_string("render_type", "dense", "The renderer type")
flags.DEFINE_string("output_dir", "", "The output directory")
flags.DEFINE_integer("batch_size", 1, "Number of images to process at once")
flags.DEFINE_float("noise_level", 0.45, "Noise level to apply on the instruction")
FLAGS = flags.FLAGS

def read_image(fname):
    img = Image.open(fname)
    img = np.array(img)
    img = img[np.newaxis,:,:,np.newaxis].astype(np.int32)
    return img

def save_image(fname, data):
    imsave(fname, data)

def main(argv):

    # find list of inputs from positional arguments
    inputs = []
    for fname in argv:
        if fname.endswith('.png'):
            if os.path.exists(fname):
                inputs.append(fname)
            else:
                print('Input %s does not exist' % fname)
                return
    # check we have something to do
    if len(inputs) == 0:
        print('No input pattern. Nothing to do!')
        return

    # parameters
    render_type = FLAGS.render_type
    batch_size  = FLAGS.batch_size
    noise_level = FLAGS.noise_level
    render_weights = render_type
    if '_' in render_type:
        render_type = render_type.split('_')[0]
    output_dir  = FLAGS.output_dir
    if output_dir == '':
        output_dir = '.'

    # create graph
    t_input = tf.placeholder(tf.int32, shape = (batch_size, 20, 20, 1))
    t_img = rendnet.network(t_input, {
        'render_type': render_type,
        'noise_level': noise_level
    })

    # run graph on inputs
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement=True
    with tf.Session(config = config) as sess:

        # load rendering network into session
        rendnet.load_weights(sess, render_weights)

        # compute renderings by batch
        batch_start = 0
        batch_end   = len(inputs)
        while batch_start < batch_end:
            # load batch
            batch = []
            names = []
            for i in range(batch_size):
                img_idx = min(batch_end - 1, batch_start + i)
                fname = inputs[img_idx]
                img = read_image(fname)
                batch.append(img)
                names.append(fname)
                # pdb.set_trace()
            img_input = np.concatenate(batch)

            # compute renderings
            img_data = sess.run([t_img], feed_dict = { t_input: img_input })
            img_data = img_data[0]

            # save to file
            for i in range(batch_size):
                if batch_start + i >= batch_end:
                    continue
                img = np.squeeze(img_data[i, :, :, :])
                img = np.maximum(0, np.minimum(255, img * 255)).astype(np.uint8)
                fname = os.path.join(output_dir, os.path.basename(names[i]))
                save_image(fname, img)
                # pdb.set_trace()
                print('Saving %s' % fname)


            # udpate batch position
            batch_start += batch_size

    print('Done with %d files' % len(inputs))

if __name__ == '__main__':
    tf.app.run()
