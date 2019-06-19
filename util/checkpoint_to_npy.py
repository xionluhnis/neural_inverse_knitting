#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import os

flags = tf.app.flags
flags.DEFINE_string("input",  "", "The model checkpoint")
flags.DEFINE_string("output", "", "The output numpy file")
FLAGS = flags.FLAGS

def main(_):

    if FLAGS.input == '':
        print('You must specify --input value (--output is optional)')
        return

    # check input exists
    if not os.path.exists(FLAGS.input + '.meta'):
        print('Input %s.meta does not exist' % FLAGS.input)
        return

    # load metagraph and its weights
    meta = tf.train.import_meta_graph(FLAGS.input + '.meta',
        clear_devices=True)
    var_list = tf.get_collection('trainable_variables')

    # extract variable values
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement=True
    with tf.Session(config = config) as sess:
        meta.restore(sess, FLAGS.input)
        val_list = sess.run(var_list)

    # store variable in dict
    data = dict()
    for i in range(len(var_list)):
        var = var_list[i]
        val = val_list[i]
        data[var.name] = val

    # save whole data to file
    fname = FLAGS.output
    if fname == '':
        fname = 'output.npy'
    elif not fname.endswith('.npy'):
        fname += '.npy'
    np.save(fname, data)
    print('Saved %s' % fname)

if __name__ == '__main__':
    tf.app.run()
