import os
import tensorflow as tf
import numpy as np
from model import FeedForwardNetworks
from util import pp

def_learn_rate  = 0.0005
def_max_iter    = 150000
def_batch_size  = 2
def_im_size     = 160
def_threads     = def_batch_size
def_datapath    = './dataset'
def_seed        = 2018
def_model_type  = 'Forward'
def_ckpt_dir    = './checkpoint'
def_training    = True
def_weights     = ''
def_params      = ''
def_component   = ''
def_runit       = 'relu'
def_gram_layers = ''

flags = tf.app.flags
flags.DEFINE_float("learning_rate", def_learn_rate,     "Learning rate of for adam")
flags.DEFINE_integer("max_iter", def_max_iter,          "The size of total iterations")
flags.DEFINE_integer("batch_size", def_batch_size,      "The size of batch images")
flags.DEFINE_integer("image_size", def_im_size,         "The size of width or height of image to use")
flags.DEFINE_integer("threads", def_threads,            "The number of threads to use in the data pipeline")
flags.DEFINE_string("dataset", def_datapath,            "The dataset base directory")
flags.DEFINE_integer("seed", def_seed,                  "Random seed number")
flags.DEFINE_string("model_type", def_model_type,       "The type of model")
flags.DEFINE_string("checkpoint_dir", def_ckpt_dir,     "Directory name to save the checkpoints")
flags.DEFINE_boolean("training", def_training,          "True for training, False for testing")
flags.DEFINE_list("params", def_params,                 "Parameter map")
flags.DEFINE_list("weights", def_weights,               "Weight map")
flags.DEFINE_string("component", def_component,         "Component to train (all by default), "
                    + "valid values include: transfer | norendering | warping | nuclear | none")
flags.DEFINE_list("gram_layers", def_gram_layers,       "List of layers for the gram loss")
FLAGS = flags.FLAGS

model_dict = {
    "Forward": FeedForwardNetworks
}

def main(_):
    if not os.path.exists(FLAGS.checkpoint_dir):
    	os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.checkpoint_dir + '/train'):
        os.makedirs(FLAGS.checkpoint_dir + '/train')
    if not os.path.exists(FLAGS.checkpoint_dir + '/val'):
        os.makedirs(FLAGS.checkpoint_dir + '/val')

    NNModel = model_dict[FLAGS.model_type]
    tf.set_random_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement=True      
    # config.log_device_placement=True
    
    with tf.Session(config=config) as sess:
        for key, val in FLAGS.flag_values_dict().items():
            pp.pprint([key, getattr(FLAGS, key)])

        # object generation
        obj_model = NNModel(sess, tf_flag = FLAGS)

        # Train or Test
        if FLAGS.training:
            obj_model.train()
        else:
            obj_model.test()

if __name__ == '__main__':
    tf.app.run()
