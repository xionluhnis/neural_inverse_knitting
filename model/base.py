import os
from glob import glob
import tensorflow as tf
import re


class Model(object):
    """Abstract object representing an Reader model."""

    def __init__(self):
        self.vocab = None
        self.data = None

    def get_model_dir(self):
        model_dir = ""
        for attr in self._attrs:
            if hasattr(self, attr):
                model_dir += "_%s-%s" % (attr, getattr(self, attr))
        return model_dir

    def find_model_dir(self, checkpoint_dir):
        def model_dir(batch):
            dpath = ""
            for attr in self._attrs:
                if hasattr(self, attr):
                    if attr == 'batch':
                        value = batch
                    else:
                        value = getattr(self, attr)
                    dpath += "_%s-%s" % (attr, value)
            return dpath

        for i in range(1, 65):
            mdl_dir = model_dir(str(i))
            path = os.path.join(checkpoint_dir, mdl_dir)
            if os.path.exists(path):
                return mdl_dir
        return self.get_model_dir()


    def save(self, checkpoint_dir, global_step=None):
        self.saver = tf.train.Saver(max_to_keep=5)

        print(" [*] Saving checkpoints...")
        model_name = type(self).__name__ or "Reader"
        model_dir = self.get_model_dir()

        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(
            self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=global_step)

    def load(self, checkpoint_dir, needed = False):

        # count parameters
        param_count = 0
        for var in tf.global_variables():
            if (re.search('generator', var.name) != None):
                shape = var.get_shape()
                var_params = 1
                for dim in shape:
                    var_params *= dim.value
                param_count += var_params
        print('Generator variables: %d' % param_count)

        # temporary
        if 0:
            select_vars = [
                var for var in tf.global_variables()
                if (re.search("generator", var.name) != None)
            ]

            for var in select_vars:
                print(var.name + '\n')

            self.saver = tf.train.Saver(var_list=select_vars, max_to_keep=5)
        else:
            self.saver = tf.train.Saver(max_to_keep=5)

        print(" [*] Loading checkpoints...")
        if needed:
            model_dir = self.find_model_dir(checkpoint_dir)
        else:
            model_dir = self.get_model_dir()
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt.model_checkpoint_path)
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess,
                               os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Load SUCCESS")
            return True
        else:
            print(" [!] Load failed...")
            if needed:
                raise FileNotFoundError(checkpoint_dir)
            return False
