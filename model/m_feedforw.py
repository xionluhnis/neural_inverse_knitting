import os
import sys
import time
import tensorflow as tf
import numpy as np
import scipy
from skimage.util import crop as imcrop
import re
import pdb
import pickle
from PIL import Image

from util import Loader, merge, tf_variable_summary, tf_keypoint_summary, tf_ind_to_rgb, tf_mirror_instr, tf_mirror_image, tf_summary_confusionmat, save_instr

from . import layer_modules
from . import danet
from . import basenet
from . import rendnet

from .base import Model
from .nnlib import *


# input names
REND = 'rendering'
XFER = 'transfer'
REAL = 'real'
UNSU = 'unsup'
INST_SYNT = 'instr_synt'
INST_REAL = 'instr_real'

class Parameters:
    pass


fn_clipping01 = lambda tensor: tf.fake_quant_with_min_max_args(tensor, min=0., max=1., num_bits=8)
fn_normalize_by_max = lambda tensor: tf.divide(tensor, tf.reduce_max(tensor, axis=[1,2,3], keep_dims=True) + 1e-5)

def fn_loss_entropy(tensor, label):
    return tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=tensor,
        labels=tf.squeeze(label, squeeze_dims=[3]),
        name="entropy")

def weight_map(weights, name):
    wmap = dict()
    for token in weights:
        if len(token) > 0:
            key, val = token.split('=')
            try:
                wmap[key] = float(val)
            except ValueError:
                wmap[key] = val # probably a string
            print('%s[%s] =' % (name, key), wmap[key])
    return wmap

class FeedForwardNetworks(Model):
    """Image to Instruction Network"""

    def __init__(self, sess, tf_flag):
        """Initialize the parameters for a network.

		Args:
		  model_type: string, 
		  batch_size: int, The size of a batch [25]
		  dataset: str, The path of dataset
		"""

        # TODO: pull out more parameters from the hard coded parameters
        self.sess = sess
        self.oparam = Parameters()
        self.oparam.learning_rate = tf_flag.learning_rate
        self.oparam.max_iter = tf_flag.max_iter
        self.oparam.batch_size = tf_flag.batch_size
        self.oparam.image_size = tf_flag.image_size  # ?? necessary? need to check the network architecture dependency to fixed image size..
        self.oparam.component = tf_flag.component
        self.oparam.threads = tf_flag.threads
        self.oparam.dataset = tf_flag.dataset
        self.oparam.model_type = tf_flag.model_type
        self.oparam.checkpoint_dir = tf_flag.checkpoint_dir
        self.oparam.is_train = tf_flag.training

        # set default weights
        self.oparam.weights = {
            'loss_transfer': 1.,
            'loss_syntax*' : 0.1,
            'loss_AE': 0.1,
            'loss_xentropy*': 2.,
            'loss_feedback*': 1.,
            'loss_disentgl*': 0.1,
            'loss_vgg_percept*': 0.1,
            'loss_unsup*': 0.01
        }
        self.oparam.weights.update(weight_map(tf_flag.weights, 'weights'))

        # set default parameters
        self.oparam.params = {
            'decay_rate': 0.99,
            'decay_steps': 10000,
            'augment': 1,
            'augment_src': 'best', # 0.25,
            'augment_mirror': 0, # detrimental unless we also apply in validation?
            'resi_global': 0,
            'resi_ch': 66,
            'gen_passes': 1,
            'decoder': 1,
            'discr_img': 0, 
            'discr_latent': 0, 
            'discr_instr':0,
            'discr_type': 'l2',
            'feedback': 0,
            'mean_img': 0,
            'runit': 'relu',
            'syntax_binary': 0,
            'bMILloss': 1,
            'bloss_unsup': 0, # seems detrimental
            'bloss_ae': 1, # only to be used with auto-encoder architectures
            'bloss_disentangle': 0, # seems detrimental
            'bvggloss': 0, # not always needed
            'vgg16or19': '16',
            'bg_type': 'local',
            'bg_weight': 0.1,
            'bunet_test': 0,
            'use_resnet': 0,
            'use_renderer': 0
        }
        self.oparam.params.update(weight_map(tf_flag.params, 'params'))

        # gram parameters for style supervision
        self.oparam.params['gram_layers'] = tf_flag.gram_layers
        self.oparam.params['discr'] = 1 if (self.oparam.params['discr_img'] + 
                                        self.oparam.params['discr_latent'] +
                                        self.oparam.params['discr_instr']) >= 1 else 0
        
        # register dataset path
        self.oparam.params['dataset'] = self.oparam.dataset

        # try loading parameters
        self.load_params(not(self.oparam.is_train))

        # using mean images instead of 0.5
        if self.oparam.params['mean_img']:
            print('Using mean images in %s' % os.path.join(self.oparam.dataset, 'mean'))
            mean_path = lambda name: os.path.join(self.oparam.dataset, 'mean', name)
            mean_imgs = {
                'mean_rend' : mean_path('rendering.jpg'),
                'mean_tran' : mean_path('transfer.jpg'),
                'mean_real' : mean_path('real.jpg')
            }
            self.oparam.params.update(mean_imgs)

        # special network with only encoder paths
        if not self.oparam.params.get('decoder', 1):
            self.oparam.params.update({
                # 'use_rend': 0,
                'bloss_unsup': 0,
                'bloss_ae': 0,
                'bloss_disentangle': 0,
                'bvggloss': 0,
                'resi_ch': 0
            })

        # do not use transfer data for renderer
        if self.oparam.params.get('use_renderer', 0):
            self.oparam.params['use_tran'] = 0 # no need to load transfer data

        # use rgb data for base network (for better data augmentation)
        if self.oparam.params.get('use_resnet', 0):
            self.oparam.params['xfer_type'] = 'rgb'

        # set default rectifier unit
        runit_type = self.oparam.params['runit']
        print('Rectifier unit:', runit_type)
        set_runit(runit_type)

        # parameters used to save a checkpoint
        self.lr = self.oparam.learning_rate
        self.batch = self.oparam.batch_size
        self._attrs = ['model_type', 'lr', 'batch']

        self.options = []

        self.oparam.params['training'] = self.oparam.is_train

        if self.oparam.is_train:
            self.load_params(False) # note: do not require parameters to exist at this stage
        else:
            self.oparam.params['use_tran'] = False
            self.oparam.params['use_rend'] = False
        self.loader = Loader(self.oparam.dataset, self.oparam.batch_size, self.oparam.threads, self.oparam.params)

        if len(self.loader.fakes) > 1:
            print('\n\n/!\\ Using multiple types of fake data.\nMake sure this is intended and not an error!\n', self.loader.fakes)
    
        if self.oparam.is_train:
            self.build_model()
        else:
            self.build_model_test()

    def model_define(self,
                     X_in,
                     Y_out,
                     is_train=False):

        net = dict()

        
        # [batch, height, width, channels]
        # semantic augmentation
        self.oparam.params['is_train'] = is_train
        if is_train and self.oparam.params.get('augment_mirror', 0):
            t_cond_real = tf.greater(tf.random_uniform([self.batch]), 0.5)
            t_cond_synt = tf.greater(tf.random_uniform([self.batch]), 0.5)
            for key in X_in.keys():
                # mirroring image
                t_img = X_in[key]
                if key == 'real':
                    t_cond = t_cond_real
                else:
                    t_cond = t_cond_synt
                X_in[key] = tf.where(t_cond, tf_mirror_image(t_img), t_img)
            for key in Y_out.keys():
                # mirroring instruction
                t_inst = Y_out[key]
                if key == 'real':
                    t_cond = t_cond_real
                else:
                    t_cond = t_cond_synt
                Y_out[key] = tf.where(t_cond, tf_mirror_instr(t_inst), t_inst)

        # remove unsupervised data from dictionary if not used
        if self.oparam.params.get('use_unsup', 0) == 0:
            if 'unsup' in X_in.keys():
                del X_in[UNSU]

        # model and loss
        if self.oparam.params.get('use_renderer'):
            net = rendnet.model_composited(Y_out, X_in, self.oparam.params)
            
            if is_train:
                loss_dict_Disc, loss_dict_Gene, metrics = rendnet.total_loss(
                    net, X_in, self.oparam.params)
            
        elif self.oparam.params.get('use_resnet'):
            net = basenet.model_composited(X_in, Y_out, self.oparam.params)
            
            if is_train:
                loss_dict_Disc, loss_dict_Gene, metrics = basenet.total_loss(
                    net, Y_out, self.oparam.params)
            
        elif self.oparam.params.get('bunet_test', 0) == 1:
            if 'rend' in X_in.keys():
                del X_in['rend']
            if 'tran' in X_in.keys():
                del X_in['tran']
            net = danet.model_composited(X_in, Y_out, self.oparam.params)
            
            if is_train:
                loss_dict_Disc, loss_dict_Gene, metrics = danet.total_loss(
                    net, Y_out[INST_SYNT], Y_out[INST_REAL], self.oparam.params)
                
        elif self.oparam.params.get('bunet_test', 0) == 2: # real, rend, tran
            if self.oparam.params.get('use_cgan', 0):
                X_in['tran'] = X_in['cgan']
            net = danet.model_composited_RFI_2(X_in, Y_out, self.oparam.params)
            if is_train:
                loss_dict_Disc, loss_dict_Gene, metrics = danet.total_loss_RFI(
                                net, Y_out, self.oparam.params)
        elif self.oparam.params.get('bunet_test', 0) == 3: # complex net
            # if 'tran' in X_in.keys():
            #     del X_in['tran']
            net = danet.model_composited_RFI_complexnet(X_in, Y_out, self.oparam.params)
            if is_train:
                loss_dict_Disc, loss_dict_Gene, metrics = danet.total_loss_RFI(
                                net, Y_out, self.oparam.params)
        elif self.oparam.params.get('use_autoencoder', 0):
            net = layer_modules.model_composited(X_in, Y_out, self.oparam.params)
            
            if is_train:
                loss_dict_Disc, loss_dict_Gene, metrics = layer_modules.total_loss(
                    net, Y_out, self.oparam.params)
        else:
            raise ValueError('No model selected (use_renderer | use_resnet | bunet_test | use_autoencoder)')

        if not is_train:
            loss_dict_Disc = None
            loss_dict_Gene = None
            metrics = None
        return net, loss_dict_Disc, loss_dict_Gene, metrics

    def build_model(self):
        print('Model build')

        # @see https://www.tensorflow.org/api_docs/python/tf/data/Iterator#from_string_handle

        # iterators
        train_iter  = self.loader.iter(set_option='train')
        val_iter    = self.loader.iter(set_option='val')
        
        # handles
        self.train_handle = self.sess.run(train_iter.string_handle())
        self.val_handle   = self.sess.run(val_iter.string_handle())
        
        # create iterator switch
        self.batch_handle = tf.placeholder(tf.string, shape=[])
        batch_iter = tf.data.Iterator.from_string_handle(self.batch_handle, train_iter.output_types)

        # get effective batch
        curbatch = batch_iter.get_next()

        #import pdb
        #pdb.set_trace()
        img_size = [self.loader.batch_size, 160, 160, 1]
        lbl_size = [self.loader.batch_size, 20, 20, 1]
        # apply shapes on images and labels
        inst_synt = curbatch['synt'][-1]
        real, inst_real = curbatch['real']
        if 'unsup' in curbatch.keys():
            unsup = curbatch['unsup'][0]

        # pdb.set_trace()  # check whether unsup data structure is good enough

        # apply shapes
        for t_img in curbatch['synt'][0:-1]:
            t_img.set_shape(img_size)
        real.set_shape(img_size)
        if 'unsup' in curbatch.keys():
            unsup.set_shape(img_size)
        for t_lbl in [inst_synt, inst_real]:
            t_lbl.set_shape(lbl_size)

        self.tf_models = Parameters()
        print('Model build')
        self.tf_models.X = { REAL: real } # UNSU: unsup
        self.tf_models.Y = { INST_SYNT: inst_synt, INST_REAL: inst_real }
        # add synthetic inputs
        for i in range(len(self.loader.fakes)):
            name = self.loader.fakes[i]
            t_img = curbatch['synt'][i]
            self.tf_models.X[name] = t_img

        # replay buffer
        if self.oparam.params.get('replay_worst', 0):
            name = 'worst'
            self.tf_models.X[name] = tf.Variable(tf.ones_like(real), name = 'worst-input', dtype = tf.float32, trainable = False)
            self.tf_models.Y[name] = tf.Variable(tf.zeros_like(inst_real), name = 'worst-output', dtype = tf.int32, trainable = False)

        # Train path
        if self.oparam.is_train:
            with tf.device('/device:GPU:0'):
                self.tf_models.net, self.tf_models.loss_dict_Disc, self.tf_models.loss_dict_Gene, self.tf_models.metrics = \
                    self.model_define(
                        X_in =  self.tf_models.X,
                        Y_out = self.tf_models.Y,
                        is_train = self.oparam.is_train)
        else:
            return  # Test phase

        # dispatching global losses from name* and *name weights
        def dispatch_weights():
            new_weights = dict()
            for name, value in self.oparam.weights.items():
                if name.endswith('*'):
                    prefix = name[:-1]
                    for loss_name in self.tf_models.loss_dict_Gene.keys():
                        if loss_name.startswith(prefix):
                            new_weights[loss_name] = value
                    for loss_name in self.tf_models.loss_dict_Disc.keys():
                        if loss_name.startswith(prefix):
                            new_weights[loss_name] = value
                if name.startswith('*'):
                    suffix = name[1:]
                    for loss_name in self.tf_models.loss_dict_Gene.keys():
                        if loss_name.endswith(suffix):
                            new_weights[loss_name] = value
                    for loss_name in self.tf_models.loss_dict_Disc.keys():
                        if loss_name.endswith(suffix):
                            new_weights[loss_name] = value
            for name, value in self.oparam.weights.items():
                for loss_name in list(self.tf_models.loss_dict_Gene.keys()) + \
                                 list(self.tf_models.loss_dict_Disc.keys()):
                    if name == loss_name:
                        new_weights[name] = value

            # applying new weights
            for name, value in new_weights.items():
                self.oparam.weights[name] = value

        dispatch_weights()

        # balance loss weights when varying the amount of data
        if self.oparam.params.get('balance_weights', 1):
            if len(self.loader.fakes) == 0:
                print('Balancing weights for real data only')
                for name in self.tf_models.loss_dict_Gene.keys():
                    if name.endswith('/real'):
                        weight = self.oparam.weights.get(name, 1.0)
                        self.oparam.weights[name] = weight * 2
                        print('- %s: %f -> %f' % (name, weight, weight * 2))

        print('Losses:')
        for name in self.tf_models.loss_dict_Gene.keys():
            weight = self.oparam.weights.get(name, 1.0)
            if weight > 0:
                print('[gen] %s (%f)' % (name, weight))
        for name in self.tf_models.loss_dict_Disc.keys():
            weight = self.oparam.weights.get(name, 1.0)
            if weight > 0:
                print('[dis] %s (%f)' % (name, weight))

        # create full losses
        self.tf_models.loss_total_gene = tf.add_n([
            tf.reduce_mean(l * self.oparam.weights.get(i, 1.0)) # default weight of 1.0
            for (i, l) in self.tf_models.loss_dict_Gene.items()
        ])
        self.tf_models.loss_main_gene = tf.add_n([
            tf.reduce_mean(l * self.oparam.weights.get(i, 1.0)) # default weight of 1.0
            for (i, l) in self.tf_models.loss_dict_Gene.items()
            # filtering generator and adapter networks, and feedback
            if 'adapt' not in i and 'gen' not in i and 'feedback' not in i
        ])
        if self.oparam.params.get('discr', 1):
            self.tf_models.loss_total_disc = tf.add_n([
                tf.reduce_mean(l * self.oparam.weights.get(i, 1.0)) # default weight of 1.0
                for (i, l) in self.tf_models.loss_dict_Disc.items()
            ])
        else:
            self.tf_models.loss_total_disc = tf.constant(0)

        # summary storage
        self.summaries = {}
        net = self.tf_models.net

        # creating dictionary of images from residual dictionary
        def res_dict_imgs(res_dict, target='real', src = None):
            if src is None:
                src = target
            if src not in net.mean_imgs:
                src = 'real'
            real_dict = dict()
            if target.startswith('*'):
                for key, value in res_dict.items():
                    if key.endswith(target[1:]):
                        # real_dict[key] = net.mean_imgs[src] + value
                        real_dict[key] = value
            elif target in res_dict:
                # real_dict[target] = net.mean_imgs[src] + res_dict[target]
                real_dict[target] = res_dict[target]
            return real_dict

        use_renderer = self.oparam.params.get('use_renderer', 0)
        # visual summary
        self.summaries['images'] = dict()
        images = {
            'inputs' : net.imgs,
            'res-inps' : net.resi_imgs,
            'res-outs' : net.resi_outs,
            'ae' : res_dict_imgs(net.resi_outs, 'real'),
            'adapt' : res_dict_imgs(net.resi_outs, '*_real'),
            'generator' : res_dict_imgs(net.resi_outs, '*_gen')
        }
        for name in net.discr.keys():
            images['discr-' + name] = net.discr[name] # discriminator outputs
        for cat, data_dict in images.items():
            for name, tf_img in data_dict.items():
                sum_name = cat + '/' + name
                if cat != 'inputs' and use_renderer == 0:
                    tf_img = tf_img + 0.5
                self.summaries['images'][sum_name] = tf.summary.image(
                    sum_name, fn_clipping01(tf_img), max_outputs = 5)
        images = {
            'gt' : self.tf_models.Y,
            'outputs': dict(),
            'outputs-adapt': dict(),
            'outputs-gen': dict()
        }
        for name, t_instr in net.instr.items():
            if '_real' in name:
                images['outputs-adapt'][name] = t_instr
            elif '_gen' in name:
                images['outputs-gen'][name] = t_instr
            else:
                images['outputs'][name] = t_instr
        for cat, data_dict in images.items():
            for name, tf_img in data_dict.items():
                if 'feedback' in name:
                    sum_name = 'feedback/' + name.replace('_feedback', '')
                else:
                    sum_name = cat + '/' + name
                # label = fn_clipping01(tf_ind_to_rgb(tf_img))
                label = tf_ind_to_rgb(tf_img)
                self.summaries['images'][sum_name] = tf.summary.image(
                    sum_name, label, max_outputs = 5)

        for name, t_bg in net.bg.items():
            sum_name = 'bg/' + name
            self.summaries['images'][sum_name] = tf.summary.image(sum_name, tf.cast(t_bg, tf.float32), max_outputs = 5)

        # loss summary
        self.summaries['scalar'] = dict()
        self.summaries['scalar']['total_loss'] = tf.summary.scalar("loss_total", self.tf_models.loss_total_gene)
        for loss_name, tf_loss in dict(self.tf_models.loss_dict_Gene, **self.tf_models.loss_dict_Disc).items():
            # skip losses whose weights are disabled
            weight = self.oparam.weights.get(loss_name, 1.0)
            if weight > 0.0:
                self.summaries['scalar'][loss_name] = tf.summary.scalar(loss_name, tf.reduce_mean(tf_loss * weight))

        # metric summary
        for metric_name, tf_metric in self.tf_models.metrics.items():
            if metric_name.startswith('confusionmat'):
                self.summaries['images'][metric_name] = tf.summary.image(metric_name,
                                                            tf_summary_confusionmat(tf_metric,
                                                            numlabel=layer_modules.prog_ch,
                                                            tag=metric_name,
                                                            ), 
                                                            max_outputs = 5)
                        # tf_summary_confusionmat(tf_metric, 
                        #                     numlabel=layer_modules.prog_ch,
                        #                     tag=metric_name)
            else:
                self.summaries['scalar'][metric_name] = tf.summary.scalar(metric_name, tf_metric)

        # # gradient summary
        # t_grad_var = self.tf_models.net.fake_imgs['rend']
        # for loss_name, t_loss in self.tf_models.loss_dict_Gene.items():
        #     grad_name = 'gradient_rend2real_' + loss_name
        #     t_grad = tf.gradients(t_loss, t_grad_var)
        #     # skip if there's no contribution from that loss
        #     if t_grad[0] is None or self.oparam.weights.get(loss_name, 1.0) <= 0.0:
        #         continue
        #     grad_sum = tf_variable_summary(var = t_grad, name = grad_name)
        #     self.summaries['scalar'][grad_name] = grad_sum
        
        # activation summary
        # self.summaries['activation'] = dict()
        # for t_out in runit_list:
        #     act_name = 'activation/' + t_out.name.replace(':0', '')
        #     act_sum  = tf_variable_summary(var = t_out, name = act_name)
        #     self.summaries['activation'][act_name] = act_sum

        # _ = tf.summary.image('error_map',
        # tf.transpose(self.tf_models.loss, perm=[1,2,3,0]),
        # max_outputs=5) # Concatenate row-wise.

    def build_model_test(self):
        print('Model build')
        # iterators # handles
        test_iter  = self.loader.iter(set_option='test')
        self.test_handle = self.sess.run(test_iter.string_handle())
        
        # create iterator switch
        self.batch_handle = tf.placeholder(tf.string, shape=[])
        batch_iter = tf.data.Iterator.from_string_handle(self.batch_handle, test_iter.output_types)

        # get effective batch
        curbatch = batch_iter.get_next()

        img_size = [self.loader.batch_size, 160, 160, 1]
        lbl_size = [self.loader.batch_size, 20, 20, 1]
        # apply shapes on images and labels
        real, inst_real, self.input_names = curbatch['real']
        
        # apply shapes
        real.set_shape(img_size)
        inst_real.set_shape(lbl_size)

        self.tf_models = Parameters()
        print('Model build')
        self.tf_models.X = { REAL: real }
        self.tf_models.Y = { INST_REAL: inst_real }
        
        # Test path
        with tf.device('/device:GPU:0'):
            self.tf_models.net, _, _, _ = self.model_define(
                                        X_in = self.tf_models.X, 
                                        Y_out = self.tf_models.Y, 
                                        is_train = False)

    def train(self):
        """Train a network"""
        self.step = tf.train.get_or_create_global_step()

        lr = tf.train.exponential_decay(
            self.oparam.learning_rate,
            global_step=self.step,
            decay_steps=self.oparam.params.get('decay_steps', 50000), # 10k
            decay_rate=self.oparam.params.get('decay_rate', 0.3), # 0.99
            staircase=True)

        # Initialize optimizers
        use_discr = self.oparam.params.get('discr', 1)

        def create_train_op(lr, loss, tvars, global_step):
            optim = tf.train.AdamOptimizer(
                lr, beta1=0.5, epsilon=1e-4)
            grads_and_vars = optim.compute_gradients(
                loss, tvars, colocate_gradients_with_ops=True)
            return optim.apply_gradients(
                grads_and_vars, global_step = global_step)

        # as well as batch normalization (until it becomes a dependency of discriminator)
        base_deps = []
        runit_type = self.oparam.params.get('runit', 'relu')
        if 1: # just for now 'bn' in runit_type or 'in' in runit_type:
            base_deps.extend(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
        else:
            base_deps = None

        # replay buffer
        replay_worst = self.oparam.params.get('replay_worst', 0)
        if replay_worst:
            replay_deps = []
            net = self.tf_models.net
            with tf.variable_scope('replay_worst', tf.AUTO_REUSE):
                worst_type = self.oparam.params.get('worst_type', 'fg')
                assert worst_type in net.acc, 'Invalid worst type'
                # compute index of worst sample from current batch
                acc = net.acc[worst_type]
                real_accs = tf.concat([acc[REAL], acc['worst']], axis = 0)
                real_inps = tf.concat([self.tf_models.X[REAL], self.tf_models.X['worst']], axis = 0)
                real_outs = tf.concat([self.tf_models.Y[INST_REAL], self.tf_models.Y['worst']], axis = 0)
                _, worst_idx = tf.nn.top_k(-tf.squeeze(real_accs), self.loader.batch_size)
                # update worst buffer for input
                worst_inps = tf.gather(real_inps, worst_idx)
                dep = self.tf_models.X['worst'].assign(worst_inps, read_value = False)
                replay_deps.append(dep)
                # update worst vuffer for output
                worst_outs = tf.gather(real_outs, worst_idx)
                dep = self.tf_models.Y['worst'].assign(worst_outs, read_value = False)
                replay_deps.append(dep)
            # add to base dependencies
            if base_deps is None:
                base_deps = replay_deps
            else:
                base_deps += replay_deps

        # load rendering network (for loss)
        if self.oparam.params.get('use_hosyntax', 0):
            rendnet.load_weights(self.sess, self.oparam.params.get('render_type', 'dense'))

        with tf.name_scope("generator_train"):
            gen_tvars = [
                var for var in tf.trainable_variables()
                if (re.search("generator", var.name) != None)
            ]
            for var in gen_tvars:
                print('gen var %s' % var.name)
            # gen_pre_tvars = list(filter(lambda var: 'gen' not in var.name and 'adapt' not in var.name, gen_tvars))

            gen_deps = []
            # if use_discr:
            #     gen_deps.append(self.dis_train_op)
            if base_deps is not None:
                gen_deps.extend(base_deps)
            else:
                gen_deps = None

            # must ensure that discriminator is done
            with tf.control_dependencies(gen_deps):
                self.gen_train_op = create_train_op(lr,
                                self.tf_models.loss_total_gene, gen_tvars, self.step)
                # self.gen_pretrain_op = create_train_op(lr,
                    # self.tf_models.loss_main_gene, gen_pre_tvars, self.step)

        if use_discr:
            dis_deps = []
            if gen_deps is not None:
                dis_deps.extend(gen_deps)
            dis_deps.append(self.gen_train_op)

            with tf.name_scope("discriminator_train"):
                dis_tvars = [
                    var for var in tf.trainable_variables()
                    if (re.search("discriminator", var.name) != None)
                ]
                for var in dis_tvars:
                    print('dis var %s' % var.name)

                # we must ensure that base dependencies are met
                with tf.control_dependencies(dis_deps):
                    self.dis_train_op = create_train_op(lr * 0.5,
                                    self.tf_models.loss_total_disc, dis_tvars, None)


        # summaries for Tensorboard
        self.summaries['scalar']['learning_rate'] = tf.summary.scalar('learning_rate', lr)
        # images_summary = tf.summary.merge(self.summaries['images'].values())
        loss_summary = tf.summary.merge(list(self.summaries['scalar'].values()))
        val1_summary = tf.summary.merge(list(self.summaries['images'].values()) + [loss_summary])
        val2_summary = tf.summary.merge_all()

        train_writer = tf.summary.FileWriter(self.oparam.checkpoint_dir + '/train', self.sess.graph)
        val_writer   = tf.summary.FileWriter(self.oparam.checkpoint_dir + '/val', self.sess.graph)

        # Training start
        tf.local_variables_initializer().run() # for metrics (accuracy)
        tf.global_variables_initializer().run() # for network parameters
        
        
        ## Save all the parameters
        self.load(self.oparam.checkpoint_dir)
        with open(os.path.join(self.oparam.checkpoint_dir, 
                                'params.pkl'), 'wb') as f:
            pickle.dump(self.oparam.params, f)

        start_time = time.time()
        start_iter = int(self.step.eval() + 1)

        train_setup = { self.batch_handle: self.train_handle }
        val_setup   = { self.batch_handle: self.val_handle }

        global_step = 0

        while global_step < self.oparam.max_iter:
            try:
                # Training the network
                global_step = tf.train.global_step(self.sess, tf.train.get_global_step())

                # save the intermediate model
                if global_step != 0 and global_step % 10000 == 0:
                    self.save(self.oparam.checkpoint_dir, tf.train.get_global_step())

                # Status check with validation data
                if global_step !=0 and global_step % 500 == 0:
                    # single validation step
                    if global_step % 1000 == 0:
                        val_summary = val2_summary
                    else:
                        val_summary = val1_summary

                    # compute validation summary and loss information
                    if use_discr:
                        summary_str, loss_probe_d, loss_probe_g = self.sess.run([
                            val_summary,
                            self.tf_models.loss_total_disc,
                            self.tf_models.loss_total_gene
                        ], feed_dict = val_setup)
                    else:
                        loss_probe_d = 0
                        summary_str, loss_probe_g = self.sess.run([
                            val_summary,
                            self.tf_models.loss_total_gene
                        ], feed_dict = val_setup)
                    val_writer.add_summary(summary_str, global_step)

                    print("Iter: [%2d/%7d] time: %4.4f, vloss: [d %.4f, g %.4f]"
                        % (global_step, self.oparam.max_iter, time.time() - start_time, loss_probe_d, loss_probe_g))

                # training operation
                if use_discr:
                    # Run generator N passes
                    # if global_step == 0:
                    #     # for _ in range(199):
                    #         # print('.')
                    #     _ = self.sess.run([self.gen_train_op], feed_dict = train_setup)
                    # else:
                    Ngen = int(self.oparam.params.get('gen_passes', 2.0))
                    for g in range(Ngen):
                        _ = self.sess.run([self.gen_train_op], feed_dict = train_setup)

                    # Run generator+discriminator pass
#                     if global_step >= 5000:
                        # for iter in range(10): # Warm start discrimantor
                    _ = self.sess.run([self.dis_train_op], feed_dict = train_setup)
                    # else:
                        # _ = self.sess.run([self.dis_train_op], feed_dict = train_setup)

                    if global_step % 100 == 0:
                        summary_str, loss_tr_d, loss_tr_g = self.sess.run([
                            loss_summary,
                            self.tf_models.loss_total_disc,
                            self.tf_models.loss_total_gene,
                        ], feed_dict = train_setup)
                    else:
                        loss_tr_d, loss_tr_g = self.sess.run([
                        self.tf_models.loss_total_disc,
                        self.tf_models.loss_total_gene,
                        ], feed_dict = train_setup)
                else:
                    loss_tr_d = 0.0
                    summary_str, loss_tr_g, _ = self.sess.run([
                        loss_summary,
                        self.tf_models.loss_total_gene,
                        self.gen_train_op
                    ], feed_dict = train_setup)
                

                # Status check with training data
                if global_step % 10 < 1:
                    '''
                    print("Iter: [%2d/%7d] time: %4.4f, loss: [d %.4f, g %.4f]" %
                          (global_step, self.oparam.max_iter, time.time() - start_time,
                           loss_tr_d, loss_tr_g))
                    '''
                    print("Iter: [%2d/%7d] time: %4.4f, loss: [d %.4f, g %.4f]" %
                          (global_step, self.oparam.max_iter, time.time() - start_time, loss_tr_d, loss_tr_g))

                if global_step % 100 == 0:
                    train_writer.add_summary(summary_str, global_step)
                    
                # Flush summary
                # writer.flush()
            except tf.errors.OutOfRangeError: # if data loader is done
                break


        print('Training ends.')
        self.save(self.oparam.checkpoint_dir, global_step)

        train_writer.close()
        val_writer.close()

    def test_imgs(self, fnames_img, name="test_imgs"):
        pass

    def test(self, name="test"):
        tf.global_variables_initializer().run() # for network parameters
        self.load(self.oparam.checkpoint_dir, True)
        
        import cv2
        def fn_rescaleimg(x):
            x += 0.5
            x[x > 1] = 1.
            x[x < 0] = 0.
            return x*255.
            
        
        svpath = os.path.join(self.oparam.checkpoint_dir, 'eval')
        fn_path = lambda x: os.path.join(svpath, x)
        if not os.path.exists(svpath):
            os.makedirs(svpath)

        test_setup = { self.batch_handle: self.test_handle }
        
        lst_eval_tensors = [
            self.input_names, # file name (no extension)
            self.tf_models.net.instr['real'], # output label map 20x20x1
            tf.nn.softmax(self.tf_models.net.logits['real']), # softmax of logits 20x20x17
        ]
        if 'real' in self.tf_models.net.resi_outs.keys():
            lst_eval_tensors.append(self.tf_models.net.resi_outs['real']) # regul image 160x160x1
            sgpath = os.path.join(svpath, 'gen')
            if not os.path.exists(sgpath):
                os.makedirs(sgpath)
        
        cnt1 = 0
        cnt2 = 0

        show_info = self.oparam.params.get('show_confidence', 0)
        while 1:
            try:
                rst = self.sess.run(lst_eval_tensors, feed_dict = test_setup)
                names = rst[0]
                labels = rst[1]
                probs = rst[2]

                for i in range(names.shape[0]):
                    fname = str(names[i], encoding='utf-8')
                    if show_info:
                        p = probs[i] # p is 20x20x17
                        max_p = np.amax(p, axis = -1)
                        conf_mean = np.mean(max_p)
                        conf_std  = np.std(max_p)
                        print('%d %s (conf: m=%f, s=%f)' % (cnt1 + 1, fname, conf_mean, conf_std))
                    else:
                        sys.stdout.write("\r%d %s" % (cnt1 + 1, fname))
                        sys.stdout.flush()
                    fpath = os.path.join(svpath, fname + '.png')
                    save_instr(fpath, labels[i])

                    # cv2.imwrite(fpath, rst[1][i])
                    cnt1 += 1
                    if 'real' in self.tf_models.net.resi_outs.keys():
                        regul = rst[3]
                        fpath = os.path.join(sgpath, fname + '.png')
                        cv2.imwrite(fpath, fn_rescaleimg(regul[i]))
                        cnt2 += 1

                # ## Do something with rst
                # for eachimg in rst[0]:
                #     curfname = fn_path('inst_%06d.png' % (cnt1))
                #     cv2.imwrite(curfname, eachimg)
                #     cnt1 += 1
                
                    # # generated image
                # if 'real' in self.tf_models.net.resi_outs.keys():
                    # for eachimg in rst[1]:
                    #     curfname = fn_path('gen_%06d.png' % (cnt2))
                    #     cv2.imwrite(curfname, fn_rescaleimg(eachimg))
                    #     cnt2 += 1
                    

            except tf.errors.OutOfRangeError: # if data loader is done
                break

        print('\nProcessing Done!')
        return
    
    def load_params(self, needed = False):
        fname = os.path.join(self.oparam.checkpoint_dir, 'params.pkl')
        try:
            with open(fname, 'rb') as f:
                new_params = pickle.load(f)
                self.oparam.params.update(new_params)
                if needed:
                    self.oparam.params['is_train'] = False
                print("Loaded parameters from %s" % fname)
                for key, value in self.oparam.params.items():
                    print('-', key, '=', value)
        except:
            if needed:
                print("[!] Error loading parameters from %s" % fname)
                raise
