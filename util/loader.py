import os
from skimage.io import imread, imsave
import numpy as np
from numpy.random import choice, permutation
import cv2
from math import ceil

import tensorflow as tf
from tensorflow.data import Dataset, TextLineDataset
from tensorflow.contrib.data import shuffle_and_repeat

from time import gmtime, strftime
from skimage.transform import warp, AffineTransform
from skimage.filters import gaussian
import pdb

from PIL import Image

def read_image(fname, expand = True, aug_scale = True):
    # img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(fname, cv2.IMREAD_COLOR)
    if img is None:
        print('Image %s not found!' % fname)
    img = np.squeeze(img[:,:,choice(3)])

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # to grayscale
    img = img.astype(np.float32) / 255. # BGR format
    if aug_scale:
        randscale = 0.2*(np.random.rand(1)-0.6)
        img = np.minimum(img*(1.+randscale), 1.)
        img = img.astype(np.float32)

    # cv2.resize(img, (0, 0), fx=scale, fy=scale)
    if expand:
        return np.expand_dims(img, axis=2) # we need three dimensions (not two) even though grayscale
    else:
        return img

def read_instr(fname):
    img = np.array(Image.open(fname)).astype(np.int32)
    return np.expand_dims(img, axis=2)

def read_points(fname, from_matlab = True):
    points = np.loadtxt(fname, delimiter = ',')
    if from_matlab:
        points = points - 1.0
    return points.astype(np.float32)

def random_crop(img, points, params):
    '''
    Warp an image using points for cropping.
    Randomly perturb those points for augmentation.

    Input is HxW, output is 160x160
    Points is 4x2, [TL, TR, BR, BL] with points[:,0] being x, points[:, 1] being y
    '''
    # compute perturbed points for cropping
    x = points[:, 0]
    y = points[:, 1]
    tx = x[1] - x[0]
    bx = x[2] - x[3]
    ly = y[3] - y[0]
    ry = y[2] - y[1]
    rng_scale = params.get('augment_scale', 0.05) # 20x20 instructions
    rng = np.array([[tx, ly], [tx, ry], [bx, ry], [bx, ly]], dtype = np.float32) * rng_scale
    src = points + np.random.uniform(- rng*0.5, rng*0.5).astype(np.float32)
    # dst = np.array([[0, 0], [0, 159], [159, 159], [159, 0]], dtype = np.float32)
    dst = np.array([[0, 0], [159, 0], [159, 159], [0, 159]], dtype = np.float32)

    # compute homography
    H = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, H, (160, 160))
    return np.expand_dims(warped, axis=2)

def count_lines(fname):
    return sum(1 for line in open(fname) if line.rstrip())

class Loader(object):

    def __init__(self, dataset_path, batch_size = 8, num_threads = 1, params = dict()):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.params = params

        unsup_path = 'unsupervised'
        transfer_type = params.get('xfer_type', 'gray')
        # build paths / directories
        fake_paths = {
            'rend': 'rendering',
            'tran': 'transfer/Cable1_019_0_19/' + transfer_type,
            'cgan': 'transfer/cyclegan/' + transfer_type
        }
        self.fake_dirs = {
            name: os.path.join(self.dataset_path, path) for (name, path) in fake_paths.items()
                if self.params.get('use_' + name, name == 'rend' or name == 'tran')
        }
        for name, path in self.fake_dirs.items():
            print('Using %s from %s' % (name, path))

        self.fakes = list(self.fake_dirs.keys())
        self.real_dir = os.path.join(self.dataset_path, 'real')
        self.inst_dir = os.path.join(self.dataset_path, 'instruction')
        self.unsup_dir = os.path.join(self.dataset_path, unsup_path)

        # dataset from files
        labels = ['train', 'val']
        self.datasets = { label: self.pipeline(label, num_threads) for label in labels }
        self.datasets['test'] = self.test_pipeline(num_threads)

    def pipeline(self, name, num_threads):
        if not self.params.get('training', 1):
            return None
        synt_fname = os.path.join(self.dataset_path, name + '_synt.txt')
        real_fname = os.path.join(self.dataset_path, name + '_real.txt')
        unsup_fname = os.path.join(self.dataset_path, 'train_unsup.txt')

        num_synt, num_real, num_unsup = [count_lines(fname) for fname in [synt_fname, real_fname, unsup_fname]]
        ratio = num_synt / float(num_real)

        # extract directories
        fake_dirs, real_dir, inst_dir = self.fake_dirs, self.real_dir, self.inst_dir

        # dataset creation
        with tf.name_scope('dataset'):
            synt, real, unsup = [TextLineDataset(name) for name in [synt_fname, real_fname, unsup_fname]]

            # @see https://www.tensorflow.org/api_docs/python/tf/contrib/data/shuffle_and_repeat
            #synt.apply(shuffle_and_repeat(buffer_size = num_synt)) #, count = 1))
            #real.apply(shuffle_and_repeat(buffer_size = num_real)) #, count = ceil(ratio)))

            synt = synt.shuffle(num_synt).repeat()
            real = real.shuffle(num_real).repeat()
            unsup = unsup.shuffle(num_unsup).repeat()

            # map to corresonding files
            # synthetic data
            def name2synt(name):
                fakes = [
                    read_image(os.path.join(path, name.decode() + '.jpg'))
                    for path in fake_dirs.values()
                ]
                inst = read_instr(os.path.join(inst_dir, name.decode() + '.png'))
                return fakes + [inst]

            synt_types = [tf.float32 for _ in self.fakes] + [tf.int32]
            synt = synt.map(lambda name: tf.py_func(name2synt, [name], synt_types), num_parallel_calls = num_threads)

            # real data
            augment = self.params.get('augment', 1)
            def name2real(name):
                inst = read_instr(os.path.join(inst_dir, name.decode() + '.png'))
                if augment:
                    src_dir = self.params.get('augment_src', 'best')
                    # print('{}/{}/{}'.format(real_dir, str(src_dir), name.decode() + '.JPG'))
                    full = read_image(os.path.join(real_dir, str(src_dir), 'rgb', name.decode() + '.jpg'), False)
                    pnts = read_points(os.path.join(real_dir, str(src_dir), 'points', name.decode() + '.txt'))
                    if isinstance(src_dir, float):
                        pnts *= src_dir
                    real = random_crop(full, pnts, self.params)
                    # TODO add mirror augmentation
                else:
                    real = read_image(os.path.join(real_dir, name.decode() + '.jpg'))
                return real, inst
            real = real.map(lambda name: tuple(tf.py_func(name2real, [name], [tf.float32, tf.int32])), num_parallel_calls = num_threads)

            # unsup data
            def name2unsup(name):
                if augment:
                    # print('{}/{}/{}'.format(real_dir, str(src_dir), name.decode() + '.JPG'))
                    img = read_image(os.path.join(self.unsup_dir, name.decode() + '.jpg'), False)
                    imsz = img.shape # y,x,c
                    # [TL, TR, BR, BL]
                    real = random_crop(img, 
                            np.array([[5,5],[imsz[1]-5,5],[imsz[1]-5,imsz[0]-5],[5,imsz[0]-5]], dtype = np.float32), self.params)
                else:
                    real = read_image(os.path.join(self.unsup_dir, name.decode() + '.jpg'))
                return real

#             unsup = unsup.map(lambda name: tuple(tf.py_func(name2unsup, [name], [tf.float32])), num_parallel_calls = num_threads)

            # zip all, batch and prefetch
            #dataset = Dataset.zip((rend, xfer, real, inst_synt, inst_real))
            dataset = Dataset.zip({ 'synt': synt, 'real': real }) # , 'unsup': unsup
            dataset = dataset.batch(self.batch_size, drop_remainder = True) # we need full batches!
            dataset = dataset.prefetch(self.batch_size * 2)
            return dataset

    def test_pipeline(self, num_threads):
        real_fname = os.path.join(self.dataset_path, 'test_real.txt')

        # extract directories
        real_dir, inst_dir = self.real_dir, self.inst_dir

        # count lines
        num_real = count_lines(real_fname)

        # dataset creation
        with tf.name_scope('dataset'):
            real = TextLineDataset(real_fname)

            # @see https://www.tensorflow.org/api_docs/python/tf/contrib/data/shuffle_and_repeat
            #synt.apply(shuffle_and_repeat(buffer_size = num_synt)) #, count = 1))
            #real.apply(shuffle_and_repeat(buffer_size = num_real)) #, count = ceil(ratio)))

            real = real.shuffle(num_real) # no repetition! .repeat()

            # real data only
            augment = 0 # self.params.get('augment', 0)
            def name2real(name):
                inst = read_instr(os.path.join(inst_dir, name.decode() + '.png'))
                if augment:
                    src_dir = self.params.get('augment_src', 'best')
                    # print('{}/{}/{}'.format(real_dir, str(src_dir), name.decode() + '.JPG'))
                    full = read_image(os.path.join(real_dir, str(src_dir), 'rgb', name.decode() + '.jpg'), False)
                    pnts = read_points(os.path.join(real_dir, str(src_dir), 'points', name.decode() + '.txt'))
                    if isinstance(src_dir, float):
                        pnts *= src_dir
                    self.params['augment_scale'] = 0.
                    real = random_crop(full, pnts, self.params)
                else:
                    real = read_image(os.path.join(real_dir, '160x160', 'gray', name.decode() + '.jpg'))
                return real, inst, name.decode()
            real = real.map(lambda name: tuple(tf.py_func(name2real, [name], [tf.float32, tf.int32, tf.string])), num_parallel_calls = num_threads)

            #dataset = Dataset.zip((rend, xfer, real, inst_synt, inst_real))
            dataset = Dataset.zip({ 'real': real })
            dataset = dataset.batch(self.batch_size, drop_remainder = True) # we need full batches!
            dataset = dataset.prefetch(self.batch_size * 2)
            return dataset

    def iter(self, set_option='train'):
        dataset = self.datasets[set_option]
        # create iterator
        return dataset.make_one_shot_iterator()
        #return iterator.get_next()

