from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import optimizers
from keras.callbacks import Callback
from sklearn.utils import shuffle
from vgg16 import vgg_16, vgg_16_conv, vgg_16_fc
from transformation import transform, image_transform

import ctypes
import imp
import logging
import numpy as np
import os
import re
import shutil
import six
import sys
import time
import argparse
import math
import h5py
import json

def load_dataset(args):
  train_fn = '%s/train_joints.csv' % args.datadir
  test_fn = '%s/test_joints.csv' % args.datadir
  train_dl = np.array([l.strip() for l in open(train_fn).readlines()])
  test_dl = np.array([l.strip() for l in open(test_fn).readlines()])

  return train_dl, test_dl

def create_result_dir(args):
  result_dir = 'results/{}_{}'.format(
    'vgg16', time.strftime('%Y-%m-%d_%H-%M-%S'))
  if args.debug > 0:
    result_dir = 'debug/{}_{}'.format(
    'vgg16', time.strftime('%Y-%m-%d_%H-%M-%S'))
  if os.path.exists(result_dir):
    result_dir += '_{}'.format(np.random.randint(100))
  if not os.path.exists(result_dir):
    os.makedirs(result_dir)

  log_fn = '%s/log.txt' % result_dir
  logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s',
    filename=log_fn, level=logging.DEBUG)
  logging.info(args)

  args.log_fn = log_fn
  args.result_dir = result_dir

def get_optimizer(args):
  if 'opt' in args:
    # prepare optimizer
    if args.opt == 'Adagrad':
      optimizer = optimizers.Adagrad(lr=args.lr)
    elif args.opt == 'SGD':
      optimizer = optimizers.SGD(lr=args.lr, momentum=0.9)
    elif args.opt == 'Adam':
      optimizer = optimizers.Adam()
    else:
      raise Exception('No optimizer is selected')

    return optimizer

  else:
    print('No optimizer generated. Using default Adam')
    return optimizers.Adam(lr=args.lr)

def training(args, model, train_dl):
  nb_datapoints = train_dl.shape[0]
  index = np.arange(nb_datapoints)
  image_index = open('{}/index.txt'.format(args.result_dir), 'a+')

  for epoch in range(1, args.epoch + 1):
    #shuffle the training set before generating batches
    print('Shuffling training set...')

    np.random.shuffle(index)

    #devide training set into batches
    print('Training epoch{}...'.format(epoch))
    logging.info('Training epoch{}...'.format(epoch))
    nb_batch = int(math.ceil(len(train_dl)/args.batchsize))
    for batch in range(nb_batch):
      image_index.write('E{} B{}:'.format(epoch, batch+1))
      index_batch = index[batch*args.batchsize:(batch+1)*args.batchsize]
      for i in index_batch:
        image_index.write(' {}'.format(i))

      train_batch = [train_dl[i] for i in index_batch]
      images_batch, joints_batch = transform(args, train_batch)
      for i in range(args.times_per_batch):
        loss, acc = model.train_on_batch(images_batch, joints_batch)
        print('E{} B{} Iter{}: loss:{}'.format(epoch, batch+1, i+1, loss))
        logging.info('E{} B{} Iter{}: loss:{}'.format(epoch, batch+1, i+1, loss))
      image_index.write('\n')
    image_index.write('\n')

  image_index.close()
    
def generate_bottleneck_features(args, conv_model, fc_model, images_batch, joints_batch):
  all_bottleneck_features = 0
  all_joints_info = 0

  for nb_dl, dl in enumerate(train_dl):
    image, joint = image_transform(args, dl.split(','))
    image = np.expand_dims(image, axis=0)
    joint = np.expand_dims(joint, axis=0)
    bottleneck_features = conv_model.predict(image)
    if (nb_dl+1) % 1000 == 0:
      print('{}k images processed'.format((nb_dl+1)/1000))
    if nb_dl == 0:
      all_bottleneck_features = bottleneck_features
      all_joints_info = joint
    else:
      all_bottleneck_features = np.concatenate((all_bottleneck_features, bottleneck_features), axis=0)
      all_joints_info = np.concatenate((all_joints_info, joint), axis=0)

  print('Saving bottleneck features...')
  logging.info('Saving bottleneck features...')
  np.save(open('all_bottleneck_features.npy', 'w'), all_bottleneck_features)
  np.save(open('all_joints_info.npy', 'w'), all_joints_info)

def train_fc_layers(args):
  train_data = np.load(open('all_bottleneck_features.npy'))
  train_joints = np.load(open('all_joints_info.npy'))
  print(train_data.shape)
  print(train_joints.shape)

  input_shape = train_data.shape[1:]
  fc_model = vgg_16_fc(input_shape, args.joints_num)
  fc_model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy'])

  fc_model.fit(train_data, train_joints, batch_size=args.batchsize, nb_epoch=args.epoch, shuffle=True)

def load_pretrain_weights(args, model):
  f = h5py.File(args.weights_path)

  for k in range(f.attrs['nb_layers']):
    if k >= args.untrainable_layers:
      break
    g = f['layer_{}'.format(k)]
    weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    model.layers[k].set_weights(weights)
  f.close()
  logging.info('Model loaded.')
  print('Model loaded')

  # set the first 25 layers to untrainable
  for layer in model.layers[0:args.untrainable_layers]:
    layer.trainable = False

  return model

if __name__ == '__main__':
  sys.path.append('../../scripts')  # to resume from result dir
  sys.path.append('../../models')  # to resume from result dir
  sys.path.append('models')  # to resume from result dir

  parser = argparse.ArgumentParser()
  parser.add_argument('--epoch', type=int)
  parser.add_argument('--batchsize', type=int)
  parser.add_argument('--times_per_batch', type=int)
  parser.add_argument('--lr', type=float)
  parser.add_argument('--datadir')
  parser.add_argument('--partial_dataset', type=int)
  parser.add_argument('--channel', type=int)
  parser.add_argument('--size', type=int)
  parser.add_argument('--min_dim', type=int)
  parser.add_argument('--cropping', type=int)
  parser.add_argument('--crop_pad_inf', type=float)
  parser.add_argument('--crop_pad_sup', type=float)
  parser.add_argument('--shift', type=int)
  parser.add_argument('--joints_num', type=int)
  parser.add_argument('--opt')
  parser.add_argument('--weights_path')
  parser.add_argument('--untrainable_layers', type=int)
  parser.add_argument('--debug', type=int)
  args = parser.parse_args()

  # create result dir
  create_result_dir(args)

  # load datasets
  train_dl, test_dl = load_dataset(args)

  # load pre-trained model and train part of the model
  if args.weights_path:
    # get prediction from conv layers
    model = vgg_16(args.joints_num)
    model = load_pretrain_weights(args, model)
    opt = get_optimizer(args)
    model.compile(optimizer=opt,
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    if args.partial_dataset:
      train_dl = train_dl[:args.partial_dataset]

    start_time = time.time()
    training(args, model, train_dl)
    end_time = time.time()

    logging.info('Training finished. Time used {}h'.format((end_time - start_time)/3600))
    model.save_weights('{}/model_weights.h5'.format(args.result_dir))
    model_structure = model.to_json()
    with open('{}/model_structure.json'.format(args.result_dir), 'w') as output:
      output.write(model_structure)