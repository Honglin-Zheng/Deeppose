from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD

def vgg_16(joints_num):
  model = Sequential()
  model.add(ZeroPadding2D((1,1), input_shape=(3,224,224)))
  model.add(Convolution2D(64,3,3, activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(64,3,3, activation='relu'))
  model.add(MaxPooling2D((2,2), strides=(2,2)))

  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(128,3,3, activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(128,3,3, activation='relu'))
  model.add(MaxPooling2D((2,2), strides=(2,2)))

  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(256,3,3, activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(256,3,3, activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(256,3,3, activation='relu'))
  model.add(MaxPooling2D((2,2), strides=(2,2)))

  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(512,3,3, activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(512,3,3, activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(512,3,3, activation='relu'))
  model.add(MaxPooling2D((2,2), strides=(2,2)))

  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(512,3,3, activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(512,3,3, activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(512,3,3, activation='relu'))
  model.add(MaxPooling2D((2,2), strides=(2,2)))

  model.add(Flatten())
  model.add(Dense(4096, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(4096, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(joints_num*2, activation='relu'))

  return model

def vgg_16_conv():
  model = Sequential()
  model.add(ZeroPadding2D((1,1), input_shape=(3,224,224)))
  model.add(Convolution2D(64,3,3, activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(64,3,3, activation='relu'))
  model.add(MaxPooling2D((2,2), strides=(2,2)))

  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(128,3,3, activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(128,3,3, activation='relu'))
  model.add(MaxPooling2D((2,2), strides=(2,2)))

  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(256,3,3, activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(256,3,3, activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(256,3,3, activation='relu'))
  model.add(MaxPooling2D((2,2), strides=(2,2)))

  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(512,3,3, activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(512,3,3, activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(512,3,3, activation='relu'))
  model.add(MaxPooling2D((2,2), strides=(2,2)))

  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(512,3,3, activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(512,3,3, activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(512,3,3, activation='relu'))
  model.add(MaxPooling2D((2,2), strides=(2,2)))

  return model

def vgg_16_fc(ip_shape, joints_num):
  model = Sequential()
  model.add(Flatten(input_shape=ip_shape))
  model.add(Dense(4096, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(4096, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(joints_num*2, activation='softmax'))

  return model