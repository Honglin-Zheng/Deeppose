import cv2 as cv
import json
import numpy as np
import os

def image_resize(image, joints, new_size):
  orig_h, orig_w = image.shape[:2]
  joints[0::2] = joints[0::2] / float(orig_w) * new_size
  joints[1::2] = joints[1::2] / float(orig_h) * new_size
  image = cv.resize(image, (new_size, new_size), interpolation=cv.INTER_NEAREST).astype(np.float32)

  return image, joints

def image_cropping(image, joints, args):
  # image cropping
  joints = joints.reshape((len(joints) // 2, 2))
  posi_joints = [(j[0], j[1]) for j in joints if j[0] > 0 and j[1] > 0]
  x, y, w, h = cv.boundingRect(np.asarray([posi_joints]))
  if w < args.min_dim:
      w = args.min_dim
  if h < args.min_dim:
      h = args.min_dim

  # bounding rect extending
  inf, sup = args.crop_pad_inf, args.crop_pad_sup
  r = sup - inf
  pad_w_r = np.random.rand() * r + inf  # inf~sup
  pad_h_r = np.random.rand() * r + inf  # inf~sup
  x -= (w * pad_w_r - w) / 2
  y -= (h * pad_h_r - h) / 2
  w *= pad_w_r
  h *= pad_h_r

  # shifting
  x += np.random.rand() * args.shift * 2 - args.shift
  y += np.random.rand() * args.shift * 2 - args.shift

  # clipping
  x, y, w, h = [int(z) for z in [x, y, w, h]]
  x = np.clip(x, 0, image.shape[1] - 1)
  y = np.clip(y, 0, image.shape[0] - 1)
  w = np.clip(w, 1, image.shape[1] - (x + 1))
  h = np.clip(h, 1, image.shape[0] - (y + 1))
  image = image[y:y + h, x:x + w]

  # joint shifting
  joints = np.asarray([(j[0] - x, j[1] - y) for j in joints])
  joints = joints.flatten()

  return image, joints

def image_transform(args, datum):
  img_fn = '%s/images/%s' % (args.datadir, datum[0])
  if not os.path.exists(img_fn):
    raise Exception('%s does not exist' % img_fn)

  image = cv.imread(img_fn)
  joints = np.asarray([int(float(p)) for p in datum[1:]])

  if args.cropping == 1:
    image, joints = image_cropping(image, joints, args)
  if args.size > 0:
    image, joints = image_resize(image, joints, args.size)

  # joint pos centerization
  # h, w, c = image.shape
  # center_pt = np.array([w / 2, h / 2], dtype=np.float32)  # x,y order
  # joints = list(zip(joints[0::2], joints[1::2]))

  # posi_joints = [(j[0], j[1]) for j in joints if j[0] >= 0 and j[1] >= 0]
  # x, y, ww, hh = cv.boundingRect(np.asarray([posi_joints]))
  # bbox = [(x, y), (x + ww, y + hh)]

  # joints = np.array(joints, dtype=np.float32) - center_pt
  # joints[:, 0] /= w
  # joints[:, 1] /= h
  # joints = joints.flatten()

  image = image.transpose((2,0,1))
  # image = np.expand_dims(image, axis=0)
  return image, joints

def transform(args, dataset):
  images = []
  joints = []
  for i, data in enumerate(dataset):
    datum = data.split(',')
    img, jts = image_transform(args, datum)
    images.append(img)
    joints.append(jts)

  return np.asarray(images), np.asarray(joints)
