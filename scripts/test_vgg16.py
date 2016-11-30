from keras.models import model_from_json
from transformation import transform, image_transform
# from train_vgg16 import get_optimizer
import numpy as np
import cv2 as cv
import argparse
import logging
import math
import sys

def load_model(args, json_structure, weights):
  json_file = open(json_structure, 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  model = model_from_json(loaded_model_json)
  model.load_weights(weights)

  # opt = get_optimizer(args)
  model.compile(optimizer='Adam',
                loss='mean_squared_error',
                metrics=['accuracy'])

  return model

def test_model(args, test_dl):
  json_structure = '%s/model_structure.json' % args.modeldir
  weights = '%s/model_weights.h5' % args.modeldir

  print('Loading model...')
  model = load_model(args, json_structure, weights)
  print('Model loaded!')

  log_fn = '%s/test_log.txt' % args.modeldir
  logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s',
    filename=log_fn, level=logging.DEBUG)
  logging.info(args)

  nb_batch = int(math.ceil(len(test_dl)/args.batchsize))
  sum_loss = 0
  prediction_file = open('%s/predict_joints.csv'%args.modeldir, 'w')
  scale_groundtruth_file = open('%s/scale_groundtruth_joints.csv'%args.modeldir, 'w')

  for batch in range(nb_batch):
    test_batch = test_dl[batch*args.batchsize:(batch+1)*args.batchsize]
    test_images_batch, test_joints_batch = transform(args, test_batch)
    loss, acc = model.test_on_batch(test_images_batch, test_joints_batch)
    print('B{}: loss:{}'.format(batch+1, loss))
    logging.info('B{}: loss:{}'.format(batch+1, loss))
    sum_loss += loss

    predict_joints = model.predict_on_batch(test_images_batch)
    for i, test_datapoint in enumerate(test_batch):
      filename = test_datapoint.split(',')[0]
      msg = '{},{}\n'.format(filename, ','.join([str(j) for j in predict_joints[i].tolist()]))
      gt_msg = '{},{}\n'.format(filename, ','.join([str(j) for j in test_joints_batch[i].tolist()]))
      prediction_file.write(msg)
      scale_groundtruth_file.write(gt_msg)

  prediction_file.close()
  scale_groundtruth_file.close()

  avg_loss = sum_loss/nb_batch
  print('Average loss:{}'.format(avg_loss))
  logging.info('Average loss:{}'.format(avg_loss))

def draw_limb(img, joints, i, j, color):
  cv.line(img, joints[i], joints[j], (255, 255, 255), thickness=2, lineType=16)
  cv.line(img, joints[i], joints[j], color, thickness=1, lineType=16)
  return img

def draw_joints(img, joints, groundtruth=True, text_scale=0.5):
  h, w, c = img.shape

  if groundtruth:
    # left hand to left elbow
    img = draw_limb(img, joints, 0, 1, (0, 255, 0))
    img = draw_limb(img, joints, 1, 2, (0, 255, 0))
    img = draw_limb(img, joints, 4, 5, (0, 255, 0))
    img = draw_limb(img, joints, 5, 6, (0, 255, 0))
    img = draw_limb(img, joints, 2, 4, (0, 255, 0))
    neck = tuple((np.array(joints[2]) + np.array(joints[4])) // 2)
    joints.append(neck)
    img = draw_limb(img, joints, 3, 7, (0, 255, 0))
    joints.pop()

  # all joint points
    for j, joint in enumerate(joints):
      cv.circle(img, joint, 5, (0, 255, 0), -1)
      cv.circle(img, joint, 3, (0, 255, 0), -1)
      cv.putText(img, '%d' % j, joint, cv.FONT_HERSHEY_SIMPLEX, text_scale,
                 (0, 0, 0), thickness=3, lineType=16)
      cv.putText(img, '%d' % j, joint, cv.FONT_HERSHEY_SIMPLEX, text_scale,
                 (255, 255, 255), thickness=1, lineType=16)

  else:
    img = draw_limb(img, joints, 0, 1, (0, 0, 255))
    img = draw_limb(img, joints, 1, 2, (0, 0, 255))
    img = draw_limb(img, joints, 4, 5, (0, 0, 255))
    img = draw_limb(img, joints, 5, 6, (0, 0, 255))
    img = draw_limb(img, joints, 2, 4, (0, 0, 255))
    neck = tuple((np.array(joints[2]) + np.array(joints[4])) // 2)
    joints.append(neck)
    img = draw_limb(img, joints, 3, 7, (0, 0, 255))
    joints.pop()

  # all joint points
    for j, joint in enumerate(joints):
      cv.circle(img, joint, 5, (0, 0, 255), -1)
      cv.circle(img, joint, 3, (0, 0, 255), -1)
      cv.putText(img, '%d' % j, joint, cv.FONT_HERSHEY_SIMPLEX, text_scale,
                 (0, 0, 0), thickness=3, lineType=16)
      cv.putText(img, '%d' % j, joint, cv.FONT_HERSHEY_SIMPLEX, text_scale,
                 (255, 255, 255), thickness=1, lineType=16)

  return img

def plot_joints_limbs(args):
  image_index = args.draw_image-1
  predict_joints = '%s/predict_joints.csv' % args.predictdir
  test_joints = '%s/test_joints.csv' % args.datadir

  predict_data_point = np.array([l.strip() for l in open(predict_joints).readlines()])[image_index]
  test_data_point = np.array([l.strip() for l in open(test_joints).readlines()])[image_index]

  predict_datum = predict_data_point.split(',')
  test_datum = test_data_point.split(',')

  # adjust the image according to the ground truth joints
  image, groundtruth_joints = image_transform(args, test_datum)
  image = image.transpose((1,2,0))

  groundtruth_joints = groundtruth_joints.tolist()
  groundtruth_joints = list(zip(groundtruth_joints[0::2], groundtruth_joints[1::2]))

  joints = np.asarray([float(p) for p in predict_datum[1:]])
  #joints[0::2] = joints[0::2] * (float(480)/224)
  #joints[1::2] = joints[1::2] * (float(720)/224)
  joints = joints * (float(480)/224)

  joints = [int(p) for p in joints]
  joints = list(zip(joints[0::2], joints[1::2]))

  # draw groundtruth
  image = draw_joints(image, groundtruth_joints)
  # draw prediction
  image = draw_joints(image, joints, False)
  cv.imwrite('test_imgs/image.jpg', image)

if __name__ == '__main__':
  sys.path.append('scripts')

  parser = argparse.ArgumentParser()
  parser.add_argument('--opt')
  parser.add_argument('--batchsize', type=int)
  parser.add_argument('--datadir')
  parser.add_argument('--modeldir')
  parser.add_argument('--predictdir')
  parser.add_argument('--partial_dataset', type=int)
  parser.add_argument('--channel', type=int)
  parser.add_argument('--size', type=int)
  parser.add_argument('--min_dim', type=int)
  parser.add_argument('--cropping', type=int)
  parser.add_argument('--crop_pad_inf', type=float)
  parser.add_argument('--crop_pad_sup', type=float)
  parser.add_argument('--shift', type=int)
  parser.add_argument('--joints_num', type=int)
  parser.add_argument('--draw_image', type=int)
  args = parser.parse_args()

  test_fn = '%s/test_joints.csv' % args.datadir
  test_dl = np.array([l.strip() for l in open(test_fn).readlines()])

  if args.partial_dataset:
    test_dl = test_dl[:args.partial_dataset]

  if args.draw_image:
    plot_joints_limbs(args)
  else:
    test_model(args, test_dl)
  # datum = train_dl.split(',')
  # image, joints = image_transform(args, datum)
  # image = image.transpose((1,2,0))
  # print(joints)
  # plot_joints_limbs(image, joints)

  # img_fn = 'data/FLIC-full/images/%s' % (datum[0])
  # joints = np.asarray([int(float(p)) for p in datum[1:]])
  # plot_joints_limbs(img_fn, joints)


