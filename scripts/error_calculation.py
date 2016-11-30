import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import math

def plot_error_graph(loss_array):
  num_dp = loss_array.shape[1]
  colors = ['blue', 'green', 'red', 'yellow', 'magenta']
  labels = ['pred', 'pred*(480/224)', 'pred*(720/224)', 'pred[x]*(480/224)\npred[y]*(720/224)']
  patches = []
  for color_index, loss in enumerate(loss_array):
    patch = mpatches.Patch(color=colors[color_index], label=labels[color_index])
    patches.append(patch)
    for i, l in enumerate(loss):
      x1 = l
      y1 = (float(i)/num_dp) * 100
      plt.scatter(x1, y1, s=2, color=colors[color_index])

  plt.legend(handles=patches, loc='lower right')
  axes = plt.gca()
  axes.set_xlabel('root mean square error')
  axes.set_ylabel('% of data less than error')
  plt.show()

def one_joint_loss(pd_joints, gt_joints):
  diff = pd_joints - gt_joints
  square = diff * diff
  sum = np.sum(square)
  mse = sum/pd_joints.shape[0]
  rms = math.sqrt(mse)
  return rms

def calculate_loss(pd_file, gt_file):
  pd = np.array([l.strip() for l in open(pd_file).readlines()])
  gt = np.array([l.strip() for l in open(gt_file).readlines()])

  # if pd.shape[0] != gt.shape[0]:
  #   print('There are not equal cases in prediciton or ground truth')
  #   pass

  num_dp = pd.shape[0]
  loss = np.zeros(num_dp)
  exp1_loss = np.zeros(num_dp)
  exp2_loss = np.zeros(num_dp)
  exp3_loss = np.zeros(num_dp)
  exp4_loss = np.zeros(num_dp)
  exp5_loss = np.zeros(num_dp)

  for i, pd_data in enumerate(pd):
    pd_datum = pd_data.split(',')
    gt_datum = gt[i].split(',')

    pd_joints = np.asarray([float(p) for p in pd_datum[1:]])
    gt_joints = np.asarray([float(p) for p in gt_datum[1:]])

    loss[i] = one_joint_loss(pd_joints, gt_joints)

    exp1_pd_joints = gt_joints
    exp1_loss[i] = one_joint_loss(exp1_pd_joints, gt_joints)

    exp2_pd_joints = pd_joints * (float(480)/224)
    exp2_loss[i] = one_joint_loss(exp2_pd_joints, gt_joints)

    exp3_pd_joints = pd_joints * (float(720)/224)
    exp3_loss[i] = one_joint_loss(exp3_pd_joints, gt_joints)

    # exp4_pd_joints = pd_joints * 4
    # exp4_loss[i] = one_joint_loss(exp4_pd_joints, gt_joints)

    # exp5_pd_joints = pd_joints * 5
    # exp5_loss[i] = one_joint_loss(exp5_pd_joints, gt_joints)

    exp4_pd_joints = pd_joints
    exp4_pd_joints[0::2] = pd_joints[0::2] * (float(480)/224)
    exp4_pd_joints[1::2] = pd_joints[1::2] * (float(720)/224)
    exp4_loss[i] = one_joint_loss(exp4_pd_joints, gt_joints)


  return np.asarray([loss, exp2_loss, exp3_loss, exp4_loss])

def draw_graph(pd_file, gt_file):
  loss_array = np.sort(calculate_loss(pd_file, gt_file))
  plot_error_graph(loss_array)

draw_graph('predict_joints.csv', 'scale_groundtruth_joints.csv')