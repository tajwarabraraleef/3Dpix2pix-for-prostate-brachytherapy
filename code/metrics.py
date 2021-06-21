'''
Evaluation and loss metrics used for 3D pix2pix for automating seed planning for prostate brachytherapy.

Tajwar Abrar Aleef - tajwaraleef@ece.ubc.ca
Robotics and Control Laboratory, University of British Columbia
'''

import keras.backend as K
import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_curve, auc

# This is the combined Needle and L1 loss
def needle_and_l1_loss():
	def loss(y_true, y_pred):
		return K.mean(l1_loss(y_true, y_pred)) + K.mean(needle_loss(y_true, y_pred))
	return loss

# Code for calculating Needle loss based on the prohibited needle patterns given by the 4 kernels
def needle_loss(y_true, y_pred):

	kernel1 = [1, 1, 1],
	kernel1 = tf.expand_dims(kernel1, axis=2)
	kernel1 = tf.expand_dims(kernel1, axis=3)

	loss1 = tf.nn.convolution(tf.cast(y_pred, tf.float32), tf.cast(kernel1, tf.float32), strides=None, padding='SAME')
	loss1 = loss1 - (tf.cast(K.sum(kernel1), tf.float32) - 1.2)
	loss1 = tf.nn.relu(loss1)

	kernel2 = [[1], [1], [1], [1]]
	kernel2 = tf.expand_dims(kernel2, axis=2)
	kernel2 = tf.expand_dims(kernel2, axis=3)

	loss2 = tf.nn.convolution(tf.cast(y_pred, tf.float32), tf.cast(kernel2, tf.float32), strides=None, padding='SAME')
	loss2 = loss2 - (tf.cast(K.sum(kernel2), tf.float32) - 1.2)
	loss2 = tf.nn.relu(loss2)

	kernel3 = [[1, 0], [1, 1], [1, 0]]
	kernel3 = np.squeeze(kernel3)
	kernel3 = tf.expand_dims(kernel3, axis=2)
	kernel3 = tf.expand_dims(kernel3, axis=3)

	loss3 = tf.nn.convolution(tf.cast(y_pred, tf.float32), tf.cast(kernel3, tf.float32), strides=None, padding='SAME')
	loss3 = loss3 - (tf.cast(K.sum(kernel3), tf.float32) - 1.2)
	loss3 = tf.nn.relu(loss3)

	kernel4 = [[1, 1], [1, 1]]
	kernel4 = np.squeeze(kernel4)
	kernel4 = tf.expand_dims(kernel4, axis=2)
	kernel4 = tf.expand_dims(kernel4, axis=3)

	loss4 = tf.nn.convolution(tf.cast(y_pred, tf.float32), tf.cast(kernel4, tf.float32), strides=None,
	                          padding='SAME')  # convolves to finds pixels that doesnt follow adjacency rule
	loss4 = loss4 - (tf.cast(K.sum(kernel4), tf.float32) - 1.2)
	loss4 = tf.nn.relu(loss4)

	return K.sum(loss1 + loss2 + loss3 + loss4, axis=[-3, -2, -1])

# For calculating L1 loss
def l1_loss(y_true, y_pred):
	return K.sum(K.abs(y_pred - y_true), axis=[-3, -2, -1])

# For calculating Feature loss
def feature_loss(y_true, y_pred):
	return K.mean(K.sum(K.abs(y_pred - y_true), axis=-1))

# Checks the prohibited needle patterns given by the 4 kernels
def prohibited_needle_check(threshold, y_pred, smooth=0.00001):

	temp = y_pred
	temp = tf.clip_by_value(temp, 0, threshold)  # converting to 0-0.5, anything above 0.5 is 0.5
	temp = tf.where(tf.equal(temp, threshold), tf.ones_like(temp), tf.zeros_like(temp))

	kernel1 = [1, 1, 1],
	kernel1 = tf.expand_dims(kernel1, axis=2)
	kernel1 = tf.expand_dims(kernel1, axis=3)
	pn1 = tf.nn.convolution(tf.cast(temp, tf.float32), tf.cast(kernel1, tf.float32), strides=None, padding='SAME')
	pn1 = tf.clip_by_value(pn1, 0, 3)
	pn1 = tf.where(tf.equal(pn1, 3), x=tf.ones_like(pn1), y=tf.zeros_like(pn1), name=None)

	kernel2 = [[1], [1], [1], [1]]
	kernel2 = tf.expand_dims(kernel2, axis=2)
	kernel2 = tf.expand_dims(kernel2, axis=3)
	pn2 = tf.nn.convolution(tf.cast(temp, tf.float32), tf.cast(kernel2, tf.float32), strides=None, padding='SAME')
	pn2 = tf.clip_by_value(pn2, 0, 4)
	pn2 = tf.where(tf.equal(pn2, 4), x=tf.ones_like(pn2), y=tf.zeros_like(pn2), name=None)

	kernel3 = [[1, 0], [1, 1], [1, 0]]
	kernel3 = np.squeeze(kernel3)
	kernel3 = tf.expand_dims(kernel3, axis=2)
	kernel3 = tf.expand_dims(kernel3, axis=3)
	pn3 = tf.nn.convolution(tf.cast(temp, tf.float32), tf.cast(kernel3, tf.float32), strides=None, padding='SAME')
	pn3 = tf.clip_by_value(pn3, 0, 4)
	pn3 = tf.where(tf.equal(pn3, 4), x=tf.ones_like(pn3), y=tf.zeros_like(pn3), name=None)

	kernel4 = [[1, 1], [1, 1]]
	kernel4 = np.squeeze(kernel4)
	kernel4 = tf.expand_dims(kernel4, axis=2)
	kernel4 = tf.expand_dims(kernel4, axis=3)
	pn4 = tf.nn.convolution(tf.cast(temp, tf.float32), tf.cast(kernel4, tf.float32), strides=None, padding='SAME')
	pn4 = tf.clip_by_value(pn4, 0, 4)
	pn4 = tf.where(tf.equal(pn4, 4), x=tf.ones_like(pn4), y=tf.zeros_like(pn4), name=None)

	total = K.sum(pn1 + pn2 + pn3 + pn4)

	return total

# Finds difference in needles predicted vs GT
def needle_difference(y_gt, y_pr, operation_point):
	return np.mean(abs(np.sum(y_gt > 0.6, axis=(1, 2, 3)) - np.sum(y_pr > operation_point, axis=(1, 2, 3))))

# Calculated evaluation metrics based on operating point (threshold)
def use_operating_points(operation_point, y_gt, y_pr):

	# Find AUC, Accuracy, Specificity, Sensitivity, Dice
	auc_ = []
	accuracy = []
	specificity = []
	sensitivity = []
	dice = []

	for i in range(np.shape(y_gt)[0]):
		fpr, tpr, _ = roc_curve(y_gt[i, :, :, :].flatten(), y_pr[i, :, :, :].flatten())
		auc_.append(auc(fpr, tpr))

		y_temp = y_pr[i, :, :, :].flatten()
		y_temp = y_temp >= operation_point
		TP, FP, TN, FN = perf_measure(y_gt[i, :, :, :].flatten(), y_temp)
		accuracy.append((TP + TN) / (TP + FP + TN + FN))
		fpr_temp = (FP / (FP + TN))
		tpr_temp = (TP / (TP + FN))

		specificity.append((1 - fpr_temp))
		sensitivity.append(tpr_temp)
		dice.append((2 * TP) / ((2 * TP) + FP + FN))

	return operation_point, np.mean(auc_, axis=0), np.mean(accuracy, axis=0), np.mean(specificity, axis=0), np.mean(
		sensitivity, axis=0), np.mean(dice, axis=0)

def perf_measure(y_actual, y_hat):

	TP = np.logical_and(y_actual,y_hat)
	FP = np.logical_and(y_hat,abs(y_actual-1))
	TN = np.logical_and(abs(y_hat-1),abs(y_actual-1))
	FN = np.logical_and(y_actual,abs(y_hat-1))

	return(np.sum(TP), np.sum(FP), np.sum(TN), np.sum(FN))

# Finds the number of needles and seeds used
def seed_needle_count(seeds_pred, seeds_gt, operating_point = 0.5):
	seeds_pred = np.squeeze(np.float32(seeds_pred>= operating_point))
	seeds_gt = np.squeeze(np.float32(seeds_gt >= 0.6))

	# number of seeds
	seeds_pred_no = np.sum(seeds_pred, axis=(0, 1, 2))
	seeds_gt_no = np.sum(seeds_gt, axis=(0, 1, 2))

	# number of needles
	needles_pred_no = np.sum(seeds_pred, axis=-1)
	needles_gt_no = np.sum(seeds_gt, axis=-1)

	needles_pred_no = needles_pred_no >= 1
	needles_gt_no = needles_gt_no >= 1

	needles_pred_no = np.sum(needles_pred_no, axis=(0, 1))
	needles_gt_no = np.sum(needles_gt_no, axis=(0, 1))

	return seeds_pred_no, seeds_gt_no, needles_pred_no, needles_gt_no