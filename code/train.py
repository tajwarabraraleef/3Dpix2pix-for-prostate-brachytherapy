'''
Training and prediction code for 3D pix2pix for automating seed planning for prostate brachytherapy.

In this code, the generator learns to map from PTV and CTV contours to needle plans.

Tajwar Abrar Aleef - tajwaraleef@ece.ubc.ca
Robotics and Control Laboratory, University of British Columbia
'''

import time
from keras.utils import generic_utils
from keras.optimizers import Adam, SGD
import metrics
import data_utils
import plot_functions
import pandas as pd
from resnet3D import Resnet3DBuilder_Discriminator, Resnet3DBuilder_Generator
from skimage.transform import resize
from keras.models import Model
from keras.layers import Input, Concatenate
import numpy as np
import keras
from keras.layers import Lambda
import os
from keras.models import load_model
import tensorflow as tf
import keras.backend as K

os.environ["KERAS_BACKEND"] = "tensorflow"
image_data_format = "channels_last"
K.set_image_data_format(image_data_format)

###
# Set parameters
###
model_name = "IPCAI"
training_type = 1  # 1 for training, 2 for saving predictions
use_all_plans = 1  # use all plans variations for training

PTVfilename = "PTV.npy"
CTVfilename = "CTV.npy"
seedsfilename = "Seed_plans.txt"
load_path = "../data/"
logging_dir = '../'
data_utils.log_folder(model_name, logging_dir)

batch_size = 1 # 16 in actual work
total_epoch = 1000
lr = 1E-5  # learning rate
B1 = 0.5  # beta 1 momentum parameter
B2 = 0.99  # beta 2 momentum parameter

save_model_interval = 100  # save model after every 100 epochs
save_plot_interval = 5  # save result plot after every 5 epochs
operation_point = 0.65  # probability threshold of output

###
# Load training and validation data (set save_all to 1 to plot loaded data)
###
Y_train, X_train, Y_val, X_val = data_utils.load_data(load_path, PTVfilename, CTVfilename, seedsfilename, use_all_plans,
                                                      save_all=1, shuffle=True)

print('\nTraining samples: ' + str(len(X_train)))
print('Validation samples: ' + str(len(X_val)))

###
# Augmentation step (setting left and right hand side of data as two different samples
###
half_dimen = 32
X_train = np.concatenate((X_train[:, :, 0:half_dimen, :, :], np.flip(X_train[:, :, half_dimen:, :, :], 2)), axis=0)
Y_train = np.concatenate((Y_train[:, :, 0:6, :], Y_train[:, :, 0:6, :]), axis=0)

X_val = X_val[:, :, 0:half_dimen, :, :]
Y_val = (Y_val[:, :, 0:6, :])

batch_size_per_epoch = np.floor(len(X_train) / batch_size)

print("\nNumber of training samples: %s" % len(X_train))
print("Number of batches: %s" % batch_size)
print("Number of batches per epoch: %s\n" % int(batch_size_per_epoch))

epoch_size = batch_size_per_epoch * batch_size

input_img_dim = X_train.shape[-4:]  # Reading shape of X_train without the number of samples, for eg 64x64x10x1

print("X train Max: " + str(np.max(X_train)) + ", Min: " + str(np.min(X_train)))
print("X val Max: " + str(np.max(X_val)) + ", Min: " + str(np.min(X_val)))
print("Y train Max: " + str(np.max(Y_train)) + ", Min: " + str(np.min(Y_train)))
print("Y val Max: " + str(np.max(Y_val)) + ", Min: " + str(np.min(Y_val)))

print('\nShape of X_train: ' + str(np.shape(X_train)))
print('Shape of Y_train: ' + str(np.shape(Y_train)))
print('\nShape of X_val: ' + str(np.shape(X_val)))
print('Shape of Y_val: ' + str(np.shape(Y_val)))

###
# Training Starts here
###
if training_type == 1:

	cgan_optimizer = Adam(lr=lr, beta_1=B1, beta_2=B2, epsilon=1e-08)
	discri_optimizer = Adam(lr=lr, beta_1=B1, beta_2=B2, epsilon=1e-08)

	## Generator/Encoder network
	generator_model = Resnet3DBuilder_Generator.build_generator((64, 32, 10, 2),
	                                                                       2)  # based on input dimension of PTV and CTV
	generator_model.summary()

	## Discriminator network
	discriminator_model = Resnet3DBuilder_Discriminator.build_discriminator((64, 32, 10, 3),
	                                                             2)  # based on input dimension of PTV, CTV, and Needle plan
	discriminator_model.summary()
	discriminator_model.trainable = False

	## cGAN model
	gen_input = Input(shape=input_img_dim, name="cGAN_input")
	generated_image = generator_model(gen_input)

	# Resize and concat block
	out = Lambda(lambda image: tf.image.resize(image, (64, 32), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
	                                           preserve_aspect_ratio=False))(generated_image)
	out = Concatenate(axis=-1)([out, out, out, out, out, out, out, out, out, out])
	out = Lambda(lambda image: K.expand_dims(image, axis=-1))(out)
	out = Concatenate(axis=-1)([gen_input, out])

	discri_output = discriminator_model(out)

	# For getting Dith feature for the feature loss calculation
	Dith_layer = Model(inputs=[discriminator_model.get_input_at(0)],
	                           outputs=[discriminator_model.get_layer(name='flatten_1').output])
	Dith_layer_features = Dith_layer(out)

	cGAN_model = Model(inputs=[gen_input], outputs=[generated_image, discri_output, Dith_layer_features], name="cGAN")

	cGAN_model.summary()

	## Loss function
	adversarial_loss = 'binary_crossentropy'
	feature_loss = metrics.feature_loss

	alpha = 100
	beta = 10

	loss = [metrics.needle_and_l1_loss(), adversarial_loss, feature_loss]
	loss_weights = [alpha, beta, beta]

	cGAN_model.compile(loss=loss, loss_weights=loss_weights, optimizer=cgan_optimizer)
	discriminator_model.trainable = True
	discriminator_model.compile(loss=adversarial_loss, optimizer=discri_optimizer)

	max_auc = 0
	max_dice = 0

	train_operating_point = []
	train_aucs = []
	train_accuracy = []
	train_sensitivity = []
	train_specificity = []
	train_dice = []
	train_needle_diff = []
	train_prohibited_needles = []

	val_operating_point = []
	val_aucs = []
	val_accuracy = []
	val_sensitivity = []
	val_specificity = []
	val_dice = []
	val_needle_diff = []
	val_prohibited_needles = []

	epochs_count = []
	disc_loss_all = []
	gen_loss_total = []
	gen_loss_needlel1 = []
	gen_loss_feature = []
	gen_loss_adv = []

	print("Start training")

	df = pd.DataFrame(
		columns=['epoch', 'Train OP', 'Val OP', 'Train AUC', 'Val AUC', 'Train Y Dice', 'Val Y Dice', 'Train Sensi',
		         'Val Sensi', 'Train Speci', 'Val Speci', 'Train Needles Diff', 'Val Needles Diff',
		         'Train Adjacent Needles', 'Val Adjacent Needles', 'D Loss', 'G Total', 'G Needle+L1', 'G Adversarial',
		         'G Feature'])

	for e in range(total_epoch):

		# Initialize progbar and batch counter
		progbar = generic_utils.Progbar(epoch_size)
		batch_counter = 1
		start = time.time()

		# Reading batch of images
		for batch_i, (Y_train_batch, X_train_batch, indx) in enumerate(
				data_utils.generate_batch(Y_train, X_train, batch_size)):

			# GT set as 1 for adversarial loss (we want Generator to produce realistic images)
			Y_cGAN_gt = np.zeros((X_train_batch.shape[0], 2), dtype=np.uint8)
			Y_cGAN_gt[:, 1] = 1

			# Freeze the discriminator
			discriminator_model.trainable = False

			# Resize and Concat block
			X_disc = (
				resize(Y_train_batch, (len(X_train_batch), np.shape(X_train_batch)[1], np.shape(X_train_batch)[2], 1),
				       order=0, preserve_range=True))
			X_disc = np.tile(X_disc, (1, 1, 1, np.shape(X_train_batch)[3]))
			X_disc = np.expand_dims(X_disc, axis=-1)
			X_disc = np.concatenate((X_train_batch, X_disc), axis=-1)
			features_gt = Dith_layer.predict(X_disc)

			# Train generator
			gen_loss = cGAN_model.train_on_batch(X_train_batch, [Y_train_batch, Y_cGAN_gt, features_gt])

			# Unfreeze the discriminator
			discriminator_model.trainable = True

			# Create X with PTV CTV and needle plans for Discriminator's input
			X_disc, y_disc = data_utils.get_disc_batch(Y_train_batch, X_train_batch, generator_model, batch_counter)

			# Train Discriminator
			disc_loss = discriminator_model.train_on_batch(X_disc, y_disc)

			progbar.add(batch_size,
			            values=[("D logloss", disc_loss), ("G Tot", gen_loss[0]), ("G Needle+L1", gen_loss[1]),
			                    ("G Adversarial", gen_loss[2]), ("G Feature", gen_loss[3])])

			batch_counter += 1

		disc_loss_all.append(disc_loss)
		gen_loss_total.append(gen_loss[0])
		gen_loss_needlel1.append(gen_loss[1])
		gen_loss_feature.append((gen_loss[2]))
		gen_loss_adv.append((gen_loss[3]))

		print(model_name)
		print("")
		print('Epoch %s/%s, Time: %s' % (e + 1, total_epoch, time.time() - start))

		# Predicting results using trained model for current epoch
		y_pred_train = generator_model.predict(X_train, verbose=0)  # ?,10,6,10,1
		y_pred_val = generator_model.predict(X_val, verbose=0)  # ?,10,6,10,1

		gt_thresholded_train = Y_train >= 0.6
		gt_thresholded_val = Y_val >= 0.6

		# Training evaluation metrics
		_, roc_auc_train, accuracy_train, specificity_train, sensitivity_train, dice_train = \
			metrics.use_operating_points(
			operation_point, gt_thresholded_train, y_pred_train)
		needle_diff_train = metrics.needle_difference(gt_thresholded_train, y_pred_train, operation_point)
		prohibited_needles_train = metrics.prohibited_needle_check(operation_point, y_pred_train)

		print(
			"Training Operating Point: {:.4f}, AUC: {:.4f}, Accuracy: {:.4f}, Sensitivity: {:.4f}, Specificity: {:.4f}, "
			"Dice: {:.4f}, Needle Diff: {:.2f}, Prohibited Needles: {:.2f}".format(operation_point, roc_auc_train,
				accuracy_train, sensitivity_train, specificity_train, dice_train, needle_diff_train,
				prohibited_needles_train))

		train_operating_point.append(operation_point)
		train_aucs.append(roc_auc_train)
		train_accuracy.append(accuracy_train)
		train_sensitivity.append(sensitivity_train)
		train_specificity.append(specificity_train)
		train_dice.append(dice_train)
		train_needle_diff.append(needle_diff_train)
		train_prohibited_needles.append(prohibited_needles_train)

		# Validation evaluation metrics
		_, roc_auc_val, accuracy_val, specificity_val, sensitivity_val, dice_val = metrics.use_operating_points(
			operation_point, gt_thresholded_val, y_pred_val)
		needle_diff_val = metrics.needle_difference(gt_thresholded_val, y_pred_val, operation_point)
		prohibited_needles_val = metrics.prohibited_needle_check(operation_point, y_pred_val)

		print("Validation Operating Point: {:.4f}, AUC: {:.4f}, Accuracy: {:.4f}, Sensitivity: {:.4f}, Specificity: {:.4f}, "
	    "Dice: {:.4f}, Needle Diff: {:.2f}, Needle Adj: {:.2f}".format(operation_point, roc_auc_val,
			accuracy_val, sensitivity_val, specificity_val, dice_val, needle_diff_val, prohibited_needles_val))

		val_operating_point.append(operation_point)
		val_aucs.append(roc_auc_val)
		val_accuracy.append(accuracy_val)
		val_sensitivity.append(sensitivity_val)
		val_specificity.append(specificity_val)
		val_dice.append(dice_val)
		val_needle_diff.append(needle_diff_val)
		val_prohibited_needles.append(prohibited_needles_val)

		epochs_count.append(e)

		# Save generated needles for visualization
		if e % save_plot_interval == 0:
			plot_functions.plot_predicted_needles(operation_point, Y_train_batch, X_train_batch, generator_model,
			                                      "Training_epoch_" + str(e), logging_dir, model_name)

			idx = np.random.choice(X_val.shape[0], batch_size, replace=False)
			plot_functions.plot_predicted_needles(operation_point, Y_val[idx], X_val[idx], generator_model,
			                                      "Validation_epoch_" + str(e), logging_dir, model_name)

		# Saving all metrics as a csv file
		new_row = {'epoch': e, 'Train OP': train_operating_point[e], 'Val OP': val_operating_point[e],
		           'Train AUC': train_aucs[e], 'Val AUC': val_aucs[e], 'Train Y Dice': train_dice[e],
		           'Val Y Dice': val_dice[e], 'Train Sensi': train_sensitivity[e], 'Val Sensi': val_sensitivity[e],
		           'Train Speci': train_specificity[e], 'Val Speci': val_specificity[e],
		           'Train Needles Diff': train_needle_diff[e], 'Val Needles Diff': val_needle_diff[e],
		           'Train Adjacent Needles': train_prohibited_needles[e].numpy(),
		           'Val Adjacent Needles': val_prohibited_needles[e].numpy(), 'D Loss': disc_loss_all[e],
		           'G Total': gen_loss_total[e], 'G Needle+L1': gen_loss_needlel1[e], 'G Adversarial': gen_loss_adv[e],
		           'G Feature': gen_loss_feature[e]}

		df = df.append(new_row, ignore_index=True)
		df.to_csv(os.path.join(logging_dir, 'models/%s/all_metrics.csv' % (model_name)), index=0)

		# Saving model if current epoch is producing better results
		if (roc_auc_val > max_auc):
			print(('\nValidation AUC improved from %f to %f at epoch %f\nSaving Top AUC weights....\n') % (
				max_auc, roc_auc_val, e))
			max_auc = roc_auc_val
			# Saving weights of network based on highest AUC
			gen_weights_path = os.path.join(logging_dir, 'models/%s/gen_weights_top_auc.h5' % (model_name))
			generator_model.save(gen_weights_path, overwrite=True)

		if (dice_val > max_dice):
			print(('\nValidation DICE improved from %f to %f at epoch %f\nSaving Top DICE weights....\n') % (
				max_dice, dice_val, e))
			max_dice = dice_val
			# Saving weights of network based on highest Dice
			gen_weights_path = os.path.join(logging_dir, 'models/%s/gen_weights_top_dice.h5' % (model_name))
			generator_model.save(gen_weights_path, overwrite=True)

		# Save weights after set intervals
		if e % save_model_interval == 0:
			gen_weights_path = os.path.join(logging_dir, 'models/%s/gen_weights_epoch%s.h5' % (model_name, e))
			generator_model.save(gen_weights_path, overwrite=True)

		# Plot training metrics and losses
		plot_functions.plot_training_history(train_prohibited_needles, val_prohibited_needles, train_aucs, val_aucs,
		                                     train_accuracy, val_accuracy, train_sensitivity, val_sensitivity,
		                                     train_specificity, val_specificity, train_dice, val_dice, epochs_count,
		                                     logging_dir, model_name)

		plot_functions.plot_training_losses(disc_loss_all, gen_loss_total, gen_loss_needlel1, gen_loss_feature,
		                                    gen_loss_adv, epochs_count, logging_dir, model_name)
###
## For predicting and saving from val/test set
###
if (training_type == 2):
	model_name = '/XX/'
	weightfile = '/gen_weights.h5'
	model_path = logging_dir + model_name + weightfile

	generator = load_model(model_path, compile=False, custom_objects={'keras': keras})
	print('Model loaded')

	# needles_train = generator.predict(x=X_train, batch_size=16, verbose=2)
	needles_val = generator.predict(x=X_val, batch_size=16, verbose=2)

	print('Prediction complete, saving....')
	# np.savez(logging_dir + model_name + '/predicted_output_TRAIN.npz', name1=X_train[:, :, :, :, 0:1],
	#          name2=needles_train, name3=Y_train)
	np.savez(logging_dir + model_name + '/predicted_output_VAL.npz', name1=X_val[:, :, :, :, 0:1], name2=needles_val,
	         name3=Y_val)

	print(np.shape(needles_val))
