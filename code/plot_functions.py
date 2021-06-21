'''
Plotting code for 3D pix2pix for automating seed planning for prostate brachytherapy.

Tajwar Abrar Aleef - tajwaraleef@ece.ubc.ca
Robotics and Control Laboratory, University of British Columbia
'''


import os
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import data_utils
import metrics


def plot_predicted_needles(operating_point, Y_gt, X_input, generator_model, suffix, logging_dir, model_name, number=1):

    """
    :param operating_point: Threshold for output
    :param Y_gt:
    :param X_input:
    :param generator_model:
    :param suffix:
    :param logging_dir:
    :param model_name:
    :param number: Number of cases to plot for visualization purpose
    :return:
    """

    Y_gen = generator_model.predict(X_input[:number])
    X_input = X_input[:number]
    Y_gen = Y_gen[:number]
    Y_gt = Y_gt[:number]

    path = os.path.join(os.path.join(logging_dir, "models/%s/visualization" % (model_name)))
    try: os.mkdir(path)
    except OSError: error_ = 0
    else: print("Successfully created the directory %s " % path)

    Y_gen = np.expand_dims(np.repeat(Y_gen, 10, axis=3), axis=-1)
    Y_gt = np.expand_dims(np.repeat(Y_gt, 10, axis=3), axis=-1)

    for j in range(number):

        Y_gen_thresholded = Y_gen[j, :, :, 0] >= operating_point
        Y_gt_thresholded = Y_gt[j, :, :, 0] >= 0.6

        fpr, tpr, _ = roc_curve(Y_gt_thresholded.flatten(),  Y_gen[j, :, :, 0].flatten())
        roc_auc = auc(fpr, tpr)

        TP, FP, TN, FN = metrics.perf_measure(Y_gt_thresholded.flatten(), Y_gen_thresholded.flatten())
        fpr_temp = (FP / (FP + TN))
        tpr_temp = (TP / (TP + FN))

        accuracy = (TP + TN) / (TP + FP + TN + FN)
        specificity = ((1 - fpr_temp))
        sensitivity = (tpr_temp)
        dice = (2 * TP) / ((2 * TP) + FP + FN)

        rows = 4 + 1
        cols = 1

        X_data = X_input[j,:,:,:,:].reshape((X_input[j,:,:,:,:].shape[0], X_input[j,:,:,:,:].shape[1] * X_input[j,:,:,:,:].shape[2], X_input[j,:,:,:,:].shape[3]),
                                order='F')

        seeds_pred_no, seeds_gt_no, needles_pred_no, needles_gt_no = metrics.seed_needle_count(Y_gen[j, :, :, :, 0],  Y_gt[j, :, :, :, 0],operating_point)

        plt.figure(figsize=(2*(rows+4), 2*rows))
        cnt = 0
        # display input
        cnt += 1
        ax = plt.subplot(rows, cols, cnt)
        plt.imshow(X_data[:, :, 0], cmap = 'gray')
        ax.get_xaxis().set_visible(False)
        ax.set_ylabel('PTV')
        ax.get_yaxis().set_visible(True)
        ax.get_yaxis().set_ticks([])
        ax.get_yaxis().set_ticklabels([])

        cnt += 1
        ax = plt.subplot(rows, cols, cnt)
        plt.imshow(X_data[:, :, 1], cmap = 'gray')
        ax.get_xaxis().set_visible(False)
        ax.set_ylabel('CTV')
        ax.get_yaxis().set_visible(True)
        ax.get_yaxis().set_ticks([])
        ax.get_yaxis().set_ticklabels([])

        # display predicted
        cnt += 1
        ax = plt.subplot(rows, cols, cnt)
        plt.imshow(Y_gen[j, :, :, 0, 0], cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.set_ylabel('Predicted')
        ax.get_yaxis().set_visible(True)
        ax.get_yaxis().set_ticks([])
        ax.get_yaxis().set_ticklabels([])

        cnt += 1
        ax = plt.subplot(rows, cols, cnt)
        plt.imshow(Y_gt[j, :, :, 0, 0], cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.set_ylabel('GT')
        ax.get_yaxis().set_visible(True)
        ax.get_yaxis().set_ticks([])
        ax.get_yaxis().set_ticklabels([])

        cnt += 1
        ax = plt.subplot(rows, cols, cnt)
        plt.imshow(Y_gen[j, :, :, 0, 0] >= operating_point, 'gray', interpolation= 'none')
        plt.imshow(Y_gt[j, :, :, 0, 0] >= 0.6, 'jet', interpolation='none', alpha=0.7)
        ax.get_xaxis().set_visible(False)
        ax.set_ylabel('Diff')
        ax.get_yaxis().set_visible(True)
        ax.get_yaxis().set_ticks([])
        ax.get_yaxis().set_ticklabels([])

        plt.suptitle(
            'AUC: {:.4f}, Accuracy: {:.4f}, Sensitivity: {:.4f}, Specificity: {:.4f}, Dice: {:.4f}, Operating Point: {:.4f}'
            '\n GT: Needles: {:.1f} PRED: Needles: {:.1f} '.format(
                roc_auc, accuracy, sensitivity, specificity, dice, operating_point, needles_gt_no, needles_pred_no))

        plt.savefig(os.path.join(logging_dir, "models/%s/visualization/%s_%s.png" % (model_name, suffix, str(j))))
        plt.clf()
        plt.close()

# Saves all the metrics and loss vs epoch plots
def	plot_training_history(train_operating_point, val_operating_point, train_aucs, val_aucs,
                                         train_accuracy, val_accuracy, train_sensitivity, val_sensitivity,
                                         train_specificity, val_specificity, train_dice, val_dice, epochs_count,
                                         logging_dir, model_name):

    rows = 2
    cols = 3
    plt.figure(figsize=(cols + 20, rows + 10))

    plt.subplot(rows, cols, 1)
    plt.plot(epochs_count, train_operating_point, label='Training')
    plt.plot(epochs_count, val_operating_point, color='orange', label='Validation')
    plt.ylim([-0.1, 10])
    plt.grid(True)
    plt.xlabel('Epochs')
    plt.ylabel('Prohibited Needle')
    plt.title('Prohibited Needle vs Epochs')
    plt.legend(loc="lower right")

    plt.subplot(rows, cols, 2)
    plt.plot(epochs_count, train_aucs, label='Training')
    plt.plot(epochs_count, val_aucs, color='orange', label='Validation')
    plt.ylim([-0.1, 1.1])
    plt.grid(True)
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.title('AUC vs Epochs')
    plt.legend(loc="lower right")

    plt.subplot(rows, cols, 3)
    plt.plot(epochs_count, train_accuracy, label='Training')
    plt.plot(epochs_count, val_accuracy, color='orange', label='Validation')
    plt.ylim([-0.1, 1.1])
    plt.grid(True)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epochs')
    plt.legend(loc="lower right")

    plt.subplot(rows, cols, 4)
    plt.plot(epochs_count, train_sensitivity, label='Training')
    plt.plot(epochs_count, val_sensitivity, color='orange', label='Validation')
    plt.ylim([-0.1, 1.1])
    plt.grid(True)
    plt.xlabel('Epochs')
    plt.ylabel('Sensitivity')
    plt.title('Sensitivity vs Epochs')
    plt.legend(loc="lower right")

    plt.subplot(rows, cols, 5)
    plt.plot(epochs_count, train_specificity, label='Training')
    plt.plot(epochs_count, val_specificity, color='orange', label='Validation')
    plt.ylim([-0.1, 1.1])
    plt.grid(True)
    plt.xlabel('Epochs')
    plt.ylabel('Specificity')
    plt.title('Specificity vs Epochs')
    plt.legend(loc="lower right")

    plt.subplot(rows, cols, 6)
    plt.plot(epochs_count, train_dice, label='Training')
    plt.plot(epochs_count, val_dice, color='orange', label='Validation')
    plt.ylim([-0.1, 1.1])
    plt.grid(True)
    plt.xlabel('Epochs')
    plt.ylabel('Dice')
    plt.title('Dice vs Epochs')
    plt.legend(loc="lower right")

    plt.savefig(os.path.join(logging_dir, "models/%s/visualization/1_Evaluation_Metrics.png" % (model_name)))
    plt.close()

# Saves training loss for generator and discriminator
def plot_training_losses(disc_loss_all, gen_loss_total, gen_loss_1, gen_loss_2, gen_loss_3, epochs_count,
                                         logging_dir, model_name):
    rows = 1
    cols = 2
    plt.figure(figsize=(cols + 20, rows + 10))

    plt.subplot(rows, cols, 1)
    plt.plot(epochs_count, disc_loss_all)
    plt.grid(True)
    plt.xlabel('Epochs')
    plt.ylabel('Discr. Loss')
    plt.title('Discr. Loss vs Epochs')

    plt.subplot(rows, cols, 2)
    #plt.plot(epochs_count, gen_loss_total, label='Total Gen Loss')
    plt.plot(epochs_count, gen_loss_1, color='orange', label='Needle+L1 Loss')
    plt.plot(epochs_count, gen_loss_2, color='purple', label='Feature Loss')
    plt.plot(epochs_count, gen_loss_3, color='green', label='Adver Loss')
    plt.grid(True)
    plt.xlabel('Epochs')
    plt.ylabel('Gen. Loss')
    plt.title('Gen. Loss vs Epochs')
    plt.legend(loc="lower right")

    plt.savefig(os.path.join(logging_dir, "models/%s/visualization/1_Evaluation_Loss.png" % (model_name)))
    plt.close()



# Plot all processed data
def plot_all_data(X_data, Y_data, logging_dir, type, strt):

    """

    :param X_data: PTV + CTV
    :param Y_data: Plans
    :param logging_dir: Where to save
    :param type: Training or validation
    :param strt: Starting index, set to 0 to plot all
    :return: Saves plotted data
    """

    path = logging_dir + "visualization/" + type + '/'
    data_utils.create_dir(path)

    Y_needles = np.sum(Y_data>0.6, axis= -2)
    Y_needles = np.repeat(Y_needles, Y_data.shape[3], axis=-1) #repeating to match same dimension as input

    cols = 1
    rows = 4

    #Reshaping for plotting
    X_data = X_data.reshape((X_data.shape[0], X_data.shape[1], X_data.shape[2]*X_data.shape[3], X_data.shape[4]), order='F')
    Y_data = Y_data.reshape((Y_data.shape[0], Y_data.shape[1], Y_data.shape[2]*Y_data.shape[3]), order='F')
    Y_needles = Y_needles.reshape((Y_needles.shape[0], Y_needles.shape[1], Y_needles.shape[2] * Y_needles.shape[3]), order='F')

    print(X_data.shape)
    print(Y_data.shape)

    for j in range(strt,len(X_data)):

        plt.figure(figsize=(cols * 14, rows))

        cnt = 0
        cnt += 1
        ax = plt.subplot(rows, cols, cnt)
        plt.imshow(X_data[j, :, :, 0], cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.set_ylabel('PTV')
        ax.get_yaxis().set_visible(True)
        ax.get_yaxis().set_ticks([])
        ax.get_yaxis().set_ticklabels([])

        cnt += 1
        ax = plt.subplot(rows, cols, cnt)
        plt.imshow(X_data[j, :, :, 1], cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.set_ylabel('CTV')
        ax.get_yaxis().set_visible(True)
        ax.get_yaxis().set_ticks([])
        ax.get_yaxis().set_ticklabels([])

        cnt += 1
        ax = plt.subplot(rows, cols, cnt)
        plt.imshow(Y_data[j, :, :] >= 0.6, cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.set_ylabel('Seed plan')
        ax.get_yaxis().set_visible(True)
        ax.get_yaxis().set_ticks([])
        ax.get_yaxis().set_ticklabels([])

        cnt += 1
        ax = plt.subplot(rows, cols, cnt)
        plt.imshow(Y_needles[j, :, :] >= 1, cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.set_ylabel('Needle plan')
        ax.get_yaxis().set_visible(True)
        ax.get_yaxis().set_ticks([])
        ax.get_yaxis().set_ticklabels([])

        #plt.show()
        plt.savefig((path + str(j) + "_index.png"))
        plt.clf()
        plt.close()
        print('Saving ' + type + ' data: ' + str(j))




