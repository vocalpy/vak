import sys
import os
import pickle
from glob import glob
from configparser import ConfigParser, NoOptionError
from datetime import datetime
import scipy.io as cpio
import tensorflow as tf
import numpy as np
import joblib

import cnn_bilstm.utils
import cnn_bilstm.metrics
from cnn_bilstm.model import CNNBiLSTM

config_file = sys.argv[1]
if not config_file.endswith('.ini'):
    raise ValueError('{} is not a valid config file, must have .ini extension'
                     .format(config_file))
config = ConfigParser()
config.read(config_file)
print('Using definitions in: ' + config_file)
results_dirname = config['OUTPUT']['results_dir_made_by_main_script']
print('Results will be saved in: ' + results_dirname)
if not os.path.isdir(results_dirname):
    raise FileNotFoundError('{} directory is not found.'
                            .format(results_dirname))
batch_size = int(config['NETWORK']['batch_size'])
time_steps = int(config['NETWORK']['time_steps'])

TRAIN_SET_DURS = [int(element)
                  for element in
                  config['TRAIN']['train_set_durs'].split(',')]
print('Durations: ' + str(TRAIN_SET_DURS))
num_replicates = int(config['TRAIN']['replicates'])
print('Replicates: ' + str(num_replicates))
REPLICATES = range(num_replicates)

labelset = list(config['DATA']['labelset'])
skip_files_with_labels_not_in_labelset = config.getboolean(
    'DATA',
    'skip_files_with_labels_not_in_labelset')
labels_mapping_file = os.path.join(results_dirname, 'labels_mapping')
with open(labels_mapping_file, 'rb') as labels_map_file_obj:
    labels_mapping = pickle.load(labels_map_file_obj)
    n_syllables = len(labels_mapping)
input_vec_size = joblib.load(
    os.path.join(
        results_dirname,
        'X_train')).shape[-1]
print('vec_size: ' + str(input_vec_size))
data_folder = config['TRAIN']['test_data_path'];
data_folder = data_folder[:-14]
spect_list = glob(data_folder + '*.spect')
true_labels = np.zeros((len(spect_list),), dtype=np.object) 
estimates = np.zeros((len(TRAIN_SET_DURS),num_replicates,len(spect_list),), dtype=np.object)
for dur_ind, train_set_dur in enumerate(TRAIN_SET_DURS):
  Y_pred_test_this_dur = []
  Y_pred_train_this_dur = []
  Y_pred_test_labels_this_dur = []
  Y_pred_train_labels_this_dur = []
  for replicate in REPLICATES:
    print("getting train and test error for "
                  "training set with duration of {} seconds, "
                  "replicate {}".format(train_set_dur, replicate))
    training_records_dir = os.path.join(results_dirname,
        (
        'records_for_training_set_with_duration_of_'
        + str(
            train_set_dur) + '_sec_replicate_'
        + str(replicate))
        )
    checkpoint_filename = ('checkpoint_train_set_dur_'
       + str(train_set_dur) +
       '_sec_replicate_'
       + str(replicate))
    meta_file = glob(os.path.join(training_records_dir, 'checkpoint*meta*'))[0]
    data_file = glob(os.path.join(training_records_dir, 'checkpoint*data*'))[0]
    #input_vec_size = X_train_subset.shape[-1]  # number of columns
    model = CNNBiLSTM(n_syllables=n_syllables,
        input_vec_size=input_vec_size,
        batch_size=batch_size)
    with tf.Session(graph=model.graph) as sess:
      tf.logging.set_verbosity(tf.logging.ERROR)

      model.restore(sess=sess,
        meta_file=meta_file,
        data_file=data_file)
      for fnum in range(len(spect_list)):
        data = joblib.load(spect_list[fnum])
        Xd = data['spect'].T
        Yd = data['labeled_timebins']
        (Xd_batch,Yd_batch,num_batches_val) = cnn_bilstm.utils.reshape_data_for_batching(Xd,
             Yd,
             batch_size,
             time_steps,
             input_vec_size)
        if 'Y_pred_train' in locals():
          del Y_pred_train

        for b in range(num_batches_val):  # "b" is "batch number"
          #d = {X: Xd_batch[:, b * time_steps: (b + 1) * time_steps, :], lng: [time_steps] * batch_size}
          d = {model.X:Xd_batch[:, b * time_steps: (b + 1) * time_steps, :],
                         model.lng: [time_steps] * batch_size}
          if 'Y_pred_test' in locals():
            #preds = sess.run(eval_op, feed_dict=d)[1]
            preds = sess.run(model.predict, feed_dict=d)
            #print(np.shape(preds))
            preds = preds.reshape( -1) #batch_size,
            #print(batch_size)
            #print(np.shape(preds))
            #print(np.shape(Y_pred_test))
            Y_pred_test = np.concatenate((Y_pred_test, preds), axis=0)
            print(np.shape(Y_pred_test))
          else:
            Y_pred_test = sess.run(model.predict, feed_dict=d)
            print(np.shape(Y_pred_test))
            Y_pred_test = Y_pred_test.reshape( -1) #batch_size,
            print(np.shape(Y_pred_test))
        Y_pred_test = Y_pred_test.ravel()
        Y_pred_test = Y_pred_test[0:len(Yd)]
        estimates[dur_ind][replicate][fnum] = Y_pred_test
        if replicate == 0 and dur_ind == 0:
          true_labels[fnum] = Yd.ravel()
        print('Duration: ' + str(dur_ind) + '/' + str(len(TRAIN_SET_DURS)) +
                  ' replicate: ' + str(replicate) + '/'+ str(num_replicates) + 
                  'file: ' + str(fnum) + '/' + str(len(spect_list)))

fname = os.path.join(results_dirname,'summary_per_file.mat')
cpio.savemat(fname ,{'estimates':estimates, 'true_labels':true_labels})
fname = os.path.join(results_dirname,'summary_per_file')
joblib.dump({'estimates':estimates, 'true_labels':true_labels}, fname)

