import os
import pickle
import sys
from configparser import ConfigParser
from glob import glob

import joblib
import numpy as np
import scipy.io as cpio
import tensorflow as tf

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
	for replicate in REPLICATES:
	#train_set_dur = TRAIN_SET_DURS[replicate]
	#replicate = 0
		print("getting train and test error for "
              "training set with duration of {} seconds, "
              "replicate {}".format(train_set_dur, replicate))
		training_records_dir = os.path.join(results_dirname,
                                            ('records_for_training_set_with_duration_of_'
                                             + str(train_set_dur) + '_sec_replicate_'
                                             + str(replicate))
                                            )
		checkpoint_filename = ('checkpoint_train_set_dur_'
                               + str(train_set_dur) +
                               '_sec_replicate_'
                               + str(replicate))
		meta_file = glob(os.path.join(training_records_dir, 'checkpoint*meta*'))[0]
		data_file = glob(os.path.join(training_records_dir, 'checkpoint*data*'))[0]

		with tf.Session(graph=tf.Graph()) as sess:
			tf.logging.set_verbosity(tf.logging.ERROR)
			saver = tf.train.import_meta_graph(meta_file)
			saver.restore(sess, data_file[:-20])
			# Retrieve the Ops we 'remembered'.
			logits = tf.get_collection("logits")[0]
			X = tf.get_collection("specs")[0]
			Y = tf.get_collection("labels")[0]
			lng = tf.get_collection("lng")[0]
			# Add an Op that chooses the top k predictions.
			eval_op = tf.nn.top_k(logits)
			for fnum in range(len(spect_list)):
				data = joblib.load(spect_list[fnum])
				Xd = data['spect'].T
				Yd = data['labeled_timebins']
				(Xd_batch,Yd_batch,num_batches_val) = tweetynet.utils.reshape_data_for_batching(Xd,
                                                                                                     Yd,
                                                                                                     batch_size,
                                                                                                     time_steps,
                                                                                                     input_vec_size)
				if 'Y_pred_test' in locals():
					del Y_pred_test
				for b in range(num_batches_val):  # "b" is "batch number"
					d = {X: Xd_batch[:, b * time_steps: (b + 1) * time_steps, :], lng: [time_steps] * batch_size}
					if 'Y_pred_test' in locals():
						preds = sess.run(eval_op, feed_dict=d)[1]
						preds = preds.reshape(batch_size, -1)
						#print(np.shape(preds))
						#print(np.shape(Y_pred_test))
						Y_pred_test = np.concatenate((Y_pred_test, preds), axis=1)
					else:
						Y_pred_test = sess.run(eval_op, feed_dict=d)[1]
						Y_pred_test = Y_pred_test.reshape(batch_size, -1)
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
