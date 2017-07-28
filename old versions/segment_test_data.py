import tensorflow as tf
import numpy as np
import scipy.io as cpio
import os
import time
import matplotlib.pyplot as plt
import glob

# Define the folder that contains all data files
# Each data file contains the variables:
#    s: The spectrogram [size = 513 x time_steps]
#    f: Frequencies [size = 513]
#    t: Time steps
#    labels: The tagging data [size = time_steps]
data_directory = '/Users/yardenc/Documents/Experiments/Imaging/CanaryData/lrb853_15/mat'
#data_directory = '/Users/yardenc/Documents/Experiments/Imaging/Data/CanaryData/lrb853_15/movs/wav/mat'

# This folder must also contain a matlab file 'file_list.mat' with cell array 'keys' that holds the data file names
data_list = cpio.loadmat(data_directory + '/file_list.mat')
number_of_files = len(data_list['keys'][0])
# The folder for saving training checkpoints
training_records_dir = '/Users/yardenc/Documents/Experiments/Imaging/CanaryData/lrb853_15/training_records'
#training_records_dir = '/Users/yardenc/Documents/Experiments/Imaging/Data/CanaryData/lrb853_15/training_records'

test_data_directory = '/Users/yardenc/Documents/Experiments/Imaging/CanaryData/lrb853_15/mat/test_data'

# Parameters
input_vec_size = 513 #= lstm_size
batch_size = 11
n_lstm_layers = 2
n_syllables = 39 #including zero
learning_rate = 0.001
n_max_iter = 14001
time_steps = 370
window_time_steps = 11

os.chdir(test_data_directory)
file_list = glob.glob('*.mat')
file_num = 502

# Evaluate training set from a saved checkpoint
with tf.Session(graph=tf.Graph()) as sess:
    saver = tf.train.import_meta_graph(
        os.path.join(training_records_dir, "checkpoint-12746.meta"))
    saver.restore(
        sess, os.path.join(training_records_dir, "checkpoint-12746"))

    # Retrieve the Ops we 'remembered'.
    logits = tf.get_collection("logits")[0]
    X = tf.get_collection("specs")[0]
    Y = tf.get_collection("labels")[0]
    lng = tf.get_collection("lng")[0]

    # Add an Op that chooses the top k predictions.
    eval_op = tf.nn.top_k(logits)

    # Run evaluation.
    # load current training file
    #for fname in file_list:
    fname = file_list[file_num]
    print fname

    data = cpio.loadmat(test_data_directory + '/' + fname)
    data1 = np.transpose(data['s'])
    print data1.shape
    temp_n = data1.shape[0]/batch_size
    rows_to_append = (temp_n + 1)*batch_size - data1.shape[0]
    data1 = np.append(data1,np.zeros((rows_to_append,input_vec_size)),axis=0)
    print data1.shape
    temp_n = temp_n + 1
    data1 = data1[0:temp_n*batch_size].reshape((batch_size,temp_n,-1))
    print data1.shape
    d = {X: data1, lng:[temp_n]*batch_size} #*batch_size
    pred = sess.run(eval_op,feed_dict = d) #eval_op
    estim = np.squeeze(pred[1]).reshape(-1)
        #estimates[file_num] = estim
        #file_num = file_num + 1

    plt.figure()
    plt.plot(np.squeeze(pred[1]).reshape(-1))
    #plt.plot(np.sum(data['s'],axis=0))
    plt.show()
    print np.unique(estim)
