import tensorflow as tf
import numpy as np
import scipy.io as cpio
import os
import time

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

# Parameters
input_vec_size = 513 #= lstm_size
batch_size = 3
n_lstm_layers = 2
n_syllables = 28 #including zero
learning_rate = 0.001
n_max_iter = 10001
time_steps = 100

# The inference graph
def label_inference_graph(spectrogram, num_hidden, num_layers, seq_length):
    #First convolutional layers
     # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=tf.reshape(spectrogram,[batch_size,time_steps,input_vec_size,1]),
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 8], strides=[1,8])
    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 8], strides=[1,8])
    # Second the dynamic bi-directional, multi-layered LSTM
    with tf.name_scope('biRNN'):
        with tf.variable_scope('fwd'):
            lstm_f1 = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0, state_is_tuple=True,reuse=None)
            lstm_f2 = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0, state_is_tuple=True,reuse=None)
            #lstm_f3 = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0, state_is_tuple=True,reuse=None)
            cells_f = tf.contrib.rnn.MultiRNNCell([lstm_f1,lstm_f2], state_is_tuple=True)
        with tf.variable_scope('bck'):
            lstm_b1 = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0, state_is_tuple=True,reuse=None)
            lstm_b2 = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0, state_is_tuple=True,reuse=None)
            #lstm_b3 = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0, state_is_tuple=True,reuse=None)
            cells_b = tf.contrib.rnn.MultiRNNCell([lstm_b1,lstm_b2], state_is_tuple=True)
        outputs, _states = tf.nn.bidirectional_dynamic_rnn(lstm_f1,lstm_b1, tf.reshape(pool2,[batch_size,time_steps-2,512]), time_major=False, dtype=tf.float32,sequence_length=seq_length)
    # Second, projection on the number of syllables creates logits
    with tf.name_scope('Projection'):
        W_f = tf.Variable(tf.random_normal([num_hidden, n_syllables]))
        W_b = tf.Variable(tf.random_normal([num_hidden, n_syllables]))
        bias = tf.Variable(tf.random_normal([n_syllables]))
    expr1 = tf.unstack(outputs[0],axis=0,num=batch_size)
    expr2 = tf.unstack(outputs[1],axis=0,num=batch_size)
    #logits = [tf.matmul(outputs[0][:,a,:],W_f) + bias + tf.matmul(outputs[1][:,a,:],W_b) for a in range(seq_length[0])]
    logits = tf.concat([tf.matmul(ex1,W_f) + bias + tf.matmul(ex2,W_b) for ex1,ex2 in zip(expr1,expr2)],0)
    return logits,outputs

# The training graph. Calculate cross entropy and loss function
def training_graph(logits, lbls, rate, lng):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits,labels = tf.concat(tf.unstack(lbls,axis=0,num=batch_size),0), name='xentropy')
    cost = tf.reduce_mean(xentropy, name='cost')
    optimizer = tf.train.AdamOptimizer(learning_rate = rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(cost, global_step=global_step)
    return train_op, cost

# Construct the full graph and add saver
full_graph = tf.Graph()
with full_graph.as_default():
        # Generate placeholders for the spectrograms and labels.
        X = tf.placeholder("float", [batch_size,time_steps,input_vec_size], name = "Xdata") # holds spectrograms
        Y = tf.placeholder("int32",[batch_size,None],name = "Ylabels") # holds labels
        lng = tf.placeholder("int32",name = "nSteps") # holds the sequence length
        tf.add_to_collection("specs", X)  # Remember this Op.
        tf.add_to_collection("labels", Y)  # Remember this Op.
        tf.add_to_collection("lng", lng)  # Remember this Op.
        # Build a Graph that computes predictions from the inference model.
        logits,outputs = label_inference_graph(X, 512, n_lstm_layers, lng) #lstm_size
        tf.add_to_collection("logits", logits)  # Remember this Op.

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op, cost = training_graph(logits, Y, learning_rate, lng)

        # Add the variable initializer Op.
        init = tf.global_variables_initializer() #initialize_all_variables()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver(max_to_keep = 10)

# Train and save checkpoint at the end of each file.
with tf.Session(graph=full_graph) as sess:
    #,config = tf.ConfigProto(intra_op_parallelism_threads = 1)
        # Run the Op to initialize the variables.
        sess.run(init)
        # Start the training loop.
        costs = []
        step = 1
        # Go over all training files
        file_num = 0
        fname = data_list['keys'][0][file_num][0][0:-3]+'mat'
        data = cpio.loadmat(data_directory + '/' + fname)
        data1 = np.transpose(data['s'])
        intY = data['labels'][0]
        for file_num in range(number_of_files-1):
            # load current training file
            fname = data_list['keys'][0][file_num+1][0][0:-3]+'mat'
            bdata = cpio.loadmat(data_directory + '/' + fname)
            bdata1 = np.transpose(bdata['s'])
            bintY = bdata['labels'][0]
            data1 = np.concatenate((data1,bdata1),axis = 0)
            intY = np.concatenate((intY,bintY),axis = 0)
            temp_n = len(intY)/batch_size
            data1 = data1[0:temp_n*batch_size].reshape((batch_size,temp_n,-1))
            intY = intY[0:temp_n*batch_size].reshape((batch_size,-1))
            iter_order = np.random.permutation(data1.shape[1]-370)
            if (len(iter_order) > n_max_iter):
                iter_order = iter_order[0:n_max_iter]
            print data1.shape, len(iter_order)
            for iternum in iter_order:
                d = {X: data1[:,iternum:iternum+time_steps,:] ,Y: intY[:,iternum+2:iternum+time_steps] ,lng:[time_steps-2]*batch_size}
                _cost,_ = sess.run((cost,train_op),feed_dict = d)
                costs.append(_cost)
                print([step,iternum,_cost])
                step = step + 1

                if (step % 100 == 0):
                    checkpoint_file = os.path.join(training_records_dir, 'checkpoint')
                    saver.save(sess, checkpoint_file, global_step=step)
                    print np.mean(costs[-10:-1])
            checkpoint_file = os.path.join(training_records_dir, 'checkpoint')
            saver.save(sess, checkpoint_file, global_step=step)
