# get accuracy on training set
# make it into one big spectrogram again
X_train_subset = X_train[train_inds, :]
Y_train_subset = Y_train[train_inds]
# need to loop through train data in chunks, can't fit on GPU all at once
# First zero pad
num_batches = X_train_subset.shape[0] // batch_size // time_steps
rows_to_append = ((num_batches + 1) * time_steps * batch_size) - X_train.shape[0]
X_train_subset = np.append(X_train_subset, np.zeros((rows_to_append, input_vec_size)),
                           axis=0)
Y_train_subset = np.append(Y_train_subset, np.zeros((rows_to_append, 1)), axis=0)
num_batches = num_batches + 1
X_train_subset = X_train_subset.reshape((batch_size, num_batches * time_steps, -1))
Y_train_subset = Y_train_subset.reshape((batch_size, num_batches * time_steps, -1))

if 'preds' in locals():
    del preds

for b in range(num_batches):  # "b" is "batch number"
    d = {X: X_train_subset[:, b * time_steps: (b + 1) * time_steps, :],
         Y: Y_train_subset[:, b * time_steps: (b + 1) * time_steps],
         lng: time_steps * batch_size}

    if 'preds' in locals():
        preds = np.concatenate((preds,
                                sess.run(eval_op,
                                         feed_dict=d)[1][:unpadded_length]))
    else:
        preds = sess.run(eval_op, feed_dict=d)[1][:unpadded_length]

Y_train_subset = Y_train[train_inds]
train_err = np.sum(preds - Y_train_subset != 0) / Y_train_subset.shape[0])
