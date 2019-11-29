import numpy as np
import tensorflow.keras.utils


class VakSequence(tensorflow.keras.utils.Sequence):
    def __init__(self, x, y, batch_size, shuffle=True):
        if len(x) != len(y):
            raise ValueError(
                'length of x does not equal length of y'
            )
        self.x = x
        self.y = y
        self.batch_size = batch_size

        self.indices = np.arange(len(x))

        self.shuffle = shuffle

        if shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.indices.shape[0] / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        inds = self.indices[index * self.batch_size: (index+1) * self.batch_size]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            np.random.shuffle(self.indices)

    @classmethod
    def from_csv(cls, csv, batch_size, shuffle=True):
        df = pd.read_csv(csv)
        x = df['spect_files']
        y = df['annot_files']
        return cls(x, y, batch_size, shuffle)
