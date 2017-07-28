# tf_syllable_segmentation_annotation
![alt text](https://github.com/yardencsGitHub/tf_syllable_segmentation_annotation/blob/master/img/sample_phrase_annotation.png)
## Jupyter notebook for tensorflow based syllable segmentation and annotation
This code utilizes the 'tensorflow' library to create, train, and use a deep neural-network algorithm for parsing and tagging birdsong spectrograms. For further details see section `Model structure`.
To install Jupyter notebook follow http://jupyter.readthedocs.io/en/latest/install.html
## Installing tensorflow
Please go to https://www.tensorflow.org/install/.
It is recommended to create an environment (e.g. using Anaconda as in https://www.tensorflow.org/install/install_mac). The code was developed with tensorflow version 1.1.0 and may be incompatible with other versions. 
## Data and folder structures
To use this code, the data has to be arranged in a certain format and in a specific folder structure.
### Spectrograms and labels
Currently, all data files must be in Matlab format. Each training file should contain 2 variables:
* s - A Nf x Nt real matrix of the spectrogram. Nt is the number of time steps. Nf is the number of frequency bins (the current code assumes 513 but this can be changed by updating the variable __input_vec_size__). 
The values in this matrix should range from 0 to 0.8 with 0 indicating low energy in the time-spectral bin.
* labels - A 1 x Nt real vector that contains the manually annotated label for each time bin. Use 0 to annotate silence or non-syllable noise. For any time bins during a syllable, use an integer. It is recommended to use the sequence [1,2, ... # of syllables] as labels and not large and sparse numbers.  
Testing or unlabeled files need to contain only the variable `s`.
### Folders and lists
The code contains variables for holding the names of four folders:
* data_directory - This folder contains all the training files. The folder must also contain a file called 'file_list.mat' that contains a Matlab's cell array called 'keys' that holds all the training file names.
* training_records_dir - This folder will hold the saved network states along the training procedure. After training, it is possible to use the last checkpoint, or any other mid-training point, to segment and annotate new data.
* test_data_directory - Contains only Matlab files with spectrograms to annotate. There is no need to create a list of files because the code will attempt to annotate any matlab file in this directory. Matlab files that do not contain a spectrogram 's' in the correct format will result in an error.
* results_dir - This folder will contain the annotations of the test data.
The code also requires specifying the name of the results file (the variable 'results_file') which will be saved in the results folder.
## Parameters
* The following parameters must be correctly defined:
  * input_vec_size - Must match the number of frequency bins in the spectrograms (current value is 513).
  * n_syllables - Must be the correct number of tags, including zero for non-syllable.
  * time_steps - The number of bins in a training snippet (current value is 87). The code concatenates all training data and trains the deep network using batches, containing snippets of length 'time_steps' from different points in the data. It is recommended to set 'time_steps' such that the snippets are of about 1 second.
* The following parameters can be changed if needed:
  * n_max_iter - The maximal number of training steps (currently 18001).
  * batch_size - The number of snippets in each training batch (currently 11)
  * learning_rate - The training step rate coefficient (currently 0.001)
Other parameters that specify the network itself can be changed in the code but require knowledge of tensorflow.
## Preparing training files
It is possible to train on any manually annotated data but there are some useful guidelines:
* __Use as many examples as possible__ - The results will just be better. Specifically, this code will not label correctly syllables it did not encounter while training and will most probably generalize to the nearest sample or ignore the syllable.
* __Use noise examples__ - This will make the code very good in ignoring noise.
* __Examples of syllables on noise are important__ - It is a good practice to start with clean recordings. The code will not perform miracles and is most likely to fail if the audio is too corrupt or masked by noise. Still, training with examples of syllables on the background of cage noises will be beneficial.
## Results of running the code
The code contains a section for evaluating performance in the training set and a section for labeling new data.
Labels of new data, in the folder set by the variable `test_data_directory`, are saved in a matlab format file whos name is defined by the variable `results_file`.
This file will contain two cell arrays:
* keys - Contains all file names.
* estimates - Contains all estimated labels.

__It is recommended to apply post processing when extracting the actual syllable tag and onset and offset timesfrom the estimates.__
## Model structure
The architecture of this deep neural network is based on these papers:
* S. Böck and M. Schedl, "Polyphonic piano note transcription with recurrent neural networks," 2012 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Kyoto, 2012, pp. 121-124.
doi: 10.1109/ICASSP.2012.6287832 (http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6287832&isnumber=6287775)
* Parascandolo, Huttunen, and Virtanen, “Recurrent Neural Networks for Polyphonic Sound Event Detection in Real Life Recordings.” (https://arxiv.org/abs/1604.00861)
The deep net. structure, used in this code, contains 3 elements:
* 2 convolutional and max pooling layers - A convolutional layer convolves the spectrogram with a set of tunable features and the max pooling is used to limit the number of parameters. These layers allow extracting local spectral and temporal features of syllables and noise.
* A long-short-term-memory recurrent layer (LSTM) - This layer allows the model to incorporate the temporal dependencies in the signal, such as canary trills and the duration of various syllables. The code contains an option to adding more LSTM layers but, since it isn't needed, those are not used.
* A projection layer -  For each time bin, this layer projects the previous layer's output on the set of possible syllables. 

