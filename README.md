# tf_syllable_segmentation_annotation
## Jupyter notebook for tensorflow based syllable segmentation and annotation
This code utilizes the 'tensorflow' library to create, train, and use a deep neural-network algorithm for parsing and tagging birdsong spectrograms. For further details see section `Model structure`.
## Data and folder structures
To use this code the data has to be arranged in a certain format and in a specific folder structure.
### Spectrograms and labels
Currently, all data files must be in Matlab format. Each training file should contain 2 variables:
* s - A Nf x Nt real matrix of the spectrogram. Nt is the number of time steps. Nf is the number of frequency bins (the current code assumes 513 but this can be changed by updating the variable __input_vec_size__). The values in this matrix should range from 0 to 0.8 with 0 indicating low energy in the time-spectral bin.
* labels - A 1 x Nt real vector that contains the manually annotated label for each time step.
Testing or unlabeled files need to contain only the variable `s`.
### Folders and lists
In the first notebook 'cell' contains variables for holding the names of two folders:
* data_directory - This folder contains all the training files.
* training_records_dir
