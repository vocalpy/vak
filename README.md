# tf_syllable_segmentation_annotation
## Jupyter notebook for tensorflow based syllable segmentation and annotation
This code utilizes the 'tensorflow' library to create, train, and use a deep neural-network algorithm for parsing and tagging birdsong spectrograms. For further details see section `Model structure`.
## Data and folder structures
To use this code the data has to be arranged in a certain format and in a specific folder structure.
### Spectrograms and labels
Currently, all data files must be in Matlab format. Each file should contain 2 variables:
* s - A Nf x Nt real matrix of the spectrogram. Nt is the number of time steps. Nf is the number of frequency bins (the current code assumes 513 but this can be changed by updating the variable)
* labels - A 1 x Nt real vector that contains the manually annotated label for each time step.
### Data for training
