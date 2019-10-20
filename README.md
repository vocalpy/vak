[![DOI](https://zenodo.org/badge/173566541.svg)](https://zenodo.org/badge/latestdoi/173566541)
[![PyPI version](https://badge.fury.io/py/vak.svg)](https://badge.fury.io/py/vak)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
# vak
## automated annotation of vocalizations for everybody
[![Build Status](https://travis-ci.com/NickleDave/vak.svg?branch=master)](https://travis-ci.com/NickleDave/vak)

## Usage
### Training models to segment and label vocalizations
Currently the easiest way to work with `vak` is through the command line.
You run it with `config.ini` files, using one of a handful of commands.
Here's the help text that prints when you run `$ vak -h` (`-h` for `help`):  
```
$ vak -h
usage: vak [-h] command configfile

vak command-line interface

positional arguments:
  command     Command to run, valid options are:
              ['prep', 'train', 'predict', 'finetune', 'learncurve']
              $ vak train ./configs/config_2018-12-17.ini
  configfile  name of config.ini file to use 
              $ vak train ./configs/config_2018-12-17.ini

optional arguments:
  -h, --help  show this help message and exit
```

As an example, you can run `vak` with a single `config.ini` file 
by using the  `train` command and passing the name of the config.ini file as an argument:  
```
(vak-env)$ vak prep ./configs/config_bird0.ini
(vak-env)$ vak train ./configs/config_bird0.ini
```  

You can then use `vak` to apply the trained model to other data with the `predict` command.
```
(vak-env)$ vak predict ./configs/config_bird0.ini
```  

For more details on how training works, see [experiments.md](doc/experiments.md), 
and for more details on the config.ini files, see [README_config.md](doc/README_config.md).

### Data and folder structures
To train models, you must supply training data in the form of audio files or 
spectrograms, and annotations for each spectrogram.

#### Spectrograms and labels
The package can generate spectrograms from `.wav` files or `.cbin` files.
It can also accept spectrograms in the form of Matlab `.mat` files.
The locations of these files are specified in the `config.ini` file as explained in 
[experiments.md](doc/experiments.md) and [README_config.md](doc/README_config.md).

## Preparing training files
It is possible to train on any manually annotated data but there are some useful guidelines:
* __Use as many examples as possible__ - The results will just be better. Specifically, this code will not label correctly syllables it did not encounter while training and will most probably generalize to the nearest sample or ignore the syllable.
* __Use noise examples__ - This will make the code very good in ignoring noise.
* __Examples of syllables on noise are important__ - It is a good practice to start with clean recordings. The code will not perform miracles and is most likely to fail if the audio is too corrupt or masked by noise. Still, training with examples of syllables on the background of cage noises will be beneficial.

### Results of running the code

__It is recommended to apply post processing when extracting the actual syllable tag and onset and offset timesfrom the estimates.__

## Predicting new labels

You can predict new labels by adding a [PREDICT] section to the `config.ini` file, and 
then running the command-line interface with the `--predict` flag, like so:  
`(vak-env)$ vak-cli --predict ./configs/config_bird0.ini`
An example of what a `config.ini` file with a [PREDICT] section is 
in the doc folder [here](./doc/template_predict.ini).

## Citation
If you use vak for a publication, please cite its DOI:
[![DOI](https://zenodo.org/badge/173566541.svg)](https://zenodo.org/badge/latestdoi/173566541)

## License
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)  
[BSD-3](./LICENSE)
