[![DOI](https://zenodo.org/badge/173566541.svg)](https://zenodo.org/badge/latestdoi/173566541)
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-3-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->
[![PyPI version](https://badge.fury.io/py/vak.svg)](https://badge.fury.io/py/vak)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Build Status](https://github.com/NickleDave/vak/actions/workflows/ci.yml/badge.svg)](https://github.com/NickleDave/vak/actions/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/gh/NickleDave/vak/branch/main/graph/badge.svg?token=9Y4XXB2ELA)](https://codecov.io/gh/NickleDave/vak)
# vak
## a neural network toolbox for animal vocalizations and bioacoustics

`vak` is a library for researchers studying animal vocalizations--such as 
birdsong, bat calls, and even human speech--although it may be useful 
to anyone working with bioacoustics data. 
While there are many important reasons to study bioacoustics, the scope of `vak` 
is limited to questions related to **vocal learning**, 
"the ability to modify acoustic and syntactic sounds, acquire new sounds via imitation, and produce vocalizations"
[(Wikipedia)](https://en.wikipedia.org/wiki/Vocal_learning). 
Research questions related to vocal learning cut across a wide range of fields 
including neuroscience, phsyiology, molecular biology, genomics, ecology, and evolution 
[(Wirthlin et al. 2019)](https://www.sciencedirect.com/science/article/pii/S0896627319308396).

`vak` has two main goals:  
1. make it easier for researchers studying animal vocalizations to 
apply neural network algorithms to their data
2. provide a common framework that will facilitate benchmarking neural 
network algorithms on tasks related to animal vocalizations

Currently the main use is automated **annotation** of vocalizations and other animal sounds, 
using artificial neural networks.
By **annotation**, we mean something like the example of annotated birdsong shown below:  
<img src="./doc/images/annotation_example_for_tutorial.png" alt="spectrogram of birdsong with syllables annotated" width="400">

You give `vak` training data in the form of audio or spectrogram files with annotations, 
and then `vak` helps you train neural network models 
and use the trained models to predict annotations for new files.

We developed `vak` to benchmark a neural network model we call [`tweetynet`](https://github.com/yardencsGitHub/tweetynet).
See pre-print here: [https://www.biorxiv.org/content/10.1101/2020.08.28.272088v2.full.pdf](https://www.biorxiv.org/content/10.1101/2020.08.28.272088v2.full.pdf)  
We would love to help you use `vak` to benchmark your own model. 
If you have questions, please feel free to [raise an issue](https://github.com/NickleDave/vak/issues).

### Installation
Short version:
```console
$ pip install vak
```
For the long version detail, please see:
https://vak.readthedocs.io/en/latest/get_started/installation.html

We currently test `vak` on Ubuntu and MacOS. We have run on Windows and 
know of other users successfully running `vak` on that operating system, 
but installation on Windows will probably require some troubleshooting.
A good place to start is by searching the [issues](https://github.com/NickleDave/vak/issues).

### Usage
#### Training models to segment and label vocalizations
Currently the easiest way to work with `vak` is through the command line.
![terminal showing vak help command output](./doc/images/terminalizer/vak-help.gif)

You run it with `config.toml` files, using one of a handful of commands.

For more details, please see the "autoannotate" tutorial here:  
https://vak.readthedocs.io/en/latest/tutorial/autoannotate.html

#### Data and folder structures
To train models, you provide training data in the form of audio or 
spectrograms files, and annotations for those files.

#### Spectrograms and labels
The package can generate spectrograms from `.wav` files or `.cbin` audio files.
It can also accept spectrograms in the form of Matlab `.mat` or Numpy `.npz` files.
The locations of these files are specified in the `config.toml` file.

The annotations are parsed by a separate library, `crowsetta`, that 
aims to handle common formats like Praat `textgrid` files, and enable 
researchers to easily work with formats they may have developed in their 
own labs. For more information please see:  
https://crowsetta.readthedocs.io/en/latest/  
https://github.com/NickleDave/crowsetta  

#### Preparing training files
It is possible to train on any manually annotated data but there are some useful guidelines:
* __Use as many examples as possible__ - The results will just be better. Specifically, this code will not label correctly syllables it did not encounter while training and will most probably generalize to the nearest sample or ignore the syllable.
* __Use noise examples__ - This will make the code very good in ignoring noise.
* __Examples of syllables on noise are important__ - It is a good practice to start with clean recordings. The code will not perform miracles and is most likely to fail if the audio is too corrupt or masked by noise. Still, training with examples of syllables on the background of cage noises will be beneficial.

### Predicting annotations for audio
You can predict annotations for audio files by creating a `config.toml` file with a [PREDICT] section.  
For more details, please see the "autoannotate" tutorial here:
https://vak.readthedocs.io/en/latest/tutorial/autoannotate.html

### Support / Contributing
Currently we are handling support through the issue tracker on GitHub:  
https://github.com/NickleDave/vak/issues  
Please raise an issue there if you run into trouble.  
That would be a great place to start if you are interested in contributing, as well.

### Citation
If you use vak for a publication, please cite its DOI:  
[![DOI](https://zenodo.org/badge/173566541.svg)](https://zenodo.org/badge/latestdoi/173566541)

### License
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)  
is [here](./LICENSE).

### Misc
#### "Why this name, vak?"
It has only three letters, so it is quick to type,
and it wasn't taken on [pypi](https://pypi.org/) yet.
Also I guess it has [something to do with speech](https://en.wikipedia.org/wiki/V%C4%81c).
"vak" rhymes with "squawk" and "talk".

#### Does your library have any poems?
[Yes.](./doc/poem.md)

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/avanikop"><img src="https://avatars.githubusercontent.com/u/39831515?v=4?s=100" width="100px;" alt=""/><br /><sub><b>avanikop</b></sub></a><br /><a href="https://github.com/NickleDave/vak/issues?q=author%3Aavanikop" title="Bug reports">🐛</a></td>
    <td align="center"><a href="http://www.lukepoeppel.com"><img src="https://avatars.githubusercontent.com/u/20927930?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Luke Poeppel</b></sub></a><br /><a href="https://github.com/NickleDave/vak/commits?author=Luke-Poeppel" title="Documentation">📖</a></td>
    <td align="center"><a href="https://yardencsgithub.github.io/"><img src="https://avatars.githubusercontent.com/u/17324841?v=4?s=100" width="100px;" alt=""/><br /><sub><b>yardencsGitHub</b></sub></a><br /><a href="https://github.com/NickleDave/vak/commits?author=yardencsGitHub" title="Code">💻</a> <a href="#ideas-yardencsGitHub" title="Ideas, Planning, & Feedback">🤔</a> <a href="#talk-yardencsGitHub" title="Talks">📢</a> <a href="#userTesting-yardencsGitHub" title="User Testing">📓</a> <a href="#question-yardencsGitHub" title="Answering Questions">💬</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!