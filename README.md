<br>
<div align="center">
<img src="https://github.com/vocalpy/vak/blob/main/doc/images/logo/vak-logo-primary.png?raw=True" width="400">
</div>

<hr>

## a neural network toolbox for animal vocalizations and bioacoustics

[![DOI](https://zenodo.org/badge/173566541.svg)](https://zenodo.org/badge/latestdoi/173566541)
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-12-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->
[![PyPI version](https://badge.fury.io/py/vak.svg)](https://badge.fury.io/py/vak)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Build Status](https://github.com/vocalpy/vak/actions/workflows/ci-linux.yml/badge.svg)](https://github.com/vocalpy/vak/actions/workflows/ci-linux.yml/badge.svg)
[![Build Status](https://github.com/vocalpy/vak/actions/workflows/ci-macos.yml/badge.svg)](https://github.com/vocalpy/vak/actions/workflows/ci-macos.yml/badge.svg)
[![codecov](https://codecov.io/gh/vocalpy/vak/branch/main/graph/badge.svg?token=9Y4XXB2ELA)](https://codecov.io/gh/vocalpy/vak)

`vak` is a library for researchers studying animal vocalizations--such as 
birdsong, bat calls, and even human speech--although it may be useful 
to anyone working with bioacoustics data. 

The library has two main goals:  
1. make it easier for researchers studying animal vocalizations to 
apply neural network algorithms to their data
2. provide a common framework that will facilitate benchmarking neural 
network algorithms on tasks related to animal vocalizations

Currently the main use is automated **annotation** of vocalizations and other animal sounds.
By **annotation**, we mean something like the example of annotated birdsong shown below:  
<p align="center">
<img src="https://github.com/vocalpy/vak/blob/main/doc/images/annotation-example.png?raw=True" 
alt="spectrogram of birdsong with syllables annotated" width="400">
</p>

You give `vak` training data in the form of audio or spectrogram files with annotations, 
and then `vak` helps you train neural network models 
and use the trained models to predict annotations for new files.

We developed `vak` to benchmark a neural network model we call [`tweetynet`](https://github.com/yardencsGitHub/tweetynet).  
Please see the eLife article here: https://elifesciences.org/articles/63853  

### Installation
Short version:

#### with `pip`

```console
$ pip install vak
```

#### with `conda`
##### on Mac and Linux

```console
$ conda install vak -c conda-forge
```

##### on Windows
On Windows, you need to add an additional channel, `pytorch`.  
You can do this by repeating the `-c` option more than once.
```console
$ conda install vak -c conda-forge -c pytorch
$ #                                 ^ notice additional channel!
```

For more details, please see:
https://vak.readthedocs.io/en/latest/get_started/installation.html

We test `vak` on Ubuntu and MacOS. We have run on Windows and 
know of other users successfully running `vak` on that operating system, 
but installation on Windows may require some troubleshooting.
A good place to start is by searching the [issues](https://github.com/vocalpy/vak/issues).

### Usage
#### Tutorial
Currently the easiest way to work with `vak` is through the command line.
![terminal showing vak help command output](https://github.com/vocalpy/vak/blob/main/doc/images/terminalizer/vak-help.gif?raw=True)

You run it with configuration files, using one of a handful of commands.

For more details, please see the "autoannotate" tutorial here:  
https://vak.readthedocs.io/en/latest/get_started/autoannotate.html

#### How can I use my data with `vak`?

Please see the How-To Guides in the documentation here:
https://vak.readthedocs.io/en/latest/howto/index.html

### Support / Contributing
We handle support through the issue tracker on GitHub:  
https://github.com/vocalpy/vak/issues  
Please raise an issue there if you run into trouble.  
That would be a great place to start if you are interested in contributing, as well.

### Citation
If you use vak for a publication, please cite its DOI:  
[![DOI](https://zenodo.org/badge/173566541.svg)](https://zenodo.org/badge/latestdoi/173566541)

### License
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)  
is [here](./LICENSE).

### About
For more on the history of `vak` please see: https://vak.readthedocs.io/en/latest/reference/about.html

#### "Why this name, vak?"
It has only three letters, so it is quick to type,
and it wasn't taken on [pypi](https://pypi.org/) yet.
Also I guess it has [something to do with speech](https://en.wikipedia.org/wiki/V%C4%81c).
"vak" rhymes with "squawk" and "talk".

#### Does your library have any poems?
[Yes.](https://vak.readthedocs.io/en/latest/poems/index.html)

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/avanikop"><img src="https://avatars.githubusercontent.com/u/39831515?v=4?s=100" width="100px;" alt=""/><br /><sub><b>avanikop</b></sub></a><br /><a href="https://github.com/vocalpy/vak/issues?q=author%3Aavanikop" title="Bug reports">🐛</a></td>
    <td align="center"><a href="http://www.lukepoeppel.com"><img src="https://avatars.githubusercontent.com/u/20927930?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Luke Poeppel</b></sub></a><br /><a href="https://github.com/vocalpy/vak/commits?author=Luke-Poeppel" title="Documentation">📖</a></td>
    <td align="center"><a href="https://yardencsgithub.github.io/"><img src="https://avatars.githubusercontent.com/u/17324841?v=4?s=100" width="100px;" alt=""/><br /><sub><b>yardencsGitHub</b></sub></a><br /><a href="https://github.com/vocalpy/vak/commits?author=yardencsGitHub" title="Code">💻</a> <a href="#ideas-yardencsGitHub" title="Ideas, Planning, & Feedback">🤔</a> <a href="#talk-yardencsGitHub" title="Talks">📢</a> <a href="#userTesting-yardencsGitHub" title="User Testing">📓</a> <a href="#question-yardencsGitHub" title="Answering Questions">💬</a></td>
    <td align="center"><a href="https://nicholdav.info/"><img src="https://avatars.githubusercontent.com/u/11934090?v=4?s=100" width="100px;" alt=""/><br /><sub><b>David Nicholson</b></sub></a><br /><a href="https://github.com/vocalpy/vak/issues?q=author%3ANickleDave" title="Bug reports">🐛</a> <a href="https://github.com/vocalpy/vak/commits?author=NickleDave" title="Code">💻</a> <a href="#data-NickleDave" title="Data">🔣</a> <a href="https://github.com/vocalpy/vak/commits?author=NickleDave" title="Documentation">📖</a> <a href="#example-NickleDave" title="Examples">💡</a> <a href="#ideas-NickleDave" title="Ideas, Planning, & Feedback">🤔</a> <a href="#infra-NickleDave" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#maintenance-NickleDave" title="Maintenance">🚧</a> <a href="#mentoring-NickleDave" title="Mentoring">🧑‍🏫</a> <a href="#projectManagement-NickleDave" title="Project Management">📆</a> <a href="https://github.com/vocalpy/vak/pulls?q=is%3Apr+reviewed-by%3ANickleDave" title="Reviewed Pull Requests">👀</a> <a href="#question-NickleDave" title="Answering Questions">💬</a> <a href="#talk-NickleDave" title="Talks">📢</a> <a href="https://github.com/vocalpy/vak/commits?author=NickleDave" title="Tests">⚠️</a> <a href="#tutorial-NickleDave" title="Tutorials">✅</a></td>
    <td align="center"><a href="https://github.com/marichard123"><img src="https://avatars.githubusercontent.com/u/30010668?v=4?s=100" width="100px;" alt=""/><br /><sub><b>marichard123</b></sub></a><br /><a href="https://github.com/vocalpy/vak/commits?author=marichard123" title="Documentation">📖</a></td>
    <td align="center"><a href="https://www.utsouthwestern.edu/labs/roberts/"><img src="https://avatars.githubusercontent.com/u/46657075?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Therese Koch</b></sub></a><br /><a href="https://github.com/vocalpy/vak/commits?author=theresekoch" title="Documentation">📖</a> <a href="https://github.com/vocalpy/vak/issues?q=author%3Atheresekoch" title="Bug reports">🐛</a></td>
    <td align="center"><a href="https://github.com/alyndanoel"><img src="https://avatars.githubusercontent.com/u/48728732?v=4?s=100" width="100px;" alt=""/><br /><sub><b>alyndanoel</b></sub></a><br /><a href="#ideas-alyndanoel" title="Ideas, Planning, & Feedback">🤔</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/adamfishbein"><img src="https://avatars.githubusercontent.com/u/70346566?v=4?s=100" width="100px;" alt=""/><br /><sub><b>adamfishbein</b></sub></a><br /><a href="https://github.com/vocalpy/vak/commits?author=adamfishbein" title="Documentation">📖</a></td>
    <td align="center"><a href="https://github.com/vivinastase"><img src="https://avatars.githubusercontent.com/u/25927299?v=4?s=100" width="100px;" alt=""/><br /><sub><b>vivinastase</b></sub></a><br /><a href="https://github.com/vocalpy/vak/issues?q=author%3Avivinastase" title="Bug reports">🐛</a> <a href="#userTesting-vivinastase" title="User Testing">📓</a></td>
    <td align="center"><a href="https://github.com/kaiyaprovost"><img src="https://avatars.githubusercontent.com/u/17089935?v=4?s=100" width="100px;" alt=""/><br /><sub><b>kaiyaprovost</b></sub></a><br /><a href="https://github.com/vocalpy/vak/commits?author=kaiyaprovost" title="Code">💻</a> <a href="#ideas-kaiyaprovost" title="Ideas, Planning, & Feedback">🤔</a></td>
    <td align="center"><a href="https://github.com/ymk12345"><img src="https://avatars.githubusercontent.com/u/47306876?v=4?s=100" width="100px;" alt=""/><br /><sub><b>ymk12345</b></sub></a><br /><a href="https://github.com/vocalpy/vak/issues?q=author%3Aymk12345" title="Bug reports">🐛</a> <a href="https://github.com/vocalpy/vak/commits?author=ymk12345" title="Documentation">📖</a></td>
    <td align="center"><a href="http://www.xavierhinaut.com"><img src="https://avatars.githubusercontent.com/u/9768731?v=4?s=100" width="100px;" alt=""/><br /><sub><b>neuronalX</b></sub></a><br /><a href="https://github.com/vocalpy/vak/issues?q=author%3AneuronalX" title="Bug reports">🐛</a> <a href="https://github.com/vocalpy/vak/commits?author=neuronalX" title="Documentation">📖</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
