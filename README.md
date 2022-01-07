[![DOI](https://zenodo.org/badge/173566541.svg)](https://zenodo.org/badge/latestdoi/173566541)
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-5-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->
[![PyPI version](https://badge.fury.io/py/vak.svg)](https://badge.fury.io/py/vak)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Build Status](https://github.com/NickleDave/vak/actions/workflows/ci-linux.yml/badge.svg)](https://github.com/NickleDave/vak/actions/workflows/ci-linux.yml/badge.svg)
[![Build Status](https://github.com/NickleDave/vak/actions/workflows/ci-macos.yml/badge.svg)](https://github.com/NickleDave/vak/actions/workflows/ci-macos.yml/badge.svg)
[![codecov](https://codecov.io/gh/NickleDave/vak/branch/main/graph/badge.svg?token=9Y4XXB2ELA)](https://codecov.io/gh/NickleDave/vak)
# vak
## a neural network toolbox for animal vocalizations and bioacoustics

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
<img src="./doc/images/annotation_example.png" alt="spectrogram of birdsong with syllables annotated" width="400">

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
For the long version, please see:
https://vak.readthedocs.io/en/latest/get_started/installation.html

We currently test `vak` on Ubuntu and MacOS. We have run on Windows and 
know of other users successfully running `vak` on that operating system, 
but installation on Windows will probably require some troubleshooting.
A good place to start is by searching the [issues](https://github.com/NickleDave/vak/issues).

### Usage
#### Tutorial
Currently the easiest way to work with `vak` is through the command line.
![terminal showing vak help command output](./doc/images/terminalizer/vak-help.gif)

You run it with `config.toml` files, using one of a handful of commands.

For more details, please see the "autoannotate" tutorial here:  
https://vak.readthedocs.io/en/latest/tutorial/autoannotate.html

#### How can I use my data with `vak`?

Please see the How-To Guides in the documentation here:
https://vak.readthedocs.io/en/latest/howto/howto.html

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
For more on the history of `vak` please see: https://vak.readthedocs.io/en/latest/reference/about.html

#### "Why this name, vak?"
It has only three letters, so it is quick to type,
and it wasn't taken on [pypi](https://pypi.org/) yet.
Also I guess it has [something to do with speech](https://en.wikipedia.org/wiki/V%C4%81c).
"vak" rhymes with "squawk" and "talk".

#### Does your library have any poems?
[Yes.](https://vak.readthedocs.io/en/latest//poems/poems.html)

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
    <td align="center"><a href="https://nicholdav.info/"><img src="https://avatars.githubusercontent.com/u/11934090?v=4?s=100" width="100px;" alt=""/><br /><sub><b>David Nicholson</b></sub></a><br /><a href="https://github.com/NickleDave/vak/issues?q=author%3ANickleDave" title="Bug reports">🐛</a> <a href="https://github.com/NickleDave/vak/commits?author=NickleDave" title="Code">💻</a> <a href="#data-NickleDave" title="Data">🔣</a> <a href="https://github.com/NickleDave/vak/commits?author=NickleDave" title="Documentation">📖</a> <a href="#example-NickleDave" title="Examples">💡</a> <a href="#ideas-NickleDave" title="Ideas, Planning, & Feedback">🤔</a> <a href="#infra-NickleDave" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#maintenance-NickleDave" title="Maintenance">🚧</a> <a href="#mentoring-NickleDave" title="Mentoring">🧑‍🏫</a> <a href="#projectManagement-NickleDave" title="Project Management">📆</a> <a href="https://github.com/NickleDave/vak/pulls?q=is%3Apr+reviewed-by%3ANickleDave" title="Reviewed Pull Requests">👀</a> <a href="#question-NickleDave" title="Answering Questions">💬</a> <a href="#talk-NickleDave" title="Talks">📢</a> <a href="https://github.com/NickleDave/vak/commits?author=NickleDave" title="Tests">⚠️</a> <a href="#tutorial-NickleDave" title="Tutorials">✅</a></td>
    <td align="center"><a href="https://github.com/marichard123"><img src="https://avatars.githubusercontent.com/u/30010668?v=4?s=100" width="100px;" alt=""/><br /><sub><b>marichard123</b></sub></a><br /><a href="https://github.com/NickleDave/vak/commits?author=marichard123" title="Documentation">📖</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!