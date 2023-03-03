<br>
<div align="center">
<img src="https://github.com/vocalpy/vak/blob/main/doc/images/logo/vak-logo-primary.png?raw=True" width="400">
</div>

<hr>

## A neural network framework for animal acoustic communication and bioacoustics

[![DOI](https://zenodo.org/badge/173566541.svg)](https://zenodo.org/badge/latestdoi/173566541)
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-19-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->
[![PyPI version](https://badge.fury.io/py/vak.svg)](https://badge.fury.io/py/vak)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Build Status](https://github.com/vocalpy/vak/actions/workflows/ci-linux.yml/badge.svg)](https://github.com/vocalpy/vak/actions/workflows/ci-linux.yml/badge.svg)
[![codecov](https://codecov.io/gh/vocalpy/vak/branch/main/graph/badge.svg?token=9Y4XXB2ELA)](https://codecov.io/gh/vocalpy/vak)

`vak` is a Python framework for neural network models, 
designed for researchers studying animal acoustic communication and bioacoustics.
Many people will be familiar with work in this area on 
animal vocalizations such as birdsong, bat calls, and even human speech.
Neural network models have provided a powerful new tool for researchers in this area, 
as in many other fields.

The library has two main goals:  
1. Make it easier for researchers studying animal vocalizations to 
apply neural network algorithms to their data
2. Provide a common framework that will facilitate benchmarking neural 
network algorithms on tasks related to animal vocalizations

Currently, the main use is an automatic *annotation* of vocalizations and other animal sounds. 
By *annotation*, we mean something like the example of annotated birdsong shown below:
<p align="center">
<img src="https://github.com/vocalpy/vak/blob/main/doc/images/annotation-example.png?raw=True" 
alt="spectrogram of birdsong with syllables annotated" width="400">
</p>

You give `vak` training data in the form of audio or spectrogram files with annotations, 
and then `vak` helps you train neural network models 
and use the trained models to predict annotations for new files.

We developed `vak` to benchmark a neural network model we call [`tweetynet`](https://github.com/yardencsGitHub/tweetynet).  
Please see the eLife article here: https://elifesciences.org/articles/63853  

For more background on animal acoustic communication and deep learning, 
and how these intersect with related fields like 
computational ethology and neuroscience,
please see the ["About"](#About) section below.

### Installation
Short version:

#### with `pip`

```console
$ pip install vak
```

#### with `conda`
```console
$ conda install vak -c pytorch -c conda-forge
$ #                  ^ notice additional channel!
```

Notice that for `conda` you specify two channels, 
and that the `pytorch` channel should come first, 
so it takes priority when installing the dependencies `pytorch` and `torchvision`.

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
faq.html#faq
### Support / Contributing

For help, please begin by checking out the Frequently Asked Questions:  
https://vak.readthedocs.io/en/latest/faq.html.

To ask a question about vak, discuss its development, 
or share how you are using it, 
please start a new "Q&A" topic on the VocalPy forum 
with the vak tag:  
<https://forum.vocalpy.org/>

To report a bug, or to request a feature, 
please use the issue tracker on GitHub:  
<https://github.com/vocalpy/vak/issues>

For a guide on how you can contribute to `vak`, please see:
https://vak.readthedocs.io/en/latest/development/index.html

### Citation
If you use vak for a publication, please cite its DOI:  
[![DOI](https://zenodo.org/badge/173566541.svg)](https://zenodo.org/badge/latestdoi/173566541)

### License
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)  
is [here](./LICENSE).

### About

Are humans unique among animals? 
We speak languages, but is speech somehow like other animal behaviors, such as birdsong? 
Questions like these are answered by studying how animals communicate with sound. 
This research requires cutting edge computational methods and big team science across a wide range of disciplines, 
including ecology, ethology, bioacoustics, psychology, neuroscience, linguistics, and genomics [^1][^2][^3]. 
As in many other domains, this research is being revolutionized by deep learning algorithms [^1][^2][^3]. 
Deep neural network models enable answering questions that were previously impossible to address, 
in part because these models automate analysis of very large datasets. 
Within the study of animal acoustic communication, multiple models have been proposed for similar tasks, 
often implemented as research code with different libraries, such as Keras and Pytorch. 
This situation has created a real need for a framework that allows researchers to easily benchmark models 
and apply trained models to their own data. To address this need, we developed vak.
We originally developed vak to benchmark a neural network model, TweetyNet [^4][^5], 
that automates annotation of birdsong by segmenting spectrograms. 
TweetyNet and vak have been used in both neuroscience [^6][^7][^8] and bioacoustics [^9]. 
For additional background and papers that have used `vak`, 
please see: https://vak.readthedocs.io/en/latest/reference/about.html

[^1]: https://www.frontiersin.org/articles/10.3389/fnbeh.2021.811737/full 
[^2]: https://peerj.com/articles/13152/ 
[^3]: https://www.jneurosci.org/content/42/45/8514 
[^4]: https://elifesciences.org/articles/63853 
[^5]: https://github.com/yardencsGitHub/tweetynet
[^6]: https://www.nature.com/articles/s41586-020-2397-3 
[^7]: https://elifesciences.org/articles/67855
[^8]: https://elifesciences.org/articles/75691
[^9]: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0278522

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
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/avanikop"><img src="https://avatars.githubusercontent.com/u/39831515?v=4?s=100" width="100px;" alt="avanikop"/><br /><sub><b>avanikop</b></sub></a><br /><a href="https://github.com/vocalpy/vak/issues?q=author%3Aavanikop" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://www.lukepoeppel.com"><img src="https://avatars.githubusercontent.com/u/20927930?v=4?s=100" width="100px;" alt="Luke Poeppel"/><br /><sub><b>Luke Poeppel</b></sub></a><br /><a href="https://github.com/vocalpy/vak/commits?author=Luke-Poeppel" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://yardencsgithub.github.io/"><img src="https://avatars.githubusercontent.com/u/17324841?v=4?s=100" width="100px;" alt="yardencsGitHub"/><br /><sub><b>yardencsGitHub</b></sub></a><br /><a href="https://github.com/vocalpy/vak/commits?author=yardencsGitHub" title="Code">💻</a> <a href="#ideas-yardencsGitHub" title="Ideas, Planning, & Feedback">🤔</a> <a href="#talk-yardencsGitHub" title="Talks">📢</a> <a href="#userTesting-yardencsGitHub" title="User Testing">📓</a> <a href="#question-yardencsGitHub" title="Answering Questions">💬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://nicholdav.info/"><img src="https://avatars.githubusercontent.com/u/11934090?v=4?s=100" width="100px;" alt="David Nicholson"/><br /><sub><b>David Nicholson</b></sub></a><br /><a href="https://github.com/vocalpy/vak/issues?q=author%3ANickleDave" title="Bug reports">🐛</a> <a href="https://github.com/vocalpy/vak/commits?author=NickleDave" title="Code">💻</a> <a href="#data-NickleDave" title="Data">🔣</a> <a href="https://github.com/vocalpy/vak/commits?author=NickleDave" title="Documentation">📖</a> <a href="#example-NickleDave" title="Examples">💡</a> <a href="#ideas-NickleDave" title="Ideas, Planning, & Feedback">🤔</a> <a href="#infra-NickleDave" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#maintenance-NickleDave" title="Maintenance">🚧</a> <a href="#mentoring-NickleDave" title="Mentoring">🧑‍🏫</a> <a href="#projectManagement-NickleDave" title="Project Management">📆</a> <a href="https://github.com/vocalpy/vak/pulls?q=is%3Apr+reviewed-by%3ANickleDave" title="Reviewed Pull Requests">👀</a> <a href="#question-NickleDave" title="Answering Questions">💬</a> <a href="#talk-NickleDave" title="Talks">📢</a> <a href="https://github.com/vocalpy/vak/commits?author=NickleDave" title="Tests">⚠️</a> <a href="#tutorial-NickleDave" title="Tutorials">✅</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/marichard123"><img src="https://avatars.githubusercontent.com/u/30010668?v=4?s=100" width="100px;" alt="marichard123"/><br /><sub><b>marichard123</b></sub></a><br /><a href="https://github.com/vocalpy/vak/commits?author=marichard123" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://www.utsouthwestern.edu/labs/roberts/"><img src="https://avatars.githubusercontent.com/u/46657075?v=4?s=100" width="100px;" alt="Therese Koch"/><br /><sub><b>Therese Koch</b></sub></a><br /><a href="https://github.com/vocalpy/vak/commits?author=theresekoch" title="Documentation">📖</a> <a href="https://github.com/vocalpy/vak/issues?q=author%3Atheresekoch" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/alyndanoel"><img src="https://avatars.githubusercontent.com/u/48728732?v=4?s=100" width="100px;" alt="alyndanoel"/><br /><sub><b>alyndanoel</b></sub></a><br /><a href="#ideas-alyndanoel" title="Ideas, Planning, & Feedback">🤔</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/adamfishbein"><img src="https://avatars.githubusercontent.com/u/70346566?v=4?s=100" width="100px;" alt="adamfishbein"/><br /><sub><b>adamfishbein</b></sub></a><br /><a href="https://github.com/vocalpy/vak/commits?author=adamfishbein" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/vivinastase"><img src="https://avatars.githubusercontent.com/u/25927299?v=4?s=100" width="100px;" alt="vivinastase"/><br /><sub><b>vivinastase</b></sub></a><br /><a href="https://github.com/vocalpy/vak/issues?q=author%3Avivinastase" title="Bug reports">🐛</a> <a href="#userTesting-vivinastase" title="User Testing">📓</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/kaiyaprovost"><img src="https://avatars.githubusercontent.com/u/17089935?v=4?s=100" width="100px;" alt="kaiyaprovost"/><br /><sub><b>kaiyaprovost</b></sub></a><br /><a href="https://github.com/vocalpy/vak/commits?author=kaiyaprovost" title="Code">💻</a> <a href="#ideas-kaiyaprovost" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ymk12345"><img src="https://avatars.githubusercontent.com/u/47306876?v=4?s=100" width="100px;" alt="ymk12345"/><br /><sub><b>ymk12345</b></sub></a><br /><a href="https://github.com/vocalpy/vak/issues?q=author%3Aymk12345" title="Bug reports">🐛</a> <a href="https://github.com/vocalpy/vak/commits?author=ymk12345" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://www.xavierhinaut.com"><img src="https://avatars.githubusercontent.com/u/9768731?v=4?s=100" width="100px;" alt="neuronalX"/><br /><sub><b>neuronalX</b></sub></a><br /><a href="https://github.com/vocalpy/vak/issues?q=author%3AneuronalX" title="Bug reports">🐛</a> <a href="https://github.com/vocalpy/vak/commits?author=neuronalX" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/akn0717"><img src="https://avatars.githubusercontent.com/u/59268707?v=4?s=100" width="100px;" alt="Khoa"/><br /><sub><b>Khoa</b></sub></a><br /><a href="https://github.com/vocalpy/vak/commits?author=akn0717" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/sthaar"><img src="https://avatars.githubusercontent.com/u/24313958?v=4?s=100" width="100px;" alt="sthaar"/><br /><sub><b>sthaar</b></sub></a><br /><a href="https://github.com/vocalpy/vak/commits?author=sthaar" title="Documentation">📖</a> <a href="https://github.com/vocalpy/vak/issues?q=author%3Asthaar" title="Bug reports">🐛</a> <a href="#ideas-sthaar" title="Ideas, Planning, & Feedback">🤔</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/yangzheng-121"><img src="https://avatars.githubusercontent.com/u/104445992?v=4?s=100" width="100px;" alt="yangzheng-121"/><br /><sub><b>yangzheng-121</b></sub></a><br /><a href="https://github.com/vocalpy/vak/issues?q=author%3Ayangzheng-121" title="Bug reports">🐛</a> <a href="#ideas-yangzheng-121" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/lmpascual"><img src="https://avatars.githubusercontent.com/u/62260534?v=4?s=100" width="100px;" alt="lmpascual"/><br /><sub><b>lmpascual</b></sub></a><br /><a href="https://github.com/vocalpy/vak/commits?author=lmpascual" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ItamarFruchter"><img src="https://avatars.githubusercontent.com/u/19908942?v=4?s=100" width="100px;" alt="ItamarFruchter"/><br /><sub><b>ItamarFruchter</b></sub></a><br /><a href="https://github.com/vocalpy/vak/commits?author=ItamarFruchter" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/kalleknast"><img src="https://avatars.githubusercontent.com/u/12412777?v=4?s=100" width="100px;" alt="Hjalmar K. Turesson"/><br /><sub><b>Hjalmar K. Turesson</b></sub></a><br /><a href="https://github.com/vocalpy/vak/issues?q=author%3Akalleknast" title="Bug reports">🐛</a> <a href="#ideas-kalleknast" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/nhoglen"><img src="https://avatars.githubusercontent.com/u/13972140?v=4?s=100" width="100px;" alt="nhoglen"/><br /><sub><b>nhoglen</b></sub></a><br /><a href="https://github.com/vocalpy/vak/issues?q=author%3Anhoglen" title="Bug reports">🐛</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
