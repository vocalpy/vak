(about)=

# About `vak`

The `vak` library has two main goals:

1. make it easier for researchers studying animal vocalizations to
   apply neural network algorithms to their data
2. provide a common framework for benchmarking neural
   network algorithms on tasks related to animal vocalizations

Neural network algorithms in `vak` help answer questions about **vocal learning**,
"the ability to modify acoustic and syntactic sounds,
acquire new sounds via imitation, and produce vocalizations" [^cite_wikipedia].
Research questions related to vocal learning cut across a wide range of fields
including neuroscience, phsyiology, molecular biology, genomics, ecology, and evolution [^cite_wir2019].

The library was developed by
[David Nicholson](https://nicholdav.info/)
and
[Yarden Cohen](https://yardencsgithub.github.io/)
for experiments assessing performance of
[TweetyNet](https://github.com/yardencsGitHub/tweetynet),
a neural network that automates annotation of birdsong,
by segmenting spectograms into the units of song, called syllables.

[^cite_wikipedia]: <https://en.wikipedia.org/wiki/Vocal_learning>

[^cite_wir2019]: Wirthlin M, Chang EF, Knörnschild M, Krubitzer LA, Mello CV, Miller CT,
    Pfenning AR, Vernes SC, Tchernichovski O, Yartsev MM.
    A modular approach to vocal learning: disentangling the diversity of
    a complex behavioral trait. Neuron. 2019 Oct 9;104(1):87-99.
    <https://www.sciencedirect.com/science/article/pii/S0896627319308396>
