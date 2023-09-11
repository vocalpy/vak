(about)=

# About `vak`

## Background

Are humans unique among animals? 
We speak languages, but is speech somehow like other animal behaviors, such as birdsong? 
Questions like these are answered by studying how animals communicate with sound. 
This research requires cutting edge computational methods and big team science across a wide range of disciplines, 
including ecology, ethology, bioacoustics, psychology, neuroscience, linguistics, and 
genomics [^cite_SainburgGentner2020][^cite_Stowell2022][^cite_Cohenetal2022a]. 
As in many other domains, this research is being revolutionized by deep learning algorithms 
[^cite_SainburgGentner2020][^cite_Stowell2022][^cite_Cohenetal2022a]. 
Deep neural network models enable answering questions that were previously impossible to address, 
in part because these models automate analysis of very large datasets. 

## Goals

Within the study of animal acoustic communication, multiple models have been proposed for similar tasks, 
often implemented as research code with different libraries, such as Keras and Pytorch. 
This situation has created a real need for a framework that allows researchers to easily benchmark models 
and apply trained models to their own data. To address this need, we developed vak.

The vak library has two main goals:

1. make it easier for researchers studying animal vocalizations to
   apply neural network algorithms to their data
2. provide a common framework for benchmarking neural
   network algorithms on tasks related to animal vocalizations

Although models in the vak library 
can be used more generally for bioacoustics [^cite_Stowell2022], 
our focus is on animal acoustic communication [^cite_SainburgGentner2020]. 
More colloquially, we can call this "vocal behavior", 
a term that encompasses related researcher areas:  
not only *communication* [^cite_Beecher2020] 
but also *culture* [^cite_YoungbloodLahti2018], 
and *vocal learning* [^cite_wikipedia] [^cite_wir2019].
Models in the `vak` library 
include deep learning algorithms developed for bioacoustics , 
but are designed specifically for computational studies of vocal behavior .

## Publications using vak

We originally developed vak to benchmark a neural network model, TweetyNet [^cite_Cohenetal2022b], 
that automates annotation of birdsong by segmenting spectrograms. 
TweetyNet and vak have been used in both neuroscience 
[^cite_Cohenetal2020][^cite_Goffinetetal2021][^cite_McGregoretal2022][^cite_Koparkaretal2023] 
and bioacoustics [^cite_Provostetal2022][^cite_Yangetal2023]. 

## References

[^cite_SainburgGentner2020]: Sainburg, Tim, and Timothy Q. Gentner. 
   “Toward a Computational Neuroethology of Vocal Communication: 
   From Bioacoustics to Neurophysiology, Emerging Tools and Future Directions.” 
   Frontiers in Behavioral Neuroscience 15 (December 20, 2021): 811737. https://doi.org/10.3389/fnbeh.2021.811737.
   <https://www.frontiersin.org/articles/10.3389/fnbeh.2021.811737/full>

[^cite_Stowell2022]: Stowell, Dan. 
   “Computational Bioacoustics with Deep Learning: A Review and Roadmap,” 2022, 46.
   <https://peerj.com/articles/13152/>

[^cite_Cohenetal2022a]: Cohen, Yarden, et al. 
   "Recent Advances at the Interface of Neuroscience and Artificial Neural Networks." 
   Journal of Neuroscience 42.45 (2022): 8514-8523.
   <https://www.jneurosci.org/content/42/45/8514>

[^cite_Beecher2020]: Beecher, Michael D. 
   “Animal Communication.” 
   In Oxford Research Encyclopedia of Psychology. 
   Oxford University Press, 2020. <https://doi.org/10.1093/acrefore/9780190236557.013.646>.  
   <https://oxfordre.com/psychology/view/10.1093/acrefore/9780190236557.001.0001/acrefore-9780190236557-e-646>.

[^cite_YoungbloodLahti2018]: Youngblood, Mason, and David Lahti. 
   “A Bibliometric Analysis of the Interdisciplinary Field of Cultural Evolution.” 
   Palgrave Communications 4, no. 1 (December 2018): 120. <https://doi.org/10.1057/s41599-018-0175-8>.
   <https://www.nature.com/articles/s41599-018-0175-8>.

[^cite_wikipedia]: <https://en.wikipedia.org/wiki/Vocal_learning>

[^cite_wir2019]: Wirthlin M, Chang EF, Knörnschild M, Krubitzer LA, Mello CV, Miller CT,
    Pfenning AR, Vernes SC, Tchernichovski O, Yartsev MM.
    "A modular approach to vocal learning: disentangling the diversity of
    a complex behavioral trait." Neuron. 2019 Oct 9;104(1):87-99.
    <https://www.sciencedirect.com/science/article/pii/S0896627319308396>

[^cite_Cohenetal2022b]: Cohen, Yarden, David Aaron Nicholson, Alexa Sanchioni, Emily K. Mallaber, Viktoriya Skidanova, 
   and Timothy J. Gardner. 
   "Automated annotation of birdsong with a neural network that segments spectrograms." Elife 11 (2022): e63853.
   Paper: <https://doi.org/10.7554/eLife.63853>. Code: <https://github.com/yardencsGitHub/tweetynet>.

[^cite_Cohenetal2020]: Cohen, Yarden, Jun Shen, Dawit Semu, Daniel P. Leman, William A. Liberti III, L. Nathan Perkins, 
   Derek C. Liberti, Darrell N. Kotton, and Timothy J. Gardner. 
   "Hidden neural states underlie canary song syntax." Nature 582, no. 7813 (2020): 539-544.
   <https://www.nature.com/articles/s41586-020-2397-3>

[^cite_Goffinetetal2021]: Goffinet, Jack, Samuel Brudner, Richard Mooney, and John Pearson. 
   "Low-dimensional learned feature spaces quantify individual and group differences in vocal repertoires." 
   Elife 10 (2021): e67855.
   <https://doi.org/10.7554/eLife.67855>

[^cite_McGregoretal2022]: McGregor, James N., Abigail L. Grassler, Paul I. Jaffe, Amanda Louise Jacob, 
   Michael S. Brainard, and Samuel J. Sober. 
   "Shared mechanisms of auditory and non-auditory vocal learning in the songbird brain." Elife 11 (2022): e75691.
   <https://doi.org/10.7554/eLife.75691>

[^cite_Provostetal2022]: Provost, Kaiya L., Jiaying Yang, and Bryan C. Carstens. 
   "The impacts of fine-tuning, phylogenetic distance, and sample size on big-data bioacoustics." 
   Plos one 17, no. 12 (2022): e0278522.
   Paper: <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0278522>
   Code: <https://github.com/kaiyaprovost/bioacoustics>

[^cite_Yangetal2023]: Yang, Jiaying, Bryan C. Carstens, and Kaiya L. Provost.
   "Machine learning reveals relationships between song, climate, and migration in coastal Zonotrichia leucophrys." 
   bioRxiv (2023): 2023-03.

[^cite_Koparkaretal2023]: Koparkar, A., Warren, T. L., Charlesworth, J. D., Shin, S., Brainard, M. S., & Veit, L. (2023).
   Lesions in a songbird vocal circuit increase variability in song syntax. 
   bioRxiv, 2023-07.
