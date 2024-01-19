---
hide-toc: true
---

# `vak`

## A neural network framework for researchers studying acoustic communication

```{image} images/song_with_colored_segments.png
```

`vak` is a library that makes it easier
for researchers studying animal
vocalizations---such as birdsong, bat calls,
and even human speech---to work with
neural network algorithms.

To learn more about the goals and design of vak, 
please see this talk from the SciPy 2023 conference, 
and the associated Proceedings paper 
[here](https://conference.scipy.org/proceedings/scipy2023/pdfs/david_nicholson.pdf).

<p align="center">
<a href="https://www.youtube.com/watch?v=tpL0m5UwpZM" target="_blank">
 <img src="https://img.youtube.com/vi/tpL0m5UwpZM/mqdefault.jpg" alt="Thumbnail of SciPy 2023 talk on vak" width="400" border="10" />
</a>
</p>

Currently, the main use is automated **annotation** of animal vocalizations.
By **annotation**, we mean something like this example of annotated birdsong:

```{image} images/annotation_example_for_tutorial.png
```

Please see links below for information on how to get started and how to use `vak` to
apply neural network models to your data.

### {ref}`get-started-index`

If you are new to working with `vak`,
and you're looking for installation instructions and a tutorial,
{ref}`start here <get-started-index>`.

### {ref}`howto-index`

If there is something specific you're trying to do,
like use your own spectrogram files or annotation formats with `vak`,
please check in the {ref}`howto-index`.

### Getting Help

For help, please begin by checking out the {ref}`faq`.

To ask a question about vak, discuss its development, 
or share how you are using it, 
please start a new "Q&A" topic on the VocalPy forum 
with the vak tag:  
<https://forum.vocalpy.org/>

To report a bug, or to request a feature, 
please use the issue tracker on GitHub:  
<https://github.com/vocalpy/vak/issues>

### {ref}`reference-index`

If you need to look up information about the command-line interface, configuration files, etc.,
please consult the {ref}`reference-index`.

### {ref}`devindex`

To learn about development of `vak` and how you can contribute, please see {ref}`devindex`

### {ref}`about`

For more about the goals of `vak` and its development, please see {ref}`about`.

### {ref}`poems-index`

Not enough open-source research software libraries have poems. We do, here: {ref}`poems-index`.

```{toctree}
:hidden: true
:maxdepth: 1

get_started/index
howto/index
faq
reference/index
development/index
api/index
reference/about
poems/index
```
