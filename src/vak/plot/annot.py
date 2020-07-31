"""functions for plotting annotations for vocalizations"""
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np


def plot_segments(onsets,
                  offsets,
                  y=0.5,
                  ax=None,
                  line_kwargs=None):
    """plot segments on an axis.

    Creates a collection of horizontal lines
    with the specified `onsets` and `offsets`
    all at height `y` and places them on the axes `ax`.

    Parameters
    ----------
    onsets : numpy.ndarray
        onset times of segments
    offsets : numpy.ndarray
        offset times of segments
    y : float, int
        height on y-axis at which segments should be plotted.
        Default is 0.5.
    ax : matplotlib.axes.Axes
        axes on which to plot segment. Default is None,
        in which case a new Axes instance is created
    line_kwargs : dict
        keyword arguments passed to the `LineCollection`
        that represents the segments. Default is None.
    """
    if line_kwargs is None:
        line_kwargs = {}

    if ax is None:
        fig, ax = plt.subplots
    segments = []
    for on, off in zip(onsets, offsets):
        segments.append(
            ((on, y), (off, y))
        )
    lc = LineCollection(segments, **line_kwargs)
    ax.add_collection(lc)


def plot_labels(labels,
                t,
                y=0.6,
                ax=None,
                text_kwargs=None):
    """plot labels on an axis.

    Parameters
    ----------
    labels : list, numpy.ndarray

    t : numpy.ndarray
        times at which to plot labels
    y : float, int
        height on y-axis at which segments should be plotted.
        Default is 0.5.
    ax : matplotlib.axes.Axes
        axes on which to plot segment. Default is None,
        in which case a new Axes instance is created
    text_kwargs : dict
        keyword arguments passed to the `Axes.text` method
        that plots the labels. Default is None.
    """
    if text_kwargs is None:
        text_kwargs = {}

    if ax is None:
        fig, ax = plt.subplots
    for label, t_lbl in zip(labels, t):
        ax.text(t_lbl, y, label, **text_kwargs)


def annotation(annot,
               t,
               tlim=None,
               y_segments=0.5,
               y_labels=0.6,
               line_kwargs=None,
               text_kwargs=None,
               ax=None):
    """plot segments with labels, from annotation

    Parameters
    ----------
    annot : crowsetta.Annotation
        annotation that has segments to be plotted
        (the `annot.seq.segments` attribute)
    t : numpy.ndarray
        vector of centers of time bins from spectrogram
    tlim : tuple, list
        limits of time axis (tmin, tmax) (i.e., x-axis).
        Default is None, in which case entire range of t will be plotted.
    y_segments : float
        height at which segments should be plotted.
        Default is 0.5 (assumes y-limits of 0 and 1).
    y_labels : float
        height at which labels should be plotted.
        Default is 0.6 (assumes y-limits of 0 and 1).
    line_kwargs : dict
        keyword arguments for `LineCollection`.
        Passed to the function `vak.plot.annot.segments` that plots segments
        as a `LineCollection` instance. Default is None.
    text_kwargs : dict
        keyword arguments for `matplotlib.axes.Axes.text`.
        Passed to the function `vak.plot.annot.labels` that plots labels
        using Axes.text method. Default is None.
    ax : matplotlib.axes.Axes
        axes on which to plot segments.
        Default is None, in which case
        a new figure with a single axes is created
    """
    if ax is None:
        fig, ax = plt.subplots()
        ax.set_ylim(0, 1)

    segment_centers = []
    for on, off in zip(annot.seq.onsets_s, annot.seq.offsets_s):
        segment_centers.append(
            np.mean([on, off])
        )
    plot_segments(onsets=annot.seq.onsets_s,
                  offsets=annot.seq.offsets_s,
                  y=y_segments,
                  ax=ax,
                  line_kwargs=line_kwargs)

    if tlim:
        ax.set_xlim(tlim)
        tmin, tmax = tlim

        labels = []
        segment_centers_tmp = []
        for label, segment_center in zip(annot.seq.labels, segment_centers):
            if tmin < segment_center < tmax:
                labels.append(label)
                segment_centers_tmp.append(segment_center)
        segment_centers = segment_centers_tmp
    else:
        labels = annot.seq.labels

    segment_centers = np.array(segment_centers)
    plot_labels(labels=labels,
                t=segment_centers,
                y=y_labels,
                ax=ax,
                text_kwargs=text_kwargs)
