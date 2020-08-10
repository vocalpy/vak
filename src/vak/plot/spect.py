"""functions for plotting spectrograms"""
import matplotlib.pyplot as plt

from .annot import annotation


def spect(s,
          t,
          f,
          tlim=None,
          flim=None,
          ax=None,
          imshow_kwargs=None):
    """plot spectrogram

    Parameters
    ----------
    s : numpy.ndarray
        spectrogram, a matrix
    t : numpy.ndarray
        vector of centers of time bins from spectrogram
    f : numpy.ndarray
        vector of centers of frequency bins from spectrogram
    tlim : tuple, list
        limits of time axis (min, max) (i.e., x-axis).
        Default is None, in which case entire range of t will be plotted.
    flim : tuple, list
        limits of frequency axis (min, max) (i.e., x-axis).
        Default is None, in which case entire range of f will be plotted.
        limits of time axis (min, max) (i.e., x-axis).
        Default is None, in which case entire range of t will be plotted.
    flim : tuple, list
        limits of frequency axis (min, max) (i.e., x-axis).
        Default is None, in which case entire range of f will be plotted.
    ax : matplotlib.axes.Axes
        axes on which to plot spectrgraom
    imshow_kwargs : dict
        keyword arguments passed to matplotlib.axes.Axes.imshow method
        used to plot spectrogram. Default is None.
    """
    if imshow_kwargs is None:
        imshow_kwargs = {}

    if ax is None:
        fig, ax = plt.subplots()

    extent = [t.min(), t.max(), f.min(), f.max()]

    ax.imshow(s,
              aspect='auto',
              origin='lower',
              extent=extent,
              **imshow_kwargs)

    if tlim is not None:
        ax.set_xlim(tlim)

    if flim is not None:
        ax.set_ylim(flim)


def spect_annot(s,
                t,
                f,
                annot,
                tlim=None,
                flim=None,
                figsize=(20, 7.5),
                imshow_kwargs=None,
                line_kwargs=None,
                text_kwargs=None,
                ):
    """plot a spectrogram with annotated segments below it.
    Convenience function that calls `vak.plot.spect` and `vak.plot.annotation`

    Parameters
    ----------
    s : numpy.ndarray
        spectrogram, a matrix
    t : numpy.ndarray
        vector of centers of time bins from spectrogram
    f : numpy.ndarray
        vector of centers of frequency bins from spectrogram
    annot : crowsetta.Annotation
        annotation that has segments to be plotted
        (the `annot.seq.segments` attribute)    figsize
    tlim : tuple, list
        limits of time axis (min, max) (i.e., x-axis).
        Default is None, in which case entire range of t will be plotted.
    flim : tuple, list
        limits of frequency axis (min, max) (i.e., x-axis).
        Default is None, in which case entire range of f will be plotted.
    figsize : tuple, list
        figure size (width, height) in inches. Default is (20, 7.5).
    imshow_kwargs : dict
        keyword arguments that will get passed to `matplotlib.axes.Axes.imshow`
        when using that method to plot spectrogram.
    line_kwargs : dict
        keyword arguments for `LineCollection`.
        Passed to the function `vak.plot.annot.segments` that plots segments
        as a `LineCollection` instance. Default is None.
    text_kwargs : dict
        keyword arguments for `matplotlib.axes.Axes.text`.
        Passed to the function `vak.plot.annot.labels` that plots labels
        using Axes.text method.
        Defaults are defined as `vak.plot.annot.DEFAULT_TEXT_KWARGS`.

    Returns
    -------
    fig, spect_ax, annot_ax :
        matplotlib Figure and Axes instances.
        The spect_ax is the axes containing the spectrogram
        and the annot_ax is the axes containing the
        annotated segments.
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3)
    spect_ax = fig.add_subplot(gs[:2, :])
    annot_ax = fig.add_subplot(gs[2, :])

    spect(s, t, f, tlim, flim, ax=spect_ax, imshow_kwargs=imshow_kwargs)

    annotation(annot, t, tlim, ax=annot_ax, line_kwargs=line_kwargs, text_kwargs=text_kwargs)

    return fig, spect_ax, annot_ax
