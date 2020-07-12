"""default item transforms used with the different command-line interface commands

"item" transforms because they apply transforms to input parameters
and then return them in an "item" (dictionary)
that is turn returned by the __getitem__ method of a vak.VocalDataset.
Having the transform return a dictionary makes it possible to avoid
coupling the VocalDataset __getitem__ implementation to the transforms
needed for specific neural network models, e.g., whether the returned
output includes a mask to crop off padding that was added.
"""
import torchvision.transforms

from . import transforms as vak_transforms


class TrainItemTransform:
    """default transform used when training models"""
    def __init__(self,
                 spect_standardizer=None,):
        if spect_standardizer is not None:
            if isinstance(spect_standardizer, vak_transforms.StandardizeSpect):
                source_transform = [spect_standardizer]
            else:
                raise TypeError(
                    f'invalid type for spect_standardizer: {type(spect_standardizer)}. '
                    'Should be an instance of vak.transforms.StandardizeSpect'
                )
        else:
            source_transform = []

        source_transform.extend(
            [
                vak_transforms.ToFloatTensor(),
                vak_transforms.AddChannel(),
            ]
        )
        self.source_transform = torchvision.transforms.Compose(source_transform)
        self.annot_transform = vak_transforms.ToLongTensor()

    def __call__(self, source, annot, spect_path=None):
        source = self.source_transform(source)
        annot = self.annot_transform(annot)
        item = {
            'source': source,
            'annot': annot,
        }

        if spect_path is not None:
            item['spect_path'] = spect_path

        return item


class EvalItemTransform:
    """default transform used when evaluating models

    Returned item includes "source" spectrogram reshaped into a stack of windows,
    with padded added to make reshaping possible, and annotation also padded and
    reshaped.
    If return_padding_mask is True, item includes 'padding_mask' that
    can be used to crop off any predictions made on the padding.
    """
    def __init__(self,
                 window_size,
                 spect_standardizer=None,
                 padval=0.,
                 return_padding_mask=True,
                 channel_dim=1):
        if spect_standardizer is not None:
            if not isinstance(spect_standardizer, vak_transforms.StandardizeSpect):
                raise TypeError(
                    f'invalid type for spect_standardizer: {type(spect_standardizer)}. '
                    'Should be an instance of vak.transforms.StandardizeSpect'
                )
        self.spect_standardizer = spect_standardizer

        self.pad_to_window = vak_transforms.PadToWindow(window_size,
                                                        padval,
                                                        return_padding_mask=return_padding_mask)

        self.source_transform_after_pad = torchvision.transforms.Compose([
            vak_transforms.ViewAsWindowBatch(window_size),
            vak_transforms.ToFloatTensor(),
            # below, add channel at first dimension because windows become batch
            vak_transforms.AddChannel(channel_dim=channel_dim),
        ])

        self.annot_transform = vak_transforms.ToLongTensor()

    def __call__(self, source, annot, spect_path=None):
        if self.spect_standardizer:
            source = self.spect_standardizer(source)

        if self.pad_to_window.return_padding_mask:
            source, padding_mask = self.pad_to_window(source)
        else:
            source = self.pad_to_window(source)
            padding_mask = None
        source = self.source_transform_after_pad(source)

        annot = self.annot_transform(annot)

        item = {
            'source': source,
            'annot': annot,
        }

        if padding_mask is not None:
            item['padding_mask'] = padding_mask

        if spect_path is not None:
            item['spect_path'] = spect_path

        return item


class PredictItemTransform:
    """default transform used when using trained models to make predictions.

    Returned item includes "source" spectrogram reshaped into a stack of windows,
    with padded added to make reshaping possible.
    If return_padding_mask is True, item includes 'padding_mask' that
    can be used to crop off any predictions made on the padding.
    """
    def __init__(self,
                 window_size,
                 spect_standardizer=None,
                 padval=0.,
                 return_padding_mask=True,
                 channel_dim=1):
        if spect_standardizer is not None:
            if not isinstance(spect_standardizer, vak_transforms.StandardizeSpect):
                raise TypeError(
                    f'invalid type for spect_standardizer: {type(spect_standardizer)}. '
                    'Should be an instance of vak.transforms.StandardizeSpect'
                )
        self.spect_standardizer = spect_standardizer

        self.pad_to_window = vak_transforms.PadToWindow(window_size,
                                                        padval,
                                                        return_padding_mask=return_padding_mask)

        self.source_transform_after_pad = torchvision.transforms.Compose([
            vak_transforms.ViewAsWindowBatch(window_size),
            vak_transforms.ToFloatTensor(),
            # below, add channel at first dimension because windows become batch
            vak_transforms.AddChannel(channel_dim=channel_dim),
        ])

    def __call__(self, source, spect_path=None):
        if self.spect_standardizer:
            source = self.spect_standardizer(source)

        if self.pad_to_window.return_padding_mask:
            source, padding_mask = self.pad_to_window(source)
        else:
            source = self.pad_to_window(source)
            padding_mask = None

        source = self.source_transform_after_pad(source)

        item = {
            'source': source,
        }

        if padding_mask is not None:
            item['padding_mask'] = padding_mask

        if spect_path is not None:
            item['spect_path'] = spect_path

        return item


def get_defaults(mode,
                 spect_standardizer=None,
                 window_size=None,
                 padval=0.,
                 return_padding_mask=False,
                 ):
    """get default transforms

    Parameters
    ----------
    mode : str
        one of {'train', 'eval', 'predict'}. Determines set of transforms.
    spect_standardizer : vak.transforms.StandardizeSpect
        instance that has already been fit to dataset, using fit_df method.
        Default is None, in which case no standardization transform is applied.
    window_size : int
        width of window in number of elements. Argument to PadToWindow transform.
    padval : float
        value to pad with. Added to end of array, the "right side" if 2-dimensional.
        Argument to PadToWindow transform. Default is 0.
    return_padding_mask : bool
        if True, the dictionary returned by ItemTransform classes will include
        a boolean vector to use for cropping back down to size before padding.
        padding_mask has size equal to width of padded array, i.e. original size
        plus padding at the end, and has values of 1 where
        columns in padded are from the original array,
        and values of 0 where columns were added for padding.

    Returns
    -------
    transform, target_transform : callable
        one or more vak transforms to be applied to inputs x and, during training, the target y.
        If more than one transform, they are combined into an instance of torchvision.transforms.Compose.
        Note that when mode is 'predict', the target transform is None.
    """
    # regardless of mode, transform always starts with StandardizeSpect, if used
    if spect_standardizer is not None:
        if not isinstance(spect_standardizer, vak_transforms.StandardizeSpect):
            raise TypeError(
                f'invalid type for spect_standardizer: {type(spect_standardizer)}. '
                'Should be an instance of vak.transforms.StandardizeSpect'
            )

    if mode == 'train':
        if spect_standardizer is not None:
            transform = [spect_standardizer]
        else:
            transform = []


        transform.extend([
            vak_transforms.ToFloatTensor(),
            vak_transforms.AddChannel(),
        ])
        transform = torchvision.transforms.Compose(transform)

        target_transform = vak_transforms.ToLongTensor()
        return transform, target_transform

    elif mode == 'predict':
        item_transform = PredictItemTransform(
            spect_standardizer=spect_standardizer,
            window_size=window_size,
            padval=padval,
            return_padding_mask=return_padding_mask,
        )
        return item_transform

    elif mode == 'eval':
        item_transform = EvalItemTransform(
            spect_standardizer=spect_standardizer,
            window_size=window_size,
            padval=padval,
            return_padding_mask=return_padding_mask,
        )
        return item_transform

    else:
        raise ValueError(
            f'invalid mode: {mode}'
        )
