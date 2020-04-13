import torchvision.transforms

from . import transforms as vak_transforms


def get_defaults(mode,
                 spect_standardizer=None,
                 window_size=None,
                 return_padding_mask=False,
                 ):
    """get default transforms

    Parameters
    ----------
    mode : str
        one of {'train', 'predict'}. Determines set of transforms.
    spect_standardizer : vak.transforms.StandardizeSpect
        instance that has already been fit to dataset, using fit_df method.
        Default is None, in which case no standardization transform is applied.

    Returns
    -------
    transform, target_transform : callable
        one or more vak transforms to be applied to inputs x and, during training, the target y.
        If more than one transform, they are combined into an instance of torchvision.transforms.Compose.
        Note that when mode is 'predict', the target transform is None.
    """
    # regardless of mode, transform always starts with StandardizeSpect, if used
    if spect_standardizer is not None:
        if isinstance(spect_standardizer, vak_transforms.StandardizeSpect):
            transform = [spect_standardizer]
        else:
            raise TypeError(
                f'invalid type for spect_standardizer: {type(spect_standardizer)}. '
                'Should be an instance of vak.transforms.StandardizeSpect'
            )
    else:
        transform = []

    if mode == 'train':
        transform.extend(
            [
                vak_transforms.ToFloatTensor(),
                vak_transforms.AddChannel(),
            ]
        )

        target_transform = vak_transforms.ToLongTensor()
    elif mode == 'predict':
        transform.extend(
            [
                vak_transforms.PadToWindow(window_size, return_padding_mask),
                vak_transforms.ReshapeToWindow(window_size),
                vak_transforms.ToFloatTensor(),
                vak_transforms.AddChannel(channel_dim=1),  # add channel at first dimension because windows become batch
            ]
        )
        target_transform = None
    else:
        raise ValueError(
            f'invalid mode: {mode}'
        )

    transform = torchvision.transforms.Compose(transform)
    return transform, target_transform
