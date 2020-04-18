import torchvision.transforms

from . import defaults
from . import transforms as vak_transforms


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

    if mode == 'train' or mode == 'predict':
        if spect_standardizer is not None:
            transform = [spect_standardizer]
        else:
            transform = []

        if mode == 'train':
            transform.extend([
                vak_transforms.ToFloatTensor(),
                vak_transforms.AddChannel(),
            ])
            transform = torchvision.transforms.Compose(transform)

            target_transform = vak_transforms.ToLongTensor()
            return transform, target_transform

        elif mode == 'predict':
            transform.extend([
                spect_standardizer,
                vak_transforms.PadToWindow(window_size, return_padding_mask),
                vak_transforms.ViewAsWindowBatch(window_size),
                vak_transforms.ToFloatTensor(),
                vak_transforms.AddChannel(channel_dim=1),  # add channel at first dimension because windows become batch
            ])
            transform = torchvision.transforms.Compose(transform)

            target_transform = None
            return transform, target_transform

    elif mode == 'eval':
        item_transform = defaults.EvalItemTransform(
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
