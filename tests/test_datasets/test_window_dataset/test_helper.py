import numpy as np
import pandas as pd
import pytest

import vak.datasets


@pytest.mark.parametrize(
    'config_type, model_name, audio_format, spect_format, annot_format, window_size, crop_dur',
    [
        ('learncurve', 'teenytweetynet', 'cbin', None, 'notmat', 22, 4.0),
        ('learncurve', 'teenytweetynet', 'cbin', None, 'notmat', 22, 6.0),
        ('learncurve', 'teenytweetynet', 'cbin', None, 'notmat', 44, 4.0),
        ('learncurve', 'teenytweetynet', 'cbin', None, 'notmat', 44, 6.0),
    ]
)
def test_crop_vectors_keep_classes(config_type, model_name, audio_format, spect_format, annot_format,
                                   window_size, crop_dur, specific_config):
    toml_path = specific_config(config_type,
                                model_name,
                                audio_format=audio_format,
                                spect_format=spect_format,
                                annot_format=annot_format)
    # ---- set-up (there's a lot so I'm marking it) ----
    cfg = vak.config.parse.from_toml_path(toml_path)
    cmd_cfg = getattr(cfg, config_type)  # "command config", i.e., cli command, [TRAIN] or [LEARNCURVE] section
    csv_path = getattr(cmd_cfg, 'dataset_path')

    df = pd.read_csv(csv_path)
    df_split = df[df.split == 'train']

    # stuff we need just to be able to instantiate window dataset
    labelmap = vak.labels.to_map(cfg.prep.labelset, map_unlabeled=True)

    timebin_dur = vak.io.dataframe.validate_and_get_timebin_dur(df)

    (source_ids_before,
     source_inds_before,
     window_inds_before,
     lbl_tb) = vak.datasets.window_dataset.helper._vectors_from_df(
        df_split,
        window_size=window_size,
        crop_to_dur=True if crop_dur else False,
        labelmap=labelmap,
    )

    # ---- actually get result we want to test
    (
        source_ids,
        source_inds,
        window_inds,
    ) = vak.datasets.window_dataset.helper.crop_vectors_keep_classes(
        lbl_tb,
        source_ids_before,
        source_inds_before,
        window_inds_before,
        crop_dur,
        timebin_dur,
        labelmap,
        window_size,
    )

    for vector_name, vector in zip(
        ('source_ids', 'source_inds', 'window_inds'),
        (source_ids, source_inds, window_inds)
    ):
        assert isinstance(vector, np.ndarray)
    assert source_ids.shape[-1] == source_inds.shape[-1]
    assert np.isclose(source_ids.shape[-1] * timebin_dur, crop_dur)

    # test that valid window indices is strictly less than or equal to source_ids
    window_inds_no_invalid = window_inds[window_inds != vak.datasets.WindowDataset.INVALID_WINDOW_VAL]
    assert window_inds_no_invalid.shape[-1] <= source_ids.shape[-1]

    # test we preserved unique classes
    assert np.array_equal(
        np.unique(lbl_tb[window_inds]),
        np.unique(lbl_tb)
    )


@pytest.mark.parametrize(
    'config_type, model_name, audio_format, spect_format, annot_format, window_size, crop_dur',
    [
        ('learncurve', 'teenytweetynet', 'cbin', None, 'notmat', 22, None),
        ('learncurve', 'teenytweetynet', 'cbin', None, 'notmat', 22, 4.0),
        ('learncurve', 'teenytweetynet', 'cbin', None, 'notmat', 44, None),
        ('learncurve', 'teenytweetynet', 'cbin', None, 'notmat', 44, 4.0),
    ]
)
def test__vectors_from_df(config_type, model_name, audio_format, spect_format, annot_format,
                          window_size, crop_dur, specific_config):
    """Test the helper function ``_vectors_from_df`` that
    generates the vectors representing windows,
    *without* removing the elements markes as invalid start indices
    from ``window_inds``.
    """
    toml_path = specific_config(config_type,
                                model_name,
                                audio_format=audio_format,
                                spect_format=spect_format,
                                annot_format=annot_format)
    cfg = vak.config.parse.from_toml_path(toml_path)

    # stuff we need just to be able to instantiate window dataset
    labelmap = vak.labels.to_map(cfg.prep.labelset, map_unlabeled=True)

    cmd_cfg = getattr(cfg, config_type)  # "command config", i.e., cli command, [TRAIN] or [LEARNCURVE] section
    csv_path = getattr(cmd_cfg, 'dataset_path')
    df = pd.read_csv(csv_path)
    df = df[df.split == 'train']

    (source_ids,
     source_inds,
     window_inds,
     lbl_tb) = vak.datasets.window_dataset.helper._vectors_from_df(
        df,
        window_size=window_size,
        crop_to_dur=True if crop_dur else False,
        labelmap=labelmap,
    )

    for vector_name, vector in zip(
        ('source_ids', 'source_inds', 'window_inds', 'lbl_tb'),
        (source_ids, source_inds, window_inds)
    ):
        assert isinstance(vector, np.ndarray)

    assert source_ids.shape == source_inds.shape == window_inds.shape

    n_source_files_in_split = len(df)
    window_inds_no_invalid = window_inds[window_inds != vak.datasets.WindowDataset.INVALID_WINDOW_VAL]
    # For every source file there will be (window_size - 1) invalid indices for a window to start at.
    # Think of the last valid window: all bins in that window except the first are invalid
    assert window_inds_no_invalid.shape[-1] == window_inds.shape[-1] - (n_source_files_in_split * (window_size - 1))

    assert np.array_equal(
        np.unique(source_ids),
        np.arange(n_source_files_in_split)
    )

    if crop_dur:
        assert lbl_tb.shape == source_ids.shape == source_inds.shape == window_inds.shape
    else:
        assert lbl_tb is None


@pytest.mark.parametrize(
    'config_type, model_name, audio_format, spect_format, annot_format, window_size, crop_dur',
    [
        ('learncurve', 'teenytweetynet', 'cbin', None, 'notmat', 22, None),
        ('learncurve', 'teenytweetynet', 'cbin', None, 'notmat', 22, 4.0),
        ('learncurve', 'teenytweetynet', 'cbin', None, 'notmat', 44, None),
        ('learncurve', 'teenytweetynet', 'cbin', None, 'notmat', 44, 4.0),
    ]
)
def test_vectors_from_df(config_type, model_name, audio_format, spect_format, annot_format,
                         window_size, crop_dur, specific_config):
    toml_path = specific_config(config_type,
                                model_name,
                                audio_format=audio_format,
                                spect_format=spect_format,
                                annot_format=annot_format)
    cfg = vak.config.parse.from_toml_path(toml_path)

    # stuff we need just to be able to instantiate window dataset
    labelmap = vak.labels.to_map(cfg.prep.labelset, map_unlabeled=True)

    cmd_cfg = getattr(cfg, config_type)  # "command config", i.e., cli command, [TRAIN] or [LEARNCURVE] section
    csv_path = getattr(cmd_cfg, 'dataset_path')
    df = pd.read_csv(csv_path)

    if crop_dur:
        timebin_dur = vak.io.dataframe.validate_and_get_timebin_dur(df)
    else:
        timebin_dur = None

    source_ids, source_inds, window_inds = vak.datasets.window_dataset.helper.vectors_from_df(
        df,
        'train',
        window_size,
        crop_dur=crop_dur,
        labelmap=labelmap,
        timebin_dur=timebin_dur,
    )

    for vector_name, vector in zip(
        ('source_ids', 'source_inds', 'window_inds'),
        (source_ids, source_inds, window_inds)
    ):
        assert isinstance(vector, np.ndarray)

    assert source_ids.shape == source_inds.shape
    n_source_files_in_split = len(df[df.split == 'train'])
    # For every source file there will be (window_size - 1) invalid indices for a window to start at.
    # Think of the last valid window: all bins in that window except the first are invalid
    n_total_invalid_start_inds = n_source_files_in_split * (window_size - 1)
    if crop_dur:
        assert window_inds.shape[-1] <= source_ids.shape[-1]
    else:
        assert window_inds.shape[-1] == source_inds.shape[-1] - n_total_invalid_start_inds


@pytest.mark.parametrize(
    'config_type, model_name, audio_format, spect_format, annot_format, window_size, crop_dur',
    [
        ('learncurve', 'teenytweetynet', 'cbin', None, 'notmat', 22, None),
        ('learncurve', 'teenytweetynet', 'cbin', None, 'notmat', 22, 4.0),
        ('learncurve', 'teenytweetynet', 'cbin', None, 'notmat', 44, None),
        ('learncurve', 'teenytweetynet', 'cbin', None, 'notmat', 44, 4.0),
    ]
)
def test_vectors_from_csv(config_type, model_name, audio_format, spect_format, annot_format,
                          window_size, crop_dur, specific_config):
    toml_path = specific_config(config_type,
                                model_name,
                                audio_format=audio_format,
                                spect_format=spect_format,
                                annot_format=annot_format)
    cfg = vak.config.parse.from_toml_path(toml_path)

    # stuff we need just to be able to instantiate window dataset
    labelmap = vak.labels.to_map(cfg.prep.labelset, map_unlabeled=True)

    cmd_cfg = getattr(cfg, config_type)  # "command config", i.e., cli command, [TRAIN] or [LEARNCURVE] section
    csv_path = getattr(cmd_cfg, 'dataset_path')
    df = pd.read_csv(csv_path)  # we use ``df`` in asserts below

    if crop_dur:
        timebin_dur = vak.io.dataframe.validate_and_get_timebin_dur(df)
    else:
        timebin_dur = None

    source_ids, source_inds, window_inds = vak.datasets.window_dataset.helper.vectors_from_csv_path(
        csv_path,
        'train',
        window_size,
        crop_dur=crop_dur,
        timebin_dur=timebin_dur,
        labelmap=labelmap,
    )

    for vector_name, vector in zip(
        ('source_ids', 'source_inds', 'window_inds'),
        (source_ids, source_inds, window_inds)
    ):
        assert isinstance(vector, np.ndarray)
    assert source_ids.shape == source_inds.shape
    n_source_files_in_split = len(df[df.split == 'train'])
    # For every source file there will be (window_size - 1) invalid indices for a window to start at.
    # Think of the last valid window: all bins in that window except the first are invalid
    n_total_invalid_start_inds = n_source_files_in_split * (window_size - 1)
    if crop_dur:
        assert window_inds.shape[-1] <= source_ids.shape[-1]
    else:
        assert window_inds.shape[-1] == source_inds.shape[-1] - n_total_invalid_start_inds
