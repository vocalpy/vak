import pandas as pd
import pytest

import vak


@pytest.mark.parametrize(
    'config_type, model_name, audio_format, spect_format, annot_format, input_type',
    [
        ('train', 'TweetyNet', 'cbin', None, 'notmat', 'spect'),
        ('predict', 'TweetyNet', 'cbin', None, 'notmat', 'spect'),
        ('eval', 'TweetyNet', 'cbin', None, 'notmat', 'spect'),
        ('train', 'TweetyNet', None, 'mat', 'yarden', 'spect'),
        ('learncurve', 'TweetyNet', 'cbin', None, 'notmat', 'spect'),
        # TODO: add audio cases
    ]
)
def test_assign_samples_to_splits(
        config_type, model_name, audio_format, spect_format, annot_format,
        input_type, tmp_path, specific_config_toml_path, specific_source_files_df,
):
    toml_path = specific_config_toml_path(
        config_type,
        model_name,
        annot_format,
        audio_format,
        spect_format,
    )

    cfg = vak.config.Config.from_toml_path(toml_path)

    # ---- set up ----
    tmp_dataset_path = tmp_path / 'dataset_dir'
    tmp_dataset_path.mkdir()

    purpose = config_type

    source_files_df = specific_source_files_df(
        config_type,
        model_name,
        annot_format,
        audio_format,
        spect_format,
    )

    out = vak.prep.frame_classification.assign_samples_to_splits(
        purpose,
        source_files_df,
        tmp_dataset_path,
        cfg.prep.train_dur,
        cfg.prep.val_dur,
        cfg.prep.test_dur,
        cfg.prep.labelset,
    )

    assert isinstance(out, pd.DataFrame)
    assert 'split' in out.columns
    if purpose == 'predict':
        assert all(val == 'predict' for val in out['split'].values)
    elif purpose == 'eval':
        assert all(val == 'test' for val in out['split'].values)
    else:
        split_vals = out['split'].values.tolist()
        assert all(
            [
                split_name in split_vals
                for split_name in ('train', 'val', 'test')
                if hasattr(cfg.prep, f'{split_name}_dur') and getattr(cfg.prep, f'{split_name}_dur') is not None
            ]
        )
