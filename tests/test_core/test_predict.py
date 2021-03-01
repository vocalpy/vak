"""tests for vak.core.predict module"""
import pytest

import vak.config
import vak.core.predict


# written as separate function so we can re-use in tests/unit/test_cli/test_predict.py
def predict_output_matches_expected(output_dir, annot_csv_filename):
    annot_csv = output_dir.joinpath(annot_csv_filename)
    assert annot_csv.exists()

    return True


@pytest.mark.parametrize(
    'audio_format, spect_format, annot_format',
    [
        ('cbin', None, 'notmat'),
        ('wav', None, 'koumura'),
    ]
)
def test_predict(audio_format,
                 spect_format,
                 annot_format,
                 specific_config,
                 tmp_path,
                 device):
    output_dir = tmp_path.joinpath(f'test_predict_{audio_format}_{spect_format}_{annot_format}')
    output_dir.mkdir()

    options_to_change = [
        {'section': 'PREDICT',
         'option': 'output_dir',
         'value': str(output_dir)},
        {'section': 'PREDICT',
         'option': 'device',
         'value': device}
    ]

    toml_path = specific_config(config_type='predict',
                                audio_format=audio_format,
                                annot_format=annot_format,
                                options_to_change=options_to_change)
    cfg = vak.config.parse.from_toml_path(toml_path)

    model_config_map = vak.config.models.map_from_path(toml_path, cfg.predict.models)

    vak.core.predict(csv_path=cfg.predict.csv_path,
                     checkpoint_path=cfg.predict.checkpoint_path,
                     labelmap_path=cfg.predict.labelmap_path,
                     model_config_map=model_config_map,
                     window_size=cfg.dataloader.window_size,
                     num_workers=cfg.predict.num_workers,
                     spect_key=cfg.spect_params.spect_key,
                     timebins_key=cfg.spect_params.timebins_key,
                     spect_scaler_path=cfg.predict.spect_scaler_path,
                     device=cfg.predict.device,
                     annot_csv_filename=cfg.predict.annot_csv_filename,
                     output_dir=cfg.predict.output_dir,
                     min_segment_dur=cfg.predict.min_segment_dur,
                     majority_vote=cfg.predict.majority_vote,
                     logger=None
                     )

    assert predict_output_matches_expected(output_dir, cfg.predict.annot_csv_filename)
