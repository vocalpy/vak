"""tests for vak.cli.prep module"""
from pathlib import Path

import pytest

import vak.config
import vak.constants
import vak.core.train
import vak.paths
import vak.io.spect

from . import cli_asserts

@pytest.mark.parametrize(
    'config_type, audio_format, spect_format, annot_format',
    [
        ('eval', 'cbin', None, 'notmat'),
        ('learncurve', 'cbin', None, 'notmat'),
        ('predict', 'cbin', None, 'notmat'),
        ('predict', 'wav', None, 'koumura'),
        ('train', 'cbin', None, 'notmat'),
        ('train', 'wav', None, 'koumura'),
        ('train', None, 'mat', 'yarden'),
    ]
)
def test_purpose_from_toml(config_type,
                           audio_format,
                           spect_format,
                           annot_format,
                           specific_config,
                           tmp_path):
    toml_path = specific_config(config_type=config_type,
                                audio_format=audio_format,
                                annot_format=annot_format,
                                spect_format=spect_format)
    config_toml = vak.config.parse._load_toml_from_path(toml_path)
    vak.cli.prep.purpose_from_toml(config_toml)



@pytest.mark.parametrize(
    'config_type, audio_format, spect_format, annot_format',
    [
        ('eval', 'cbin', None, 'notmat'),
        ('learncurve', 'cbin', None, 'notmat'),
        ('predict', 'cbin', None, 'notmat'),
        ('predict', 'wav', None, 'koumura'),
        ('train', 'cbin', None, 'notmat'),
        ('train', 'wav', None, 'koumura'),
        ('train', None, 'mat', 'yarden'),
    ]
)
def test_prep(config_type,
              audio_format,
              spect_format,
              annot_format,
              specific_config,
              tmp_path):
    output_dir = tmp_path.joinpath(f'test_prep_{config_type}_{audio_format}_{spect_format}_{annot_format}')
    output_dir.mkdir()

    options_to_change = {
        'section': 'PREP',
        'option': 'output_dir',
        'value': str(output_dir)
    }
    toml_path = specific_config(config_type=config_type,
                                audio_format=audio_format,
                                annot_format=annot_format,
                                spect_format=spect_format,
                                options_to_change=options_to_change)

    vak.cli.prep(toml_path)

    cfg = vak.config.parse.from_toml_path(toml_path)
    command_section = getattr(cfg, config_type)
    csv_path = getattr(command_section, 'csv_path')
    # we don't bother checking whether csv is as expected
    # because that's already tested by `test_io.test_spect`, `test_io.test_dataframe`, etc.
    assert Path(csv_path).exists()

    assert cli_asserts.log_file_created(command='prep', output_path=cfg.prep.output_dir)
