"""sub-package with function that parses config.ini file and returns config object"""
from .parse import parse_config
from .data import parse_data_config
from .output import parse_output_config
from .predict import parse_predict_config
from .spectrogram import parse_spect_config
from .train import parse_train_config

