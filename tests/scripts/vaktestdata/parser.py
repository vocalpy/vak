"""CLI parser for generate test data script"""
import argparse

from . import constants


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--step',
        choices=constants.GENERATE_TEST_DATA_STEPS,
        help=f"Which step of generating test data to perform, one of: {constants.GENERATE_TEST_DATA_STEPS}",
        default='all'
    )
    parser.add_argument(
        '--commands',
        choices=('train', 'learncurve', 'eval', 'predict', 'train_continue'),
        help=f"Space-separated list of commands to run for 'results' step",
        nargs="+",
        default=('train', 'learncurve', 'eval', 'predict', 'train_continue')
    )
    parser.add_argument(
        '--single-train-result',
        action = argparse.BooleanOptionalAction,
        help=(
            "If --single-train-result, require there be a single results directory "
            "from any training config when looking for them to use in other configs. "
            "If --no-single-train-result, allow multiple and use the most recent."
        )
    )
    return parser
