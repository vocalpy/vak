"""Constants used by vaktestdata and scripts that rely on it."""
import json
import pathlib

from .config_metadata import ConfigMetadata

HERE = pathlib.Path(__file__).parent
TEST_DATA_ROOT = HERE / ".." / ".." / "data_for_tests"
CONFIG_METADATA_JSON_PATH = TEST_DATA_ROOT / "configs" / "configs.json"
with CONFIG_METADATA_JSON_PATH.open('r') as fp:
    CONFIG_METADATA_LIST = json.load(fp)['config_metadata']
CONFIG_METADATA = [
    ConfigMetadata(**config_metadata_dict)
    for config_metadata_dict in CONFIG_METADATA_LIST
]
GENERATED_TEST_DATA_ROOT = TEST_DATA_ROOT / "generated"

GENERATED_SPECT_OUTPUT_DIR = GENERATED_TEST_DATA_ROOT / "spect-output-dir"
GENERATED_SOURCE_FILES_CSV_DIR = GENERATED_TEST_DATA_ROOT / "source-files-csv"
GENERATED_SOURCE_FILES_WITH_SPLITS_CSV_DIR = GENERATED_TEST_DATA_ROOT / "source-files-with-splits-csv"

GENERATED_TEST_CONFIGS_ROOT = GENERATED_TEST_DATA_ROOT / "configs"

# convention is that all the config.toml files in tests/data_for_tests/configs
# that should be run when generating test data
# have filenames of the form `{MODEL}_{COMMAND}_audio_{FORMAT}_annot_{FORMAT}.toml'
# **or** `{MODEL}_{COMMAND}_spect_{FORMAT}_annot_{FORMAT}_config.ini'
# e.g., 'TweetyNet_learncurve_audio_cbin_annot_notmat.toml'.
# Below, we iterate over model names
# so glob doesn't pick up static configs that are just used for testing,
# like 'invalid_option_config.toml`
TEST_CONFIGS_ROOT = TEST_DATA_ROOT.joinpath("configs")

# the sub-directories that will get made inside `./tests/data_for_tests/generated`
TOP_LEVEL_DIRS = [
    "prep",
    "results",
]

# need to run 'train' config before we run 'predict'
# so we can add checkpoints, etc., from training to predict
COMMANDS = (
    "train",
    "learncurve",
    "eval",
    "predict",
    "train_continue",
)

GENERATE_TEST_DATA_STEPS = (
    'prep',
    'results',
    'all',
)
