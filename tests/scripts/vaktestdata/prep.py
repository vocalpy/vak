# Do this here to suppress warnings before we import vak
import warnings
from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
import vak

def run_prep(config_paths):
    """run ``vak prep`` to generate data for testing"""
    for config_path in config_paths:
        if not config_path.exists():
            raise FileNotFoundError(f"{config_path} not found")
        print(
            f"\nRunning vak prep to generate data for tests, using config:\n{config_path.name}"
        )
        vak.cli.prep.prep(toml_path=config_path)
