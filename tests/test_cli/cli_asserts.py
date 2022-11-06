"""assertions common to cli functions"""
import vak


def toml_config_file_copied_to_results_path(results_path, toml_path):
    assert results_path.joinpath(toml_path.name).exists()
    return True


def log_file_created(command, output_path):
    log_file = sorted(output_path.glob(f"{command}*.log"))
    assert len(log_file) == 1
    return True


def log_file_contains_version(command, output_path):
    log_file = sorted(output_path.glob(f"{command}*.log"))
    assert len(log_file) == 1
    log_file = log_file[-1]
    with log_file.open('r') as fp:
        lines = fp.read().splitlines()
    assert lines[0].endswith(f'vak version: {vak.__about__.__version__}')
    return True