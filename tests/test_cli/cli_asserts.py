"""assertions common to cli functions"""


def toml_config_file_copied_to_results_path(results_path, toml_path):
    assert results_path.joinpath(toml_path.name).exists()
    return True


def log_file_created(command, output_path):
    log_file = sorted(output_path.glob(f"{command}*.log"))
    assert len(log_file) == 1

    return True
