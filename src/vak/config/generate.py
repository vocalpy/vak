import importlib.resources
import pathlib
import shutil

CONFIGFILE_KIND_FILENAME_MAP = {
    "train": "configfile_train.toml",
    "eval": "configfile_eval.toml",
    "predict": "configfile_predict.toml",
    "learncurve": "configfile_learncurve.toml",
}


def generate(
    kind: str,
    add_prep: bool = False,
    dst: str | pathlib.Path = pathlib.Path.cwd(),
) -> None:
    """Generate a TOML configuration file

    This is the function called by 
    :func:`vak.cli.cli.generate` 
    when a user runs the command ``vak configfile``
    using the command-line interface.

    Parameters
    ----------
    kind : str
        The kind of TOML configuration file to generate.
        One of: ``{'train', 'eval', 'predict', 'learncurve'}``
    add_prep : bool
        If True, add a ``[vak.prep]`` table to the 
        TOML configuration file.
    dst : string, pathlib.Path
        Destination for the generated configuration file.
        Either a full path including filename, 
        or a directory, in which case a default filename 
        will be used.
        The default `dst` is the current working directory.
    """
    dst = pathlib.Path(dst)
    if not dst.is_dir() and dst.exists():
        raise ValueError(
            f"Destination for generated config file `dst` is already a file that exists:\n{dst}\n"
            "Please specify a value for the `--dst` argument that will not overwrite an existing file."
        )
    filename = CONFIGFILE_KIND_FILENAME_MAP[kind]
    src = pathlib.Path(
        importlib.resources.files("vak.config").joinpath(filename)
    )
    shutil.copy(src, dst)
