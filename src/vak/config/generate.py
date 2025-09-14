import importlib.resources
import pathlib

import tomlkit


CONFIGFILE_KIND_FILENAME_MAP = {
    "train": "configfile_train.toml",
    "eval": "configfile_eval.toml",
    "predict": "configfile_predict.toml",
    "learncurve": "configfile_learncurve.toml",
}

# next line: can't use `.items()`, we'll get `RuntimeError` about dictionary changed sized during iteration
for key in list(CONFIGFILE_KIND_FILENAME_MAP.keys()):
    val = CONFIGFILE_KIND_FILENAME_MAP[key]
    CONFIGFILE_KIND_FILENAME_MAP[f"{key}_prep"] = val.replace(key, f"{key}_prep")


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
    
    # for now, we "add a prep section" by using a naming convention
    # and loading an existing toml file that has a `[vak.prep]` table
    if add_prep:
        kind = f"{kind}_prep"

    try:
        src_filename = CONFIGFILE_KIND_FILENAME_MAP[kind]
    except KeyError:
        raise ValueError(
            f"Invalid kind: {kind}"
        )
    
    src_path = pathlib.Path(
        importlib.resources.files("vak.config._toml_config_templates").joinpath(src_filename)
    )
    # even though we are loading an existing file,
    # we use tomlkit to load and dump.
    # TODO: add "interactive" arg and use tomlkit with `input` to interactively build config file
    with src_path.open("r") as fp:
        tomldoc = tomlkit.load(fp)

    if dst.is_dir():
        dst_path = dst / src_filename
    else:
        dst_path = dst

    with dst_path.open("w") as fp:
        tomlkit.dump(tomldoc, fp)
