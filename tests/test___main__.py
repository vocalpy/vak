import os
import subprocess
from unittest import mock

import pytest

import vak


DUMMY_CONFIGFILE = './configs/config_2018-12-17.toml'


@pytest.mark.parametrize(
    'args_list',
    [
        ['prep', DUMMY_CONFIGFILE],
        ['train', DUMMY_CONFIGFILE],
        ['learncurve', DUMMY_CONFIGFILE],
        ['eval', DUMMY_CONFIGFILE],
        ['predict', DUMMY_CONFIGFILE],
        ['configfile', 'train', '--add-prep', '--dst', DUMMY_CONFIGFILE]
    ]
)
def test_main(args_list):
    """Test that :func:`vak.__main__.main` calls the function we expect through :func:`vak.cli.cli`
    
    Notes
    -----
    We mock these and call it a unit test 
    because actually calling and running :func:vak.cli.prep` 
    would be expensive. 
    
    The exception is `vak configfile` 
    that we test directly (in other test functions below).
    """
    command = args_list[0]
    mock_cli_function = mock.Mock(name=f'mock_{command}')
    with mock.patch.dict(
        vak.cli.cli.CLI_COMMAND_FUNCTION_MAP, {command: mock_cli_function}
    ):
        # wAFAICT e can't do this with `subprocess` since the function won't be mocked in the subprocess,
        # so we need to test indirectly with `arg_list` passed into `main`
        vak.__main__.main(args_list)
        mock_cli_function.assert_called()


def test___main__prints_help_with_no_args(parser, capsys):
    """Test that if we don't pass in any args, we get """
    parser.print_help()
    expected_output = capsys.readouterr().out.rstrip()

    # doing this by calling a `subprocess` is slow but lets us test the CLI directly
    result = subprocess.run("vak", capture_output=True, text=True)  # call `vak` at CLI with no help
    output = result.stdout.rstrip()
     
    assert output == expected_output


@pytest.mark.parametrize(
    'kind, add_prep, dst_name',
    [
        # ---- train
        (
            "train",
            False,
            None
        ),
        (
            "train",
            True,
            None
        ),
        (
            "train",
            False,
            "configs-dir"
        ),
        (
            "train",
            True,
            "configs-dir"
        ),
        (
            "train",
            False,
            "configs-dir/config.toml"
        ),
        (
            "train",
            True,
            "configs-dir/config.toml"
        ),
        # ---- eval
               (
            "eval",
            False,
            None
        ),
        (
            "eval",
            True,
            None
        ),
        (
            "eval",
            False,
            "configs-dir"
        ),
        (
            "eval",
            True,
            "configs-dir"
        ),
        (
            "eval",
            False,
            "configs-dir/config.toml"
        ),
        (
            "eval",
            True,
            "configs-dir/config.toml"
        ),
        # ---- predict
               (
            "predict",
            False,
            None
        ),
        (
            "predict",
            True,
            None
        ),
        (
            "predict",
            False,
            "configs-dir"
        ),
        (
            "predict",
            True,
            "configs-dir"
        ),
        (
            "predict",
            False,
            "configs-dir/config.toml"
        ),
        (
            "predict",
            True,
            "configs-dir/config.toml"
        ),
        # ---- learncurve
               (
            "learncurve",
            False,
            None
        ),
        (
            "learncurve",
            True,
            None
        ),
        (
            "learncurve",
            False,
            "configs-dir"
        ),
        (
            "learncurve",
            True,
            "configs-dir"
        ),
        (
            "learncurve",
            False,
            "configs-dir/config.toml"
        ),
        (
            "learncurve",
            True,
            "configs-dir/config.toml"
        ),
    ]
)
def test_configfile_command(kind, add_prep, dst_name, tmp_path):
    """Test :func:`vak.config.generate.generate`"""
    if dst_name is None:
        dst = tmp_path / "tmp-dst-None"
    else:
        dst = tmp_path / dst_name
    if dst.suffix == ".toml":
        # if dst ends with a toml extension
        # then its *parent* is the dir we need to make
        dst.parent.mkdir()
    else:
        dst.mkdir()

    if dst_name is None:
        os.chdir(dst)

    args = ["vak", "configfile", kind]
    if add_prep:
        args = args + ["--add-prep"]
    if dst_name is not None:
        args = args + ["--dst", str(dst)]
    subprocess.run(args)

    if dst.is_dir():
        # we need to get the actual generated TOML
        generated_toml_path = sorted(dst.glob("*toml"))
        assert len(generated_toml_path) == 1
        generated_toml_path = generated_toml_path[0]
    else:
        generated_toml_path = dst
        # next line: the rest of the assertions would fail if this one did
        # but we're being super explicit here:
        # if we specified a file name for dst then it should exist as a file
        assert generated_toml_path.exists()

    # we can't load with `vak.config.Config.from_toml_path`
    # because the generated config doesn't have a [vak.dataset.path] key-value pair yet,
    # and the corresponding attrs class that represents that table will throw an error.
    # So we load as a Python dict and check the expected keys are there.
    # I don't have any better ideas at the moment for how to test
    cfg_dict = vak.config.load._load_toml_from_path(generated_toml_path)
    # N.B. that `vak.config.load._load_toml_from_path` accesses the top-level key "vak"
    # and returns the result of that, so we don't need to do something like `cfg_dict["vak"]["prep"]`
    assert kind in cfg_dict
    if add_prep:
        assert "prep" in cfg_dict
    else:
        assert "prep" not in cfg_dict
