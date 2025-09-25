import os
import tempfile

import pytest

import vak.config.generate


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
def test_generate(kind, add_prep, dst_name, tmp_path):
    """Test :func:`vak.config.generate.generate`"""
    # FIXME: handle case where `dst` is a filename -- handle .toml extension
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
        vak.config.generate.generate(kind=kind, add_prep=add_prep)
    else:
        vak.config.generate.generate(kind=kind, add_prep=add_prep, dst=dst)

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


def test_generate_raises(tmp_path):
    dst = tmp_path / "fake.config.toml"
    with dst.open("w") as fp:
        fp.write("[fake.config]")
    with pytest.raises(FileExistsError):
        vak.config.generate.generate("train", add_prep=True, dst=dst)
