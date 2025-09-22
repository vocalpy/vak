import os

import pytest

import vak.config.generate


@pytest.mark.parametrize(
    'kind, add_prep, dst_name',
    [
        (
            "train",
            False,
            None
        )
    ]
)
def test_generate(kind, add_prep, dst_name, tmp_path):
    """Test :func:`vak.config.generate.generate`"""
    # FIXME: handle case where `dst` is a filename -- handle .toml extension
    if dst_name is None:
        dst = tmp_path / "tmp-dst-None"
    else:
        dst = tmp_path / dst_name
    dst.mkdir()

    if dst_name is None:
        os.chdir(dst)
        vak.config.generate.generate(kind=kind, add_prep=add_prep)
    else:
        dst = tmp_path / dst
        vak.config.generate.generate(kind=kind, add_prep=add_prep, dst=dst)

    if dst.is_dir():
        # we need to get the actual generated TOML
        generated_toml_path = sorted(dst.glob("*toml"))
        assert len(generated_toml_path) == 1
        generated_toml_path = generated_toml_path[0]
    else:
        generated_toml_path = dst

    cfg = vak.config.Config.from_toml_path(generated_toml_path)
    assert hasattr(cfg, kind)
    if add_prep:
        assert hasattr(cfg, "prep")
    else:
        assert not hasattr(cfg, "prep")


def test_generate_raises():
    # FIXME: test we raise error if dst already exists
    assert False