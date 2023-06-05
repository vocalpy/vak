import pytest

import vak.converters


@pytest.mark.parametrize(
    ("toml_value", "labelset"),
    [
        ("abc", {"a", "b", "c"}),
        ("1235", {"1", "2", "3", "5"}),
        ("range: 1-3, 5", {"1", "2", "3", "5"}),
        ([1, 2, 3, 5], {"1", "2", "3", "5"}),
        (["a", "b", "c"], {"a", "b", "c"}),
        (["range: 1-3", "noise"], {"1", "2", "3", "noise"}),
    ],
)
def test_labelset_from_toml_value(toml_value, labelset):
    """test that function behaves as specified in docstring"""
    assert vak.converters.labelset_to_set(toml_value) == labelset


def test_labelset_from_toml_value_raises():
    """test that an invalid value in a labelset list raises a TypeError"""
    labelset_with_float = [1, "a", 0.3]
    with pytest.raises(TypeError):
        _ = vak.converters.labelset_to_set(labelset_with_float)
