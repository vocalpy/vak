# adopted from attrs library
# https://github.com/python-attrs/attrs/blob/master/tests/utils.py
from attr import Attribute
from attr._make import NOTHING


def simple_attr(
    name,
    default=NOTHING,
    validator=None,
    repr=True,
    eq=True,
    hash=None,
    init=True,
    converter=None,
    kw_only=False,
    inherited=False,
):
    """
    Return an attribute with a name and no other bells and whistles.
    """
    return Attribute(
        name=name,
        default=default,
        validator=validator,
        repr=repr,
        cmp=None,
        eq=eq,
        hash=hash,
        init=init,
        converter=converter,
        kw_only=kw_only,
        inherited=inherited,
    )
