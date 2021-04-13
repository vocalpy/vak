import sys


def test_installed():
    # makes sure the entry point loads properly
    if "vak" in sys.modules:
        sys.modules.pop("vak")
    import vak

    models = [name for name, class_ in sorted(vak.models.find())]
    assert "TeenyTweetyNetModel" in models
