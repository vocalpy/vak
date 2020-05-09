"""module that contains helper function to load metrics

Metrics in separate packages should make themselves available to vak by including
'vak.metrics' in the entry_points of their setup.py file.

For more detail on entry points in Python, see:
https://packaging.python.org/guides/creating-and-discovering-plugins/#using-package-metadata
https://setuptools.readthedocs.io/en/latest/setuptools.html#dynamic-discovery-of-services-and-plugins
https://amir.rachum.com/blog/2017/07/28/python-entry-points/
"""
from .. import entry_points

METRICS_ENTRY_POINT = 'vak.models'


def find():
    """find installed vak.metrics

    returns generator that yields metric name and function for loading
    """
    for entrypoint in entry_points._iter(METRICS_ENTRY_POINT):
        yield entrypoint.name, entrypoint.load()
