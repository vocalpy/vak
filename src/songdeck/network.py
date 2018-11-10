"""module that contains helper function to load networks, and
the AbstractSongdeckNetwork class.
Networks in separate packages should subclass this class, and
then make themselves available to songdeck by including
'songdeck.network' in the entry_points value of their setup.py
file.

The reason for subclassing is simply to set a
standard interface for networks to use. songdeck will check
that any network implements all the methods listed here,
since it requires those methods to carry out commands
in its command-line interface.

For example, if you had a package `grunet` containing a class
`GRUnet`, then that package would include the following in its
setup.py file:

setup(
    ...
    entry_points={'songdeck.network': 'GRUnet = grunet.model:GRUnet'},
    ...
)

For more detail on entry points in Python, see:
https://packaging.python.org/guides/creating-and-discovering-plugins/#using-package-metadata
https://setuptools.readthedocs.io/en/latest/setuptools.html#dynamic-discovery-of-services-and-plugins
https://amir.rachum.com/blog/2017/07/28/python-entry-points/
"""
import pkg_resources


def _load():
    networks = {
        entry_point.name: entry_point.load()
        for entry_point
        in pkg_resources.iter_entry_points('songdeck.network')
    }
    return networks


class AbstractSongdeckNetwork:
    def __init__(self):
        pass

    def inference(self):
        """inference method, forward pass through network
        with output that is used with loss function for
        training (e.g., softmax layer)"""
        pass

    def optimize(self):
        """optimize method, that runs optimizer on one cost value, given
        ground truth as expected output"""
        pass

    def error(self):
        """error method, that returns mean error given as input a spectrogram
        and the true and predicted labels"""

    def train(self):
        """method that uses error and optimize methods to iteratively
        train the network"""
        pass

    def predict(self):
        """predict method, that (usually) returns argmax(inference)
        for all outputs"""
        pass

    def save(self):
        """method to save network"""
        pass

    def restore(self):
        pass
