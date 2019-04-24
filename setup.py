#!/usr/bin/env python
# -*- coding: utf-8 -*-

# adopted from Kenneth Reitz
# https://github.com/kennethreitz/setup.py
# released under MIT license
# (https://github.com/kennethreitz/setup.py/blob/master/LICENSE)

# Note: To use the 'upload' functionality of this file, you must:
#   $ pip install twine

import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

# Package meta-data.
NAME = 'vak'
DESCRIPTION = 'library to work with neural networks that segment and annotate vocalizations'
URL = 'https://github.com/NickleDave/vak'
EMAIL = 'dnicho4@emory.edu'
AUTHOR = 'David Nicholson, Yarden Cohen'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = '0.1.0a8'
LICENSE = 'BSD'

REQUIRED = [
    'tensorflow', 'numpy', 'scipy', 'matplotlib', 'joblib', 'tqdm', 'attrs', 'crowsetta',
]

dev_deps = [
    'twine',
]

test_deps = [
    'vak_test_net', 'tweetynet>=0.1.1a4',
]

# this is here so that the .travis.yml script can install
# dependencies just for the tests by running
# pip install .[tests]
EXTRAS = {
    'tests': test_deps,
    'dev': dev_deps,
}

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    with open(os.path.join(here, NAME, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION


class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system(
            '{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')

        sys.exit()


# Where the magic happens:
setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(where="src", exclude=('tests',)),
    package_dir={"": "src"},
    # scripts=['src/bin/vak-cli.py'],
    entry_points={
        'console_scripts': ['vak=vak.__main__:main'],
    },
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license=LICENSE,
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: BSD License',
        'Development Status :: 4 - Beta',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
    # $ setup.py publish support.
    cmdclass={
        'upload': UploadCommand,
    },
)
