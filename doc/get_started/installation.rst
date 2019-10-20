.. toctree::
   :maxdepth: 2
   :caption: Contents:

Installation
============

with Python package manager `pip`
---------------------------------

To install, run the following command at the command line:

.. code-block:: console

    you@your-computer: ~/Documents $ pip install vak

(just type the `pip install vak` part)

with `conda` package manager
----------------------------

`vak` will be available through a conda channel shortly, and eventually through conda-forge.
Keep an eye on these issues on GitHub:

| https://github.com/NickleDave/vak/issues/69
| https://github.com/NickleDave/vak/issues/70

in a virtual environment
------------------------
Before you install, you'll probably want to set up a virtual environment
(for an explanation of why, see https://www.geeksforgeeks.org/python-virtual-environment/).
Creating a virtual environment is not as hard as it might sound;
here's a primer on Python tools: https://realpython.com/python-virtual-environments-a-primer/
For many scientific packages that depend on libraries written in
languages besides Python, you may find it easier to use
a platform dedicated to managing those dependencies, such as
[Anaconda](https://www.anaconda.com/download) (which is free).
You can use the `conda` command-line tool that they develop
to create environments and install the scientific libraries that this package
depends on. In addition, using `conda` to install the dependencies may give you some performance gains
(see https://www.anaconda.com/blog/developer-blog/tensorflow-in-anaconda/).
Here's how you'd set up a `conda` environment:

.. code-block:: console

    you@your-computer: ~/Documents $ conda create -n vak-env python=3.6 numpy scipy joblib tensorflow-gpu ipython jupyter
    you@your-computer: ~/Documents $ source activate vak-env

(You don't have to `source` on Windows: `> activate vak-env`)

You can then use `pip` inside a `conda` environment:

.. code-block:: console

    (vak-env)/home/you/code/ $ pip install vak

from source
-----------
You can also work with a local copy of the code.
It's possible to install the local copy with `pip` so that you can still edit
the code, and then have its behavior as an installed library reflect those edits.
  * Clone the repo from Github using the version control tool `git`:

    .. code-block:: console

        (vak-env) you@your-computer: ~/Documents $ git clone https://github.com/NickleDave/vak

    (you can install `git` from Github or using `conda`.)
  * Install the package with `pip` using the `-e` flag (for `editable`).

  .. code-block:: console

      $ (vak-env) you@your-computer: ~/Documents $ cd vak
      $ (vak-env) you@your-computer: ~/Documents $ pip install -e .

