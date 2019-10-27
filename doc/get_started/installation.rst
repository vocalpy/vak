.. toctree::
   :maxdepth: 2
   :caption: Contents:


Installation
============

with ``conda`` environment and package manager
--------------------------------------------

Currently, installing ``vak`` requires use of the ``conda`` tool to create a virtual environment and
install the libraries that ``vak`` depends on into that environment. The easiest way to use ``conda`` is to install the
Anaconda platform (https://www.anaconda.com/download) (which is free).
For a more detailed explanation of why you would use a virtual environment, please see
`Why use a virtual environment?`_.

Here are the steps to follow after installing Anaconda:

.. code-block:: console

    you@your-computer: ~/Documents $ conda create -n vak-env python=3.7
    you@your-computer: ~/Documents $ source activate vak-env

(You don't have to `source` on Windows: ``> activate vak-env``)

Once you create and activate the environment, you can install `vak`:

.. code-block:: console
    (vak-env) you@your-computer: ~/Documents $ conda install vak -c nickledave

Eventually `vak` will be available through the conda-forge channel.
Keep an eye on this issue on GitHub:

| https://github.com/NickleDave/vak/issues/70

with ``pip`` package manager
--------------------------------------------

We strongly suggest using ``conda`` to install, but you can also use ``pip``.
Again we recommend installing into a ``conda`` environment:

.. code-block:: console

    (vak-env)/home/you/code/ $ pip install vak

Note that if you use ``pip``, and you are using a GPU, you will need to ensure that the
installation of ``Tensorflow`` is using the system install of binaries that it depends on, such as `libcuda.so`.
``conda`` makes it possible to install ``tensorflow-gpu`` which itself uses ``cudatoolkit``
and other dependencies installed into the environment to avoid dealing with system-wide installs of binaries.

from source
-----------
You can also work with a local copy of the code.
It's possible to install the local copy with ``pip`` so that you can still edit
the code, and then have its behavior as an installed library reflect those edits.
  * Clone the repo from Github using the version control tool ``git``:

    .. code-block:: console

        (vak-env) you@your-computer: ~/Documents $ git clone https://github.com/NickleDave/vak

    (you can install `git` from Github, with your operating system package manager, or using ``conda``.)
  * Install the package with `pip` using the `-e` flag (for ``editable``).

Why use a virtual environment?
------------------------------
For an explanation of virtual environments, please see
https://realpython.com/python-virtual-environments-a-primer/.
For data science packages that depend on libraries written in
languages besides Python, you may find it easier to use
a platform dedicated to managing those dependencies, such as
Anaconda(https://www.anaconda.com/download) (which is free).
You can use the ``conda`` command-line tool that they develop
to create environments and install the libraries that this package
depends on. Here is an in-depth look at using `conda` to manage environments:
https://www.freecodecamp.org/news/why-you-need-python-environments-and-how-to-manage-them-with-conda-85f155f4353c/.
Further detail about `conda` and how it relates to other tools like
`virtualenv` and `pip` can be found in this blog post:
https://jakevdp.github.io/blog/2016/08/25/conda-myths-and-misconceptions/.