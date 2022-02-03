.. _installation:

============
Installation
============

``vak`` can be installed with the package managers ``conda`` or ``pip``.
We recommend ``conda`` for most users. For more detail, please see :ref:`why-conda`

We also recommend installing ``vak`` into a virtual environment.
For an explanation, please see :ref:`why-virtualenv`.

with ``conda``
==============

on Mac and Linux
----------------
To create a new virtual environment that contains ``vak`` and all the libraries it depends on, run:

.. code-block:: shell

   $ conda create --name vak-env vak -c conda-forge

To install ``vak`` into an existing environment, run:

.. code-block:: shell

   (existing-env) $ conda install vak -c conda-forge

on Windows
----------

On Windows, you need to add an additional channel, ``pytorch``.
You can do this by repeating the `-c` option more than once.

E.g, to install ``vak`` into an existing environment on Windows:

.. code-block:: powershell

   (existing-env) C:\Users\You> conda install vak -c conda-forge -c pytorch


with ``pip``
============

To install ``vak`` with ``pip``, run:

.. code-block:: shell

   (.venv) /home/you/code/ $ pip install vak

Installing a neural network model
=================================

Finally you'll want to install a neural network model to train!
Currently this is done with ``pip``. You can use ``pip`` inside a ``conda`` environment.

.. code-block:: shell

   (vak-env) /home/you/code/ $ pip install tweetynet


.. _why-conda:

Why do we recommend ``conda``?
==============================

Many libraries for data science packages have dependencies
written in languages besides Python. There are platforms
dedicated to managing these dependencies that you may find it easier to use.
For example, Anaconda (https://www.anaconda.com/download) (which is free).
You can use the ``conda`` command-line tool that they develop
to create environments and install the libraries that this package
depends on. Here is an in-depth look at using `conda` to manage environments:
https://www.freecodecamp.org/news/why-you-need-python-environments-and-how-to-manage-them-with-conda-85f155f4353c/.
Further detail about `conda` and how it relates to other tools like
`virtualenv` and `pip` can be found in this blog post:
https://jakevdp.github.io/blog/2016/08/25/conda-myths-and-misconceptions/.

``vak`` depends on several widely-used libraries from the Python data science ecosystem.
Currently, the easiest way to install these libraries across operating systems
(Linux, Mac, and Windows) is to use the ``conda`` tool.
It will help you create what is a called a "virtual environment",
and then install the libraries that ``vak`` depends on into that environment.
The easiest way to use ``conda`` is to install the
Anaconda platform (https://www.anaconda.com/download) (which is free).

The main reason we use ``conda`` to install other dependencies,
instead of letting ``pip`` handle that,
is because ``conda`` makes it easier to work with GPUs.
For example, using ``conda`` avoids the need to install and configure drivers for NVIDIA.
In contrast, if you install just with ``pip``, and you are using a GPU,
you may need to ensure that the installation of ``PyTorch`` is using the system install of binaries
that it depends on, such as ``libcuda.so``.
``conda`` makes it possible to install ``cudatoolkit`` and other dependencies into a virtual environment
to avoid dealing with system-wide installs of binaries.

.. _why-virtualenv:

Why use a virtual environment?
==============================
Virtual environments makes it possible to install the software libraries that
a program depends on, known as "dependencies", so that
they can be isolated from the dependencies of other programs.
This avoids many issues, like when two programs depend on two
different versions of the same library.
For an in-depth explanation of virtual environments, please see this
`guide from the Turing Way <https://the-turing-way.netlify.app/reproducible-research/renv.html>`_.
For a Python specific guide, see https://realpython.com/python-virtual-environments-a-primer/ or
https://dev.to/bowmanjd/python-tools-for-managing-virtual-environments-3bko.
