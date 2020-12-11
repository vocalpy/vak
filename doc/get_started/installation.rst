.. _installation:

Installation
============

Prerequisites: ``conda`` environment and package manager
--------------------------------------------------------

``vak`` depends on several widely-used libraries from the Python data science ecosystem.
Currently, the easiest way to install these libraries across operating systems
(Linux, Mac, and Windows) is to use the ``conda`` tool.
It will help you create what is a called a "virtual environment",
and then install the libraries that ``vak`` depends on into that environment.
The easiest way to use ``conda`` is to install the
Anaconda platform (https://www.anaconda.com/download) (which is free).
For a more detailed explanation of why you would use a virtual environment, please see
:ref:`why-virtualenv`.

Creating a virtual environment
------------------------------

Here are the steps to create a virtual environment after installing Anaconda.
You should execute these in a terminal by entering the following commands at the prompt
(indicated by the ``$`` below):

  * Create a virtual environment for ``vak``:

    .. code-block:: console

        (base) you@your-computer: ~/Documents $ conda create -n vak-env python=3.6
        (base) you@your-computer: ~/Documents $ conda activate vak-env
        (vak-env) you@your-computer: ~/Documents $

    There's no command in the last line above. It's there simply to show how the prompt changes
    when you activate the environment. The name in parentheses indicates which environment is activated.

  * Install the dependencies using ``conda`` (not ``pip``, because this can cause issues on some platforms)

    .. code-block:: console

        (vak-env) you@your-computer: ~/Documents $ conda install pytorch torchvision cudatoolkit -c pytorch
        (vak-env) you@your-computer: ~/Documents $ conda install attrs dask joblib matplotlib pandas scipy toml tqdm

Once you have created the virtual environment, activated it, and installed the necessary dependencies,
you can install ``vak``.

Installation with ``pip`` package manager
-----------------------------------------

You can use the Python package manager ``pip`` to install ``vak`` into the ``conda`` virtual environment
you created.

.. code-block:: console

    (vak-env)/home/you/code/ $ pip install vak

The main reason we use ``conda`` to install other dependencies, instead of letting ``pip`` handle that,
is because ``conda`` makes it easier to work with GPUs.
For example, using ``conda`` avoids the need to install and configure drivers for NVIDIA.
In contrast, if you install just with ``pip``, and you are using a GPU,
you may need to ensure that the installation of ``PyTorch`` is using the system install of binaries
that it depends on, such as ``libcuda.so``.
``conda`` makes it possible to install ``cudatoolkit`` and other dependencies into a virtual environment
to avoid dealing with system-wide installs of binaries.

Installation from source, e.g. for development
----------------------------------------------
You can also work with a local copy of the code.
It's possible to install the local copy with ``pip`` so that you can still edit
the code, and then have its behavior as an installed library reflect those edits.

When working with ``vak`` this way, you'll most likely still want to isolate
it within a virtual environment using ``conda``.

Here's the steps to set up a development environment:
  * Create a virtual environment for ``vak`` with the necessary dependencies (if you haven't already):

    .. code-block:: console

        you@your-computer: ~/Documents $ conda create -n vak-env python=3.6
        you@your-computer: ~/Documents $ conda activate vak-env
        (vak-env) you@your-computer: ~/Documents $ conda install pytorch torchvision cudatoolkit -c pytorch
        (vak-env) you@your-computer: ~/Documents $ conda install attrs dask joblib matplotlib pandas scipy toml tqdm

  * Clone the repo from Github using the version control tool ``git``:

    .. code-block:: console

        you@your-computer: ~/Documents $ git clone https://github.com/NickleDave/vak

    (you can install ``git`` from `Github <https://help.github.com/en/github/getting-started-with-github/set-up-git>`_,
    with your operating system package manager, or using ``conda``.)

  * In the root of the ``vak`` directory cloned by ``git``, use ``pip`` to install the package,
    making the install ``editable`` with the ``-e`` option.

    .. code-block:: console

        (vak-env) you@your-computer: ~/Documents $ cd vak
        (vak-env) you@your-computer: ~/Documents/vak $ pip install -e .

    Note that ``pip`` may install some other dependencies for development -- that's okay.

Eventually ``vak`` will be available through the conda-forge channel,
meaning you can install with a single command into a ``conda`` environment.
Keep an eye on this issue on GitHub:

| https://github.com/NickleDave/vak/issues/70

.. _why-virtualenv:

Why use a virtual environment?
------------------------------
Virtual environments makes it possible to install the software libraries that
a program depends on, known as "dependencies", so that
they can be isolated from the dependencies of other programs.
This avoids many issues, like when two programs depend on two
different versions of the same library.
For an in-depth explanation of virtual environments, please see
https://realpython.com/python-virtual-environments-a-primer/.

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
