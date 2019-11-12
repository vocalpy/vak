Automated Annotation
====================

The main use for ``vak`` is to automate annotation of vocalizations.
Using ``vak`` for this task consists of three steps:
- ``prep``aring a training dataset
- ``train``ing a neural network
- and then using the trained network to ``predict`` annotations for other data

As you might guess from the ``monoscript`` typeface, the three steps
correspond to actual commands you will run with ``vak``.

This tutorial walks you through how you would do that, by
changing a few options in configuration files, that you then use to run
``vak`` from the command line. If you're not sure what
is meant by "configuration file" or "command line",
don't worry, it will all be explained in the following sections.

Use of ``vak`` from the command line
------------------------------------
``vak`` uses a command-line interface, meaning you run it from the terminal,
also known as the shell. If you don't have experience with the shell, we
suggest working through this beginner-friendly tutorial from the Carpentries:

https://link.com

Although it might seem a bit daunting at first, you can actually work quite
efficiently in the shell once you get familiar with the cryptic commands.
There's only a handful you need on a regular basis.

Why command line?
~~~~~~~~~~~~~~~~~

A strength of the shell is that it lets you write scripts, so that whatever
you do with data is (more) reproducible. That includes the things you'll do
with your data when you're telling ``vak`` how to use it to train a neural
network. In a machine learning context, you need to reproduce the same steps
when preparing the data you want to apply the trained network to, so you can
predict its annotation.

The ``vak`` command-line interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With those preliminary comments out of the way, we introduce the command-line
interface. Basically any time you run ``vak``, what you type at the prompt
will have the following form:

.. code-block:: console

   $ vak command config.ini

where ``command`` will be an actual command, like ``prep``, and ``config.ini``
will be an actual ``.ini`` file in which you specify the options for the different
commands.
In what follows we introduce

The ``.ini`` files are set up so that each section corresponds to one
of the commands. For example, there is a section called ``[PREP]`` where you
specify options