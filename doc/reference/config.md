(config)=
# Configuration files

This document contains the specification
for the `.toml` configuration files used
when running `vak` commands through the command-line interface,
as described in {ref}`cli`.

A `.toml` configuration file is split up into sections.
The sections and their valid options
are represented in the `vak` code
by classes.
To ensure that the code and this documentation
do not go out of sync,
the options are presented below
exactly as documented in the code
for each class.

## Valid section names

Following is the set of valid section names:
`{eval, learncurve, predict, prep, train}`.
In the code, these names correspond to attributes
of the main `Config` class, as shown below.

The only other valid section name
is the name of a class
representing a neural network.
For such sections to be recognized as valid,
the model must be installed via the `vak.models`
entry point, so that it can be recognized by the function
`vak.config.validators.is_valid_model_name`.

```{eval-rst}
.. autoclass:: vak.config.config.Config
```

## Valid Options by Section

Each section of the `.toml` config
has a set of option names
that are considered valid.
Valid options for each section are presented below.

(ref-config-prep)=
### `[vak.prep]` section

```{eval-rst}
.. autoclass:: vak.config.prep.PrepConfig
```

(ref-config-spect-params)=
### `[vak.prep.spect_params]` section

```{eval-rst}
.. autoclass:: vak.config.spect_params.SpectParamsConfig
```

(ref-config-train)=
### `[vak.train]` section

```{eval-rst}
.. autoclass:: vak.config.train.TrainConfig
```

(ref-config-eval)=
### `[vak.eval]` section

```{eval-rst}
.. autoclass:: vak.config.eval.EvalConfig
```

(ref-config-predict)=
### `[vak.predict]` section

```{eval-rst}
.. autoclass:: vak.config.predict.PredictConfig
```

(ref-config-learncurve)=
### `[vak.learncurve]` section

```{eval-rst}
.. autoclass:: vak.config.learncurve.LearncurveConfig
```
