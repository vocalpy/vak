import torch

import vak.models.registry


# ---- mock networks ---------------------------------------------------------------------------------------------------
class MockNetwork(torch.nn.Module):
    """Network used just to test vak.models.base.Model"""
    def __init__(self, n_classes=10):
        super().__init__()
        # set these attributes so we can easily test config works
        self.n_classes = n_classes
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(100, 50),
            torch.nn.Linear(50, 25),
            torch.nn.Linear(25, n_classes),
        )

    def forward(self, x):
        return self.layers(x)


class MockEncoder(torch.nn.Module):
    """Network used for testing.

    This network is put into a
    ``dict`` with ``MockDecoder`` to test
    that specifying ``network`` as a ``dict`` works.
    """
    def __init__(self, input_size=100, hidden_size=50):
        super().__init__()
        # set these attributes so we can easily test config works
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, x):
        return self.layers(x)


class MockDecoder(torch.nn.Module):
    """Network used just to test vak.models.base.Model.
    Unlike ``MockNetwork``, this network will be put into a
    ``dict`` with ``MockDecoder`` to test
    that specifying ``network`` as a ``dict`` works.
    """
    def __init__(self, hidden_size=50, output_size=10):
        super().__init__()
        # set these attributes so we can easily test config works
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.layers(x)


# ---- mock metrics ----------------------------------------------------------------------------------------------------
class MockAcc:
    """Mock metric used for testing"""
    def __init__(self, average='macro'):
        self.average = average

    def __call__(self, y: torch.Tensor, y_pred: torch.Tensor):
        sample_acc = y == y_pred
        if self.average == 'macro':
            return sample_acc.mean()
        elif self.average == 'micro':
            return NotImplemented


# ---- mock model families ---------------------------------------------------------------------------------------------
class UnregisteredMockModelFamily(vak.models.Model):
    """A model family defined only for tests.
    Used to test :func:`vak.models.registry.model_family`.
    """
    def __init__(self, network, optimizer, loss, metrics):
        super().__init__(
            network=network, loss=loss, optimizer=optimizer, metrics=metrics
        )

    def training_step(self, *args, **kwargs):
        pass

    def validation_step(self, *args, **kwargs):
        pass

    @classmethod
    def from_config(cls, config: dict):
        """Return an initialized model instance from a config ``dict``."""
        network, loss, optimizer, metrics = cls.attributes_from_config(config)
        return cls(
            network=network,
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
        )


# Make a "copy" of UnregisteredModelFamily that we *do* register
# so we can use it to test `vak.models.decorator.model` and other functions
# that require a registered ModelFamily.
# Used when testing :func:`vak.models.decorator.model` -- we need a model in the registry to test
# and we don't want to have to deal with the idiosyncrasies of actual model families
MockModelFamily = type('MockModelFamily',
                       UnregisteredMockModelFamily.__bases__,
                       dict(UnregisteredMockModelFamily.__dict__))
vak.models.registry.model_family(MockModelFamily)


# ---- mock models -----------------------------------------------------------------------------------------------------
class MockModel:
    """Model definition used for testing :func:`vak.models.decorator.model`"""
    network = MockNetwork
    loss = torch.nn.CrossEntropyLoss
    optimizer = torch.optim.SGD
    metrics = {'acc': MockAcc}
    default_config = {
        'optimizer': {'lr': 0.003}
    }


class MockEncoderDecoderModel:
    """Model definition used for testing :func:`vak.models.decorator.model`.
    Specifically tests that `network` works with a ``dict``"""
    network = {'MockEncoder': MockEncoder, 'MockDecoder': MockDecoder}
    loss = torch.nn.TripletMarginWithDistanceLoss
    optimizer = torch.optim.Adam
    metrics = {
        'loss': torch.nn.TripletMarginWithDistanceLoss
    }
    default_config = {
        'optimizer': {'lr': 0.003}
    }


# pytest.mark.parametrize vals for test_init_with_definition
class OtherNetwork(torch.nn.Module):
    pass


class OtherOptimizer(torch.optim.Optimizer):
    pass


def other_loss_func(targets, y_pred):
    return


other_metrics_dict = {
    'other_metric': lambda x: x
}
