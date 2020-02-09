from collections import defaultdict

import torch
import torch.nn.modules.loss
import torch.optim
from tqdm import tqdm

from ..util.general import get_default_device


class Model:
    """lightweight model class that adds methods for training and evaluation
    to PyTorch neural networks.

    Sets forth a consistent API that models should implement to work with vak.

    Class Attributes
    ----------------
    network : torch.nn.Module
        subclass that implements a neural network
    loss : callable
        that implements a loss function, often called "criterion" in PyTorch code.
    optimizer : callable
        that optimizes model by iteratively updating trainable parameters.
    metrics : dict
        where keys are metric names, and values are callables that compute that a metric,
        e.g. accuracy. Metrics should accept arguments y_pred and y_true.

    Attributes
    ----------
    device : str
        device on which to place tensors. One of {"cuda", "cpu}.

    Methods
    -------
    fit : fit a model by training it with supplied data for a specified number of epochs
    evaluate : evaluate a model by computing specified metrics on supplied data
    predict : return predictions of model, i.e. output when fed with supplied data
    compile : returns instance of model with attributes set to specified arguments

    Private Methods
    ---------------
    _train : helper method, called by the fit method on each epoch.
        Fits the model by iterating once through all the training data.
        Override this method if you need to implement your own training method.
    _eval : helper method, called by the evaluate method, and called by the fit
        method for validation after each epoch. Evaluates the model by iterating
        through eval_data and computing the specified metrics for each batch.
        Override this method if you need to implement your own validation and test method.
    _predict : helper method, called by the predict method on each epoch.
        Uses the model to make predictions, by iterating through pred_data
        and returning the outputs of each batch fed into it.
        Override this method if you need to implement your own predict method.
    """

    REQUIRED_SUBCLASS_ATTRIBUTES = [
        'network',
        'optimizer',
        'loss',
        'metrics',
    ]

    def __init__(self,
                 network,
                 loss,
                 optimizer,
                 metrics):
        self.network = network
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.device = None

    def _train(self, train_data):
        """helper method, called by the fit method on each epoch.
        Iterates once through train_data, using it to update model parameters.
        Override this method if you need to implement your own training method.

        Parameters
        ----------
        train_data : torch.util.Dataloader
            instance that will be iterated over.
        """
        self.network.train()

        progress_bar = tqdm(train_data)
        for ind, batch in enumerate(progress_bar):
            x, y = batch[0].to(self.device), batch[1].to(self.device)
            y_pred = self.network.forward(x)
            self.optimizer.zero_grad()
            loss = self.loss(y_pred, y)
            loss.backward()
            self.optimizer.step()
            progress_bar.set_description(
                f'batch {ind}, loss: {loss.item():.4f}'
            )

    def _eval(self, eval_data):
        """helper method, called by the evaluate method, and called by the fit
        method for validation after each epoch. Evaluates the model by iterating
        through eval_data and computing the specified metrics for each batch.
        Override this method if you need to implement your own validation and test method.

        Parameters
        ----------
        eval_data : torch.util.Dataloader
            instance that will be iterated over.
        """
        self.network.eval()

        metric_vals = defaultdict(list)

        n_batches = 0

        progress_bar = tqdm(eval_data)
        with torch.no_grad():
            for ind, batch in enumerate(progress_bar):
                x, y = batch[0].to(self.device), batch[1].to(self.device)
                y_pred = self.network.forward(x)
                for metric_name, metric_callable in self.metrics.items():
                    metric_vals[metric_name].append(
                        metric_callable(y_pred, y)
                    )
                n_batches += 1
                progress_bar.set_description(
                    f'batch {ind} / {len(eval_data)}'
                )

        for metric_name in metric_vals.keys():
            if metric_name in ['loss', 'acc']:
                avg_metric_val = (
                        torch.tensor(metric_vals[metric_name]).sum() / n_batches
                ).item()
                metric_vals[metric_name] = avg_metric_val
            else:
                raise NotImplementedError(
                    f'calculation of metric across batches not yet implemented for {metric_name}'
                )

        return metric_vals

    def _predict(self, pred_data):
        """helper method, called by the predict method on each epoch.
        Uses the model to make predictions, by iterating through pred_data
        and returning the outputs of each batch fed into it.
        Override this method if you need to implement your own predict method.

        Parameters
        ----------
        pred_data : torch.util.Dataloader
            instance that will be iterated over.
        """
        self.network.eval()

        y_pred_all = []

        progress_bar = tqdm(pred_data)

        with torch.no_grad():
            for ind, batch in enumerate(progress_bar):
                x, y = batch[0].to(self.device), batch[1].to(self.device)
                y_pred = self.network.forward(x)
                y_pred_all.append(y_pred)

                progress_bar.set_description(
                    f'batch {ind} / {len(pred_data)}'
                )

        return torch.cat(y_pred_all)

    def save(self, ckpt_path, epoch, **kwargs):
        """save model state to a checkpoint file.

        Saves epoch, network state_dict, optimizer state_dict, and
        any keyword arguments to a checkpoint file with the name
        specified by ckpt_path.

        Parameters
        ----------
        ckpt_path : str, Path
            path including filename that should be used to save checkpoint
        epoch : int
            epoch on which this checkpoint was saved
        """
        ckpt = {
            'epoch': epoch,
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        ckpt.update(**kwargs)
        torch.save(ckpt, ckpt_path)

    def load(self, ckpt_path):
        """load model state dict from a checkpoint file.

        Loads state_dicts into network and optimizer
        from a checkpoint file with the name
        specified by ckpt_path.

        Parameters
        ----------
        ckpt_path : str, Path
            path including filename from which to load checkpoint
        """
        ckpt = torch.load(ckpt_path)
        self.network.load_state_dict(ckpt['network_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    def fit(self,
            train_data,
            num_epochs,
            ckpt_root,
            val_data=None,
            val_step=1,
            checkpoint_step=1,
            save_best_only=True,
            patience=None,
            single_ckpt=True,
            device=None
            ):
        if save_best_only:
            if val_data is None:
                raise ValueError(
                    'save_best_only is True but no validation dataset was provided to measure accuracy'
                )

        if patience is not None:
            if val_data is None:
                raise ValueError(
                    f'patience set to {patience}, but no validation dataset was provided to measure accuracy'
                )

        if device is None:
            device = get_default_device()
        self.device = device

        if save_best_only or patience:
            max_val_acc = 0

        if patience:
            patience_counter = 0

        self.network.to(self.device)

        for epoch in range(1, num_epochs + 1):
            print(f'epoch {epoch} / {num_epochs}')
            self._train(train_data)

            if single_ckpt:
                ckpt_path = ckpt_root.joinpath('checkpoint.pt')
            else:
                ckpt_path = ckpt_root.joinpath(f'checkpoint_epoch{epoch}.pt')

            if val_data is not None:
                if epoch % val_step == 0:
                    print('computing metrics on validation set')
                    metric_vals = self._eval(val_data)
                    print(
                        ', '.join([f'{k}: {v:.4f}' for k, v in metric_vals.items()])
                    )

                    if patience or epoch % checkpoint_step == 0:
                        epoch_acc = metric_vals['acc']
                        if patience:
                            if epoch_acc > max_val_acc:
                                max_val_acc = metric_vals['acc']
                                patience_counter = 0
                                print(
                                    f'accuracy improved, saving checkpoint'
                                )
                                self.save(ckpt_path, epoch)
                            else:
                                patience_counter += 1
                                if patience_counter > patience:
                                    print(
                                        f'early stopping, validation accuracy has not improved in {patience} epochs')
                                    if not save_best_only:
                                        self.save(ckpt_path, epoch)
                                break
                        elif epoch % checkpoint_step == 0:
                            if save_best_only:
                                if epoch_acc < max_val_acc:
                                    continue
                            self.save(ckpt_path, epoch)

            elif epoch % checkpoint_step == 0:  # but we don't have validation data
                self.save(ckpt_path, epoch)

    def evaluate(self,
                 eval_data,
                 device=None):
        if device is None:
            device = get_default_device()
        self.device = device
        self.network.to(self.device)
        return self._eval(eval_data)

    def predict(self,
                pred_data,
                device=None):
        if device is None:
            device = get_default_device()
        self.device = device
        self.network.to(self.device)
        return self._predict(pred_data)

    @classmethod
    def from_config(cls, config):
        """any model that inherits from this class should do whatever it needs to
        in this factory method to create the network, optimizer, and loss, and
        then pass those to the init function

        Parameters
        ----------
        config : dict
            presumably mapping 'network', 'optimizer' and 'loss' to kwargs that
            determine parameters for each of those, e.g. dropout rate for the
            network, learning rate for the optimizer, etc.

        Returns
        -------
        model : Model
            instance of a model that subclasses this Model
        """
        raise NotImplementedError
