from collections import defaultdict

import torch
import torch.nn.modules.loss
import torch.optim
from tqdm import tqdm

from ..device import get_default as get_default_device
from ..labeled_timebins import lbl_tb2labels
from ..logging import log_or_print


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
                 metrics,
                 logger=None,
                 summary_writer=None,
                 global_step=0):
        self.network = network
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        self.logger = logger
        self.summary_writer = summary_writer
        self.global_step = global_step  # used for summary writer

        # attributes set by fit / _train methods
        self.device = None
        self.ckpt_path = None
        self.max_val_acc = 0
        self.max_val_acc_ckpt_path = None
        self.patience = None
        self.patience_counter = 0

    def _train(self,
               train_data,
               epoch,
               val_data=None,
               val_step=None,
               ckpt_step=None,
               ):
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
                f'Epoch {epoch}, batch {ind}. Loss: {loss.item():.4f}. Global step: {self.global_step}'
            )

            if self.summary_writer is not None:
                self.summary_writer.add_scalar('loss/train', loss.item(), self.global_step)
            self.global_step += 1

            if val_data is not None:
                if self.global_step % val_step == 0:
                    log_or_print(f'Step {self.global_step} is a validation step; computing metrics on validation set',
                                 logger=self.logger, level='info')
                    metric_vals = self._eval(val_data)
                    self.network.train()  # because _eval calls network.eval()
                    log_or_print(msg=', '.join([f'{metric_name}: {metric_value:.4f}'
                                                for metric_name, metric_value in metric_vals.items()
                                                if metric_name.startswith('avg_')]),
                                 logger=self.logger, level='info')

                    if self.summary_writer is not None:
                        for metric_name, metric_value in metric_vals.items():
                            if metric_name.startswith('avg_'):
                                self.summary_writer.add_scalar(f'{metric_name}/val',
                                                               metric_value,
                                                               self.global_step)

                    current_val_acc = metric_vals['avg_acc']
                    if current_val_acc > self.max_val_acc:
                        self.max_val_acc = current_val_acc
                        log_or_print(msg=f'Accuracy on validation set improved. Saving max-val-acc checkpoint.',
                                     logger=self.logger, level='info')
                        self.save(self.max_val_acc_ckpt_path, epoch=epoch, global_step=self.global_step)
                        if self.patience:
                            self.patience_counter = 0
                    else:  # if accuracy did not improve
                        if self.patience:
                            self.patience_counter += 1
                            if self.patience_counter > self.patience:
                                log_or_print(
                                    'Stopping training early, '
                                    f'accuracy has not improved in {self.patience} validation steps.',
                                    logger=self.logger, level='info')
                                # save "backup" checkpoint upon stopping; don't save over "max-val-acc" checkpoint
                                self.save(self.ckpt_path, epoch=epoch, global_step=self.global_step)
                                progress_bar.close()
                                break
                            else:
                                log_or_print(
                                    f'Accuracy has not improved in {self.patience_counter} validation steps. '
                                    f'Not saving max-val-acc checkpoint for this validation step.',
                                    logger=self.logger, level='info')
                        else:  # patience is None. We still log that we are not saving checkpoint.
                            log_or_print(
                                'Accuracy is less than maximum validation accuracy so far. '
                                'Not saving max-val-acc checkpoint.',
                                logger=self.logger, level='info')

            # below can be true regardless of whether we have val_data and/or current epoch is a val_epoch
            if self.global_step % ckpt_step == 0:
                log_or_print(f'Step {self.global_step} is a checkpoint step.',
                             logger=self.logger, level='info')
                self.save(self.ckpt_path, epoch=epoch, global_step=self.global_step)

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
                x, y = batch['source'].to(self.device), batch['annot'].to(self.device)
                # remove "batch" dimension added by collate_fn to x
                # we keep for y because loss still expects the first dimension to be batch
                if x.ndim == 5:
                    if x.shape[0] == 1:
                        x = torch.squeeze(x, dim=0)
                else:
                    raise ValueError(
                        f'invalid shape for x: {x.shape}'
                    )

                out = self.network.forward(x)
                # permute and flatten out
                # so that it has shape (1, number classes, number of time bins)
                # ** NOTICE ** just calling out.reshape(1, out.shape(1), -1) does not work, it will change the data
                out = out.permute(1, 0, 2)
                out = torch.flatten(out, start_dim=1)
                out = torch.unsqueeze(out, dim=0)
                # reduce to predictions, assuming class dimension is 1
                y_pred = torch.argmax(out, dim=1)  # y_pred has dims (batch size 1, predicted label per time bin)

                if 'padding_mask' in batch:
                    padding_mask = batch['padding_mask']  # boolean: 1 where valid, 0 where padding
                    # remove "batch" dimension added by collate_fn
                    # because this extra dimension just makes it confusing to use the mask as indices
                    if padding_mask.ndim == 2:
                        if padding_mask.shape[0] == 1:
                            padding_mask = torch.squeeze(padding_mask, dim=0)
                    else:
                        raise ValueError(
                            f'invalid shape for padding mask: {padding_mask.shape}'
                        )

                    out = out[:, :, padding_mask]
                    y_pred = y_pred[:, padding_mask]

                if (any(['levenshtein' in metric_name for metric_name in self.metrics.keys()]) or
                        any(['segment_error_rate' in metric_name for metric_name in self.metrics.keys()])):
                    y_labels = lbl_tb2labels(y.cpu().numpy(), eval_data.dataset.labelmap)
                    y_pred_labels = lbl_tb2labels(y_pred.cpu().numpy(), eval_data.dataset.labelmap)
                else:
                    y_labels = None
                    y_pred_labels = None

                for metric_name, metric_callable in self.metrics.items():
                    if metric_name == 'loss':
                        metric_vals[metric_name].append(
                            metric_callable(out, y)
                        )
                    elif metric_name == 'acc':
                        metric_vals[metric_name].append(
                            metric_callable(y_pred, y)
                        )
                    elif metric_name == 'levenshtein':
                        metric_vals[metric_name].append(
                            metric_callable(y_pred_labels, y_labels)
                        )
                    elif metric_name == 'segment_error_rate':
                        metric_vals[metric_name].append(
                            metric_callable(y_pred_labels, y_labels)
                        )
                    else:
                        raise NotImplementedError(
                            f'calculation of metric not yet implemented for {metric_name}'
                        )

                n_batches += 1
                progress_bar.set_description(
                    f'batch {ind} / {len(eval_data)}'
                )

        # ---- compute metrics averaged across batches -----------------------------------------------------------------
        # iterate over list of keys, to avoid "dictionary changed size" error when adding average metrics
        for metric_name in list(metric_vals):
            if metric_name in ['loss', 'acc', 'levenshtein', 'segment_error_rate']:
                avg_metric_val = (
                        torch.tensor(metric_vals[metric_name]).sum().cpu().numpy() / n_batches
                ).item()
                metric_vals[f'avg_{metric_name}'] = avg_metric_val
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
        preds = {}
        self.network.eval()

        progress_bar = tqdm(pred_data)

        with torch.no_grad():
            for ind, batch in enumerate(progress_bar):
                x, spect_path = batch['source'].to(self.device), batch['spect_path']
                if isinstance(spect_path, list) and len(spect_path) == 1:
                    spect_path = spect_path[0]
                if x.ndim == 5:
                    if x.shape[0] == 1:
                        x = torch.squeeze(x, dim=0)
                y_pred = self.network.forward(x)
                preds[spect_path] = y_pred
                progress_bar.set_description(
                    f'batch {ind} / {len(pred_data)}'
                )

        return preds

    def save(self, ckpt_path, **kwargs):
        """save model state to a checkpoint file.

        Saves network state_dict, optimizer state_dict, and
        any keyword arguments to a checkpoint file with the name
        specified by ckpt_path.

        Parameters
        ----------
        ckpt_path : str, Path
            path including filename that should be used to save checkpoint
        kwargs :
            keyword arguments; if there are any, they will be added to checkpoint
        """
        ckpt = {
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        ckpt.update(**kwargs)
        log_or_print(
            f'Saving checkpoint at:\n{ckpt_path} ',
            logger=self.logger, level='info')
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
        log_or_print(
            f'Loading checkpoint from:\n{ckpt_path} ',
            logger=self.logger, level='info')
        ckpt = torch.load(ckpt_path)
        self.network.load_state_dict(ckpt['network_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    def fit(self,
            train_data,
            num_epochs,
            ckpt_root,
            val_data=None,
            val_step=None,
            ckpt_step=None,
            patience=None,
            device=None
            ):
        # ---- pre-conditions ----------
        if val_data is None:
            if patience is not None:
                    raise ValueError(
                        f'patience set to {patience}, but no validation dataset was provided to measure accuracy'
                    )
            if val_step is not None:
                raise ValueError(
                    f'val_step set to {val_step}, but no validation dataset was provided to measure accuracy'
                )

        # ---- set attributes ----------
        if device is None:
            device = get_default_device()
        self.device = device

        # note there can be up to two checkpoint paths.
        # this first one is the "backup" checkpoint, saved intermittently (with frequency determined by ckpt_step)
        # and also saved at the end of training
        self.ckpt_path = ckpt_root.joinpath('checkpoint.pt')

        if val_data is not None:
            # this is the second checkpoint path, saved when accuracy improves on the validation set
            self.max_val_acc_ckpt_path = ckpt_root.joinpath('max-val-acc-checkpoint.pt')
            self.max_val_acc = 0

        if patience is not None:
            self.patience = patience
            self.patience_counter = 0

        self.network.to(self.device)

        # ---- actually do fitting ----------
        for epoch in range(1, num_epochs + 1):
            log_or_print(f'epoch {epoch} / {num_epochs}', logger=self.logger, level='info')
            self._train(train_data,
                        epoch,
                        val_data,
                        val_step,
                        ckpt_step)
            if patience is not None:
                if self.patience_counter > self.patience:
                    # need to break here too, not just inside _train function
                    break

        if epoch == num_epochs:  # save at end, if we complete all epochs (not if we stopped because of patience)
            log_or_print('Completed last epoch.', logger=self.logger, level='info')
            self.save(self.ckpt_path, epoch=epoch, global_step=self.global_step)

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
    def from_config(cls, config, logger=None):
        """any model that inherits from this class should do whatever it needs to
        in this factory method to create the network, optimizer, and loss, and
        then pass those to the init function

        Parameters
        ----------
        config : dict
            presumably mapping 'network', 'optimizer' and 'loss' to kwargs that
            determine parameters for each of those, e.g. dropout rate for the
            network, learning rate for the optimizer, etc.
        logger : logging.Logger
            instance returned by vak.logging.get_logger.
            Default is None, in which case messages are just sent to print function.

        Returns
        -------
        model : Model
            instance of a model that subclasses this Model
        """
        raise NotImplementedError
