import logging
from copy import deepcopy
from typing import Any, Optional, Union, List, Dict

import tabulate
from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import LoggerCollection
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.tuner.lr_finder import _LRFinder
from torch.utils.data import DataLoader

from speech_recognition.modules.metrics import plot_confusion_matrix


class CrossValidationTrainer:
    # Copied or Adapted from:
    # https://github.com/PyTorchLightning/pytorch-lightning/issues/839

    def __init__(self, *args, **kwargs):
        self.trainer = Trainer(*args, **kwargs)
        self.fast_dev_run = True
        self.overall_test_results_array = []
        self.current_fold = None
        self.class_names = None

    @staticmethod
    def _update_logger(logger, fold_idx):
        if hasattr(logger, 'experiment_name'):
            logger_key = 'experiment_name'
        elif hasattr(logger, 'name'):
            logger_key = 'name'
        else:
            raise AttributeError('The logger associated with the trainer '
                                 'should have an `experiment_name` or `name` '
                                 'attribute.')
        new_experiment_name = getattr(logger, logger_key) + f'/{fold_idx}'
        try:
            setattr(logger, logger_key, new_experiment_name)
        except AttributeError:
            pass

    def update_logger(self, trainer, fold_idx):
        if not isinstance(trainer.logger, LoggerCollection):
            _loggers = [trainer.logger]
        else:
            _loggers = trainer.logger

        # Update loggers:
        for _logger in _loggers:
            self._update_logger(_logger, fold_idx)

    @staticmethod
    def update_modelcheckpoint(model_ckpt_callback, fold_idx):
        _default_filename = '{epoch}-{step}'
        _suffix = f'_fold{fold_idx}'
        if model_ckpt_callback.filename is None:
            new_filename = _default_filename + _suffix
        else:
            new_filename = model_ckpt_callback.filename + _suffix
        setattr(model_ckpt_callback, 'filename', new_filename)

    def tune(
        self,
        model: LightningModule,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
        datamodule: Optional[LightningDataModule] = None,
        scale_batch_size_kwargs: Optional[Dict[str, Any]] = None,
        lr_find_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Optional[Union[int, _LRFinder]]]:
        pass

    @staticmethod
    def get_metric_table_from_test_metrics(test_metrics):
        metric_table = []
        for name, metric in test_metrics.items():
            metric_table.append((name, metric.compute().item()))
        return metric_table

    def test_end_callback(self, caller, test_metrics):
        metric_table = self.get_metric_table_from_test_metrics(test_metrics)

        logging.info(f"\nFold: {self.current_fold} "
                     f"Test Metrics:\n{tabulate.tabulate(metric_table)}")

        confusion_matrix = caller.test_confusion.compute()
        caller.test_confusion.reset()

        if self.class_names is None:
            self.class_names = caller.get_class_names()

        confusion_plot = plot_confusion_matrix(
            confusion_matrix.cpu().numpy(), caller.get_class_names()
        )

        confusion_plot.savefig(f"fold_{self.current_fold}_test_confusion.png")
        confusion_plot.savefig(f"fold_{self.current_fold}_test_confusion.pdf")

        # roc_fpr, roc_tpr, roc_thresholds = self.test_roc.compute()
        caller.test_roc.reset()
        self.overall_test_results_array += [(metric_table,
                                             confusion_matrix.cpu().numpy())]

    def overall_test_results(self):
        accum_table = None
        overall_confusion = None
        count_folds = len(self.overall_test_results_array)
        for metric_table, confusion_matrix in self.overall_test_results_array:
            if accum_table is None:
                accum_table = metric_table
            else:
                for i in range(len(accum_table)):
                    name, metric = accum_table[i]
                    iter_name, iter_metric = metric_table[i]
                    assert name == iter_name
                    accum_table[i] = (name, metric + iter_metric)

            if overall_confusion is None:
                overall_confusion = confusion_matrix
            else:
                overall_confusion += confusion_matrix

        overall_table = []
        for name, metric in accum_table:
            overall_table += [(name, metric / count_folds)]  # Mean value

        logging.info(f"Overall Test Metrics:"
                     f"\n{tabulate.tabulate(overall_table)}")

        confusion_plot = plot_confusion_matrix(
            overall_confusion, self.class_names
        )

        confusion_plot.savefig(f"test_confusion.png")
        confusion_plot.savefig(f"test_confusion.pdf")

    # Do all in fit
    def fit(
        self,
        model: LightningModule,
        train_dataloader: Any = None,
        val_dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
        datamodule: Optional[LightningDataModule] = None,
    ) -> None:
        working_model = deepcopy(model)
        working_model.setup(None)
        working_trainer = deepcopy(self.trainer)
        working_trainer.tune(working_model)
        loader_model = deepcopy(working_model)
        for fold, (train_loader, val_loader, test_loader) \
            in enumerate(zip(loader_model.train_dataloader(),
                             loader_model.val_dataloader(),
                             loader_model.test_dataloader())):
            fold += 1  # We want natural enumeration
            self.current_fold = fold
            model_copy = deepcopy(model)
            model_copy.register_test_end_callback_function(self.test_end_callback)
            trainer_copy = deepcopy(working_trainer)
            self.update_logger(trainer_copy, fold)
            for callback in trainer_copy.callbacks:
                if isinstance(callback, ModelCheckpoint):
                    self.update_modelcheckpoint(callback, fold)
            trainer_copy.fit(model_copy,
                             train_dataloader=train_loader,
                             val_dataloaders=val_loader)
            trainer_copy.validate(model_copy, val_dataloaders=val_loader)
            trainer_copy.test(model_copy, test_dataloaders=test_loader)

        self.overall_test_results()

        # TODO: Summaries
        # TODO: Show fold in logger
        # TODO: num_fold in right place

    def validate(
        self,
        model: Optional[LightningModule] = None,
        val_dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
        ckpt_path: Optional[str] = 'best',
        verbose: bool = True,
        datamodule: Optional[LightningDataModule] = None,
    ):
        pass

    def test(
        self,
        model: Optional[LightningModule] = None,
        test_dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
        ckpt_path: Optional[str] = 'best',
        verbose: bool = True,
        datamodule: Optional[LightningDataModule] = None,
    ):
        pass
