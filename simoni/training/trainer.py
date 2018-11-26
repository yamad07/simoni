import torch
from typing import Dict
import os
import logging
from simoni.models import Model
import torch.utils.data as data
import comet_ml

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self,
            models: Dict[str, Model],
            optimizers: Dict[str, torch.optim.Optimizer],
            verbose: bool = False,
            save_model_path : str = './models',
            save_model_interval: int = 10,
            comet_experiment_module: comet_ml.Experiment = None
            ):

        self.models = models
        self.optimizers = optimizers
        self.verbose = verbose
        self.save_model_path = save_model_path
        self.save_model_interval = save_model_interval
        self.comet_experiment_module = comet_experiment_module

    def train(self, n_epochs: int, train_data: torch.utils.data.Dataset, batch_size: int):
        for epoch in range(n_epochs):
            logger.info('Epoch: {}/{}'.format(epoch, n_epochs))
            self._train_epoch(epoch, train_data, batch_size)

    def _train_epoch(self, epoch: int, train_data: torch.utils.data.Dataset, batch_size: int):

        train_dataloader = data.DataLoader(train_data,
                batch_size=batch_size,
                shuffle=True)

        self._init_optims()
        self._init_models()

        for batch in train_dataloader:

            self._train_step(batch)

        if self.verbose:
           self._logging_metrics_to_std()

        if epoch % self.save_model_interval == 0:
            self._save_models(epoch)

        if self.comet_experiment_module is not None:
            self._logging_metrices_to_comet()

    def _init_optims(self):
        for optim in self.optimizers.values():
            optim.zero_grad()

    def _init_models(self):
        for model in self.models.values():
            model.train()

    def _train_step(self, batch):
        output = self.models['model'](*batch)
        loss = output['loss']
        loss.backward()
        self.optimizers['optim'].step()

        return loss.item()

    def _logging_metrics_to_std(self):
        for model_name, model in self.models.items():
            logger.info('Metrics of {}'.format(model_name))
            metrics = model.get_metrics()
            for metrics_name, value in metrics.items():
                logger.info('{}: {}'.format(metrics_name, value))

    def _logging_metrices_to_comet(self):
        for model_name, model in self.models.items():
            metrics = model.get_metrics()
            for metrics_name, value in metrics.items():
                self.comet_experiment_module.log_metric(metrics_name, value)

    def _save_models(self, epoch):
        for model_name, model in self.models.items():
            torch.save(model.state_dict(), os.path.join(self.save_model_path, '{}_{}.pth'.format(model_name, epoch)))
