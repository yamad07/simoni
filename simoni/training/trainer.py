import torch
from typing import Dict
import logging
from simoni.models import Model

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self,
            models: Dict[str, Model],
            optimizers: Dict[str, torch.optim.Optimizer],
            verbose: bool,
            save_model_path : str ='./models',
            save_model_interval: int = 10
            ):

        self.models = models
        self.optimizers = optimizers
        self.verbose = verbose
        self.save_model_path = save_model_path
        self.save_model_interval = save_model_interval

    def train(self, n_epochs: int):
        for epoch in n_epochs:
            logger.info('Epoch: {}/{}'.format(epoch, n_epochs))
            self._train_epoch(epoch)

    def _train_epoch(self, epoch: int):

        train_dataloader = data.DataLoader(self.train_data,
                batch_size=self.batch_size,
                shuffle=True)

        self.model.train()
        self._init_optims()

        for batch in train_dataloader:

            self._train_step(batch)

            if self.verbose:
                self._logging_metrics(self.model)

            if epoch % self.save_model_interval == 0:
                self._save_models(epoch)

    def _init_optims():
        for optim in self.optimizers.values:
            optim.zero_grad()

    def _train_step(self, batch):
        output = self.model(batch)
        loss = output['loss']
        loss.backward()
        self.optimizers['optim'].step()

        return loss.item()

    def _logging_metrics(self, output):
        for model_name, model in self.models.items():
            logger.info('Metrics of {}'.format(model_name))
            metrics = model.get_metrics()
            for metrics_name, value in metrics.items():
                logger.info('{}: {}'.format(metrics_name, value))
        return self.model.get_metrics()

    def _save_models(self, epoch):
        for model_name, model in self.models.items():
            torch.save(model.state_dict(), os.path.join(self.save_model_path, '{}_{}.pth'.format(model_name, epoch)))
