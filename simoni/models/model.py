import abc
import torch.nn as nn


class Model(nn.Module):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def forward(self, *input):
        raise NotImplementedError

    @abc.abstractmethod
    def get_metrics(self):
        raise NotImplementedError
