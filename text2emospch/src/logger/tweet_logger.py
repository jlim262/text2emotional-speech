
import os
import logging
import torch
import numpy as np

from sklearn import metrics
from built.logger import LoggerBase
from built.registry import Registry


@Registry.register(category="hooks")
class BaseLogger(LoggerBase):
    def log_extras(self, inputs: any, targets: any, outputs: any):
        pass
