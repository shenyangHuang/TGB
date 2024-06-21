"""
An Early Stopping Module
"""
import os
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

class EarlyStopMonitor(object):
    
    def __init__(self, save_model_dir: str, save_model_id: str, 
                tolerance: float=1e-10, patience: int=5,
                higher_better: bool=True):
        r"""
        Early Stopping Monitor
        :param: save_model_path: strc, where to save the model
        :param: save_model_id: str, an id to save the model with
        :param: tolerance: float, the amount of tolerance of the early stopper
        :param: patience: int, how many round to wait
        :param: higher_better: whether higher_value of the a metric is better
        """
        self.tolerance = tolerance
        self.patience = patience
        self.higher_better = higher_better
        self.counter = 0
        self.best_sofar = None
        self.best_epoch = 0
        self.epoch_idx = 1

        self.save_model_dir = save_model_dir
        if not os.path.exists(self.save_model_dir):
            os.mkdir(self.save_model_dir)
            print('INFO: Create directory {}'.format(save_model_dir))
        Path(self.save_model_dir).mkdir(parents=True, exist_ok=True)
        self.save_model_id = save_model_id

    def get_best_model_path(self):
        r"""
        return the path of the best model
        """
        return self.save_model_dir + '/{}.pth'.format(self.save_model_id)
    
    def step_check(self, curr_metric: float, models_dict: dict):
        r"""
        execute the early stop strategy
        :param: metric: a metric to evaluate the early stopping on
        :param: models_dict: a dictionary containing all models to be saved
        """
        if not self.higher_better:
            curr_metric *= -1
        
        if (self.best_sofar is None) or ((curr_metric - self.best_sofar) / np.abs(self.best_sofar) > self.tolerance):
            # first iteration or observing an improvement
            self.best_sofar = curr_metric
            print("INFO: save a checkpoint...")
            self.save_checkpoint(models_dict)
            self.counter = 0
            self.best_epoch = self.epoch_idx
        else:
            # no improvement observed
            self.counter += 1
        
        self.epoch_idx += 1
        
        return self.counter >= self.patience
    
    def save_checkpoint(self, models_dict: dict):
        r"""
        save models as a checkpoint
        :param: models_dict: a dictionary containing all models to be saved 
        """
        model_path = self.get_best_model_path()
        print("INFO: save the model to {}".format(model_path))
        model_names = list(models_dict.keys())
        model_components = list(models_dict.values())
        torch.save({model_names[i]: model_components[i].state_dict() for i in range(len(model_names))}, 
                    model_path)

    def load_checkpoint(self, models_dict: dict):
        r"""
        save models from the checkpoint
        :param: models_dict: a dictionary containing all models
        """
        model_path = self.get_best_model_path()
        print("INFO: load the model of epoch {} from {}".format(self.best_epoch, model_path))
        checkpoint = torch.load(model_path)
        for model_name, model in models_dict.items():
            model.load_state_dict(checkpoint[model_name])
        


        