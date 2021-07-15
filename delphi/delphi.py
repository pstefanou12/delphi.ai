'''
Parent class for models to train in delphi trainer.
'''


import torch as ch
from torch.optim import SGD, Adam, lr_scheduler
import numpy as np

from abc import ABC, abstractmethod


# CONSTANTS 
CYCLIC = 'cyclic'
COSINE = 'cosine'
LINEAR = 'linear'

class delphi(ch.nn.Module, ABC):
    '''
    Parent/abstract class for models to bas passed into trainer.
    '''
    def __init__(self, args, model, store=None, table=None, schema=None): 
        '''
        Args: 
            args (cox.utils.Parameters) : parameter object holding hyperparameters
        '''
        super().__init__()
        self.args = args
        self.model = model
        self.store, self.table, self.schema = store, table, schema
        # if store provided, create logs table for experiment
        if self.store is not None:
            self.store.add_table(self.table, self.schema)
        self.writer = self.store.tensorboard if self.store else None
        # algorithm optimizer and scheduler
        self.optimizer, self.scheduler = None, None

    def make_optimizer_and_schedule(self):
        """
        Create optimizer (ch.nn.optim) and scheduler (ch.nn.optim.lr_scheduler module)
        for SGD procedure. 
        """
        # initialize optimizer, scheduler, and then get parameters
        param_list = self.model.parameters() if self.update_params is None else self.params

        # check for Adam optimizer
        if self.args.adam: 
            self.optimizer = Adam(param_list, betas=self.args.betas, lr=self.args.lr, weight_decay=self.args.weight_decay, 
            amsgrad=self.args.amsgrad)
        else: 
            # SGD optimizer
            self.optimizer = SGD(param_list, self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
            # cyclic learning rate scheduler
            if self.args.custom_lr_multiplier == CYCLIC and self.M is not None:
                lr_func = lambda t: np.interp([t], [0, self.M*4//15, self.M], [0, 1, 0])[0]
                self.schedule = lr_scheduler.LambdaLR(self.optimizer, lr_func)
            # cosine annealing scheduler
            elif self.args.custom_lr_multiplier == COSINE and self.M is not None:
                schedule = lr_scheduler.CosineAnnealingLR(self.optimizer, self.M)
            elif self.args.custom_lr_multiplier:
                cs = self.args.custom_lr_multiplier
                periods = eval(cs) if type(cs) is str else cs
                # constant linear interpolation
                if self.args.lr_interpolation == LINEAR:
                    lr_func = lambda t: np.interp([t], *zip(*periods))[0]
                # custom lr interpolation
                else:
                    def lr_func(ep):
                        for (milestone, lr) in reversed(periods):
                            if ep >= milestone: return lr
                        return 1.0
                self.schedule = lr_scheduler.LambdaLR(self.optimizer, lr_func)
            # step learning rate
            elif self.args.step_lr:
                self.schedule = lr_scheduler.StepLR(self.optimizer, step_size=self.args.step_lr, 
                gamma=self.args.step_lr_gamma, verbose=self.args.verbose)
            
        # if checkpoint load  optimizer and scheduler
        if self.checkpoint:
            self.optimizer.load_state_dict(self.checkpoint['optimizer'])
            try:
                schedule.load_state_dict(checkpoint['schedule'])
            # if can't load scheduler state, take epoch steps
            except:
                steps_to_take = checkpoint['epoch']
                print('Could not load schedule (was probably LambdaLR).'
                    f' Stepping {steps_to_take} times instead...')
                for i in range(steps_to_take):
                    self.schedule.step()

    @abstractmethod
    def pretrain_hook(self):
        '''
        Hook called before training procedure begins.
        '''
        pass 

    @abstractmethod
    def train_step(self, i, batch):
        '''
        Training step for defined model.
        Args: 
            i (int) : gradient step or epoch number
            batch (Iterable) : iterable of inputs that 
        '''
        pass 

    @abstractmethod
    def val_step(self, i, batch):
        '''
        Valdation step for defined model. 
        '''
        pass 

    @abstractmethod
    def iteration_hook(self, i, loop_type, loss, prec1, prec5, batch):
        '''
        Iteration hook for defined model. Method is called after each 
        training update.
        Args:
            loop_type (str) : 'train' or 'val'; indicating type of loop
            loss (ch.Tensor) : loss for that iteration
            prec1 (float) : accuracy for top prediction
            prec5 (float) : accuracy for top-5 predictions
        '''
        pass 

    @abstractmethod
    def epoch_hook(self, i, loop_type, loss, prec1, prec5, batch):
        '''
        Epoch hook for defined model. Method is called after each 
        complete iteration through dataset.
        '''
        pass 

    @abstractmethod 
    def post_train_hook(self):
        '''
        Post training hook, called after sgd procedures completes. 
        '''
        pass
