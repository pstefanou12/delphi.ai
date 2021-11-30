'''
Parent class for models to train in delphi trainer.
'''


import torch as ch
from torch.optim import SGD, Adam, lr_scheduler
import numpy as np

from abc import ABC, abstractmethod


# CONSTANTS 
ADAM = 'adam'
CYCLIC = 'cyclic'
COSINE = 'cosine'
LINEAR = 'linear'


class delphi:
    '''
    Parent/abstract class for models to be passed into trainer.
    '''
    def __init__(self, args, custom_lr_multiplier: str, lr_interpolation:str, step_lr:int, step_lr_gamma: float, store=None, table=None, schema=None): 
        '''
        Args: 
            args (cox.utils.Parameters) : parameter object holding hyperparameters
        '''
        super().__init__()
        self.args = args
        self.store, self.table, self.schema = store, table, schema
        # if store provided, create logs table for experiment
        if self.store is not None:
            self.store.add_table(self.table, self.schema)
        self.writer = self.store.tensorboard if self.store else None
        self.params = None
        # algorithm optimizer and scheduler
        self.optimizer, self.schedule = None, None
        self.checkpoint = None
        self.M = self.args.steps if self.args.steps else self.args.epochs

        # set attribute for learning rate scheduler
        if custom_lr_multiplier: 
            self.args.__setattr__('custom_lr_multiplier', custom_lr_multiplier)
            if lr_interpolation: 
                self.args.__setattr__('lr_interpolation', lr_interpolation)
        else: 
            self.args.__setattr__('step_lr', step_lr)
            self.args.__setattr__('step_lr_gamma', step_lr_gamma)

    def make_optimizer_and_schedule(self):
        """
        Create optimizer (ch.nn.optim) and scheduler (ch.nn.optim.lr_scheduler module)
        for SGD procedure. 
        """
        if self.model is None and self.params is None: raise ValueError('need to inititalize model of update params')
        # initialize optimizer, scheduler, and then get parameters
        # param_list = self.model.parameters() if self.params is None else self.params
        # setup optimizer
        if self.args.custom_lr_multiplier == ADAM:  # adam
            self.optimizer = Adam(self.parameters, lr=self.args.lr, weight_decay=self.args.weight_decay)
        else: 
            # SGD optimizer
            self.optimizer = SGD(self.parameters, self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)

            # setup learning rate scheduler
            if self.args.custom_lr_multiplier == CYCLIC and self.M is not None: # cyclic
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

    def pretrain_hook(self):
        '''
        Hook called before training procedure begins.
        '''
        pass 

    def train_step(self, i, batch):
        '''
        Training step for defined model.
        Args: 
            i (int) : gradient step or epoch number
            batch (Iterable) : iterable of inputs that 
        '''
        pass 

    def val_step(self, i, batch):
        '''
        Valdation step for defined model. 
        '''
        pass 

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

    def epoch_hook(self, i, loop_type, loss, prec1, prec5, batch):
        '''
        Epoch hook for defined model. Method is called after each 
        complete iteration through dataset.
        '''
        pass 

    def post_training_hook(self, val_loader):
        '''
        Post training hook, called after sgd procedures completes. By default returns True, 
        so that procedure terminates by default after one trial.
        '''
        return True

    def description(self, epoch, i, loop_msg, loss, prec1, prec5):
        '''
        Returns string description for model at each iteration.
        '''
        return '{} Epoch: {} | Loss: {} | Train Prec 1:  {} | Train Prec5: {} ||'.format(epoch, loop_msg, round(float(loss.avg), 4), round(float(prec1.avg), 4), round(float(prec5.avg), 4))

    @property 
    def val_loader(self): 
        '''
        Default validation loader property.
        '''
        return None

    @property
    def parameters(self): 
        if self.params: 
            return self.params 
        return self.model.parameters()
