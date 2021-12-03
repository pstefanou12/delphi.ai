"""
General training format for training models with SGD/backprop.
"""

import time
import os
import warnings
import dill
import numpy as np
import torch as ch
from torch import Tensor
import cox
from typing import Any, Iterable, Callable
from abc import ABC
from time import time

from .delphi import delphi
from . import oracle
from .utils.helpers import has_attr, ckpt_at_epoch, type_of_script, AverageMeter, accuracy, setup_store_with_metadata, ProcedureComplete
from .utils import constants as consts

# CONSTANTS
TRAIN = 'train'
VAL = 'val'
INFINITY = float('inf')

EVAL_LOGS_SCHEMA = {
    'test_prec1':float,
    'test_loss':float,
    'time':float
}

# determine running environment
if type_of_script() in {'jupyter', 'colab'}: 
    from tqdm.auto import tqdm
else: 
    from tqdm import tqdm

class Trainer: 
    """
    Flexible trainer class for training models in Pytorch.
    """
    def __init__(self, 
                model: delphi,
                max_iter: int,
                trials: int=1,
                tol: float=1e-3, 
                early_stopping: bool=False,
                n_iter_no_change: int=5,
                disable_no_grad: bool=False,
                verbose: bool=False) -> None:
        """
        Train models. 
        Args: 
            model (delphi) : delphi model to train 
            max_iter (int): maximum number of epocsh to perform on data
            trials (int): number of trials to pform procedure
            tol (float): the tolerance for the stopping criterion
            early_stopping (bool): whether to use a stopping criterion based on the validation set
            n_iter_no_change (int): number of iteration with no improvement to wait before stopping
            disable_no_grad (bool) : if True, then even model evaluation will be
                run with autograd enabled (otherwise it will be wrapped in a ch.no_grad())
            verbose (bool) : print iterator output as procedure progresses
        """
        assert isinstance(model, delphi), "model type: {} is incompatible with Trainer class".format(type(model))
        self.model = model
        assert isinstance(max_iter, int), "max_iter is type {}, Trainer expects type int".format(type(max_iter))
        self.max_iter = max_iter 
        assert isinstance(trials, int), "trials is type {}, Trainer expects type int".format(type(tol))
        self.trials = trials 

        # procedure early termination parameters
        assert isinstance(tol, float), "tol is type {}, Trainer expects type float".format(type(tol))
        self.tol = tol 
        assert isinstance(early_stopping, bool), "early_stopping is type {}, Trainer expects type bool".format(type(early_stopping))
        self.early_stopping = early_stopping
        assert isinstance(n_iter_no_change, int), "n_iter_no_change is type {}, Trainer expects type int".format(type(n_iter_no_change))
        self.n_iter_no_change = n_iter_no_change
        self.no_improvement_count = 0
        self.best_loss = -INFINITY

        assert isinstance(disable_no_grad, bool), "disable_no_grad is type {}, Trainer expects type bool".format(type(disable_no_grad))
        self.disable_no_grad = disable_no_grad
        # print log output or not
        assert isinstance(verbose, bool), "verbose is type {}, Trainer expects type bool".format(type(verbose))
        self.verbose = verbose

    def eval_model(self, loader):
        """
        Evaluate a model for standard (and optionally adversarial) accuracy.
        Args:
            loader (Iterable) : a dataloader serving batches from the test set
            store (cox.Store) : store for saving results in (via tensorboardX)
        Returns: 
            schema with model performance metrics
        """
        train_loader, val_loader = loaders
        # start timer
        start_time = time.time()

        # if store provided, 
        if store is not None:
            store.add_table('eval', consts.EVAL_LOGS_SCHEMA)

        # add 
        writer = store.tensorboard if store else None
        test_prec1, test_loss = self.model_loop(VAL, loader, 1)

        # log info 
        if store:
            log_info = {
                'test_prec1': test_prec1,
                'test_loss': test_loss,
                'time': time.time() - start_time
            }
            store['eval'].append_row(log_info)
        return log_info

    def train_model(self, loaders):
        """
        Train model. 
        Args: 
            loaders (Iterable) : iterable with the train and validation set DataLoaders
        Returns: 
            Trained model
        """
        train_loader, val_loader = loaders
        # check to make sure that the model's trainer has data in it
        if len(train_loader.dataset) == 0: 
            raise Exception('No Datapoints in Train Loader')

        # keep track of whether procedure is done or not
        done = False
        t_start = time()
        for trial in range(self.trials):
            # PRETRAIN HOOK
            self.model.pretrain_hook()
            
            # make optimizer and scheduler for training neural network
            self.model.make_optimizer_and_schedule()
            
            if self.model.checkpoint:
                epoch = self.model.checkpoint['epoch']
                best_prec1 = self.model.checkpoint['prec1'] if 'prec1' in self.model.checkpoint else self.model_loop(VAL, val_loader)[0]
        
            # do training loops until performing enough gradient steps or epochs
            for epoch in range(1, self.max_iter + 1):
                # TRAIN LOOP
                loss, prec1, prec5 = self.model_loop(TRAIN, train_loader, epoch)
                
                # VALIDATION LOOP
                if val_loader is not None:
                    with ch.no_grad():
                        loss, prec1, prec5 = self.model_loop(VAL, val_loader, epoch)

                    # check for early completion
                    if self.early_stopping: 
                        if self.tol > -INFINITY and loss > self.best_loss - self.tol:
                            self.no_improvement_count += 1
                        else: 
                            self.no_improvement_count = 0

                        if loss < self.best_loss: 
                            self.best_loss = loss

                    if self.no_improvement_count >= self.n_iter_no_change:
                        if self.verbose: 
                            print("Convergence after %d epochs took %.2f seconds" % (epoch, time() - t_start))
                        return self.model

            # POST TRAINING HOOK     
            self.model.post_training_hook()
        return self.model
                
    def model_loop(self, loop_type, loader, epoch):
        """
        *Internal method* (refer to the train_model and eval_model functions for
        how to train and evaluate models).
        Runs a single epoch of either training or evaluating.
        Args:
            loop_type ('train' or 'val') : whether we are training or evaluating
            loader (iterable) : an iterable loader
            epoch (int) : which epoch we are currently on
            writer : tensorboardX writer (optional)
        Returns:
            The average top1 accuracy and the average loss across the epoch.
        """
        # check loop type 
        if not loop_type in ['train', 'val']: 
            err_msg = "loop type must be in {0} must be 'train' or 'val".format(loop_type)
            raise ValueError(err_msg)
        # train or val loop
        is_train = (loop_type == 'train')
        loop_msg = 'Train' if is_train else 'Val'
        
        # if is_train, put model into train mode, else eval mode
        # self.model = self.model.model.train() if is_train else self.model.model.eval()
        loss_, prec1_, prec5_ = AverageMeter(), AverageMeter(), AverageMeter()
        
        # iterator
        iterator = tqdm(enumerate(loader), total=len(loader), leave=False, bar_format='{l_bar}{bar}{r_bar}') if self.verbose else enumerate(loader) 
        for i, batch in iterator:
            self.model.optimizer.zero_grad()
            loss, prec1, prec5 = self.model(batch)
            # update average meters
            loss_.update(loss)
            if prec1 is not None: prec1_.update(prec1)
            if prec5 is not None: prec5_.update(prec5)
            # regularize
            reg_term = self.model.regularize(batch)
            loss = loss + reg_term

            # if training loop, perform training step
            if is_train:
                loss.backward()
                self.model.optimizer.step()
                
                if self.model.schedule is not None: self.model.schedule.step()
            
            # ITERATOR DESCRIPTION
            if self.verbose:
                desc = self.model.description(epoch, i, loop_msg, loss_, prec1_, prec5_)
                iterator.set_description(desc)
           
            # ITERATION HOOK 
            self.model.iteration_hook(i, loop_type, loss, prec1, prec5, batch)
        # EPOCH HOOK
        self.model.epoch_hook(epoch, loop_type, loss, prec1, prec5)

        return loss_.avg, prec1_.avg, prec5_.avg

