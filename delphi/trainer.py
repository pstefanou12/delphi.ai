"""
Flexible, parent trainer class for experimentation. By default the class is programmed 
to train Robust CV models. Nevertheless, the flexible nature of the framework allows it 
to accomodate all the mmodesl with the delphi.ai package.
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

from . import oracle
from .utils.helpers import has_attr, ckpt_at_epoch, type_of_script, AverageMeter, accuracy, setup_store_with_metadata, ProcedureComplete
from .utils import constants as consts

# CONSTANTS
TRAIN = 'train'
VAL = 'val'

EVAL_LOGS_SCHEMA = {
    'test_prec1':float,
    'test_loss':float,
    'time':float
}

# determine running environment
if type_of_script() in {'jupyter', 'colab'}: 
    from tqdm.autonotebook import tqdm 
else: 
    from tqdm import tqdm

class Trainer: 
    """
    Flexible trainer class for training models in Pytorch.
    """
    def __init__(self, 
                model: Any,
                disable_no_grad: bool = False,
                verbose: bool = True) -> None:
        """
        Train models. 
        Args: 
            args (object) : A python object for arguments, implementing
                ``getattr()`` and ``setattr()`` and having the following
                attributes. See :attr:`delphi.defaults.TRAINING_ARGS` for a 
                list of arguments, and you can use
                :meth:`delphi.defaults.check_and_fill_args` to make sure that
                all required arguments are filled and to fill missing args with
                reasonable defaults:
                epochs (int, *required*)
                    number of epochs to train for
                lr (float, *required*)
                    learning rate for SGD optimizer
                weight_decay (float, *required*)
                    weight decay for SGD optimizer
                momentum (float, *required*)
                    momentum parameter for SGD optimizer
                step_lr (int)
                    if given, drop learning rate by 10x every `step_lr` steps
                custom_lr_multplier (str)
                    If given, use a custom LR schedule, formed by multiplying the
                        original ``lr`` (format: [(epoch, LR_MULTIPLIER),...])
                lr_interpolation (str)
                    How to drop the learning rate, either ``step`` or ``linear``,
                        ignored unless ``custom_lr_multiplier`` is provided.
                log_iters (int, *required*)
                    How frequently (in epochs) to save training logs
                save_ckpt_iters (int, *required*)
                    How frequently (in epochs) to save checkpoints (if -1, then only
                    save latest and best ckpts)
                eps (float or str, *required if adv_train or adv_eval*)
                    float (or float-parseable string) for the adv attack budget
                use_best (int or bool, *required if adv_train or adv_eval*) :
                    If True/1, use the best (in terms of loss) PGD step as the
                    attack, if False/0 use the last step
                custom_accuracy (function)
                    If given, should be a function that takes in model outputs
                    and model targets and outputs a top1 and top5 accuracy, will 
                    displayed instead of conventional accuracies
                regularizer (function, optional) 
                    If given, this function of `model, input, target` returns a
                    (scalar) that is added on to the training loss without being
                    subject to adversarial attack
            model (Any) : model to train 
            pretrain_hook (Callable) : procedure pretrian hook, default is None; useful for initializing model parameters, etc.
            train_step (Callable) : procedure train step, including both the forward and the backward step, default 
            is multi-class train step with CE Loss
            val_step (Callable) : procedure val step 
            epoch_hook (Callable) : procedure epoch hook, default is None; useful for custom logging, etc.
            post_train_hook (Callable) : post training hook, called after the model has been trained
            checkpoint (dict) : a loaded checkpoint previously saved by this library
                (if resuming from checkpoint)
            store (cox.Store) : a cox store for logging training progress
            params (list) : list of parameters to use for training, if None
                then all parameters in the model are used (useful for transfer
                learning)
            disable_no_grad (bool) : if True, then even model evaluation will be
                run with autograd enabled (otherwise it will be wrapped in a ch.no_grad())
            verbose (bool) : print iterator output as procedure progresses
        """
        # INSTANCE VARIABLES
        self.model = model
        self.disable_no_grad = disable_no_grad
        # number of periods/epochs for learning rate schedulers
        self.M = self.model.args.epochs if self.model.args.epochs else self.model.args.steps
        # print log output or not
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
        # separate loaders
        train_loader, val_loader = loaders

        if self.model.args.steps: 
            self.M = int(self.M / len(train_loader))

        # PRETRAIN HOOK
        if hasattr(self.model, 'pretrain_hook'): self.model.pretrain_hook()
        
        # make optimizer and scheduler for training neural network
        self.model.make_optimizer_and_schedule()
        
        if self.model.checkpoint:
            epoch = self.model.checkpoint['epoch']
            best_prec1 = self.model.checkpoint['prec1'] if 'prec1' in self.model.checkpoint else self.model_loop(VAL, val_loader)[0]
        
        # keep track of the start time
        start_time = time.time()
        # do training loops until performing enough gradient steps or epochs
        for epoch in range(1, self.M + 1):
            try: 
                self.model_loop(TRAIN, train_loader, epoch)
            # if raising ProcedureComplete, then terminate
            except ProcedureComplete: 
                return self.model
            # raise error
            except Exception as e: 
                raise e
            # evaluate model on validation set, if there is one
            if val_loader is not None:
                ctx = ch.enable_grad() if self.disable_no_grad else ch.no_grad()
                with ctx:
                    self.model_loop(VAL, val_loader, epoch)

        # POST TRAINING HOOK     
        if hasattr(self.model, 'post_training_hook'): self.model.post_training_hook()
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
       
        # iterator
        iterator = enumerate(loader) if self.model.args.steps else tqdm(enumerate(loader), total=len(loader), leave=False) 
        for i, batch in iterator:
            # if training loop, perform training step
            if is_train:
                loss, prec1, prec5 = self.model.train_step(i, batch)
            else: 
                loss, prec1, prec5 = self.model.val_step(i, batch)

            # iterator description
            if self.verbose and hasattr(self.model, 'description'):
                desc = self.model.description(epoch, i, loop_msg)
                iterator.set_description(desc)

            # iteration hook 
            if hasattr(self.model, 'iteration_hook'): self.model.iteration_hook(i, loop_type, loss, prec1, prec5, batch)

        # EPOCH HOOK
        if hasattr(self.model, 'epoch_hook'): self.model.epoch_hook(i, loop_type, loss, prec1, prec5, batch)

