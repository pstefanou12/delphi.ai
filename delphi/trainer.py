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
from torch.optim import SGD, Adam, lr_scheduler
import cox
from typing import Any, Iterable, Callable
from abc import ABC

from . import oracle
from .utils.helpers import has_attr, ckpt_at_epoch, AverageMeter, accuracy, type_of_script, setup_store_with_metadata, ProcedureComplete
from .utils import constants as consts

# CONSTANTS 
CYCLIC='cyclic'
COSINE='cosine'
ADAM='adam'
LINEAR='linear'
TRAIN='train'
VAL='val'
EVAL_LOGS_TABLE='eval'
CKPT_NAME_LATEST='checkpoint.pt.latest'
CKPT_NAME_BEST='checkpoint.pt.best'
JUPYTER = 'jupyter'
TERMINAL = 'terminal'
IPYTHON = 'ipython'
ZMQ='zmqshell'
COLAB='google.colab'

LOGS_SCHEMA = {
    'epoch':int,
    'val_prec1':float,
    'val_loss':float,
    'train_prec1':float,
    'train_loss':float,
    'time':float
}

EVAL_LOGS_SCHEMA = {
    'test_prec1':float,
    'test_loss':float,
    'time':float
}

# HELPER FUNCTIONS 
def type_of_script():
    """
    Check the program's running environment.
    """
    try:
        ipy_str = str(type(get_ipython()))
        if ZMQ in ipy_str:
            return JUPYTER
        if TERMINEL in ipy_str:
            return IPYTHON
        if COLAB in ipy_str: 
            return COLAB
    except:
        return TERMINAL

# determine running environment
from tqdm.autonotebook import tqdm as tqdm if type_of_script() in set(JUPYTER, COLAB) else from tqdm import tqdm

class Trainer: 
    """
    Flexible trainer class for training models in Pytorch.
    """
    def __init__(self, 
                args: cox.Parameters, 
                model: Any, 
                checkpoint: dict = None, 
                parallel: bool = False, 
                cuda: bool = False, 
                dp_device_ids = None, 
                store: cox.store.Store=None, 
                table: str = 'table', 
                params: Iterable = None, 
                disable_no_grad: bool = False) -> None:
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
                epoch hook (function, optional)
                    Similar to iteration_hook but called every epoch instead, and
                    given arguments `model, log_info` where `log_info` is a
                    dictionary with keys `epoch, nat_prec1, adv_prec1, nat_loss,
                    adv_loss, train_prec1, train_loss`.
            model (Any) : model to train 
            train_loader: (ch.utils.data.DataLoader) : training set data loader 
            val_loader: (ch.utils.data.DataLoader): validation set data loader 
            pretrain_hook (Callable) : procedure pretrian hook, default is None; useful for initializing model parameters, etc.
            train_step: (Callable) : procedure train step, including both the forward and the backward step, default 
            is multi-class train step with CE Loss
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
        """
        # INSTANCE VARIABLES
        self.args = args 
        self.model = model 
        self.train_loader, self.val_loader = train_loader, val_loader
        self.checkpoint = checkpoint 
        self.parallel = parallel 
        self.device = device 
        self.dp_device_ids = dp_device_ids 
        self.store = store 
        self.table = table 
        self.params = params 
        self.disable_no_grad = disable_no_grad

        # if store provided, create logs table for experiment
        if self.store is not None:
            self.store.add_table(self.table, consts.LOGS_SCHEMA)
        self.writer = self.store.tensorboard if self.store else None

        # number of periods/epochs for learning rate schedulers
        self.M = self.args.epochs if self.args.epochs else self.args.steps
        self.make_optimizer_and_schedule()

        # hooks for training procedure
        self.pretrain_hook = pretrain_hook
        self.train_step = train_step
        self.iteration_hook = iteration_hook 
        self.epoch_hook = epoch_hook
        self.post_train_hook post_train_hook

        # run model in parallel model
        assert not hasattr(self.model, "module"), "model is already in DataParallel."
        if self.parallel and next(model.parameters()).is_cuda:
            model = ch.nn.DataParallel(model, device_ids=self.dp_device_ids)

    def make_optimizer_and_schedule(self):
        """
        Create optimizer (ch.nn.optim) and scheduler (ch.nn.optim.lr_scheduler module)
        for SGD procedure. 
        """
        # initialize optimizer, scheduler, and then get parameters
        param_list = self.model.parameters() if self.params is None else self.params

        # check for Adam optimizer
        if self.args.adam: 
            optimizer = Adam(param_list, betas=self.args.betas, lr=self.args.lr, weight_decay=self.args.weight_decay, 
            amsgrad=self.args.amsgrad)
        else: 
            # SGD optimizer
            self.optimizer = SGD(param_list, self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
            # cyclic learning rate scheduler
            if self.args.custom_lr_multiplier == CYCLIC and self.M is not None:
                lr_func = lambda t: np.interp([t], [0, self.M*4//15, self.M], [0, 1, 0])[0]
                self.schedule = lr_scheduler.LambdaLR(optimizer, lr_func)
            # cosine annealing scheduler
            elif self.args.custom_lr_multiplier == COSINE and self.M is not None:
                schedule = lr_scheduler.CosineAnnealingLR(optimizer, self.M)
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
                self.schedule = lr_scheduler.StepLR(optimizer, step_size=self.args.step_lr, 
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
        table = EVAL_LOGS_TABLE if table is None else table
        if store is not None:
            store.add_table(table, consts.EVAL_LOGS_SCHEMA)

        # add 
        writer = store.tensorboard if store else None
        test_prec1, test_loss = self.model_loop(VAL, loader)

        # log info 
        if store:
            log_info = {
                'test_prec1': test_prec1,
                'test_loss': test_loss,
                'time': time.time() - start_time
            }
            table = EVAL_LOGS_TABLE if table is None else table # table name
            store[table].append_row(log_info)
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

        best_prec1, epoch = (0, 0)
        if checkpoint:
            epoch = checkpoint['epoch']
            best_prec1 = checkpoint['prec1'] if 'prec1' in checkpoint delse self.model_loop(VAL, val_loader)[0]

        # keep track of the start time
        counter, start_time = 0, time.time()
        # do training loops until performing enough gradient steps or epochs
        while counter < self.M:
            try: 
                train_prec1, train_loss = model_loop(TRAIN, train_loader)
            # if raising ProcedureComplete, then terminate
            except ProcedureComplete: 
                return model
            # raise error
            except Exception as e: 
                raise e

            # EPOCH HOOK
            if hasattr(self, 'epoch_hook'): self.epoch_hook(counter)
                
            # update procedure step counter
            counter += (len(train_loader) if has_attr(self.args, 'steps') else 1)
            
        if hasattr(self, 'post_training_hook'): self.post_training_hook()
        return model
                
    def model_loop(self, loop_type, loader):
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
    
        # model stats
        losses, top1, top5 =  AverageMeter(), AverageMeter(), AverageMeter()
        model = model.train() if not isinstance(model, ch.distributions.distribution.Distribution) and is_train else model.eval()
        
        # iterator
        self.iterator = enumerate(loader) if args.steps else tqdm(enumerate(loader), total=len(loader), leave=False) 
        for i, batch in iterator:
            # zero gradient for model parameters
            self.optimizer.zero_grad()
            # if training loop, perform training step
            if is_train:
                self.train_step(i, batch)
            # iteration hook 
            self.iteration_hook(i, loop_type, *batch)
            
            # increment number of gradients
            if steps is not None: 
                steps += 1
                if schedule: schedule.step()

        # LOSS AND ACCURACY
        return top1.avg, losses.avg

    def pretrain_hook(self): 
        """
        Default pre-train hook. Does nothing by default
        """
        pass

    def train_step(self, i, batch)
        """
        Default train step is written assuming that the 
        default model is a neural network using cross 
        entropy loss. 
        """ 
        inp, targ = batch
        inp, targ = inp.to(device), targ.to(device)
        output = self.model(inp)

        # AttackerModel returns both output and final input
        if isinstance(output, tuple):
            model_logits, _ = output

        # regularizer 
        reg_term = 0.0
        if has_attr(self.args, "regularizer") and isinstance(model, ch.nn.Module):
            reg_term = self.args.regularizer(model, inp, targ)

        # calculate loss and regularize
        loss = ch.nn.CrossEntropyLoss()(model_logits, targ)
        loss = loss + reg_term

        # backward propagation
        loss.backward()
        # update model 
        optimizer.step()

        return loss

    def iteration_hook(self, i, loop_type, *batch): 
        """
        Default iteration hook. By default, does logging for DNN training 
        for both robust and non-robust models. This hook can be very useful 
        for custom logging, checking procedure convergence, projection sets, etc. 
        Args: 
            *batch (Iterable) : iterable of 
        Returns: 

        """
        # measure accuracy and record loss
        top1_acc = float('nan')
        top5_acc = float('nan')

        losses.update(loss.item(), inp.size(0))

        # calculate accuracy metrics
        maxk = min(5, model_logits.shape[-1])
        if has_attr(self.args, "custom_accuracy"):
            prec1, prec5 = self.args.custom_accuracy(model_logits, target)
        else:
            prec1, prec5 = accuracy(model_logits, target, topk=(1, maxk))
            prec1, prec5 = prec1[0], prec5[0]

        top1.update(prec1, inp.size(0))
        top5.update(prec5, inp.size(0))
        top1_acc = top1.avg
        top5_acc = top5.avg

        # ITERATOR
        desc = ('Epoch: {0} | Loss {loss.avg:.4f} | '
                '{1}1 {top1_acc:.3f} | {1}5 {top5_acc:.3f} | '
                'Reg term: {reg} ||'.format(epoch, loop_msg,
                                            loss=losses, top1_acc=top1_acc, top5_acc=top5_acc, reg=reg_term))
        
    def epoch_hook(self, i): 
        """
        Default epoch hook. Does nothing by default. However, this hook 
        can be very useful for custom logging, etc.
        """
        # write to Tensorboard
        if self.writer is not None:
            descs = ['loss', 'top1', 'top5']
            vals = [losses, top1, top5]
            for d, v in zip(descs, vals):
                self.writer.add_scalar('_'.join([loop_type, d]), v.avg,
                                epoch)

        # check for logging/checkpoint
        last_epoch = (i == (self.args.epochs - 1))
        should_save_ckpt = (i % self.args.save_ckpt_iters == 0 or last_epoch) 
        should_log = (i % self.args.log_iters == 0 or last_epoch)

        # validation loop
        val_prec1, val_loss = 0.0, 0.0
        if should_log or should_save_ckpt: 
            ctx = ch.enable_grad() if disable_no_grad else ch.no_grad()

            # evaluate model on validation set, if there is one
            if val_loader is not None:
                with ctx:
                    val_prec1, val_loss = self.model_loop(VAL val_loader)

                # remember best prec_1 and save checkpoint
                is_best = val_prec1 > best_prec1
                best_prec1 = max(val_prec1, best_prec1)

        # CHECKPOINT -- checkpoint epoch of better DNN performance
        if should_save_ckpt or is_best:
            sd_info = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'schedule': (self.schedule and self.schedule.state_dict()),
                'epoch': i+1,
                'amp': amp.state_dict() if self.args.mixed_precision else None,
                'prec1': val_prec1
            }
            
            def save_checkpoint(filename):
                """
                Saves model checkpoint at store path with filename.
                Args: 
                    filename (str) name of file for saving model
                """
                ckpt_save_path = os.path.join(store.path, filename)
                ch.save(sd_info, ckpt_save_path, pickle_module=dill)

            # update the latest and best checkpoints (overrides old one)
            if is_best:
                save_checkpoint(CKPT_NAME_BEST)
            if should_save_ckpt: 
                # if we are at a saving epoch (or the last epoch), save a checkpoint
                save_checkpoint(ckpt_at_epoch(epoch))
                save_checkpoint(CKPT_NAME_LATEST)

        # LOG
        if should_log and store: # TODO: add custom logging hook
            log_info = {
                'epoch': i + 1,
                'val_prec1': val_prec1,
                'val_loss': val_loss,
                'train_prec1': train_prec1,
                'train_loss': train_loss,
                'time': time.time() - start_time
            }
            store[self.table].append_row(log_info)

    def post_train_hook(self): 
        """
        Default post training hook. Does nothing by default. However, this hook 
        can be useful for custom logging, saving trained models, and testing model 
        accuracy after the entire training procedure.
        """
        pass 