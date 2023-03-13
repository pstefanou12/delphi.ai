"""
Module used for training models.
"""
import torch as ch
import numpy as np
from torch.optim import SGD, Adam, lr_scheduler
from torch.utils.data import DataLoader
from time import time
from tqdm import tqdm
import copy
from typing import Iterable
from cox.store import Store

from .delphi import delphi
from .utils import constants as consts
from .utils.helpers import AverageMeter, setup_store_with_metadata, Parameters
from .utils.defaults import TRAINER_DEFAULTS, check_and_fill_args

# CONSTANTS 
BY_ALG = 'by algorithm'  # default parameter depends on algorithm
ADAM = 'adam'
CYCLIC = 'cyclic'
COSINE = 'cosine'
LINEAR = 'linear'


def make_optimizer_and_schedule(args: Parameters, 
                                params: Iterable, 
                                checkpoint: dict=None):
    """
    Create optimizer (ch.nn.optim) and scheduler (ch.nn.optim.lr_scheduler module)
    for SGD procedure. 
    """
    optimizer, schedule = None, None
    if args.cuda: params.to('cuda')
    optimizer = SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.custom_lr_multiplier == ADAM: 
        optimizer = Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    elif not args.constant: 
        eps = args.epochs
        if args.custom_lr_multiplier == CYCLIC:
            lr_func = lambda t: np.interp([t], [0, eps*4//15, eps], [0, 1, 0])[0]
            schedule = lr_scheduler.LambdaLR(optimizer, lr_func)
        elif args.custom_lr_multiplier == COSINE:
            schedule = lr_scheduler.CosineAnnealingLR(optimizer, eps)
        elif args.custom_lr_multiplier:
            cs = args.custom_lr_multiplier
            periods = eval(cs) if type(cs) is str else cs
            if args.lr_interpolation == LINEAR:
                lr_func = lambda t: np.interp([t], *zip(*periods))[0]
            else:
                def lr_func(ep):
                    for (milestone, lr) in reversed(periods):
                        if ep >= milestone: return lr
                    return 1.0
            schedule = lr_scheduler.LambdaLR(optimizer, lr_func)
        elif args.step_lr:
            schedule = lr_scheduler.StepLR(optimizer, step_size=args.step_lr, gamma=args.step_lr_gamma)
            
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        try:
            schedule.load_state_dict(checkpoint['schedule'])
        # if can't load scheduler state, take epoch steps
        except:
            steps_to_take = checkpoint['epoch']
            print('Could not load schedule (was probably LambdaLR).'
                f' Stepping {steps_to_take} times instead...')
            for i in range(steps_to_take):
                schedule.step() 

    return optimizer, schedule


def model_loop_(args: Parameters, 
                model: delphi,
                loader: ch.utils.data.DataLoader,
                epoch: int, 
                is_train: bool, 
                optimizer: ch.optim=None, 
                schedule: ch.optim.lr_scheduler=None):

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
    history_ = ch.Tensor([])
    loop_msg = 'Train' if is_train else 'Val'
    loss_, prec1_, prec5_ = AverageMeter(), AverageMeter(), AverageMeter()
    iterator = tqdm(enumerate(loader), total=len(loader), leave=False, 
                    bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') if args.verbose else enumerate(loader) 
    for i, batch in iterator:
        if is_train: optimizer.zero_grad()
        inp, targ = batch
        pred = model(inp, targ)
        loss = model.criterion(pred, targ, *model.criterion_params)

        """
        NOTE: Depending on batch size, the loss may not be shape (1x1), 
        but instead (nX1).
        """    
        if len(loss.shape) > 0: loss = loss.sum()

        loss_.update(loss)
        reg_term = model.regularize(batch)
        if args.cuda:
            reg_term = reg_term.cuda()
        loss = loss + reg_term

        if is_train:
            loss.backward()
            # import pdb; pdb.set_trace()
            model.pre_step_hook(inp)
            optimizer.step()
            if schedule is not None: schedule.step()

        if args.verbose:
            desc = model.description(epoch, i, loop_msg, loss_, prec1_, prec5_, reg_term)
            iterator.set_description(desc)
 
        model.iteration_hook(i, is_train, loss, batch)
        if is_train: 
            try: 
                history_ = ch.cat([history_, model.weight.data[None,...]])
            except: 
                history_ = ch.cat([history_, model._parameters[0]['params'][0].data[None,...]])

    model.epoch_hook(epoch, is_train, loss)

    return loss_.avg, prec1_.avg, prec5_.avg, history_


def eval_model(args: Parameters, 
               model: delphi, 
               loader: DataLoader, 
               store: Store=None):
    """
    Evaluate a model for standard (and optionally adversarial) accuracy.
    Args:
        loader (Iterable) : a dataloader serving batches from the test set
        store (cox.Store) : store for saving results in (via tensorboardX)
    Returns: 
        schema with model performance metrics
    """
    start_time = time.time()

    if store is not None:
        store.add_table('eval', consts.EVAL_LOGS_SCHEMA)

    writer = store.tensorboard if store else None
    test_prec1, test_loss = model_loop_(args, model, loader, 1)

    if store:
        log_info = {
            'test_prec1': test_prec1,
            'test_loss': test_loss,
            'time': time.time() - start_time
        }
        store['eval'].append_row(log_info)
    return log_info


def train_model(args: Parameters, 
                model: delphi, 
                train_loader: ch.utils.data.DataLoader, 
                val_loader: ch.utils.data.DataLoader, 
                rand_seed: int=0, 
                store: Store=None, 
                checkpoint=None):
    """
    Train model. 
    Args: 
        loaders (Iterable) : iterable with the train and validation set DataLoaders
    Returns: 
        Tuple(best_params, best)
    """
    if len(train_loader.dataset) == 0: 
        raise Exception('No Datapoints in Train Loader')

    args = check_and_fill_args(args, TRAINER_DEFAULTS)
    
    if store is not None: 
        store.add_table('logs', {
            'trial': int,
            'epoch': int,
            'train_loss': float, 
            'train_prec1': float,
            'train_prec5': float, 
            'val_loss': float,
            'val_prec1': float, 
            'val_prec5': float})
        setup_store_with_metadata(args, store)

    # stores model estimates after each gradient step
    history = ch.Tensor([])
    best_loss, best_params = float('inf'), None
    for trial in range(args.trials):
        ch.manual_seed(rand_seed)
        if args.verbose: print(f'trial: {trial + 1}')

        t_start = time()
        no_improvement_count = 0
        
        model.pretrain_hook(train_loader)
        optimizer, schedule = make_optimizer_and_schedule(args, model.parameters()) 
   
        if checkpoint:
            epoch = checkpoint['epoch']
            best_prec1 = checkpoint['prec1'] if 'prec1' in checkpoint else model_loop_(args, model, val_loader, epoch, False)[0]
        
        for epoch in range(1, args.epochs + 1):
            train_loss, train_prec1, train_prec5, history_ = model_loop_(args, model, train_loader, epoch, True, optimizer)
            history = ch.cat([history, history_])

            if val_loader is not None:
                with ch.no_grad():
                    val_loss, val_prec1, val_prec5, _ = model_loop_(args, model, val_loader, epoch, False)
                if args.verbose: print(f'Epoch {epoch} - Loss: {val_loss}')

            if store is not None:
                store['logs'].append_row({
                    'trial': trial,
                    'epoch': epoch,
                    'train_loss': train_loss, 
                    'train_prec1': train_prec1,
                    'train_prec5': train_prec5,
                    'val_loss': val_loss,
                    'val_prec1': val_prec1, 
                    'val_prec5': val_prec5})

            """
            NOTE: Check for training procedure convergence. If 
            no improvement in loss for args.n_iter_no_change epochs, 
            then procedure has converged.
            """
            if best_params is None or val_loss < best_loss: 
                try: 
                    best_params, best_loss = copy.copy(list(model.parameters())[0]), val_loss
                    best_params.requires_grad = False
                except: 
                    best_params, best_loss = copy.copy(list(model._parameters)), val_loss
                    # best_params['params'][0].requires_grad = False
            if args.early_stopping: 
                if ch.abs(val_loss - best_loss) <= args.tol:
                    no_improvement_count += 1
                else: 
                    no_improvement_count = 0
                if no_improvement_count >= args.n_iter_no_change:
                    if args.verbose: 
                        print("Convergence after %d epochs took %.2f seconds" % (epoch, time() - t_start))
                    break

        model.post_training_hook()
        
    if args.early_stopping and args.verbose and no_improvement_count < args.n_iter_no_change: 
        print('Procedure did not converge after %d epochs and %.2f seconds' % (epoch, time() - t_start))

    return best_params, history, copy.copy(model.parameters)
