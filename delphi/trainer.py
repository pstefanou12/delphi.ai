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


class Trainer:
    def __init__(self, 
                model: delphi): 
        self.model = model        
        self.train_costs, self.val_costs = ch.Tensor([]), ch.Tensor([])

    def model_loop_(self,
                    args: Parameters, 
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
            pred = self.model(inp, targ)
            loss = self.model.criterion(pred, targ, *self.model.criterion_params)

            """
            NOTE: Depending on batch size, the loss may not be shape (1x1), 
            but instead (nX1).
            """    
            if len(loss.shape) > 0: loss = loss.sum()

            loss_.update(loss)
            reg_term = self.model.regularize(batch)
            if args.cuda:
                reg_term = reg_term.cuda()
            loss = loss + reg_term
            if is_train:
                self.train_costs = ch.cat([self.train_costs, loss])
            else: 
                self.val_costs = ch.cat([self.val_costs, loss])

            if is_train:
                loss.backward()
                self.model.pre_step_hook(inp)
                optimizer.step()
                if schedule is not None: schedule.step()

            if args.verbose:
                desc = self.model.description(epoch, i, loop_msg, loss_, prec1_, prec5_, reg_term)
                iterator.set_description(desc)
 
            self.model.iteration_hook(i, is_train, loss, batch)
            if is_train: 
                try: 
                    history_ = ch.cat([history_, self.model.weight.data[None,...]])
                except: 
                    history_ = ch.cat([history_, self.model._parameters[0]['params'][0].data[None,...]])

        self.model.epoch_hook(epoch, is_train, loss)

        return loss_.avg, prec1_.avg, prec5_.avg, history_


    def eval_model(self, 
                    args: Parameters, 
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
        test_prec1, test_loss = self.model_loop_(args, loader, 1)

        if store:
            log_info = {
                'test_prec1': test_prec1,
                'test_loss': test_loss,
                'time': time.time() - start_time
            }
            store['eval'].append_row(log_info)
        return log_info


    def train_model(self,
                    args: Parameters, 
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
        
            self.model.pretrain_hook(train_loader)
            optimizer, schedule = self.model.make_optimizer_and_schedule(self.model.parameters()) 
   
            if checkpoint:
                epoch = checkpoint['epoch']
                best_prec1 = checkpoint['prec1'] if 'prec1' in checkpoint else self.model_loop_(args, val_loader, epoch, False)[0]
        
            for epoch in range(1, args.epochs + 1):
                train_loss, train_prec1, train_prec5, history_ = self.model_loop_(args, train_loader, epoch, True, optimizer, schedule)
                history = ch.cat([history, history_])

                if val_loader is not None:
                    with ch.no_grad():
                        val_loss, val_prec1, val_prec5, _ = self.model_loop_(args, val_loader, epoch, False)
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
                        best_params, best_loss = copy.copy(list(self.model.parameters())[0]), val_loss
                        best_params.requires_grad = False
                    except: 
                        best_params, best_loss = copy.copy(list(self.model._parameters)), val_loss
                if args.early_stopping: 
                    if ch.abs(val_loss - best_loss) <= args.tol:
                        no_improvement_count += 1
                    else: 
                        no_improvement_count = 0
                    if no_improvement_count >= args.n_iter_no_change:
                        if args.verbose: 
                            print("Convergence after %d epochs took %.2f seconds" % (epoch, time() - t_start))
                        break

            self.model.post_training_hook()
        
        if args.early_stopping and args.verbose and no_improvement_count < args.n_iter_no_change: 
            print('Procedure did not converge after %d epochs and %.2f seconds' % (epoch, time() - t_start))

        return best_params, history, copy.copy(self.model.parameters)
