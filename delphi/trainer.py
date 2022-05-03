"""
General training format for training models with SGD/backprop.
"""

import time
import torch as ch
import cox
from cox.store import Store
from time import time
from tqdm import tqdm
import copy

from .delphi import delphi
from .utils.helpers import ckpt_at_epoch, AverageMeter, setup_store_with_metadata, Parameters
from .utils.defaults import check_and_fill_args, TRAINER_DEFAULTS
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


class Trainer: 
    """
    Flexible trainer class for training models in Pytorch.
    """
    def __init__(self, 
                model: delphi,
                args: Parameters,
                store: Store=None) -> None:
        """
        Train models. 
        Args: 
            model (delphi) : delphi model to train 
            epochs(int): maximum number of epochs to perform on data
            trials (int): number of trials to pform procedure
            tol (float): the tolerance for the stopping criterion
            early_stopping (bool): whether to use a stopping criterion based on the validation set
            n_iter_no_change (int): number of iteration with no improvement to wait before stopping
            disable_no_grad (bool) : if True, then even model evaluation will be
                run with autograd enabled (otherwise it will be wrapped in a ch.no_grad())
        """
        assert isinstance(model, delphi), "model type: {} is incompatible with Trainer class".format(type(model))
        self.model = model
        # keep track of the best model based off the best nll
        best_loss, best_model = None, None

        # check and fill trainer hyperparameters
        self.args = check_and_fill_args(args, TRAINER_DEFAULTS)
        assert store is None or isinstance(store, cox.store.Store), "provided store is type: {}. expecting logging store cox.store.Store".format(type(store))
        self.store = store 
        
        assert store is None or isinstance(store, cox.store.Store), "prorvided store is type: {}. expecting logging store cox.store.Store".format(type(store))
        self.store = store 
        
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
        if self.store is not None:
            self.store.add_table('eval', consts.EVAL_LOGS_SCHEMA)

        # add 
        writer = self.store.tensorboard if self.store else None
        test_prec1, test_loss = self.model_loop(VAL, loader, 1)

        # log info 
        if self.store:
            log_info = {
                'test_prec1': test_prec1,
                'test_loss': test_loss,
                'time': time.time() - start_time
            }
            self.store['eval'].append_row(log_info)
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
        
        if self.store is not None: 
            self.store.add_table('logs', {
                'trial': int,
                'epoch': int,
                'train_loss': float, 
                'train_prec1': float,
                'train_prec5': float, 
                'val_loss': float,
                'val_prec1': float, 
                'val_prec5': float})
            # record hyperparameters
            setup_store_with_metadata(self.args, self.store)

        best_loss, best_model = INFINITY, None
        for trial in range(self.args.trials):
            if self.args.verbose: print(f'trial: {trial + 1}')
            t_start = time()
            no_improvement_count = 0

            # PRETRAIN HOOK
            self.model.pretrain_hook()
            
            # make optimizer and scheduler for training neural network
            self.model.make_optimizer_and_schedule()
            
            if self.model.checkpoint:
                epoch = self.model.checkpoint['epoch']
                best_prec1 = self.model.checkpoint['prec1'] if 'prec1' in self.model.checkpoint else self.model_loop(VAL, val_loader)[0]
        
            # do training loops until performing enough gradient steps or epochs
            for epoch in range(1, self.args.epochs + 1):
                # TRAIN LOOP
                train_loss, train_prec1, train_prec5 = self.model_loop(TRAIN, train_loader, epoch)

                                
                # VALIDATION LOOP
                if val_loader is not None:
                    with ch.no_grad():
                        val_loss, val_prec1, val_prec5 = self.model_loop(VAL, val_loader, epoch)
                    
                    if self.args.verbose: print(f'Epoch {epoch} - Loss: {val_loss}')

                # if store provided, log epoch results
                if self.store is not None:
                    self.store['logs'].append_row({
                        'trial': trial,
                        'epoch': epoch,
                        'train_loss': train_loss, 
                        'train_prec1': train_prec1,
                        'train_prec5': train_prec5,
                        'val_loss': val_loss,
                        'val_prec1': val_prec1, 
                        'val_prec5': val_prec5})

                # check for early completion
                if self.args.early_stopping: 
                    if self.args.tol > -INFINITY and ch.abs(val_loss - best_loss) <= self.args.tol:
                        no_improvement_count += 1
                    else: 
                        no_improvement_count = 0

                    if best_model is None or val_loss < best_loss: 
                        best_model, best_loss = self.model.model[:], val_loss

                    # model convergence
                    if no_improvement_count >= self.args.n_iter_no_change:
                        if self.args.verbose: 
                            print("Convergence after %d epochs took %.2f seconds" % (epoch, time() - t_start))
                        break
                    
            # POST TRAINING HOOK     
            self.model.post_training_hook()
            # update best model and best loss
            if best_model is None or val_loss < best_loss: 
                best_model, best_loss = copy.copy(self.model.model), val_loss 
        
        # inform user that SGD did not converge
        if self.args.verbose and no_improvement_count <= self.args.n_iter_no_change: 
           print('Procedure did not converge after %d epochs and %.2f seconds' % (epoch, time() - t_start))
        # set best model in delphi model object 
        self.model.model = best_model        

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
        loss_, prec1_, prec5_ = AverageMeter(), AverageMeter(), AverageMeter()
        
        # iterator
        iterator = tqdm(enumerate(loader), total=len(loader), leave=False, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') if self.args.verbose else enumerate(loader) 
        for i, batch in iterator:
            self.model.optimizer.zero_grad()
            loss, prec1, prec5 = self.model(batch)
            
            if len(loss.shape) > 0: loss = loss.sum()

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
                if self.model.schedule is not None and not self.model.args.epoch_step: self.model.schedule.step()
            
            # ITERATOR DESCRIPTION
            if self.args.verbose:
                desc = self.model.description(epoch, i, loop_msg, loss_, prec1_, prec5_, reg_term)
                iterator.set_description(desc)
 
            # ITERATION HOOK 
            self.model.iteration_hook(i, loop_type, loss, prec1, prec5, batch)
        if self.model.schedule is not None and self.model.args.epoch_step: self.model.scheduler.step() 
        # EPOCH HOOK
        self.model.epoch_hook(epoch, loop_type, loss, prec1, prec5)

        return loss_.avg, prec1_.avg, prec5_.avg

