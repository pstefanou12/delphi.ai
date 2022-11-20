'''
Parent class for models to train in delphi trainer.
'''


import torch as ch
from torch.optim import SGD, Adam, lr_scheduler
import numpy as np
import cox
from cox.store import Store
from time import time
from tqdm import tqdm
import copy
import warnings
from typing import Any, List
import collections

from .utils.helpers import ckpt_at_epoch, AverageMeter, setup_store_with_metadata, Parameters
from .utils.defaults import check_and_fill_args, TRAINER_DEFAULTS, DELPHI_DEFAULTS, check_and_fill_args
from .utils import constants as consts

# CONSTANTS 
# default parameters for delphi module (can be overridden by any class
DEFAULTS = {
        'epochs': (int, 1),
        'num_trials': (int, 3),
        'val': (float, .2),
        'lr': (float, 1e-1), 
        'step_lr': (int, 100),
        'step_lr_gamma': (float, .9), 
        'custom_lr_multiplier': (str, None), 
        'momentum': (float, 0.0), 
        'weight_decay': (float, 0.0), 
        'l1': (float, 0.0), 
        'eps': (float, 1e-5),
        'batch_size': (int, 10),
        'tol': (float, 1e-3),
        'workers': (int, 0),
        'early_stopping': (bool, False), 
        'n_iter_no_change': (int, 5)
}

# CONSTANTS
INFINITY = float('inf')

EVAL_LOGS_SCHEMA = {
    'test_prec1':float,
    'test_loss':float,
    'time':float
}

# CONSTANTS 
BY_ALG = 'by algorithm'  # default parameter depends on algorithm
ADAM = 'adam'
CYCLIC = 'cyclic'
COSINE = 'cosine'
LINEAR = 'linear'

class delphi(ch.nn.Module):
    '''
    Parent/abstract class for models to be passed into trainer.
    '''
    def __init__(self, 
                args: Parameters, 
                defaults: dict={},
                store: Store=None, 
                checkpoint=None): 
        '''
        Args: 
            args (delphi.utils.helpers.Parameters) : parameter object holding hyperparameters
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
                constant (bool)
                    Boolean indicating to have a constant learning rate for procedure
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
        '''
        super().__init__()
        self.defaults = defaults
        self.defaults.update(DELPHI_DEFAULTS)
        assert isinstance(args, Parameters), "args is type: {}. expecting args to be type delphi.utils.helpers.Parameters"
        self.args = check_and_fill_args(args, self.defaults)

        self.best_loss, self.best_model = None, None
        self.optimizer, self.schedule = None, None

        assert store is None or isinstance(store, cox.store.Store), "provided store is type: {}. expecting logging store cox.store.Store".format(type(store))
        self.store = store 
        
        assert checkpoint is None or isinstance(checkpoint, dict), "prorvided checkpoint is type: {}. expecting checkpoint dictionary".format(type(checkpoint))
        self.checkpoint = checkpoint

        self.criterion = ch.nn.CrossEntropyLoss()
        self.criterion_params = []

    def pretrain_hook(self, train_loader) -> None:
        '''
        Hook called before training procedure begins.
        '''
        pass 

    def __call__(self, inp, targ=None) -> [ch.Tensor, float, float]:
        '''
        Forward pass for the model during training/evaluation.
        Args: 
            batch (Iterable) : iterable of inputs that 
        Returns: 
            list with loss, top 1 accuracy, and top 5 accuracy
        '''
        pass 

    def pre_step_hook(self, inp) -> None: 
        '''
        Hook called after .backward call, but before taking a step 
        with the optimizer. 
        ''' 
        pass

    def iteration_hook(self, i, is_train, loss, batch) -> None:
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

    def epoch_hook(self, i, is_train, loss) -> None:
        '''
        Epoch hook for defined model. Method is called after each 
        complete iteration through dataset.
        '''
        pass 

    def post_training_hook(self) -> None:
        '''
        Post training hook, called after sgd procedures completes. By default returns True, 
        so that procedure terminates by default after one trial.
        '''
        pass

    def description(self, epoch, i, loop_msg, loss_, prec1_, prec5_, reg_term):
        '''
        Returns string description for model at each iteration.
        '''
        return ('{2} Epoch:{0} | Loss {loss.avg:.4f} | '
        'Prec1: {top1_acc:.3f} | Prec5: {top5_acc:.3f} | '
        'Reg term: {reg} ||'.format( epoch, i, loop_msg, 
        loss=loss_, top1_acc=float(prec1_.avg), top5_acc=float(prec5_.avg), reg=float(reg_term)))

    def regularize(self, batch) -> ch.Tensor:
        '''
        Regularizer method to apply to loss function. By default retruns 0.0.
        '''
        return ch.zeros(1, 1)

    def make_optimizer_and_schedule(self):
        """
        Create optimizer (ch.nn.optim) and scheduler (ch.nn.optim.lr_scheduler module)
        for SGD procedure. 
        """
        params = self._parameters if (self._parameters is not None and len(self._parameters) != 0) else [i[1] for i in self.named_parameters()]
        if isinstance(params, collections.OrderedDict): params = params.values()
        if self.args.cuda: self.to('cuda')
        self.optimizer = SGD(params, lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        if self.args.custom_lr_multiplier == ADAM: 
            self.optimizer = Adam(params, lr=self.args.lr, weight_decay=self.args.weight_decay)
        elif not self.args.constant: 
            if self.args.custom_lr_multiplier == CYCLIC and self.M is not None: # cyclic
                lr_func = lambda t: np.interp([t], [0, self.M*4//15, self.M], [0, 1, 0])[0]
                self.schedule = lr_scheduler.LambdaLR(self.optimizer, lr_func)
            elif self.args.custom_lr_multiplier == COSINE and self.M is not None:
                schedule = lr_scheduler.CosineAnnealingLR(self.optimizer, self.M)
            elif self.args.custom_lr_multiplier:
                cs = self.args.custom_lr_multiplier
                periods = eval(cs) if type(cs) is str else cs
                if self.args.lr_interpolation == LINEAR:
                    lr_func = lambda t: np.interp([t], *zip(*periods))[0]
                else:
                    def lr_func(ep):
                        for (milestone, lr) in reversed(periods):
                            if ep >= milestone: return lr
                        return 1.0
                self.schedule = lr_scheduler.LambdaLR(self.optimizer, lr_func)
            elif self.args.step_lr:
                self.schedule = lr_scheduler.StepLR(self.optimizer, step_size=self.args.step_lr, 
                gamma=self.args.step_lr_gamma)
            
        if self.checkpoint:
            self.optimizer.load_state_dict(self.checkpoint['optimizer'])
            try:
                schedule.load_state_dict(self.checkpoint['schedule'])
            # if can't load scheduler state, take epoch steps
            except:
                steps_to_take = self.checkpoint['epoch']
                print('Could not load schedule (was probably LambdaLR).'
                    f' Stepping {steps_to_take} times instead...')
                for i in range(steps_to_take):
                    self.schedule.step() 

    @property
    def model(self): 
        '''
        Model property.
        '''
        if self._model is None: 
            warnings.warn('model is None')
        return self._model

    @model.setter
    def model(self, model: Any): 
        '''
        Model setter.
        '''
        self._model = model
    
    @property
    def parameters(self):
        return self._parameters


"""
TODO: make this a static method
"""
        
def eval_model(self, loader):
    """
    Evaluate a model for standard (and optionally adversarial) accuracy.
    Args:
        loader (Iterable) : a dataloader serving batches from the test set
        store (cox.Store) : store for saving results in (via tensorboardX)
    Returns: 
        schema with model performance metrics
    """
    start_time = time.time()

    if self.store is not None:
        self.store.add_table('eval', consts.EVAL_LOGS_SCHEMA)

    writer = self.store.tensorboard if self.store else None
    test_prec1, test_loss = self.model_loop(VAL, loader, 1)

    if self.store:
        log_info = {
            'test_prec1': test_prec1,
            'test_loss': test_loss,
            'time': time.time() - start_time
        }
        self.store['eval'].append_row(log_info)
    return log_info



def train_model(args, model, train_loader, val_loader, store=None, checkpoint=None):
    """
    Train model. 
    Args: 
        loaders (Iterable) : iterable with the train and validation set DataLoaders
    Returns: 
        Trained model
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

    best_loss, best_params = INFINITY, None
    for trial in range(args.trials):
        if args.verbose: print(f'trial: {trial + 1}')

        t_start = time()
        no_improvement_count = 0

        model.pretrain_hook(train_loader)
        model.make_optimizer_and_schedule()
            
        if checkpoint:
            epoch = checkpoint['epoch']
            best_prec1 = checkpoint['prec1'] if 'prec1' in checkpoint else model_loop_(args, model, val_loader, epoch, False)[0]
        
        for epoch in range(1, args.epochs + 1):
            train_loss, train_prec1, train_prec5 = model_loop_(args, model, train_loader, epoch, True)
                                
            if val_loader is not None:
                with ch.no_grad():
                    val_loss, val_prec1, val_prec5 = model_loop_(args, model, val_loader, epoch, False)
                    
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
            if args.early_stopping: 
                if args.tol > -INFINITY and ch.abs(val_loss - best_loss) <= args.tol:
                    no_improvement_count += 1
                else: 
                    no_improvement_count = 0
                if best_params is None or val_loss < best_loss: 
                    best_params, best_loss = copy.copy(model.parameters), val_loss
                if no_improvement_count >= args.n_iter_no_change:
                    if args.verbose: 
                        print("Convergence after %d epochs took %.2f seconds" % (epoch, time() - t_start))
                    break
                    
        model.post_training_hook()
        if best_params is None or val_loss < best_loss: 
            best_params, best_loss = copy.copy(model.parameters), val_loss 
        
    if args.early_stopping and args.verbose and no_improvement_count < args.n_iter_no_change: 
        print('Procedure did not converge after %d epochs and %.2f seconds' % (epoch, time() - t_start))

    return best_params, best_loss

def model_loop_(args: Parameters, 
                model: delphi, 
                loader: ch.utils.data.DataLoader, 
                epoch: int, 
                is_train: bool):
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
    loop_msg = 'Train' if is_train else 'Val'
    loss_, prec1_, prec5_ = AverageMeter(), AverageMeter(), AverageMeter()
    iterator = tqdm(enumerate(loader), total=len(loader), leave=False, 
                    bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') if args.verbose else enumerate(loader) 
    for i, batch in iterator:
        model.optimizer.zero_grad()
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
            model.pre_step_hook(inp)
            model.optimizer.step()
                        
        if args.verbose:
            desc = model.description(epoch, i, loop_msg, loss_, prec1_, prec5_, reg_term)
            iterator.set_description(desc)
 
        model.iteration_hook(i, is_train, loss, batch)
    model.epoch_hook(epoch, is_train, loss)

    return loss_.avg, prec1_.avg, prec5_.avg