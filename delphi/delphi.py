'''
Parent class for models to train in delphi trainer.
'''

import torch as ch
import cox
from cox.store import Store
import warnings
from typing import Any, Iterable
from torch.optim import SGD, Adam, lr_scheduler
import numpy as np

from .utils.defaults import check_and_fill_args, DELPHI_DEFAULTS, check_and_fill_args
from .utils.helpers import Parameters

# CONSTANTS 
BY_ALG = 'by algorithm'  # default parameter depends on algorithm
ADAM = 'adam'
CYCLIC = 'cyclic'
COSINE = 'cosine'
LINEAR = 'linear'

EVAL_LOGS_SCHEMA = {
    'test_prec1':float,
    'test_loss':float,
    'time':float
}

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

    def make_optimizer_and_schedule(self,
                                    params: Iterable, 
                                    checkpoint: dict=None):
        """
        Create optimizer (ch.nn.optim) and scheduler (ch.nn.optim.lr_scheduler module)
        for SGD procedure. 
        """
        optimizer, schedule = None, None
        if self.args.cuda: params.to('cuda')
        optimizer = SGD(params, lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        if self.args.custom_lr_multiplier == ADAM: 
            optimizer = Adam(params, lr=self.args.lr, weight_decay=self.args.weight_decay)
        elif not self.args.constant: 
            eps = self.args.epochs
            if self.args.custom_lr_multiplier == CYCLIC:
                lr_func = lambda t: np.interp([t], [0, eps*4//15, eps], [0, 1, 0])[0]
                schedule = lr_scheduler.LambdaLR(optimizer, lr_func)
            elif self.args.custom_lr_multiplier == COSINE:
                schedule = lr_scheduler.CosineAnnealingLR(optimizer, eps)
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
                schedule = lr_scheduler.LambdaLR(optimizer, lr_func)
            elif self.args.step_lr:
                schedule = lr_scheduler.StepLR(optimizer, step_size=self.args.step_lr, gamma=self.args.step_lr_gamma)
            
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