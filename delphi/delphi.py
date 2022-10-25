'''
Parent class for models to train in delphi trainer.
'''


import torch as ch
import warnings
from typing import Any

from .utils.defaults import DELPHI_DEFAULTS, check_and_fill_args

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


class delphi:
    '''
    Parent/abstract class for models to be passed into trainer.
    '''
    def __init__(self, args): 
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
        # check to make sure that hyperparameters arre valid
        self.args = check_and_fill_args(args, DELPHI_DEFAULTS)
        self.params = None
        # algorithm optimizer and scheduler
        self._model= None

    def pretrain_hook(self) -> None:
        '''
        Hook called before training procedure begins.
        '''
        pass 

    def __call__(self, batch) -> [ch.Tensor, float, float]:
        '''
        Forward pass for the model during training/evaluation.
        Args: 
            batch (Iterable) : iterable of inputs that 
        Returns: 
            list with loss, top 1 accuracy, and top 5 accuracy
        '''
        pass 

    def pre_step_hook(self) -> None: 
        '''
        Hook called after .backward call, but before taking a step 
        with the optimizer. 
        ''' 
        pass

    def iteration_hook(self, i, loop_type, loss, prec1, prec5, batch) -> None:
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

    def epoch_hook(self, i, loop_type, loss, prec1, prec5) -> None:
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
        if self.params: 
            return self.params 
        return self.model.parameters()
