'''
Parent class for models to train in delphi trainer.
'''

import torch as ch
import cox
from cox.store import Store
from typing import Iterable
from torch.optim import SGD, LBFGS, Adam, lr_scheduler
import numpy as np

from .delphi_logger import delphiLogger
from .utils.defaults import check_and_fill_args, DELPHI_DEFAULTS, SGD_DEFAULTS, LBFGS_DEFAULTS, ADAM_DEFAULTS
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
                logger: delphiLogger,
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
        assert isinstance(args, Parameters), "args is type: {}. expecting args to be type delphi.utils.helpers.Parameters"
        self.args = check_and_fill_args(args, DELPHI_DEFAULTS)
        self.logger = logger

        self.best_loss, self.best_model = None, None
        self.optimizer, self.schedule = None, None

        assert store is None or isinstance(store, cox.store.Store), "provided store is type: {}. expecting logging store cox.store.Store".format(type(store))
        self.store = store 
        
        assert checkpoint is None or isinstance(checkpoint, dict), "prorvided checkpoint is type: {}. expecting checkpoint dictionary".format(type(checkpoint))
        self.checkpoint = checkpoint

        self.criterion = None
        self.criterion_params = []
        self.model = None
        self.optimizer = None 
        self.schedule = None

    def make_optimizer_and_schedule(self, 
                                    params: Iterable, 
                                    checkpoint: dict=None):
        """
        Comprehensive optimizer and scheduler factory with full PyTorch support.
        """
        params = list(params)
    
        self.optimizer = self._create_optimizer(params)
        self.optimizer.register_step_pre_hook(self.step_pre_hook)
        self.optimizer.register_step_post_hook(self.step_post_hook)

        self.schedule = self._create_scheduler()
    
        if checkpoint:
            self._load_checkpoint(checkpoint)
    
        return self.optimizer, self.schedule

    def _create_optimizer(self, params):
        """Create optimizer with comprehensive PyTorch parameter support"""
        optimizer_type = self._get_optimizer_type()
    
        optimizer_creators = {
            'sgd': self._create_sgd,
            'lbfgs': self._create_lbfgs,
            'adam': self._create_adam, 
        }
    
        if optimizer_type not in optimizer_creators:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
    
        return optimizer_creators[optimizer_type](params)

    def _get_optimizer_type(self):
        """Determine which optimizer to use"""
        return self.args.optimizer.lower()
    
    def _remove_none_config(self, config): 
        """Filter out None values (use PyTorch defaults)"""
        return {k: v for k, v in config.items() if v is not None}

    def _create_sgd(self, params):
        """Create SGD optimizer with all PyTorch parameters"""
        # check_and_fill_args(self.args, SGD_DEFAULTS)
        config = {
            'lr': self.args.lr,
            'momentum': getattr(self.args, 'momentum', 0),
            'dampening': getattr(self.args, 'dampening', 0),
            'weight_decay': getattr(self.args, 'weight_decay', 0),
            'nesterov': getattr(self.args, 'nesterov', False),
            'maximize': getattr(self.args, 'maximize', False),
            'foreach': getattr(self.args, 'foreach', None),
            'differentiable': getattr(self.args, 'differentiable', False),
            'fused': getattr(self.args, 'fused', None),
        }
        return  SGD(params, **self._remove_none_config(config))
    
    def _create_lbfgs(self, params):
        """Create L-BFGS optimizer with all PyTorch parameters"""
        check_and_fill_args(self.args, LBFGS_DEFAULTS)
        config = {
            'lr': getattr(self.args, 'lr', 1.0),
            'max_iter': getattr(self.args, 'max_iter', 20),
            'max_eval': getattr(self.args, 'max_eval', None),
            'tolerance_grad': getattr(self.args, 'tolerance_grad', 1e-7),
            'tolerance_change': getattr(self.args, 'tolerance_change', 1e-9),
            'history_size': getattr(self.args, 'history_size', 100),
            'line_search_fn': getattr(self.args, 'line_search_fn', None),
        }
        return  LBFGS(params, **self._remove_none_config(config) )

    def _create_adam(self, params):
        """Create Adam optimizer with all PyTorch parameters"""
        check_and_fill_args(self.args, ADAM_DEFAULTS)
        config = {
            'lr': getattr(self.args, 'lr', 1e-1),
            'betas': (
                getattr(self.args, 'beta1', 0.9),
                getattr(self.args, 'beta2', 0.999)
            ),
            'eps': getattr(self.args, 'eps', 1e-8),
            'weight_decay': getattr(self.args, 'weight_decay', 0),
            'amsgrad': getattr(self.args, 'amsgrad', False),
            'maximize': getattr(self.args, 'maximize', False),
            'foreach': getattr(self.args, 'foreach', None),
            'capturable': getattr(self.args, 'capturable', False),
            'differentiable': getattr(self.args, 'differentiable', False),
            'fused': getattr(self.args, 'fused', None),
        }
        return Adam(params, **self._remove_none_config(config))

    def _create_scheduler(self):
        """Create learning rate scheduler"""
        if getattr(self.args, 'constant', False):
            return None
    
        scheduler_type = self._get_scheduler_type()
        if not scheduler_type:
            return None
    
        scheduler_creators = {
            'cyclic': self._create_cyclic_scheduler,
            'cosine': self._create_cosine_scheduler,
            'step': self._create_step_scheduler,
            'multi_step': self._create_multi_step_scheduler,
            'exponential': self._create_exponential_scheduler,
            'reduce_on_plateau': self._create_plateau_scheduler,
        }
    
        if scheduler_type not in scheduler_creators:
            raise ValueError(f"Unsupported scheduler: {scheduler_type}")
    
        return scheduler_creators[scheduler_type]()

    def _get_scheduler_type(self):
        """Determine which scheduler to use"""
        # Priority: explicit scheduler > custom_lr_multiplier > step_lr
        if hasattr(self.args, 'scheduler') and self.args.scheduler:
            return self.args.scheduler.lower()
        return None

    def _create_cyclic_scheduler(self):
        """Create cyclic learning rate scheduler"""
        epochs = getattr(self.args, 'epochs', 100)
        lr_func = lambda t: np.interp([t], [0, epochs*4//15, epochs], [0, 1, 0])[0]
        return lr_scheduler.LambdaLR(self.optimizer, lr_func)

    def _create_cosine_scheduler(self):
        """Create cosine annealing scheduler"""
        config = {
            'T_max': getattr(self.args, 'epochs', 100),
            'eta_min': getattr(self.args, 'min_lr', 0),
        }
        return lr_scheduler.CosineAnnealingLR(self.optimizer, **self._remove_none_config(config))

    def _create_step_scheduler(self):
        """Create step LR scheduler"""

        config = {
            'step_size': getattr(self.args, 'step_lr', 100),
            'gamma': getattr(self.args, 'step_lr_gamma', 0.1),
        }
        return lr_scheduler.StepLR(self.optimizer, **self._remove_none_config(config))

    def _create_multi_step_scheduler(self):
        """Create multi-step LR scheduler"""
        config = {
            'milestones': getattr(self.args, 'milestones', [30, 60, 90]),
            'gamma': getattr(self.args, 'gamma', 0.1),
        }
        return lr_scheduler.MultiStepLR(self.optimizer, **self._remove_none_config(config))

    def _create_exponential_scheduler(self):
        """Create exponential LR scheduler"""
        config = {
            'gamma': getattr(self.args, 'gamma', 0.95),
        }
        return lr_scheduler.ExponentialLR(self.optimizer, **config)

    def _create_plateau_scheduler(self):
        """Create reduce-on-plateau scheduler"""
        config = {
            'mode': getattr(self.args, 'plateau_mode', 'min'),
            'factor': getattr(self.args, 'plateau_factor', 0.1),
            'patience': getattr(self.args, 'plateau_patience', 10),
            'threshold': getattr(self.args, 'plateau_threshold', 1e-4),
            'threshold_mode': getattr(self.args, 'plateau_threshold_mode', 'rel'),
            'cooldown': getattr(self.args, 'plateau_cooldown', 0),
            'min_lr': getattr(self.args, 'min_lr', 0),
            'eps': getattr(self.args, 'plateau_eps', 1e-8),
        }
        config = {k: v for k, v in config.items() if v is not None}
        return lr_scheduler.ReduceLROnPlateau(self.optimizer, **config)

    def _load_checkpoint(self, checkpoint):
        """Properly restore optimization state from checkpoint"""
        if 'optimizer' in checkpoint and hasattr(self, 'optimizer'):
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.logger.info("✓ Loaded optimizer state from checkpoint")
    
        if 'scheduler' in checkpoint and hasattr(self, 'schedule') and self.schedule is not None:
            self.schedule.load_state_dict(checkpoint['scheduler'])
            self.logger.info("✓ Loaded scheduler state from checkpoint")
        
            # CRITICAL: Step the scheduler to the correct position
            # The checkpoint contains the state AT THE TIME OF SAVING
            # We need to advance to the current training position
            current_epoch = checkpoint.get('epoch', 0)
            for _ in range(current_epoch):
                self.schedule.step()
            self.logger.info(f"✓ Advanced scheduler to epoch {current_epoch}")
    
        if 'random_states' in checkpoint:
            self._load_random_states(checkpoint['random_states'])
    
        if 'training_state' in checkpoint:
            self._load_training_state(checkpoint['training_state'])

    def _load_random_states(self, random_states):
        """Restore random number generator states"""
        if 'python' in random_states:
            import random
            random.setstate(random_states['python'])
    
        if 'numpy' in random_states:
            import numpy as np
            np.random.set_state(random_states['numpy'])
    
        if 'pytorch' in random_states:
            ch.set_rng_state(random_states['pytorch'])
    
        self.logger.info("✓ Restored random number generator states")

    def _load_training_state(self, training_state):
        """Restore custom training state"""
        self.start_epoch = training_state.get('epoch', 0)
        self.best_loss = training_state.get('best_loss', float('inf'))
        self.train_losses = training_state.get('train_losses', [])
        self.val_losses = training_state.get('val_losses', [])
        self.logger.info(f"✓ Resuming from epoch {self.start_epoch}")

    def pretrain_hook(self) -> None:
        '''
        Hook called before training procedure begins.
        '''
        pass 

    def step_pre_hook(self, 
                      optimizer, 
                      args, 
                      kwargs) -> None: 
        '''
        Hook called after .backward call, but before taking a step 
        with the optimizer. 
        ''' 
        pass

    def step_post_hook(self, 
                       optimizer, 
                       args, 
                       kwargs) -> None:
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

    def post_epoch_hook(self, i, is_train, loss) -> None:
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

    def description(self, epoch, i, loss_, prec1_, prec5_, reg_term):
        '''
        Returns string description for model at each iteration.
        '''
        return ('{2} Epoch:{0} | Loss {loss.avg:.4f} | '
                'Prec1: {top1_acc:.3f} | Prec5: {top5_acc:.3f} | '
                'Reg term: {reg} ||'.format( epoch, i, self.model.training, 
                loss=loss_, top1_acc=float(prec1_.avg), top5_acc=float(prec5_.avg), reg=float(reg_term)))

    def regularize(self, batch) -> ch.Tensor:
        '''
        Regularizer method to apply to loss function. By default retruns 0.0.
        '''
        return ch.zeros(1, 1)

    def parameters_(self):
        return self.parameters()