"""
Module used for training models.
"""
import torch as ch
from torch.utils.data import DataLoader 
from time import time
from tqdm import tqdm
from cox.store import Store

from .delphi import delphi
from .utils import constants as consts
from .utils.helpers import AverageMeter, setup_store_with_metadata, Parameters
from .utils.defaults import TRAINER_DEFAULTS, check_and_fill_args

STOP_REASONS = ["grad_tol", "loss_tol", "early_stop", "max_gradient_steps"]

def ensure_tuple(x):
    return x if isinstance(x, (tuple, list)) else (x,)

class Trainer:
    def __init__(self, 
                 model: delphi, 
                 args: Parameters, 
                 store=None): 
        self.model = model
        self.args = check_and_fill_args(args, TRAINER_DEFAULTS)
        self.store = store

        self.epoch, self.iterations = 0, None
        # store procedure history within lists because of torch.cat performance overhead
        self.train_losses = []
        self.val_losses = []
        self.train_param_history = []
        self.val_param_history = []
        self.grad_norms = []
        
        self.t_start = None
        self._best_loss, self._best_params = float('inf'), None
        self._final_loss, self._final_params = None, None
        self.stop_reason = None

    def make_closure(self, batch): 
        def closure(): 
            inp, targ = batch
            self.model.optimizer.zero_grad()
            
            pred = self.model(inp, targ)
            loss = self.model.criterion(*ensure_tuple(pred), targ, *self.model.criterion_params)
            if len(loss.shape) > 0: loss = loss.sum(0)
            reg_term = self.model.regularize(batch)

            if self.args.cuda:
                reg_term = reg_term.cuda()
            loss = loss + reg_term
            
            loss.backward()
            return loss 
        return closure

    def model_loop_(self, 
                    loader: DataLoader, 
                    is_train: bool):
        loop_msg = 'Train' if is_train else 'Val'
        loss_, prec1_, prec5_ = AverageMeter(), AverageMeter(), AverageMeter()
        iterator = tqdm(enumerate(loader), total=len(loader), leave=False) if self.args.verbose and self.args.tqdm else enumerate(loader)
        
        for i, batch in iterator:
            flat_params = ch.cat([param.flatten() for param in list(self.model.parameters_())]).detach().clone()
            if is_train: 
                self.train_param_history.append(flat_params)
            else: 
                self.val_param_history.append(flat_params)

            if is_train: 
                loss = self.model.optimizer.step(self.make_closure(batch))
                if self.model.schedule is not None: self.model.schedule.step()

                self.train_losses.append(loss)
                flat_grads = ch.cat([param.grad.flatten() for param in list(self.model.parameters_()) if param.grad is not None]).detach().clone() 
                grad_norm = flat_grads.norm()
                self.grad_norms.append(grad_norm.item())
                
                self.iterations += 1
            else:
                inp, targ = batch
                pred = self.model(inp, targ)
                loss = self.model.criterion(*ensure_tuple(pred), targ, *self.model.criterion_params)
                if len(loss.shape) > 0: loss = loss.sum(0)
                self.val_losses.append(loss)

            loss_.update(loss.item()) # Use item() to store scalar value
            # OPTIONAL: calculate precision metrics (prec1, prec5) here if your model supports it
            # if not, you'll need to define what these mean for your model and loss.

            # stopping criteria
            stop, reason = self.should_stop()
            if stop:
                self.stop_reason = reason
                break

            if self.args.verbose:
                if self.args.tqdm:
                    desc = self.model.description(self.epoch, i, loop_msg, loss_, prec1_, prec5_, reg_term)
                    iterator.set_description(desc)
            
            
        self.model.post_epoch_hook(self.epoch, is_train, loss)
        return loss_.avg, prec1_.avg, prec5_.avg
    
    def should_stop(self): 
        if self.stop_reason not in STOP_REASONS:
            # Criterion 1: max iterations
            if self.args.gradient_steps is not None and self.iterations >= self.args.gradient_steps:
                return True, "max_gradient_steps"

            # Criterion 2: gradient norm small (scipy-like)
            if self.args.grad_tol is not None and len(self.grad_norms) > 1 and (self.grad_norms[-1]) < self.args.grad_tol:
                return True, "grad_tol"

            # Criterion 3: loss change small
            if self.args.loss_tol is not None and len(self.train_losses) > 1:
                if abs(self.train_losses[-1] - self.train_losses[-2]) < self.args.loss_tol:
                    return True, "loss_tol"

            # Criterion 4: early stopping on val loss
            if len(self.val_losses) > self.args.patience:
                recent = self.val_losses[-self.args.patience:]
                if recent[-1] > min(recent[:-1]):
                    return True, "early_stop"

        return False, None

    def train_model(self,
                    train_loader: DataLoader, 
                    val_loader: DataLoader, 
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

        if store is not None: 
            store.add_table('logs', {
                'trial': int,
                'epoch': int,
                'train_loss': float, 
                'train_prec1': float,
                'train_prec5': float, 
                'val_loss': float,
                'val_prec1': float, 
                'val_prec5': float
            })
            setup_store_with_metadata(self.args, store)

        # stores model estimates after each gradient step
        for trial in range(self.args.trials):
            self.iterations = 0

            ch.manual_seed(rand_seed)
            if self.args.verbose: print(f'trial: {trial + 1}')

            self.t_start = time()
            self.model.pretrain_hook(train_loader)
            self.model.make_optimizer_and_schedule(self.model.parameters_()) 
   
            if checkpoint:
                epoch = checkpoint['epoch']
                best_loss, best_prec1, best_prec5 = checkpoint['prec1'] if 'prec1' in checkpoint else self.model_loop_(val_loader, False)[0]

            for epoch in range(1, self.args.epochs + 1):
                self.epoch = epoch
                train_loss, train_prec1, train_prec5 = self.model_loop_(train_loader, True)

                if val_loader is not None:
                    with ch.no_grad():
                        val_loss, val_prec1, val_prec5 = self.model_loop_(val_loader, False)

                        if val_loss < self.best_loss: 
                            self._best_loss = val_loss
                            self._best_params = self.val_param_history[-1]

                    if self.args.verbose: 
                        print(f'Epoch {epoch} - Loss: {val_loss}')

                if self.stop_reason in STOP_REASONS: 
                    if self.args.verbose:
                        print("Stopped due to %s after %d gradient steps. \n Total time: %.2f seconds" % (self.stop_reason, self.iterations, time() - self.t_start))
                    break
            
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

            self._final_loss = val_loss 
            self._final_params = self.val_param_history[-1] 

            self.model.post_training_hook()

        # convert training loss/param history to tensors
        self.train_losses = ch.tensor(self.train_losses)
        self.val_losses   = ch.tensor(self.val_losses)
        self.train_param_history = ch.stack(self.train_param_history)
        self.val_param_history = ch.stack(self.val_param_history)
        self.grad_norms = ch.Tensor(self.grad_norms)

        if self.stop_reason is None or self.stop_reason == 'max_gradient_steps': 
            print('Procedure did not converge after %d epochs in %.2f seconds' % (self.epoch, time() - self.t_start))
    
    def eval_model(self, 
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
        test_prec1, test_loss = self.model_loop_(loader, is_train=False)

        if store:
            log_info = {
                'test_prec1': test_prec1,
                'test_loss': test_loss,
                'time': time.time() - start_time
            }
            store['eval'].append_row(log_info)
        return log_info

    @property
    def best_params(self): 
        return self._best_params
    
    @property
    def best_loss(self): 
        return self._best_loss
    
    @property
    def final_params(self): 
        return self._final_params

    @property
    def final_loss(self): 
        return self._final_loss
