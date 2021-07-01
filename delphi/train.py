import time
import os
import warnings
import dill
import numpy as np
import torch as ch
from torch import Tensor
from torch.optim import SGD, Adam
from torch.optim import lr_scheduler
import cox
from typing import Any, Iterable, Callable

from . import oracle
from .utils.helpers import has_attr, ckpt_at_epoch, AverageMeter, accuracy, type_of_script, LinearUnknownVariance, setup_store_with_metadata, ProcedureComplete
from .utils import constants as consts
from .attacker import AttackerModel

# determine running environment
script = type_of_script()
if script == consts.JUPYTER:
    from tqdm.autonotebook import tqdm as tqdm
else:
    from tqdm import tqdm


# CONSTANTS 
LOGS_SCHEMA = None


class Trainer: 
    """
    Trainer class for training models in pytorch.
    """

    def __init__(self, 
                args: cox.Parameters, 
                model: Any, 
                train_loader: ch.utils.data.DataLoader, 
                val_loader: ch.utils.data.DataLoader = None,
                phi: oracle = None, 
                train_step: Callable = None,
                iteration_hook: Callable = None, 
                epoch_hook: Callable = None, 
                checkpoint: dict = None, 
                parallel: bool = False, 
                device:str = None, 
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
                iteration_hook (function, optional)
                    If given, this function is called every training iteration by
                    the training loop (useful for custom logging). The function is
                    given arguments `model, iteration #, loop_type [train/eval],
                    current_batch_ims, current_batch_labels`.
                epoch hook (function, optional)
                    Similar to iteration_hook but called every epoch instead, and
                    given arguments `model, log_info` where `log_info` is a
                    dictionary with keys `epoch, nat_prec1, adv_prec1, nat_loss,
                    adv_loss, train_prec1, train_loss`.
            model (AttackerModel) : model to train 
            train_loader: (ch.utils.data.DataLoader) : training set data loader 
            val_loader: (ch.utils.data.DataLoader): validation set data loader 
            phi: (delphi.oracle) : optional oracle object, depending on what model you are training 
            criterion (ch.Autograd.Function) : instantiated loss object for training model
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
        self.phi = phi 
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
        self.T = self.args.epochs if self.args.epochs else self.args.steps
        self.make_optimizer_and_schedule()


    def make_optimizer_and_schedule(self):
        """
        Create optimizer (ch.nn.optim.Optimizer) and scheduler (ch.nn.optim.lr_schedulers module)
        for SGD procedure. 
        """
        # initialize optimizer, scheduler, and then get parameters
        param_list = self.model.parameters() if self.params is None else self.params

        # check for Adam optimizer
        if self.args.adam: 
            optimizer = Adam(param_list, betas=args.betas, lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.amsgrad)
        else: 
            # SGD optimizer
            optimizer = SGD(param_list, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            # cyclic learning rate scheduler
            if args.custom_lr_multiplier == consts.CYCLIC and T is not None:
                lr_func = lambda t: np.interp([t], [0, T*4//15, T], [0, 1, 0])[0]
                schedule = lr_scheduler.LambdaLR(optimizer, lr_func)
            # cosine annealing scheduler
            elif args.custom_lr_multiplier == consts.COSINE and T is not None:
                schedule = lr_scheduler.CosineAnnealingLR(optimizer, T)
            elif args.custom_lr_multiplier:
                cs = args.custom_lr_multiplier
                periods = eval(cs) if type(cs) is str else cs
                # constant linear interpolation
                if args.lr_interpolation == consts.LINEAR:
                    lr_func = lambda t: np.interp([t], *zip(*periods))[0]
                # custom lr interpolation
                else:
                    def lr_func(ep):
                        for (milestone, lr) in reversed(periods):
                            if ep >= milestone: return lr
                        return 1.0
                schedule = lr_scheduler.LambdaLR(optimizer, lr_func)
            # step learning rate
            elif args.step_lr:
                schedule = lr_scheduler.StepLR(optimizer, step_size=args.step_lr, gamma=args.step_lr_gamma, verbose=args.verbose)
            
        # if checkpoint load  optimizer and scheduler
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


def eval_model(args, model, loader, store=None, device=None, table=None):
    """
    Evaluate a model for standard (and optionally adversarial) accuracy.
    Args:
        args (object) : A list of arguments---should be a python object
            implementing ``getattr()`` and ``setattr()``.
        model (AttackerModel) : model to evaluate
        loader (iterable) : a dataloader serving `(input, label)` batches from
            the validation set
        store (cox.Store) : store for saving results in (via tensorboardX)
        device (str) : optional parameter, where to run model evaluation
        table (str) : table name, `eval` by default

    Returns: 
        model (AttackerModel)
    """
    # start timer
    start_time = time.time()

    # if store provided, 
    table = consts.EVAL_LOGS_TABLE if table is None else table
    if store is not None:
        store.add_table(table, consts.EVAL_LOGS_SCHEMA)

    writer = store.tensorboard if store else None

    # put model onto device
    if device is not None: 
        model.to(device)

    # run model in parallel model
    assert not hasattr(model, "module"), "model is already in DataParallel."
    if args.parallel and next(model.parameters()).is_cuda:
        model = ch.nn.DataParallel(model)
    test_prec1, test_loss = model_loop(args, 'val', loader,
                                        model, None, ch.nn.CrossEntropyLoss(), None,  0, 0, writer, args.device)

    # log info 
    if store:
        log_info = {
            'test_prec1': test_prec1,
            'test_loss': test_loss,
            'time': time.time() - start_time
        }
        table = consts.EVAL_LOGS_TABLE if table is None else table # table name
        store[table].append_row(log_info)

    return log_info


def train_model(args, model, loaders, *, phi=None, criterion=ch.nn.CrossEntropyLoss(), checkpoint=None, parallel=False, 
                device=None, dp_device_ids=None, store=None, table=None, params=None, disable_no_grad=False):
 
    

    # put the neural network onto gpu and in parallel mode
    assert not has_attr(model, "module"), "model is already in DataParallel."
    if device is not None: 
        model = model.to(device)
    if parallel:
        model = ch.nn.DataParallel(model)

    best_prec1, epoch = (0, 0)
    if checkpoint:
        epoch = checkpoint['epoch']
        best_prec1 = checkpoint['prec1'] if 'prec1' in checkpoint \
            else model_loop(args, 'val', val_loader, model, phi, criterion, optimizer, start_epoch-1, steps, writer=None, device=args.device, schedule=schedule)[0]

    # keep track of the start time
    start_time = time.time()
    steps = 0 if args.steps else None # number of gradient steps taken
    # do training loops until performing enough gradient steps or epochs
    while (args.steps is not None and steps < args.steps) or (args.epochs is not None and epoch < args.epochs):
        try: 
            train_prec1, train_loss = model_loop(args, 'train', train_loader, model, phi, criterion, optimizer, epoch+1, steps, writer, device=args.device, schedule=schedule)
        except ProcedureComplete: 
            return model
        except Exception as e: 
            raise e

        # check for logging/checkpoint
        last_epoch = (epoch == (args.epochs - 1)) if args.epochs else False
        should_save_ckpt = (epoch % args.save_ckpt_iters == 0 or last_epoch) if args.epochs else False
        should_log = (epoch % args.log_iters == 0 or last_epoch) if args.epochs else False

        # validation loop
        val_prec1, val_loss = 0.0, 0.0
        if should_log or should_save_ckpt: 
            ctx = ch.enable_grad() if disable_no_grad else ch.no_grad()

            # evaluate model on validation set, if there is one
            if val_loader is not None:
                with ctx:
                    val_prec1, val_loss = model_loop(args, 'val', val_loader, model,
                            phi, criterion, optimizer, epoch + 1, steps, writer, device=args.device)

                # remember best prec_1 and save checkpoint
                is_best = val_prec1 > best_prec1
                best_prec1 = max(val_prec1, best_prec1)

        # CHECKPOINT -- checkpoint epoch of better DNN performance
        if should_save_ckpt or is_best:
            sd_info = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'schedule': (schedule and schedule.state_dict()),
                'epoch': epoch+1,
                'amp': amp.state_dict() if args.mixed_precision else None,
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
                save_checkpoint(consts.CKPT_NAME_BEST)
            if should_save_ckpt: 
                # if we are at a saving epoch (or the last epoch), save a checkpoint
                save_checkpoint(ckpt_at_epoch(epoch))
                save_checkpoint(consts.CKPT_NAME_LATEST)

        # LOG
        if should_log and store: # TODO: add custom logging hook
            log_info = {
                'epoch': epoch + 1,
                'val_prec1': val_prec1,
                'val_loss': val_loss,
                'train_prec1': train_prec1,
                'train_loss': train_loss,
                'time': time.time() - start_time
            }
            store[table].append_row(log_info)
        
        # UPDATE LR
        if args.epochs is not None and schedule: schedule.step()

        # EPOCH HOOK
        if has_attr(args, 'epoch_hook'): 
            args.epoch_hook(model, epoch)
            
        # GRADIENT STEP COUNTER
        epoch += 1
        if steps is not None: 
            steps += len(train_loader)
        
    # TODO: add end training hook
    return model
            
            
def model_loop(self, loop_type, loader):
    """
    *Internal function* (refer to the train_model and eval_model functions for
    how to train and evaluate models).
    Runs a single epoch of either training or evaluating.
    Args:
        loop_type ('train' or 'val') : whether we are training or evaluating
        loader (iterable) : an iterable loader of the form 
            `(image_batch, label_batch)`
        model (AttackerModel) : model to train/evaluate
        epoch (int) : which epoch we are currently on
        adv (bool) : whether to evaluate adversarially (otherwise standard)
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
    iterator = enumerate(loader) if args.steps else tqdm(enumerate(loader), total=len(loader), leave=False) 
    for i, batch in iterator:
        inp, target, output = None, None, None
        loss = 0.0
        if isinstance(model, ch.distributions.distribution.Distribution):
            loss = criterion(*optimizer.param_groups[0]['params'], *batch)
        elif isinstance(model, ch.nn.Module) or isinstance(model, AttackerModel):
            inp, target = batch
            # put data on device
            if device is not None: 
                inp, target = inp.to(device), target.to(device)
            output = model(inp)
            
            if phi is not None: 
                # model with unknown noise variance
                try:
                    loss = criterion(output, target, model.lambda_, phi)
                except Exception as e:
                    # known noise variance
                    loss = criterion(output, target, phi)


        # training step

        # 



        # measure accuracy and record loss
        top1_acc = float('nan')
        top5_acc = float('nan')

        desc = None  # description for epoch
        # censored, truncated distributions - calculate score
        if args.steps:
            steps += 1 
            if schedule: schedule.step()
        # latent variable models
        else:
            losses.update(loss.item(), inp.size(0))
            # calculate accuracy metrics
            if args.accuracy:
                # accuracy
                maxk = min(5, model_logits.shape[-1])
                if has_attr(args, "custom_accuracy"):
                    prec1, prec5 = args.custom_accuracy(model_logits, target)
                else:
                    prec1, prec5 = accuracy(model_logits, target, topk=(1, maxk))
                    prec1, prec5 = prec1[0], prec5[0]

                top1.update(prec1, inp.size(0))
                top5.update(prec5, inp.size(0))
                top1_acc = top1.avg
                top5_acc = top5.avg

            # ITERATOR
            if args.accuracy: 
                desc = ('Epoch: {0} | Loss {loss.avg:.4f} | '
                        '{1}1 {top1_acc:.3f} | {1}5 {top5_acc:.3f} | '
                        'Reg term: {reg} ||'.format(epoch, loop_msg,
                                                    loss=losses, top1_acc=top1_acc, top5_acc=top5_acc, reg=reg_term))
            else: 
                desc = ('Epoch: {0} | Loss {loss.avg:.4f} | {1}1'
                        'Reg term: {reg} ||'.format(epoch, loop_msg, loss=losses, reg=reg_term))
            
            # set tqdm description
            iterator.set_description(desc)
    
        # USER-DEFINED HOOK
        if has_attr(args, 'iteration_hook'):
            args.iteration_hook(model, optimizer, i, loop_type, inp, target)

        # increment number of gradients
        if steps is not None: 
            steps += 1
            if schedule: schedule.step()

    # write to Tensorboard
    if writer is not None:
        descs = ['loss', 'top1', 'top5']
        vals = [losses, top1, top5]
        for d, v in zip(descs, vals):
            writer.add_scalar('_'.join([loop_type, d]), v.avg,
                              epoch)
    
    # LOSS AND ACCURACY
    return top1.avg, losses.avg



class DefaultTrainStep:
    """
    Default train step is written assuming that the 
    default model is a neural network using cross 
    entropy loss. 
    """
    def __init__(self): 
        # CE Loss
        self.criterion = ch.nn.CrossEntropyLoss()

    def __call__(self, 
                args: cox.Parameters,
                model: Any, 
                optimizer: ch.optim,
                inp: Tensor, 
                targ: Tensor, 
                device: str = 'cuda') -> None: 
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
                regularizer (function, optional) 
                    If given, this function of `model, input, target` returns a
                    (scalar) that is added on to the training loss without being
                    subject to adversarial attack
            model (AttackerModel) : model to train 
            optimizer: (ch.nn.optim) : optimizer for training procedure
            inp: (torch.Tensor) : input Tensor for neural network 
            targ: (torch.Tensor) : targ classification for neural network
            device: (str) : device that the model is on
        """
        # pass data through model
        if device is not None: 
            inp, targ = inp.to(device), targ.to(device)
        output = model(inp)
        # AttackerModel returns both output and final input
        if isinstance(output, tuple):
            model_logits, _ = output

        # regularizer 
        reg_term = 0.0
        if has_attr(args, "regularizer") and isinstance(model, ch.nn.Module):
            reg_term = args.regularizer(model, inp, targ)

        # calculate loss and regularize
        loss = self.criterion(model_logits, targ)
        loss = loss + reg_term

        # zero gradient for model parameters
        optimizer.zero_grad()
        # backward propagation
        loss.backward()
        # update model 
        optimizer.step()

        if len(loss.size()) > 0: loss = loss.mean()




