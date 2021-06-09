import time
import os
import warnings
import dill
import numpy as np
import torch as ch
from torch import Tensor
from torch.optim import SGD, Adam
from torch.optim import lr_scheduler

from .utils.helpers import has_attr, ckpt_at_epoch, AverageMeter, accuracy, type_of_script, LinearUnknownVariance, setup_store_with_metadata, LinearUnknownVariance
from .utils import constants as consts

# determine running environment
script = type_of_script()
if script == consts.JUPYTER:
    from tqdm.autonotebook import tqdm as tqdm
else:
    from tqdm import tqdm


def make_optimizer_and_schedule(args, model, checkpoint, params, T=None):
    param_list = model.parameters() if params is None else params

    # run on mutliple GPUs
    if args.adam: 
        optimizer = Adam(param_list, lr=args.lr, weight_decay=args.weight_decay)
    else: 
        optimizer = SGD(param_list, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Make schedule
    schedule = None
    if args.custom_lr_multiplier == consts.CYCLIC and T is not None:
        lr_func = lambda t: np.interp([t], [0, T*4//15, T], [0, 1, 0])[0]
        schedule = lr_scheduler.LambdaLR(optimizer, lr_func)
    elif args.custom_lr_multiplier == consts.COSINE and T is not None:
        schedule = lr_scheduler.CosineAnnealingLR(optimizer, T)
    elif args.custom_lr_multiplier:
        cs = args.custom_lr_multiplier
        periods = eval(cs) if type(cs) is str else cs
        if args.lr_interpolation == consts.LINEAR:
            lr_func = lambda t: np.interp([t], *zip(*periods))[0]
        else:
            def lr_func(ep):
                for (milestone, lr) in reversed(periods):
                    if ep >= milestone: return lr
                return 1.0
        schedule = lr_scheduler.LambdaLR(optimizer, lr_func)
    elif args.step_lr:
        schedule = lr_scheduler.StepLR(optimizer, step_size=args.step_lr, gamma=args.step_lr_gamma)
        
    # Fast-forward the optimizer and the scheduler if resuming
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        try:
            schedule.load_state_dict(checkpoint['schedule'])
        except:
            steps_to_take = checkpoint['epoch']
            print('Could not load schedule (was probably LambdaLR).'
                  f' Stepping {steps_to_take} times instead...')
            for i in range(steps_to_take):
                schedule.step()
    return optimizer, schedule


def eval_model(args, model, loader, store, table=None):
    """
    Evaluate a model for standard (and optionally adversarial) accuracy.
    Args:
        args (object) : A list of arguments---should be a python object
            implementing ``getattr()`` and ``setattr()``.
        model (AttackerModel) : model to evaluate
        loader (iterable) : a dataloader serving `(input, label)` batches from
            the validation set
        store (cox.Store) : store for saving results in (via tensorboardX)
    """
    start_time = time.time()

    table = consts.EVAL_LOGS_TABLE if table is None else table

    if store is not None:
        store.add_table(table, consts.EVAL_LOGS_SCHEMA)

    writer = store.tensorboard if store else None

    # put model on device
    model.to(args.device)

    assert not hasattr(model, "module"), "model is already in DataParallel."
    if args.parallel and next(model.parameters()).is_cuda:
        model = ch.nn.DataParallel(model)
    test_prec1, test_loss, score = model_loop(args, 'val', loader,
                                        model, None, 0, 0, writer, args.device)
    log_info = {
        'test_prec1': test_prec1,
        'test_loss': test_loss,
        'time': time.time() - start_time
    }

    # Log info into the logs table
    if store:
        store[consts.EVAL_LOGS_TABLE if table is None else table].append_row(log_info)
    return log_info


def train_model(args, model, loaders, *, checkpoint=None, parallel=False, dp_device_ids=None, 
                store=None, table=None, update_params=None, disable_no_grad=False):
    table = consts.LOGS_TABLE if table is None else table

    if store is not None:
        store.add_table(table, consts.LOGS_SCHEMA)

    writer = store.tensorboard if store else None

    # data loaders
    train_loader, val_loader = loaders
    # number of periods/epochs for learning rate schedulers
    T = args.epochs if args.epochs else args.steps
    optimizer, schedule = make_optimizer_and_schedule(args, model, checkpoint, update_params, T=T)

    # put the model into parallel mode
    assert not has_attr(model, "module"), "model is already in DataParallel."

    # run on parallel GPUs
    if parallel:
        model = ch.nn.DataParallel(model)

    if isinstance(model, ch.distributions.distribution.Distribution):
        model.loc.to(args.device)
        model.covariance_matrix.to(args.device)
    else:
        model.to(args.device)

    best_prec1, epoch = (0, 0)
    if checkpoint:
        epoch = checkpoint['epoch']
        best_prec1 = checkpoint['prec1'] if 'prec1' in checkpoint \
            else model_loop(args, 'val', val_loader, model, None, start_epoch-1, steps, writer=None, device=args.device, schedule=schedule)[0]

    # keep track of the start time
    start_time = time.time()
    steps = 0 if args.steps else None # number of gradient steps taken
    # do training loops until performing enough gradient steps or epochs
    while (args.steps is not None and steps < args.steps) or (args.epochs is not None and epoch < args.epochs):
        train_prec1, train_loss, score = model_loop(args, 'train', train_loader, model, optimizer, epoch+1, steps, writer, device=args.device, schedule=schedule)

        # check for logging/checkpoint
        last_epoch = (epoch == (args.epochs - 1)) if args.epochs else (steps >= args.steps)
        should_save_ckpt = ((epoch % args.save_ckpt_iters == 0 or last_epoch) if args.epochs else (steps % args.save_ckpt_iters == 0 or last_epoch)) if args.save_ckpt_iters else False
        should_log = ((epoch % args.log_iters == 0 or last_epoch) if args.epochs else (steps % args.log_iters == 0 or last_epoch)) if args.log_iters else False

        # validation loop
        val_prec1, val_loss, val_score = 0.0, 0.0, 0.0
        if should_log or should_save_ckpt: 
            ctx = ch.enable_grad() if disable_no_grad else ch.no_grad()

            # evaluate model on validation set, if there is one
            if val_loader is not None:
                with ctx:
                    val_prec1, val_loss, val_score = model_loop(args, 'val', val_loader, model,
                            None, epoch + 1, steps, writer, device=args.device)

                # remember best prec_1 and save checkpoint
                is_best = val_prec1 > best_prec1
                best_prec1 = max(val_prec1, best_prec1)

        # save model checkpoint -- for neural networks
        if should_save_ckpt:
            sd_info = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'schedule': (schedule and schedule.state_dict()),
                'epoch': epoch+1,
                'amp': amp.state_dict() if args.mixed_precision else None,
                'prec1': val_prec1
            }
            
            def save_checkpoint(filename):
                ckpt_save_path = os.path.join(args.out_dir if not store else \
                                              store.path, filename)
                ch.save(sd_info, ckpt_save_path, pickle_module=dill)

            # If we are at a saving epoch (or the last epoch), save a checkpoint
            save_checkpoint(ckpt_at_epoch(epoch))
            # Update the latest and best checkpoints (overrides old one)
            save_checkpoint(consts.CKPT_NAME_LATEST)
            if is_best: save_checkpoint(consts.CKPT_NAME_BEST)

        # log results
        if should_log: # TODO: add custom logging hook
            # log every checkpoint
            log_info = {
                'epoch': epoch + 1,
                'val_prec1': val_prec1,
                'val_loss': val_loss,
                'train_prec1': train_prec1,
                'train_loss': train_loss,
                'time': time.time() - start_time
            }

            # log info in log table
            if store: store[table].append_row(log_info)
        
        # update lr
        if args.epochs is not None and schedule: schedule.step()

        if has_attr(args, 'epoch_hook'): 
            args.epoch_hook(model, epoch)

        # check score tolerance
        if args.score and ch.all(ch.where(ch.abs(score) < args.tol, ch.ones(1).to(args.device), ch.zeros(1).to(args.device)).bool()):
            break
            
        # increment epoch counter
        epoch += 1
        # update number of gradient steps taken
        if steps is not None: 
            steps += len(train_loader)
        
    # TODO: add end training hook
    return model
            
            
def model_loop(args, loop_type, loader, model, optimizer, epoch, steps, writer, device, schedule=None):
    # check loop type 
    if not loop_type in ['train', 'val']: 
        err_msg = "loop type must be in {0} must be 'train' or 'val".format(loop_type)
        raise ValueError(err_msg)
    # train or val loop
    is_train = (loop_type == 'train')
    loop_msg = 'Train' if is_train else 'Val'
 
    # algorithm metrics
    losses, score = AverageMeter(), AverageMeter()
    top1, top5 = AverageMeter(), AverageMeter()

    if not isinstance(model, ch.distributions.distribution.Distribution):   
        model = model.train() if is_train else model.eval()
    
    # check for custom criterion
    has_custom_criterion = has_attr(args, 'custom_criterion')
    criterion = args.custom_criterion if has_custom_criterion else ch.nn.CrossEntropyLoss()
    iterator = tqdm(enumerate(loader), total=len(loader), leave=False)

    for i, batch in iterator:
        inp, target, output = None, None, None
        loss = 0.0
        if isinstance(model, ch.distributions.distribution.Distribution):
            loss = criterion(*optimizer.param_groups[0]['params'], *batch)
        elif isinstance(model, ch.nn.Module):
            inp, target = batch
            inp, target = inp.to(device), target.to(device)
            output = model(inp)
            # attacker model returns both output anf final input
            if isinstance(output, tuple):
                output, final_inp = output
            # lambda parameter used for regression with unknown noise variance
            try:
                loss = criterion(output, target, model.lambda_)

            except Exception as e:
                loss = criterion(output, target)

        # regularizer option 
        reg_term = 0.0
        if has_attr(args, "regularizer") and isinstance(model, ch.nn.Module):
            reg_term = args.regularizer(model, inp, target)
        loss = loss + reg_term
        
        # perform backprop and take optimizer step
        if is_train:
            # import pdb; pdb.set_trace()
            optimizer.zero_grad()
            loss.backward()

            # print("weight grad: ", model.weight.grad)
            # print("bias grad: ", model.bias.grad)
            # print("lambda grad: ", model.lambda_.grad)

            # normalize gradient
            if args.norm: 
                # check if unknown variance regression
                if not isinstance(model, LinearUnknownVariance):
                    if model.bias is not None: 
                        grad = ch.cat([model.weight.grad.flatten(), model.bias.grad.flatten()])
                        norm_grad = grad / grad.norm()
                        model.weight.grad = norm_grad[:-1].reshape(model.weight.size())
                        model.bias.grad = norm_grad[-1].reshape(model.bias.size())
                    else: 
                        model.weight.grad = model.weight.grad / model.weight.grad.norm()
                else: 
                    if model.bias is not None: 
                        grad = ch.cat([model.weight.grad.flatten(), model.bias.grad.flatten(), model.lambda_.flatten()])
                        norm_grad = grad / grad.norm()
                        model.weight.grad = norm_grad[:-2].reshape(model.weight.size())
                        model.bias.grad = norm_grad[-2].reshape(model.bias.size())
                        model.lambda_.grad = norm_grad[-1].reshape(model.lambda_.size())
                    else: 
                        grad = ch.cat([model.weight.grad.flatten(), model.lambda_.flatten()])
                        norm_grad = model.weight.grad / moptimiodel.weight.grad.norm()
                        model.weight.grad = norm_grad[:-1].reshape(model.weight.size())
                        model.lambda_.grad = norm_grad[-1].reshape(model.lambda_.size())
            optimizer.step()

        if len(loss.size()) > 0: loss = loss.mean()

        model_logits = None
        if not isinstance(model, ch.distributions.distribution.Distribution):
            model_logits = output[0] if isinstance(output, tuple) else output

        # measure accuracy and record loss
        top1_acc = float('nan')
        top5_acc = float('nan')

        desc = None  # description for epoch
        try:
            # censored, truncated distributions - calculate score
            if isinstance(model, ch.distributions.distribution.Distribution):
                score.update(ch.cat([model.loc.grad, model.covariance_matrix.grad.flatten()]),
                             model.loc.size(0) + model.covariance_matrix.flatten().size(0))
                desc = ('Steps:{0} | Score: {score} \n ||'.format(epoch, loop_msg, score=[round(x, 4) for x in score.avg.tolist()]))
            # regression with unknown variance
            elif inp is not None:
                losses.update(loss.item(), inp.size(0))
                # calculate score
                if args.score:
                    score.update(ch.cat([p.grad.flatten() for p in model.parameters()]), inp.size(0))

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
                if steps is not None: 
                    # TODO: figure out better way to pass descriptions through to iterator
                    if args.accuracy: 
                        desc = ('Steps: {0} | Loss {loss.avg:.4f} | '
                                '{1}1 {top1_acc:.3f} | {1}5 {top5_acc:.3f} | '
                                'Reg term: {reg} ||'.format(steps, loop_msg,
                                                            loss=losses, top1_acc=top1_acc, top5_acc=top5_acc, reg=reg_term))
                    else: 
                        desc = ('Steps: {0} | Loss {loss.avg:.4f} | '
                                'Reg term: {reg} ||'.format(steps, loop_msg, loss=losses, reg=reg_term))
                else: 
                    if args.accuracy: 
                        desc = ('Epoch: {0} | Loss {loss.avg:.4f} | '
                                '{1}1 {top1_acc:.3f} | {1}5 {top5_acc:.3f} | '
                                'Reg term: {reg} ||'.format(epoch, loop_msg,
                                                            loss=losses, top1_acc=top1_acc, top5_acc=top5_acc, reg=reg_term))
                    else: 
                        desc = ('Epoch: {0} | Loss {loss.avg:.4f} | '
                                'Reg term: {reg} ||'.format(epoch, loop_msg, loss=losses, reg=reg_term))
        except Exception as e:
            # ITERATOR
            if steps is not None:
                desc = ('Steps: {0} | Loss {loss.avg:.4f} ||'.format(steps, loop_msg, loss=losses))
            else: 
                desc = ('Epoch:{0} | Loss {loss.avg:.4f} ||'.format(epoch, loop_msg, loss=losses))
        
        iterator.set_description(desc)
    
        # USER-DEFINED HOOK
        if has_attr(args, 'iteration_hook'):
            args.iteration_hook(model, i, loop_type, inp, target)

        # increment number of gradients
        if steps is not None: 
            steps += 1
            if schedule: schedule.step()

    if writer is not None:
        descs = ['loss', 'top1', 'top5']
        vals = [losses, top1, top5]
        for d, v in zip(descs, vals):
            writer.add_scalar('_'.join([loop_type, d]), v.avg,
                              epoch)
    
    # LOSS AND ACCURACY
    return top1.avg, losses.avg, score.avg
