import time
import os
import warnings
import dill
import numpy as np
import torch as ch
from torch import Tensor
from torch.optim import SGD
from torch.optim import lr_scheduler
import config
import IPython

from .utils.helpers import setup_store_with_metadata, has_attr, ckpt_at_epoch, AverageMeter, accuracy, type_of_script
from .utils.constants import LOGS_SCHEMA, LOGS_TABLE, CKPT_NAME_BEST, CKPT_NAME_LATEST, JUPYTER, TERMINAL, \
    IPYTHON, LINEAR, CYCLIC, COSINE

# determine running environment
script = type_of_script()
if script == JUPYTER:
    from tqdm.autonotebook import tqdm as tqdm
else:
    from tqdm import tqdm


def make_optimizer_and_schedule(model, checkpoint, params):
    param_list = model.parameters() if params is None else params

    optimizer = SGD(param_list, config.args.lr, momentum=config.args.momentum, weight_decay=config.args.weight_decay)

    # Make schedule
    schedule = None
    if config.args.custom_lr_multiplier == CYCLIC:
        eps = config.args.epochs
        lr_func = lambda t: np.interp([t], [0, eps*4//15, eps], [0, 1, 0])[0]
        schedule = lr_scheduler.LambdaLR(optimizer, lr_func)
    elif config.args.custom_lr_multiplier == COSINE:
        eps = config.args.epochs
        schedule = lr_scheduler.CosineAnnealingLR(optimizer, eps)
    elif config.args.custom_lr_multiplier:
        cs = config.args.custom_lr_multiplier
        periods = eval(cs) if type(cs) is str else cs
        if config.args.lr_interpolation == LINEAR:
            lr_func = lambda t: np.interp([t], *zip(*periods))[0]
        else:
            def lr_func(ep):
                for (milestone, lr) in reversed(periods):
                    if ep >= milestone: return lr
                return 1.0
        schedule = lr_scheduler.LambdaLR(optimizer, lr_func)
    elif config.args.step_lr:
        schedule = lr_scheduler.StepLR(optimizer, step_size=config.args.step_lr, gamma=config.args.step_lr_gamma)
        
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


def train_model(model, loaders, *, checkpoint=None, device="cpu", dp_device_ids=None,
                store=None, update_params=None, disable_no_grad=False):

    # clear jupyter/ipython output before each training run
    if script == JUPYTER or script == IPYTHON:
        IPython.display.clear_output()

    if store is not None: 
        setup_store_with_metadata(config.args, store)
        store.add_table(LOGS_TABLE, LOGS_SCHEMA)
    writer = store.tensorboard if store else None
    
    # data loaders
    train_loader, val_loader = loaders
    optimizer, schedule = make_optimizer_and_schedule(model, checkpoint, update_params)
    
    # put the model into parallel mode
    assert not has_attr(model, "module"), "model is already in DataParallel."
    if isinstance(model, ch.distributions.Distribution):
        model.loc.to(device)
        model.covariance_matrix.to(device)
    else:
        model.to(device)

    best_prec1, start_epoch = (0, 0)
    if checkpoint:
        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['prec1'] if 'prec1' in checkpoint \
            else model_loop('val', val_loader, model, None, start_epoch-1, writer=None, device=device)[0]

    # keep track of the start time
    start_time = time.time()
    for epoch in range(start_epoch, config.args.epochs):
        train_prec1, train_loss, score = model_loop('train', train_loader, model, optimizer, epoch+1, writer, device=device)

        # print("TOP 1 AVG: {}".format(train_prec1))
        # print("LOSS AVG: {}".format(train_loss))
        print("SCORE AVG: {}".format(score))
        # check score tolerance
        if ch.all(ch.where(ch.abs(score) < config.args.tol, 1, 0).bool()):
            break

        last_epoch = (epoch == (config.args.epochs - 1))

        # if neural network passed through framework, use log performance
        if config.args.should_save_ckpt:
            # evaluate on validation set
            sd_info = {
                'model':model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'schedule':(schedule and schedule.state_dict()),
                'epoch': epoch+1,
            }

            def save_checkpoint(filename):
                ckpt_save_path = os.path.join(config.args.out_dir if not store else \
                                              store.path, filename)
                ch.save(sd_info, ckpt_save_path, pickle_module=dill)

            save_its = config.args.save_ckpt_iters
            should_save_ckpt = (epoch % save_its == 0) and (save_its > 0)
            should_log = (epoch % config.args.log_iters == 0)

            if should_log or last_epoch or should_save_ckpt:
                # log + get best
                ctx = ch.enable_grad() if disable_no_grad else ch.no_grad()
                with ctx:
                    val_prec1, val_loss, score = model_loop('val', val_loader, model,
                            None, epoch + 1, writer, device=device)

                # remember best prec@1 and save checkpoint
                is_best = val_prec1 > best_prec1
                best_prec1 = max(val_prec1, best_prec1)
                sd_info['prec1'] = val_prec1

                # log every checkpoint
                log_info = {
                    'epoch':epoch + 1,
                    'val_prec1':val_prec1,
                    'val_loss':val_loss,
                    'train_prec1':train_prec1,
                    'train_loss':train_loss,
                    'time': time.time() - start_time
                }

                # Log info into the logs table
                if store: store[LOGS_TABLE].append_row(log_info)
                # If we are at a saving epoch (or the last epoch), save a checkpoint
                if should_save_ckpt or last_epoch: save_checkpoint(ckpt_at_epoch(epoch))

                # Update the latest and best checkpoints (overrides old one)
                save_checkpoint(CKPT_NAME_LATEST)
                if is_best: save_checkpoint(CKPT_NAME_BEST)
        
        # update lr
        if schedule: schedule.step()

        tqdm._instances.clear()

    # model results
    if isinstance(score, Tensor):
        print("avg score: \n {}".format(score))
    if train_loss != 0:
        print("avg loss: {}".format(train_loss))
    if train_prec1 != 0:
        print("avg top 1: {}".format(train_prec1))
    return model
            
            
def model_loop(loop_type, loader, model, optimizer, epoch, writer, device):
    # check loop type 
    if not loop_type in ['train', 'val']: 
        err_msg = "loop type must be in {0} must be 'train' or 'val".format(loop_type)
        raise ValueError(err_msg)
    is_train = (loop_type == 'train')
    
    loop_msg = 'Train' if is_train else 'Val'

    # algorithm metrics
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    score = AverageMeter()
    
    # check for custom criterion
    has_custom_criterion = has_attr(config.args, 'custom_criterion')
    criterion = config.args.custom_criterion if has_custom_criterion else ch.nn.CrossEntropyLoss()

    # clear jupyter/ipython output before each iteration
    if script == JUPYTER or script == IPYTHON:
        IPython.display.clear_output()
    
    iterator = tqdm(enumerate(loader), total=len(loader))
    for i, batch in iterator:
        inp, target, output = None, None, None
        loss = 0.0
        if isinstance(model, ch.distributions.distribution.Distribution):
            loss = criterion(*optimizer.param_groups[0]['params'], *batch)
        elif isinstance(model, ch.nn.Module):
            inp, target = batch
            inp, target = inp.to(device), target.to(device)
            output = model(inp)
            # lambda parameter used for regression with unknown noise variance
            try:
                loss = criterion(output, target, model.lambda_)
            except Exception as e:
                loss = criterion(output, target)

        # regularizer option 
        reg_term = 0.0
        if has_attr(config.args, "regularizer") and isinstance(model, ch.nn.Module):
            reg_term = config.args.regularizer(model, inp, target)
        loss = loss + reg_term
        
        # perform backprop and take optimizer step
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if len(loss.size()) > 0: loss = loss.mean()

        model_logits = output[0] if isinstance(output, tuple) else output

        # measure accuracy and record loss
        top1_acc = float('nan')
        top5_acc = float('nan')
        try:
            # score
            # censored, truncated distributions
            if isinstance(model, ch.distributions.Distribution):
                score.update(ch.cat([model.loc.grad, model.covariance_matrix.grad.flatten()]), model.loc.size(0) + model.covariance_matrix.flatten().size(0))
                desc = ('Epoch:{0} | Score: {score} \n |'.format(epoch, loop_msg, score=[round(x, 4) for x in score.avg.tolist()]))
            # regression with unknown variance
            elif inp is not None:
                losses.update(loss.item(), inp.size(0))

                # accuracy
                maxk = min(5, model_logits.shape[-1])
                if has_attr(config.args, "custom_accuracy"):
                    prec1, prec5 = config.args.custom_accuracy(model_logits, target)
                else:
                    prec1, prec5 = accuracy(model_logits, target, topk=(1, maxk))
                    prec1, prec5 = prec1[0], prec5[0]

                top1.update(prec1, inp.size(0))
                top5.update(prec5, inp.size(0))
                top1_acc = top1.avg
                top5_acc = top5.avg
                # ITERATOR
                if not config.args.var: # unknown variance
                    score.update(ch.cat([model.v.grad, model.bias.grad, model.lambda_.grad]).flatten(), inp.size(0) + 1)
                    desc = ('Epoch:{0} | Score: {score} \n | Loss {loss.avg:.4f} |'.format(
                        epoch, loop_msg, score=[round(x, 4) for x in score.avg.tolist()], loss=losses.avg))
                # regression with known variance
                else: # known variance
                    score.update(ch.cat([model.weight.grad.T, model.bias.grad.unsqueeze(0)]).flatten(), inp.size(0))
                    desc = ('Epoch:{0} | Score: {score} \n | Loss {loss.avg:.4f} |'.format(
                        epoch, loop_msg, score=[round(x, 4) for x in score.avg.tolist()], loss=losses.avg))
        except Exception as e:
            print(e)
            if isinstance(model, ch.nn.Module):
                warnings.warn('Failed to calculate the accuracy.')
            # ITERATOR
            desc = ('Epoch:{0} | Loss {loss.avg:.4f} |'.format(epoch, loop_msg, loss=losses))
        
        iterator.set_description(desc)
    
        # USER-DEFINED HOOK
        if has_attr(config.args, 'iteration_hook'):
            config.args.iteration_hook(model, i, loop_type, inp, target)

    if writer is not None:
        descs = ['loss', 'top1', 'top5']
        vals = [losses, top1, top5]
        for d, v in zip(descs, vals):
            writer.add_scalar('_'.join([loop_type, d]), v.avg,
                              epoch)
    
    # LOSS AND ACCURACY
    return top1.avg, losses.avg, score.avg




