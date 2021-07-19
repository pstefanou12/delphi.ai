"""
**For most use cases, this can just be considered an internal class and
ignored.**
This module houses the :class:`delphi.attacker.Attacker` and
:class:`delphi.attacker.AttackerModel` classes.
:class:`~delphi.attacker.Attacker` is an internal class that should not be
imported/called from outside the library.
:class:`~delphi.attacker.AttackerModel` is a "wrapper" class which is fed a
model and adds to it adversarial attack functionalities as well as other useful
options. See :meth:`delphi.attacker.AttackerModel.forward` for documentation
on which arguments AttackerModel supports, and see
:meth:`delphi.attacker.Attacker.forward` for the arguments pertaining to
adversarial examples specifically.
For a demonstration of this module in action, see the walkthrough
":doc:`../example_usage/input_space_manipulation`"
**Note 1**: :samp:`.forward()` should never be called directly but instead the
AttackerModel object itself should be called, just like with any
:samp:`nn.Module` subclass.
**Note 2**: Even though the adversarial example arguments are documented in
:meth:`delphi.attacker.Attacker.forward`, this function should never be
called directly---instead, these arguments are passed along from
:meth:`delphi.attacker.AttackerModel.forward`.
Cite: code for attacker model shamelessly taken from:
@misc{robustness,
   title={Robustness (Python Library)},
   author={Logan Engstrom and Andrew Ilyas and Hadi Salman and Shibani Santurkar and Dimitris Tsipras},
   year={2019},
   url={https://github.com/MadryLab/robustness}
}
"""

import torch as ch
import os
import dill
import time

from .utils.helpers import type_of_script, calc_est_grad, InputNormalize, accuracy, AverageMeter, has_attr, ckpt_at_epoch
from .utils import constants as consts
from . import attack_steps
from .delphi import delphi

# determine running environment
script = type_of_script()
if script == consts.JUPYTER:
    from tqdm.autonotebook import tqdm as tqdm
else:
    from tqdm import tqdm

STEPS = {
    'inf': attack_steps.LinfStep,
    '2': attack_steps.L2Step,
    'unconstrained': attack_steps.UnconstrainedStep,
    'fourier': attack_steps.FourierStep,
    'random_smooth': attack_steps.RandomStep
}
LOGS = 'logs'
LOGS_SCHEMA = {
                'epoch': int,
                'val_prec1': float,
                'val_loss': float,
                'train_prec1': float,
                'train_loss': float,
                'time': float,
            }


class Attacker(ch.nn.Module):
    """
    Attacker class, used to make adversarial examples.
    This is primarily an internal class, you probably want to be looking at
    :class:`delphi.attacker.AttackerModel`, which is how models are actually
    served (AttackerModel uses this Attacker class).
    However, the :meth:`delphi.Attacker.forward` function below
    documents the arguments supported for adversarial attacks specifically.
    """
    def __init__(self, model, dataset):
        """
        Initialize the Attacker
        Args:
            nn.Module model : the PyTorch model to attack
            Dataset dataset : dataset the model is trained on, only used to get mean and std for normalization
        """
        super(Attacker, self).__init__()
        self.normalize = InputNormalize(dataset.mean, dataset.std)
        self.model = model

    def forward(self, x, target, *_, constraint, eps, step_size, iterations,
                random_start=False, random_restarts=False, do_tqdm=False,
                targeted=False, custom_loss=None, should_normalize=True,
                orig_input=None, use_best=True, return_image=True,
                est_grad=None, mixed_precision=False):
        """
        Implementation of forward (finds adversarial examples). Note that
        this does **not** perform inference and should not be called
        directly; refer to :meth:`delphi.attacker.AttackerModel.forward`
        for the function you should actually be calling.
        Args:
            x, target (ch.tensor) : see :meth:`delphi.attacker.AttackerModel.forward`
            constraint
                ("2"|"inf"|"unconstrained"|"fourier"|:class:`~delphi.attack_steps.AttackerStep`)
                : threat model for adversarial attacks (:math:`\ell_2` ball,
                :math:`\ell_\infty` ball, :math:`[0, 1]^n`, Fourier basis, or
                custom AttackerStep subclass).
            eps (float) : radius for threat model.
            step_size (float) : step size for adversarial attacks.
            iterations (int): number of steps for adversarial attacks.
            random_start (bool) : if True, start the attack with a random step.
            random_restarts (bool) : if True, do many random restarts and
                take the worst attack (in terms of loss) per input.
            do_tqdm (bool) : if True, show a tqdm progress bar for the attack.
            targeted (bool) : if True (False), minimize (maximize) the loss.
            custom_loss (function|None) : if provided, used instead of the
                criterion as the loss to maximize/minimize during
                adversarial attack. The function should take in
                :samp:`model, x, target` and return a tuple of the form
                :samp:`loss, None`, where loss is a tensor of size N
                (per-element loss).
            should_normalize (bool) : If False, don't normalize the input
                (not recommended unless normalization is done in the
                custom_loss instead).
            orig_input (ch.tensor|None) : If not None, use this as the
                center of the perturbation set, rather than :samp:`x`.
            use_best (bool) : If True, use the best (in terms of loss)
                iterate of the attack process instead of just the last one.
            return_image (bool) : If True (default), then return the adversarial
                example as an image, otherwise return it in its parameterization
                (for example, the Fourier coefficients if 'constraint' is
                'fourier')
            est_grad (tuple|None) : If not None (default), then these are
                :samp:`(query_radius [R], num_queries [N])` to use for estimating the
                gradient instead of autograd. We use the spherical gradient
                estimator, shown below, along with antithetic sampling [#f1]_
                to reduce variance:
                :math:`\\nabla_x f(x) \\approx \\sum_{i=0}^N f(x + R\\cdot
                \\vec{\\delta_i})\\cdot \\vec{\\delta_i}`, where
                :math:`\delta_i` are randomly sampled from the unit ball.
            mixed_precision (bool) : if True, use mixed-precision calculations
                to compute the adversarial examples / do the inference.
        Returns:
            An adversarial example for x (i.e. within a feasible set
            determined by `eps` and `constraint`, but classified as:
            * `target` (if `targeted == True`)
            *  not `target` (if `targeted == False`)
        .. [#f1] This means that we actually draw :math:`N/2` random vectors
            from the unit ball, and then use :math:`\delta_{N/2+i} =
            -\delta_{i}`.
        """
        # Can provide a different input to make the feasible set around
        # instead of the initial point
        if orig_input is None: orig_input = x.detach()
        orig_input = orig_input.cuda()

        # Multiplier for gradient ascent [untargeted] or descent [targeted]
        m = -1 if targeted else 1

        # Initialize step class and attacker criterion
        criterion = ch.nn.CrossEntropyLoss(reduction='none')
        step_class = STEPS[constraint] if isinstance(constraint, str) else constraint
        step = step_class(eps=eps, orig_input=orig_input, step_size=step_size)

        def calc_loss(inp, target):
            '''
            Calculates the loss of an input with respect to target labels
            Uses custom loss (if provided) otherwise the criterion
            '''
            if should_normalize:
                inp = self.normalize(inp)
            output = self.model(inp)
            if custom_loss:
                return custom_loss(self.model, inp, target)

            return criterion(output, target), output

        # Main function for making adversarial examples
        def get_adv_examples(x):
            # Random start (to escape certain types of gradient masking)
            if random_start:
                x = step.random_perturb(x)

            iterator = range(iterations)
            if do_tqdm: iterator = tqdm(iterator)

            # Keep track of the "best" (worst-case) loss and its
            # corresponding input
            best_loss = None
            best_x = None

            # A function that updates the best loss and best input
            def replace_best(loss, bloss, x, bx):
                if bloss is None:
                    bx = x.clone().detach()
                    bloss = loss.clone().detach()
                else:
                    replace = m * bloss < m * loss
                    bx[replace] = x[replace].clone().detach()
                    bloss[replace] = loss[replace]

                return bloss, bx

            # PGD iterates
            for _ in iterator:
                x = x.clone().detach().requires_grad_(True)
                losses, out = calc_loss(step.to_image(x), target)
                assert losses.shape[0] == x.shape[0], \
                        'Shape of losses must match input!'

                loss = ch.mean(losses)

                if step.use_grad:
                    if (est_grad is None) and mixed_precision:
                        with amp.scale_loss(loss, []) as sl:
                            sl.backward()
                        grad = x.grad.detach()
                        x.grad.zero_()
                    elif (est_grad is None):
                        grad, = ch.autograd.grad(m * loss, [x])
                    else:
                        f = lambda _x, _y: m * calc_loss(step.to_image(_x), _y)[0]
                        grad = calc_est_grad(f, x, target, *est_grad)
                else:
                    grad = None

                with ch.no_grad():
                    args = [losses, best_loss, x, best_x]
                    best_loss, best_x = replace_best(*args) if use_best else (losses, x)

                    x = step.step(x, grad)
                    x = step.project(x)
                    if do_tqdm: iterator.set_description("Current loss: {l}".format(l=loss))

            # Save computation (don't compute last loss) if not use_best
            if not use_best:
                ret = x.clone().detach()
                return step.to_image(ret) if return_image else ret

            losses, _ = calc_loss(step.to_image(x), target)
            args = [losses, best_loss, x, best_x]
            best_loss, best_x = replace_best(*args)
            return step.to_image(best_x) if return_image else best_x

        # Random restarts: repeat the attack and find the worst-case
        # example for each input in the batch
        if random_restarts:
            to_ret = None

            orig_cpy = x.clone().detach()
            for _ in range(random_restarts):
                adv = get_adv_examples(orig_cpy)

                if to_ret is None:
                    to_ret = adv.detach()

                _, output = calc_loss(adv, target)
                corr, = accuracy(output, target, topk=(1,), exact=True)
                corr = corr.byte()
                misclass = ~corr
                to_ret[misclass] = adv[misclass]

            adv_ret = to_ret
        else:
            adv_ret = get_adv_examples(x)

        return adv_ret


class AttackerModel(delphi):
    """
    Wrapper class for adversarial attacks on models. Given any normal
    model (a ``ch.nn.Module`` instance), wrapping it in AttackerModel allows
    for convenient access to adversarial attacks and other applications.::
        model = ResNet50()
        model = AttackerModel(model)
        x = ch.rand(10, 3, 32, 32) # random images
        y = ch.zeros(10) # label 0
        out, new_im = model(x, y, make_adv=True) # adversarial attack
        out, new_im = model(x, y, make_adv=True, targeted=True) # targeted attack
        out = model(x) # normal inference (no label needed)
    More code examples available in the documentation for `forward`.
    For a more comprehensive overview of this class, see
    :doc:`our detailed walkthrough <../example_usage/input_space_manipulation>`.
    """
    def __init__(self, args, model, dataset, checkpoint = None, store=None, parallel=False, dp_device_ids=None, update_params=None):
       
        super(AttackerModel, self).__init__(args, store, LOGS, LOGS_SCHEMA)
        self.model = model
        self.checkpoint = checkpoint
        self.parallel = parallel 
        self.dp_device_ids = dp_device_ids
        self.update_params = update_params
        self.normalizer = InputNormalize(dataset.mean, dataset.std)
        self.attacker = Attacker(model, dataset)
        # keep track of the best performing neural network on the validation set
        self.best_prec1 = 0.0
        # training and validation set metric counters
        self.reset_metrics()
        
        if checkpoint is not None: 
            sd = self.checkpoint[state_dict_path]
            self.model.load_state_dict(sd)
 
        # put AttackerModel on GPU
        self.model = self.model.cuda() 

        # run model in parallel model
        assert not hasattr(self.model, "module"), "model is already in DataParallel."
        if self.parallel and next(self.model.parameters()).is_cuda:
            self.model = ch.nn.DataParallel(self.model, device_ids=self.dp_device_ids)

    def forward(self, inp, target=None, make_adv=False, with_latent=False,
                fake_relu=False, no_relu=False, with_image=True, **attacker_kwargs):
        """
        Main function for running inference and generating adversarial
        examples for a model.
        Parameters:
            inp (ch.tensor) : input to do inference on [N x input_shape] (e.g. NCHW)
            target (ch.tensor) : ignored if `make_adv == False`. Otherwise,
                labels for adversarial attack.
            make_adv (bool) : whether to make an adversarial example for
                the model. If true, returns a tuple of the form
                :samp:`(model_prediction, adv_input)` where
                :samp:`model_prediction` is a tensor with the *logits* from
                the network.
            with_latent (bool) : also return the second-last layer along
                with the logits. Output becomes of the form
                :samp:`((model_logits, model_layer), adv_input)` if
                :samp:`make_adv==True`, otherwise :samp:`(model_logits, model_layer)`.
            fake_relu (bool) : useful for activation maximization. If
                :samp:`True`, replace the ReLUs in the last layer with
                "fake ReLUs," which are ReLUs in the forwards pass but
                identity in the backwards pass (otherwise, maximizing a
                ReLU which is dead is impossible as there is no gradient).
            no_relu (bool) : If :samp:`True`, return the latent output with
                the (pre-ReLU) output of the second-last layer, instead of the
                post-ReLU output. Requires :samp:`fake_relu=False`, and has no
                visible effect without :samp:`with_latent=True`.
            with_image (bool) : if :samp:`False`, only return the model output
                (even if :samp:`make_adv == True`).
        """
        if make_adv:
            assert target is not None
            prev_training = bool(self.training)
            self.eval()
            adv = self.attacker(inp, target, **attacker_kwargs)
            if prev_training:
                self.train()

            inp = adv

        normalized_inp = self.normalizer(inp)

        if no_relu and (not with_latent):
            print("WARNING: 'no_relu' has no visible effect if 'with_latent is False.")
        if no_relu and fake_relu:
            raise ValueError("Options 'no_relu' and 'fake_relu' are exclusive")

        output = self.model(normalized_inp, with_latent=with_latent,
                                fake_relu=fake_relu, no_relu=no_relu)
        if with_image:
            return (output, inp)
        return output

    def reset_metrics(self):
        '''
        *INTERNAL FUNCTION* resets meters that keep track of model's 
        performance over training and validation loops.
        '''
        # training and validation set metric counters
        self.train_losses, self.train_top1, self.train_top5 =  AverageMeter(), AverageMeter(), AverageMeter()
        self.val_losses, self.val_top1, self.val_top5 =  AverageMeter(), AverageMeter(), AverageMeter()

    def step(self, batch, losses, top1, top5, is_train=False): 
        '''
        *INTERNAL FUNCTION* used for both train 
        and validation steps. 
        '''
        # unpack input and target
        inp, targ = batch
        inp, targ = inp.cuda(), targ.cuda()
        model_logits = self.model(inp)

        # AttackerModel returns both output and final input
        if isinstance(model_logits, tuple):
            model_logits, _ = output

        # regularizer 
        self.reg_term = 0.0
        if has_attr(self.args, "regularizer") and isinstance(model, ch.nn.Module):
            self.reg_term = args.regularizer(self.model, inp, targ)

        # calculate loss and regularize
        loss = ch.nn.CrossEntropyLoss()(model_logits, targ)
        loss = loss + self.reg_term

        # backprop if is train
        if is_train:
            # zero gradient for model parameters
            self.optimizer.zero_grad()
            # backward propagation
            loss.backward()
            # update model parameters
            self.optimizer.step()

        # calculate accuracy metrics
        maxk = min(5, model_logits.shape[-1])
        if has_attr(self.args, "custom_accuracy"):
            prec1, prec5 = args.custom_accuracy(model_logits, targ)
        else:
            prec1, prec5 = accuracy(model_logits, targ, topk=(1, maxk))
            prec1, prec5 = prec1[0], prec5[0]
        # udpate model metric meters
        losses.update(loss.item(), batch[0].size(0))
        top1.update(prec1, batch[0].size(0))
        top5.update(prec5, batch[0].size(0))

    def pretrain_hook(self): 
        self.reset_metrics() 
    
    def train_step(self, batch):
        self.step(batch, self.train_losses, self.train_top1, self.train_top5, is_train=True)
        
    def val_step(self, batch):
        self.step(batch, self.val_losses, self.val_top1, self.val_top5, is_train=False) 

    def iteration_hook(self, epoch, i, loop_type, batch): 
        pass 

    def description(self, epoch, i, loop_msg):
        if loop_msg == 'Train': 
            losses, top1, top5 = self.train_losses, self.train_top1, self.train_top5
        else: 
            losses, top1, top5 = self.val_losses, self.val_top1, self.val_top5

        return ('Epoch: {0} | Loss {loss.avg:.4f} | '
                    '{1}1 {top1_acc.avg:.3f} | {1}5 {top5_acc.avg:.3f} | '
                    'Reg term: {reg} ||'.format(epoch, loop_msg,
                                                loss=losses, top1_acc=top1, top5_acc=top5, reg=self.reg_term))
  
    def epoch_hook(self, epoch, loop_type): 
        # update learning rate
        if self.schedule: 
            self.schedule.step()
        if loop_type == 'Train': 
            losses, top1, top5 = self.train_losses, self.train_top1, self.train_top5
        else: 
            losses, top1, top5 = self.val_losses, self.val_top1, self.val_top5

        # write to Tensorboard
        if self.writer is not None:
            descs = ['loss', 'top1', 'top5']
            vals = [losses, top1, top5]
            for d, v in zip(descs, vals):
                self.writer.add_scalar('_'.join([loop_type, d]), v.avg, epoch)

        # check for logging/checkpoint
        last_epoch = (epoch == (self.args.epochs - 1))
        should_save_ckpt = (epoch % self.args.save_ckpt_iters == 0 or last_epoch) 
        should_log = (epoch % self.args.log_iters == 0 or last_epoch)

        # logging
        if should_log or should_save_ckpt: 
            # remember best prec_1 and save checkpoint
            is_best = self.val_top1.avg > self.best_prec1
            self.best_prec1 = max(self.val_top1.avg, self.best_prec1)

        # CHECKPOINT -- checkpoint epoch of better DNN performance
        if self.store is not None and (should_save_ckpt or is_best):
            sd_info = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'schedule': (self.schedule and self.schedule.state_dict()),
                'epoch': epoch + 1,
                'amp': amp.state_dict() if self.args.mixed_precision else None,
                'prec1': self.val_top1.avg
            }
            
            def save_checkpoint(store, filename):
                """
                Saves model checkpoint at store path with filename.
                Args: 
                    filename (str) name of file for saving model
                """
                ckpt_save_path = os.path.join(store.path, filename)
                ch.save(sd_info, ckpt_save_path, pickle_module=dill)

            # update the latest and best checkpoints (overrides old one)
            if is_best:
                save_checkpoint(self.store, consts.CKPT_NAME_BEST)
            if should_save_ckpt: 
                # if we are at a saving epoch (or the last epoch), save a checkpoint
                save_checkpoint(self.store, ckpt_at_epoch(epoch))
                save_checkpoint(self.store, consts.CKPT_NAME_LATEST)

        # LOG
        if should_log and self.store:
            log_info = {
                'epoch': epoch + 1,
                'val_prec1': self.val_top1.avg,
                'val_loss': self.val_losses.avg,
                'train_prec1': self.train_top1.avg,
                'train_loss': self.train_losses.avg,
                'time': time.time()
            }
            self.store['logs'].append_row(log_info)

        # reset model performance meters for next epoch
        self.reset_metrics()

    def post_train_hook(self):
        pass
