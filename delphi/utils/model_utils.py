"""
Helper functions for dealing with model.
"""

import os
import dill
import torch as ch
from torch import nn

from delphi.attacker import AttackerModel


class DummyModel(nn.Module):
    """A thin wrapper that ignores extra forward arguments."""

    def __init__(self, model):
        """Initialize DummyModel wrapping the given model."""
        super().__init__()
        self.model = model

    def forward(self, x, *args, **kwargs):  # pylint: disable=unused-argument
        """Forward pass delegating to the wrapped model."""
        return self.model(x)


def make_and_restore_model(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    *_,
    arch,
    args,
    dataset,
    store=None,
    parallel=False,
    dp_device_ids=None,
    update_params=None,
    resume_path=None,
    pytorch_pretrained=False,
    add_custom_forward=False,
):
    """
    Makes a model and (optionally) restores it from a checkpoint.
    Args:
        arch (str|nn.Module): Model architecture identifier or otherwise a
            torch.nn.Module instance with the classifier
        dataset (Dataset class [see datasets.py])
        resume_path (str): optional path to checkpoint saved with the
            robustness library (ignored if ``arch`` is not a string)
        pytorch_pretrained (bool): if True, try to load a standard-trained
            checkpoint from the torchvision library (throw error if failed)
        add_custom_forward (bool): ignored unless arch is an instance of
            nn.Module (and not a string). Normally, architectures should have a
            forward() function which accepts arguments ``with_latent``,
            ``fake_relu``, and ``no_relu`` to allow for adversarial manipulation
            (see `here`<https://robustness.readthedocs.io/en/latest/example_usage/
            training_lib_part_2.html#training-with-custom-architectures>
            for more info). If this argument is True, then these options will
            not be passed to forward(). (Useful if you just want to train a
            model and don't care about these arguments, and are passing in an
            arch that you don't want to edit forward() for, e.g.  a pretrained model)
    Returns:
        A tuple consisting of the model (possibly loaded with checkpoint), and the checkpoint itself
    """
    if (not isinstance(arch, str)) and add_custom_forward:
        arch = DummyModel(arch)

    classifier_model = (
        dataset.get_model(arch, pytorch_pretrained) if isinstance(arch, str) else arch
    )

    # optionally resume from a checkpoint
    checkpoint = None
    if resume_path and os.path.isfile(resume_path):
        print(f"=> loading checkpoint '{resume_path}'")
        checkpoint = ch.load(resume_path, pickle_module=dill)

        # Makes us able to load models saved with legacy versions
        if "model" not in checkpoint:
            print(f"=> loaded checkpoint '{resume_path}' (epoch {checkpoint['epoch']})")
    elif resume_path:
        raise ValueError(f"=> no checkpoint found at '{resume_path}'")

    model = AttackerModel(
        args,
        classifier_model,
        dataset,
        checkpoint=checkpoint,
        store=store,
        parallel=parallel,
        dp_device_ids=dp_device_ids,
        update_params=update_params,
    )

    return model
