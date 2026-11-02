# Author: pstefanou12@
"""Smoke tests for delphi.adversarial: imports, layers, models, attack steps."""

import torch as ch

from delphi.adversarial import attack_steps, attacker, layers
from delphi.adversarial.cifar_models import resnet


def test_fake_relu_passes_gradient_through_clamp() -> None:
    """FakeReLU clamps forward but keeps the gradient of negative inputs."""
    x = ch.tensor([-1.0, 2.0], requires_grad=True)
    out = layers.FakeReLU.apply(x)
    out.sum().backward()
    ch.testing.assert_close(out, ch.tensor([0.0, 2.0]))
    ch.testing.assert_close(x.grad, ch.ones(2))


def test_cifar_resnet_forward_shapes() -> None:
    """A CIFAR ResNet maps 32x32 images to logits, and to latents on request."""
    model = resnet.ResNet18(num_classes=10).eval()
    images = ch.rand(2, 3, 32, 32)
    with ch.no_grad():
        logits, latent = model(images, with_latent=True)
    assert logits.shape == (2, 10)
    assert latent.shape[0] == 2


def test_linf_step_stays_in_ball() -> None:
    """A step followed by a projection stays within eps of the input."""
    orig = ch.full((4, 3, 2, 2), 0.5)
    step = attack_steps.LinfStep(orig, eps=0.1, step_size=1.0)
    moved = step.project(step.step(orig, ch.randn_like(orig)))
    assert (moved - orig).abs().max() <= 0.1 + 1e-6


def test_attacker_registers_every_step() -> None:
    """Each named threat model resolves to an AttackerStep subclass."""
    for step in attacker.STEPS.values():
        assert issubclass(step, attack_steps.AttackerStep)
