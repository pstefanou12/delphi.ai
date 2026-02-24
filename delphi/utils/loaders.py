# Author: pstefanou12@
"""
Data loader wrappers providing per-epoch recomputation and real-time transformations.
"""

import torch as ch

from delphi.utils import folder
from delphi.utils.helpers import type_of_script
from delphi.utils import constants as consts

# Determine running environment.
SCRIPT = type_of_script()
if SCRIPT == consts.JUPYTER:
    from tqdm.autonotebook import tqdm
else:
    from tqdm import tqdm


# Loader wrapper (for adding custom functions to dataloader).
class PerEpochLoader:
    """
    A blend between TransformedLoader and LambdaLoader: stores the whole loader
    in memory, but recomputes it from scratch every epoch, instead of just once
    at initialization.
    """

    def __init__(self, loader, func, do_tqdm=True):
        """Initialize with a loader, transformation function, and tqdm flag."""
        self.orig_loader = loader
        self.func = func
        self.do_tqdm = do_tqdm
        self.data_loader = self.compute_loader()
        self.loader = iter(self.data_loader)

    def compute_loader(self):
        """Rebuild the transformed loader from the original loader."""
        return transformed_loader(
            self.orig_loader,
            self.func,
            None,
            self.orig_loader.num_workers,
            self.orig_loader.batch_size,
            do_tqdm=self.do_tqdm,
        )

    def __len__(self):
        return len(self.orig_loader)

    def __getattr__(self, attr):
        return getattr(self.data_loader, attr)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.loader)
        except StopIteration:
            self.data_loader = self.compute_loader()
            self.loader = iter(self.data_loader)
            raise StopIteration from None


class LambdaLoader:
    """
    This is a class that allows one to apply any given (fixed)
    transformation to the output from the loader in *real-time*.
    For instance, you could use for applications such as custom
    data augmentation and adding image/label noise.
    Note that the LambdaLoader is the final transformation that
    is applied to image-label pairs from the dataset as part of the
    loading process---i.e., other (standard) transformations such
    as data augmentation can only be applied *before* passing the
    data through the LambdaLoader.
    For more information see :ref:`our detailed walkthrough <using-custom-loaders>`
    """

    def __init__(self, loader, func):
        """Initialize with a loader and a real-time transformation function.

        Args:
            loader: PyTorch DataLoader for the dataset.
            func (Callable): fixed transformation applied to every batch
                in real-time. Takes ``(images, labels)`` and returns
                ``(images, labels)`` of the same shape.
        """
        self.data_loader = loader
        self.loader = iter(self.data_loader)
        self.func = func

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        return self

    def __getattr__(self, attr):
        return getattr(self.data_loader, attr)

    def __next__(self):
        """Return the next transformed (image, label) pair."""
        try:
            im, targ = next(self.loader)
        except StopIteration:
            self.loader = iter(self.data_loader)
            raise StopIteration from None

        return self.func(im, targ)


def transformed_loader(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    loader,
    func,
    loader_transforms,
    workers=None,
    batch_size=None,
    do_tqdm=False,
    augment=False,
    fraction=1.0,
    shuffle=True,
):
    """Apply a fixed transformation to all loader outputs once and return a new loader.

    Unlike LambdaLoader (which applies the transform at iteration time),
    this materialises the entire transformed dataset in memory before returning.

    Args:
        loader: PyTorch DataLoader for the source dataset.
        func (Callable): fixed transformation applied once to every batch.
            Takes ``(images, labels)`` and returns ``(images, labels)`` that
            may differ in the batch dimension but not in other dimensions.
        loader_transforms: torchvision transforms applied after ``func``.
        workers (int): number of workers for data fetching.
        batch_size (int): batch size for the returned loader.
        do_tqdm (bool): if True, show a tqdm progress bar while transforming.
        augment (bool): if True, the output contains both original and
            transformed image-label pairs.
        fraction (float): fraction of pairs that are transformed; the
            remainder are original pairs from ``loader``.
        shuffle (bool): whether the returned loader shuffles each epoch.

    Returns:
        A standard PyTorch DataLoader over the transformed dataset.
    """

    new_ims = []
    new_targs = []
    total_len = len(loader)
    enum_loader = enumerate(loader)

    it = enum_loader if not do_tqdm else tqdm(enum_loader, total=total_len)
    for i, (im, targ) in it:
        new_im, new_targ = func(im, targ)
        if augment or (i / float(total_len) > fraction):
            new_ims.append(im.cpu())
            new_targs.append(targ.cpu())
        if i / float(total_len) <= fraction:
            new_ims.append(new_im.cpu())
            new_targs.append(new_targ.cpu())

    dataset = folder.TensorDataset(
        ch.cat(new_ims, 0), ch.cat(new_targs, 0), transform=loader_transforms
    )
    return ch.utils.data.DataLoader(
        dataset, num_workers=workers, batch_size=batch_size, shuffle=shuffle
    )


# Keep backward-compatible alias
TransformedLoader = transformed_loader  # pylint: disable=invalid-name
