import os
import numpy as np
import torch as ch
from torch.utils.data import Subset
from torch.utils.data import DataLoader, TensorDataset

from . import folder
from .helpers import type_of_script
from . import constants as consts
# determine running environment
script = type_of_script()
if script == consts.JUPYTER:
    from tqdm.autonotebook import tqdm as tqdm
else:
    from tqdm import tqdm


## loader wrapper (for adding custom functions to dataloader)
class PerEpochLoader:
    '''
    A blend between TransformedLoader and LambdaLoader: stores the whole loader
    in memory, but recomputes it from scratch every epoch, instead of just once
    at initialization.
    '''
    def __init__(self, loader, func, do_tqdm=True):
        self.orig_loader = loader
        self.func = func
        self.do_tqdm = do_tqdm
        self.data_loader = self.compute_loader()
        self.loader = iter(self.data_loader)

    def compute_loader(self):
        return TransformedLoader(self.orig_loader, self.func, None,
                    self.orig_loader.num_workers, self.orig_loader.batch_size,
                    do_tqdm=self.do_tqdm)

    def __len__(self):
        return len(self.orig_loader)

    def __getattr__(self, attr):
        return getattr(self.data_loader, attr)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.loader)
        except StopIteration as e:
            self.data_loader = self.compute_loader()
            self.loader = iter(self.data_loader)
            raise StopIteration

        return self.func(im, targ)


class LambdaLoader:
    '''
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
    '''

    def __init__(self, loader, func):
        '''
        Args:
            loader (PyTorch dataloader) : loader for dataset (*required*).
            func (function) : fixed transformation to be applied to
                every batch in real-time (*required*). It takes in
                (images, labels) and returns (images, labels) of the
                same shape.
        '''
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
        try:
            im, targ = next(self.loader)
        except StopIteration as e:
            self.loader = iter(self.data_loader)
            raise StopIteration

        return self.func(im, targ)

    def __getattr__(self, attr):
        return getattr(self.data_loader, attr)


def TransformedLoader(loader, func, transforms, workers=None,
        batch_size=None, do_tqdm=False, augment=False, fraction=1.0,
        shuffle=True):
    '''
    This is a function that allows one to apply any given (fixed)
    transformation to the output from the loader *once*.
    For instance, you could use for applications such as assigning
    random labels to all the images (before training).
    The TransformedLoader also supports the application of additional
    transformations (such as standard data augmentation) after the fixed
    function.
    For more information see :ref:`our detailed walkthrough <using-custom-loaders>`
    Args:
        loader (PyTorch dataloader) : loader for dataset
        func (function) : fixed transformation to be applied once. It takes
        in (images, labels) and returns (images, labels) with the same shape
        in every dimension except for the first, i.e., batch dimension
        (which can be any length).
        transforms (torchvision.transforms) : transforms to apply
            to the training images from the dataset (after func) (*required*).
        workers (int) : number of workers for data fetching (*required*).
        batch_size (int) : batch size for the data loaders (*required*).
        do_tqdm (bool) : if True, show a tqdm progress bar for the attack.
        augment (bool) : if True,  the output loader contains both the original
            (untransformed), and new transformed image-label pairs.
        fraction (float): fraction of image-label pairs in the output loader
            which are transformed. The remainder is just original image-label
            pairs from loader.
        shuffle (bool) : whether or not the resulting loader should shuffle every
            epoch (defaults to True)
    Returns:
        A loader and validation loader according to the
        parameters given. These are standard PyTorch data loaders, and
        thus can just be used via:
        >>> output_loader = ds.make_loaders(loader,
                                            assign_random_labels,
                                            workers=8,
                                            batch_size=128)
        >>> for im, lab in output_loader:
        >>>     # Do stuff...
    '''

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

    dataset = folder.TensorDataset(ch.cat(new_ims, 0), ch.cat(new_targs, 0), transform=transforms)
    return ch.utils.data.DataLoader(dataset, num_workers=workers,
                        batch_size=batch_size, shuffle=shuffle)
