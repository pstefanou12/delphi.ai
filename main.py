"""
CLI interface for using delphi.
"""

import torchvision 
from torchvision import transforms
from torch.utils.data import DataLoader
import torch as ch
import torch.nn as nn
import cox
from cox.utils import Parameters
import cox.store as store
import os

from delphi import train
from delphi.cifar_models import vgg11
from delphi.utils import model_utils
from delphi.utils.datasets import CIFAR
import delphi.utils.constants as consts
import delphi.utils.data_augmentation as da
from delphi.utils.helpers import setup_store_with_metadata

# set environment variable so that stores can create output files
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

BASE_CLASSIFIER_PATH = '/home/gridsan/stefanou/RESNET-18-CIFAR-10/'

if __name__ == '__main__':
    # experiment hyperparameters
    args = Parameters({ 
        'epochs': 150,
        'num_workers': 8, 
        'batch_size': 128, 
        'lr': 1e-1, 
        'momentum': .9, 
        'weight_decay': 5e-4, 
        'save_ckpt_iters': 50,
        'should_save_ckpt': True,
        'log_iters': 1,
        'custom_lr_multiplier': consts.COSINE, 
        'validation_split': .8,
        'shuffle': True,
        'parallel': True, 
    })
    # check if cuda available
    if ch.cuda.is_available(): 
        args.__setattr__('device', 'cuda')

    print(args)

    ds = CIFAR(data_path='/home/gridsan/stefanou/')
    model, _ = model_utils.make_and_restore_model(arch='resnet18', dataset=ds)
    train_loader, val_loader = ds.make_loaders(batch_size=128, workers=8)

    # logging store
    out_store = store.Store(BASE_CLASSIFIER_PATH)
    setup_store_with_metadata(args, out_store)

    # train
    train.train_model(args, model, (train_loader, val_loader), store=out_store, parallel=args.parallel)












