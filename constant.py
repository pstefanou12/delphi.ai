import sys
sys.path.append('../..')
import torchvision 
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch as ch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Gumbel
import math
import numpy as np
import matplotlib.pyplot as plt
import cox
from cox.utils import Parameters
import cox.store as store
import seaborn as sns
import os
import config
import pickle
import pandas as pd

from delphi import train
from delphi.utils import model_utils
from delphi import grad
from delphi import oracle
from delphi.utils.datasets import CIFAR, ImageNet
import delphi.utils.constants as consts
import delphi.utils.data_augmentation as da
from delphi.utils.helpers import setup_store_with_metadata

# CONSTANTS
# set environment variable so that stores can create output files
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

# noise distributions
gumbel = Gumbel(0, 1)
num_classes = 10

# file path constants
BASE_CLASSIFIER = '/home/pstefanou/cifar-10/resnet-18/base_calibrated_'
BASE_CLASSIFIER_PATH = BASE_CLASSIFIER + '/e4476e64-c5bb-4d8a-b2bc-299aed256e88/checkpoint.pt.latest'
LOGIT_BALL_CLASSIFIER = '/home/pstefanou/cifar-10/resnet-18/trunc_ce_constant'
STANDARD_CLASSIFIER = '/home/pstefanou/cifar-10/resnet-18/ce_constant'
DATA_PATH = '/home/pstefanou/data/'
TRUNC_TRAIN_DATASET = 'trunc_train_calibrated_logit__'
TRUNC_VAL_DATASET = 'trunc_val_calibrated_logit__'
TRUNC_UNSEEN_DATASET = 'trunc_test_calibrated_logit__'

# helper dataset
class TruncatedCIFAR(Dataset):
    """
    Truncated CIFAR-10 dataset [Kri09]_.
    Original dataset has 50k training images and 10k testing images, with the
    following classes:
    * Airplane
    * Automobile
    * Bird
    * Cat
    * Deer
    * Dog
    * Frog
    * Horse
    * Ship
    * Truck
    .. [Kri09] Krizhevsky, A (2009). Learning Multiple Layers of Features
        from Tiny Images. Technical Report.
        
    Truncated dataset only includes images and labels from original dataset that fall within the truncation set.
    """
    def __init__(self, img, label, transform = None):
        """
        """
        self.img = img 
        self.label = label
        self.transform = transform

    def __getitem__(self, idx):
        """
        """
        x = self.img[idx]
        y = self.label[idx]
        # data augmentation
        if self.transform: 
            x = self.transform(x)
            
        return x, y
    
    def __len__(self): 
        return self.img.size(0)
transform_ = transforms.Compose(
    [transforms.ToTensor()])

# hyperparameters
args = Parameters({ 
    'epochs': 50,
    'workers': 8, 
    'batch_size': 128, 
    'lr': 1e-2, 
    'accuracy': True,
    'momentum': 0.0, 
    'weight_decay': 0.0, 
    'save_ckpt_iters': 10,
    'should_save_ckpt': True,
    'log_iters': 1,
    'step_lr': 1, 
    'step_lr_gamma': 1.0,
    'validation_split': .8,
    'shuffle': True,
    'parallel': False, 
    'num_samples': 1000,
    'logit_ball': 7.5,
    'trials': 3,
    'step_lr': 10, 
    'step_lr_gamma': 1.0,
    'device': 'cuda' if ch.cuda.is_available() else 'cpu'
})
LEARNING_RATES = [1e-1, 1e-2, 1e-3]

ds = CIFAR(data_path='/home/pstefanou/')

# test dataset
test_set = torchvision.datasets.CIFAR10(root='/home/pstefanou/', train=False,
                                       download=False, transform=transform_)
test_loader = ch.utils.data.DataLoader(test_set, batch_size=128,
                                         shuffle=False, num_workers=2)

# datasets 
trunc_train_loader = pd.read_pickle(DATA_PATH + TRUNC_TRAIN_DATASET + str(args.logit_ball) + '.pickle')
trunc_val_loader = pd.read_pickle(DATA_PATH + TRUNC_VAL_DATASET + str(args.logit_ball) + '.pickle')
trunc_unseen_loader = pd.read_pickle(DATA_PATH + TRUNC_UNSEEN_DATASET + str(args.logit_ball) + '.pickle')

phi = oracle.LogitBallComplement(args.logit_ball)

# generate random order of seeds for all of the trials
seeds = ch.randperm(args.trials)

for i in range(args.trials):
    for lr in LEARNING_RATES: 
        # set learning rate
        args.__setattr__('lr', lr)
        # train model using truncated ce loss 
        # logging store
        out_store = store.Store(LOGIT_BALL_CLASSIFIER)
        setup_store_with_metadata(args, out_store)
        delphi_, _ = model_utils.make_and_restore_model(arch='resnet18', dataset=ds)

        # train
        ch.manual_seed(seeds[i])
        config.args = args
        delphi_ = train.train_model(args, delphi_, (trunc_train_loader, trunc_val_loader), store=out_store, phi=phi, criterion=grad.TruncatedCE.apply)

        # test model on data sets
        delphi_unseen_results = train.eval_model(args, delphi_, trunc_unseen_loader, out_store, table='unseen')
        delphi_test_results = train.eval_model(args, delphi_, test_loader, out_store, table='test')
        delphi_train_results = train.eval_model(args, delphi_, trunc_train_loader, out_store, table='train')
        delphi_val_results = train.eval_model(args, delphi_, trunc_val_loader, out_store, table='val')
        out_store.close()

        # train model using standard cross entropy loss
        # logging store
        out_store = store.Store(STANDARD_CLASSIFIER)
        setup_store_with_metadata(args, out_store)
        standard, _ = model_utils.make_and_restore_model(arch='resnet18', dataset=ds)

        # train
        ch.manual_seed(seeds[i])
        config.args = args

        # test model on datasets
        standard_model = train.train_model(args, standard, (trunc_train_loader, trunc_val_loader), store=out_store, parallel=args.parallel)
        standard_unseen_results = train.eval_model(args, standard_model, trunc_unseen_loader, out_store, table='unseen')
        standard_test_results = train.eval_model(args, standard_model, test_loader, out_store, table='test')
        standard_train_results = train.eval_model(args, standard_model, trunc_train_loader, out_store, table='train')
        standard_val_results = train.eval_model(args, standard_model, trunc_val_loader, out_store, table='val')
        out_store.close()

























