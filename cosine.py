from delphi import train
from delphi import oracle

import torchvision 
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch as ch
from cox.utils import Parameters
import cox.store as store
import os
import config
import pickle
import pandas as pd

from delphi import grad
from delphi.utils import model_utils
from delphi.utils.datasets import CIFAR
import delphi.utils.constants as consts
import delphi.utils.data_augmentation as da
from delphi.utils.helpers import setup_store_with_metadata
# CONSTANTS
# set environment variable so that stores can create output files
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

# file path constants
LOGIT_BALL_CLASSIFIER = '/home/gridsan/stefanou/cifar-10/resnet-18/trunc_ce_cosine_100epochs'
STANDARD_CLASSIFIER = '/home/gridsan/stefanou/cifar-10/resnet-18/ce_cosine_100epochs'
DATA_PATH = '/home/gridsan/stefanou/data/'
TRUNC_TRAIN_DATASET = 'trunc_train_calibrated_logit_'
TRUNC_VAL_DATASET = 'trunc_val_calibrated_logit_'
TRUNC_UNSEEN_DATASET = 'trunc_test_calibrated_logit_'

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
    'epochs': 100,
    'workers': 8, 
    'batch_size': 128, 
    'lr': 1e-2, 
    'accuracy': True,
    'momentum': 0.9, 
    'weight_decay': 5e-4, 
    'custom_lr_multiplier': consts.COSINE,
    'should_save_ckpt': True,
    'log_iters': 1,
    'validation_split': .8,
    'shuffle': True,
    'parallel': False, 
    'num_samples': 1000,
    'logit_ball': 12.0,
    'trials': 3,
    'step_lr': 1, 
    'step_lr_gamma': 1.0,
    'device': 'cuda' if ch.cuda.is_available() else 'cpu'
})
LEARNING_RATES = [5e-1, 3e-1, 2e-1, 1e-1, 1e-2]

ds = CIFAR(data_path='/home/gridsan/stefanou/data')

# test dataset
test_set = torchvision.datasets.CIFAR10(root='/home/gridsan/stefanou/data', train=False,
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

























