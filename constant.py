from delphi import train
from delphi import oracle

import torchvision 
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch as ch
from torch import Tensor
from cox.utils import Parameters
import cox.store as store
import os
import config
import pickle
import pandas as pd
from tqdm import tqdm

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
BASE_CLASSIFIER = '/home/gridsan/stefanou/cifar-10/resnet-18/base_constant_test/'
LOGIT_BALL_CLASSIFIER = '/home/gridsan/stefanou/cifar-10/resnet-18/trunc_constant_test/'
STANDARD_CLASSIFIER = '/home/gridsan/stefanou/cifar-10/resnet-18/standard_constant_test/'
DATA_PATH = '/home/gridsan/stefanou/data/'

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

EPOCHS = [1]
LEARNING_RATES = [1e-1, 1e-2]


def main(args):
    """
    Run parameter tuning script.
    """
    # dataset
    ds = CIFAR(data_path=DATA_PATH)
    dataset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True,
                                            download=False, transform=transform_)
    # train and validation datasets
    train_set, val_set = ch.utils.data.random_split(dataset, [45000, 5000])
    train_loader = ch.utils.data.DataLoader(train_set, batch_size=128,
                                            shuffle=True, num_workers=2)
    val_loader = ch.utils.data.DataLoader(val_set, batch_size=128,
                                            shuffle=True, num_workers=2)
    # test dataset
    test_set = torchvision.datasets.CIFAR10(root=DATA_PATH, train=False,
                                        download=False, transform=transform_)
    test_loader = ch.utils.data.DataLoader(test_set, batch_size=128,
                                            shuffle=False, num_workers=2)

    # oracle
    phi = oracle.LogitBallComplement(args.logit_ball)

    for i in range(args.trials):
        for epochs in EPOCHS: 
            # set learning rate
            args.__setattr__('epochs', epochs)
            for lr in LEARNING_RATES: 

                config.args = args

                # train base classifier on entire dataset
                out_store = store.Store(BASE_CLASSIFIER)
                base_classifier, _ = model_utils.make_and_restore_model(arch='resnet18', dataset=ds)
                ch.manual_seed(i)
                train.train_model(args, base_classifier, (train_loader, val_loader), store=out_store)

                # calibrate neural network
                temperature = calibrate(test_loader, base_classifier)

                args.__setattr__('temperature', float(temperature.item()))
                # store metadata afterwards to also track temperature
                setup_store_with_metadata(args, out_store)

                # truncate dataset 
                x_train, x_unseen_, y_train, y_unseen_ = truncate(train_loader, base_classifier, phi, temperature, 'cuda')
                x_val, x_unseen, y_val, y_unseen = truncate(val_loader, base_classifier, phi, temperature, 'cuda')

                trunc_train_loader = DataLoader(TruncatedCIFAR(x_train, y_train.long()), num_workers=args.workers, 
                    shuffle=args.shuffle, batch_size=args.batch_size)
                
                trunc_val_loader = DataLoader(TruncatedCIFAR(x_val, y_val.long()), num_workers=args.workers, 
                    shuffle=args.shuffle, batch_size=args.batch_size)

                unseen_loader = DataLoader(TruncatedCIFAR(ch.cat([x_unseen_, x_unseen]), ch.cat([y_unseen_, y_unseen]).long()), num_workers=args.workers, shuffle=args.shuffle, batch_size=args.batch_size)

                train.eval_model(args, base_classifier, unseen_loader, out_store, table='unseen')
                train.eval_model(args, base_classifier, test_loader, out_store, table='test')
                train.eval_model(args, base_classifier, trunc_train_loader, out_store, table='train')
                train.eval_model(args, base_classifier, trunc_val_loader, out_store, table='val')
                # close base classifier out store
                out_store.close()

                # train model using truncated ce loss 
                # logging store
                out_store = store.Store(LOGIT_BALL_CLASSIFIER)
                setup_store_with_metadata(args, out_store)
                delphi_, _ = model_utils.make_and_restore_model(arch='resnet18', dataset=ds)

                # train
                ch.manual_seed(i)
                delphi_ = train.train_model(args, delphi_, (trunc_train_loader, trunc_val_loader), store=out_store, phi=phi, criterion=grad.TruncatedCE.apply)

                # test model on data sets
                train.eval_model(args, delphi_, unseen_loader, out_store, table='unseen')
                train.eval_model(args, delphi_, test_loader, out_store, table='test')
                train.eval_model(args, delphi_, trunc_train_loader, out_store, table='train')
                train.eval_model(args, delphi_, trunc_val_loader, out_store, table='val')
                out_store.close()

                # train model using standard cross entropy loss
                # logging store
                out_store = store.Store(STANDARD_CLASSIFIER)
                setup_store_with_metadata(args, out_store)
                standard, _ = model_utils.make_and_restore_model(arch='resnet18', dataset=ds)

                # train
                ch.manual_seed(i)
                config.args = args

                # test model on datasets
                train.train_model(args, standard, (trunc_train_loader, trunc_val_loader), store=out_store, parallel=args.parallel)
                train.eval_model(args, standard, unseen_loader, out_store, table='unseen')
                train.eval_model(args, standard, test_loader, out_store, table='test')
                train.eval_model(args, standard, trunc_train_loader, out_store, table='train')
                train.eval_model(args, standard, trunc_val_loader, out_store, table='val')
                out_store.close()


def T_scaling(logits, temp):
    """
    Temperature scaling.
    """
    return ch.div(logits, temp)


def calibrate(test_loader, base_classifier): 
    """
    Run SGD procedure to find temperature 
    scaling parameter.
    Args: 
        test_loader (ch.nn.DataLoader) : pytorch DataLoader with test dataset
        base_classifier (AttackerModel) : AttackerModel to calibrate
    Returns: ch.Tensor with the calculated temperature scalar
    """
    temperature = ch.nn.Parameter(ch.ones(1).cuda())
    ce_loss = ch.nn.CrossEntropyLoss()

    # Removing strong_wolfe line search results in jump after 50 epochs
    optimizer = ch.optim.LBFGS([temperature], lr=0.001, max_iter=10000, line_search_fn='strong_wolfe')

    logits_list = []
    labels_list = []

    for images, labels in tqdm(test_loader, 0):
        images, labels = images.cuda(), labels.cuda()

        base_classifier.eval()
        with ch.no_grad():
            logits_list.append(base_classifier(images)[0])
            labels_list.append(labels)

    # create tensors
    logits_list = ch.cat(logits_list).cuda()
    labels_list = ch.cat(labels_list).cuda()

    def _eval():
        loss = ce_loss(T_scaling(logits_list, temperature), labels_list)
        loss.backward()
        return loss

    # run SGD
    optimizer.step(_eval)
    return temperature


def truncate(loader, base_classifier, phi, temperature, device):
    """
    Truncate dataset. 
    Args: 
        loader (ch.nn.DataLoader) : pytorch DataLoader with image dataset 
        base_classifier (AttackerModel) : pytorch AttackerModel
        phi (delphi.oracle) : oracle object 
        temperature (ch.Tensor) : temperature scaling constant
    Returns: 
        (trunc_X, unseen_X, trunc_y, unseen_y)
    """
    trunc_X, trunc_y = Tensor([]), Tensor([]) 
    trunc_test_X, trunc_test_y = Tensor([]), Tensor([]) 
    for i, batch in enumerate(loader): 

        inp, targ = batch[0].to(device), batch[1].to(device)
        logits, inp = base_classifier(inp)
        filtered = phi(ch.div(logits, temperature))
        indices = filtered.nonzero(as_tuple=False).flatten()
        test_indices = (~filtered).nonzero(as_tuple=False).flatten()
        trunc_X, trunc_y = ch.cat([trunc_X, inp[indices].cpu()]), ch.cat([trunc_y, targ[indices].cpu()])
        trunc_test_X, trunc_test_y = ch.cat([trunc_test_X, inp[test_indices].cpu()]), ch.cat([trunc_test_y, targ[test_indices].cpu()])
    # truncated test set (unseen data that the model has not been tested on)
    return (trunc_X, trunc_test_X, trunc_y.long(), trunc_test_y)


if __name__ == '__main__': 
    # hyperparameters
    args = Parameters({ 
        'workers': 8, 
        'batch_size': 128, 
        'lr': 1e-2, 
        'accuracy': True,
        'momentum': 0.9, 
        'weight_decay': 5e-4, 
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
        'step_lr': 1, 
        'step_lr_gamma': 1.0,
        'device': 'cuda' if ch.cuda.is_available() else 'cpu'
    })

    main(args)







