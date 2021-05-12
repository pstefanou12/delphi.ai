"""
Module containing all the supported datasets, which are subclasses of the
abstract class :class:`robustness.datasets.DataSet`. 
Currently supported datasets:
- CIFAR-10 (:class:`delphi.datasets.CIFAR`)
"""

import torch as ch
from torch import Tensor
from torch.utils.data import TensorDataset
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torchvision import transforms, datasets
from sklearn.linear_model import LinearRegression, LogisticRegression
import copy
import warnings

from .helpers import censored_sample_nll, cov
from . import data_augmentation as da
from .. import cifar_models
from .. import imagenet_models
from . import loaders


###
# Computer Vision Datasets: (all subclassed from dataset)
# In order:
## - CIFAR
## - Imagenet
###

# required and optional arguments for datasets
CNN_REQUIRED_ARGS = ['num_classes', 'mean', 'std',
                         'transform_train', 'transform_test']
CNN_OPTIONAL_ARGS = ['custom_class', 'label_mapping', 'custom_class_args']

CENSORED_MULTIVARIATE_NORMAL_REQUIRED_ARGS, CENSORED_MULTIVARIATE_NORMAL_OPTIONAL_ARGS = ['custom_class', 'custom_class_args'], ['label_mapping', 'transform_train', 'transform_test']
TRUNCATED_MULTIVARIATE_NORMAL_REQUIRED_ARGS, TRUNCATED_MULTIVARIATE_NORMAL_OPTIONAL_ARGS = ['custom_class', 'custom_class_args'], ['label_mapping', 'transform_train', 'transform_test']
TENSOR_REQUIRED_ARGS, TENSOR_OPTIONAL_ARGS = ['custom_class', 'custom_class_args'], ['label_mapping', 'transform_train', 'transform_test']


class DataSet(object):
    '''
    Base class for representing a dataset. Meant to be subclassed, with
    subclasses implementing the `get_model` function.
    '''

    def __init__(self, ds_name, required_args, optional_args, data_path=None, **kwargs):
        """
        Args:
            ds_name (str) : string identifier for the dataset
            data_path (str) : *optional argument, but required for CNNs* path to the dataset
            num_classes (int) : *required kwarg for CNN*, the number of classes in
                the dataset
            mean (ch.tensor) : *required kwarg for CNN*, the mean to normalize the
                dataset with (e.g.  :samp:`Tensor([0.4914, 0.4822,
                0.4465])` for CIFAR-10)
            std (ch.tensor) : *required kwarg for CNN*, the standard deviation to
                normalize the dataset with (e.g. :samp:`Tensor([0.2023,
                0.1994, 0.2010])` for CIFAR-10)

            custom_class (type) : *required kwarg*, a
                :samp:`torchvision.models` class corresponding to the
                dataset, if it exists (otherwise :samp:`None`)
            label_mapping (dict[int,str]) : *required kwarg for CNN*, a dictionary
                mapping from class numbers to human-interpretable class
                names (can be :samp:`None`)
            transform_train (torchvision.transforms) : *required kwarg*,
                transforms to apply to the training images from the
                dataset
            transform_test (torchvision.transforms) : *required kwarg*,
                transforms to apply to the validation images from the
                dataset
        """
        missing_args = set(required_args) - set(kwargs.keys())
        if len(missing_args) > 0:
            raise ValueError("Missing required args %s" % missing_args)

        extra_args = set(kwargs.keys()) - set(required_args + optional_args)
        if len(extra_args) > 0:
            raise ValueError("Got unrecognized args %s" % extra_args)
        final_kwargs = {k: kwargs.get(k, None) for k in required_args + optional_args}

        self.ds_name = ds_name
        self.data_path = data_path
        self.__dict__.update(final_kwargs)

    def override_args(self, default_args, kwargs):
        '''
        Convenience method for overriding arguments. (Internal)
        '''
        for k in kwargs:
            if not (k in default_args): continue
            req_type = type(default_args[k])
            no_nones = (default_args[k] is not None) and (kwargs[k] is not None)
            if no_nones and (not isinstance(kwargs[k], req_type)):
                raise ValueError(f"Argument {k} should have type {req_type}")
        return {**default_args, **kwargs}

    def get_model(self, arch, pretrained):
        '''
        Should be overriden by subclasses. Also, you will probably never
        need to call this function, and should instead by using
        `model_utils.make_and_restore_model </source/robustness.model_utils.html>`_.
        Args:
            arch (str) : name of architecture
            pretrained (bool): whether to try to load torchvision
                pretrained checkpoint
        Returns:
            A model with the given architecture that works for each
            dataset (e.g. with the right input/output dimensions).
        '''
        raise NotImplementedError

    def make_loaders(self, workers, batch_size, data_aug=True, subset=None,
                     subset_start=0, subset_type='rand', val_batch_size=None,
                     train=True, val=True, shuffle_train=True, shuffle_val=True, subset_seed=None):
        '''
        Args:
            workers (int) : number of workers for data fetching (*required*).
                batch_size (int) : batch size for the data loaders (*required*).
            data_aug (bool) : whether or not to do train data augmentation.
            subset (None|int) : if given, the returned training data loader
                will only use a subset of the training data; this should be a
                number specifying the number of training data points to use.
            subset_start (int) : only used if `subset` is not None; this specifies the
                starting index of the subset.
            subset_type ("rand"|"first"|"last") : only used if `subset is
                not `None`; "rand" selects the subset randomly, "first"
                uses the first `subset` images of the training data, and
                "last" uses the last `subset` images of the training data.
            seed (int) : only used if `subset == "rand"`; allows one to fix
                the random seed used to generate the subset (defaults to 1).
            val_batch_size (None|int) : if not `None`, specifies a
                different batch size for the validation set loader.
            only_val (bool) : If `True`, returns `None` in place of the
                training data loader
            shuffle_train (bool) : Whether or not to shuffle the training data
                in the returned DataLoader.
            shuffle_val (bool) : Whether or not to shuffle the test data in the
                returned DataLoader.
        Returns:
            A training loader and validation loader according to the
            parameters given. These are standard PyTorch data loaders, and
            thus can just be used via:
            >>> train_loader, val_loader = ds.make_loaders(workers=8, batch_size=128)
            >>> for im, lab in train_loader:
            >>>     # Do stuff...
        '''
        transforms = (self.transform_train, self.transform_test)
        return loaders.make_loaders(workers=workers,
                                    batch_size=batch_size,
                                    transforms=transforms,
                                    data_path=self.data_path,
                                    data_aug=data_aug,
                                    dataset=self.ds_name,
                                    label_mapping=self.label_mapping,
                                    custom_class=self.custom_class,
                                    val_batch_size=val_batch_size,
                                    subset=subset,
                                    subset_start=subset_start,
                                    subset_type=subset_type,
                                    val=val,
                                    train=train,
                                    seed=subset_seed,
                                    shuffle_train=shuffle_train,
                                    shuffle_val=shuffle_val,
                                    custom_class_args=self.custom_class_args)


class CIFAR(DataSet):
    """
    CIFAR-10 dataset [Kri09]_.
    A dataset with 50k training images and 10k testing images, with the
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
    """
    def __init__(self, data_path='/tmp/', **kwargs):
        """
        """
        ds_kwargs = {
            # 'data_path': data_path,
            'num_classes': 10,
            'mean': ch.tensor([0.4914, 0.4822, 0.4465]),
            'std': ch.tensor([0.2023, 0.1994, 0.2010]),
            'custom_class': datasets.CIFAR10,
            'label_mapping': None,
            'transform_train': da.TRAIN_TRANSFORMS_DEFAULT(32),
            'transform_test': da.TEST_TRANSFORMS_DEFAULT(32),
        }
        ds_kwargs = self.override_args(ds_kwargs, kwargs)
        super(CIFAR, self).__init__('cifar', CNN_REQUIRED_ARGS, CNN_OPTIONAL_ARGS, data_path=data_path, **ds_kwargs)

    def get_model(self, arch, pretrained):
        """
        """
        if pretrained:
            raise ValueError('CIFAR does not support pytorch_pretrained=True')
        return cifar_models.__dict__[arch](num_classes=self.num_classes)


class ImageNet(DataSet):
    '''
    ImageNet Dataset [DDS+09]_.
    Requires ImageNet in ImageFolder-readable format. 
    ImageNet can be downloaded from http://www.image-net.org. See
    `here <https://pytorch.org/docs/master/torchvision/datasets.html#torchvision.datasets.ImageFolder>`_
    for more information about the format.
    .. [DDS+09] Deng, J., Dong, W., Socher, R., Li, L., Li, K., & Fei-Fei, L. (2009). ImageNet: A large-scale hierarchical image database. 2009 IEEE Conference on Computer Vision and Pattern Recognition, 248-255.
    '''
    def __init__(self, data_path, **kwargs):
        """
        """
        ds_kwargs = {
            'num_classes': 1000,
            'mean': ch.tensor([0.485, 0.456, 0.406]),
            'std': ch.tensor([0.229, 0.224, 0.225]),
            'custom_class': None,
            'label_mapping': None,
            'transform_train': da.TRAIN_TRANSFORMS_IMAGENET,
            'transform_test': da.TEST_TRANSFORMS_IMAGENET
        }
        ds_kwargs = self.override_args(ds_kwargs, kwargs)
        super(ImageNet, self).__init__('imagenet', CNN_REQUIRED_ARGS, CNN_OPTIONAL_ARGS, data_path, **ds_kwargs)

    def get_model(self, arch, pretrained):
        """
        """
        print("pretrained: {}".format(pretrained))
        return imagenet_models.__dict__[arch](num_classes=self.num_classes, 
                                        pretrained=pretrained)


class CensoredNormal(ch.utils.data.Dataset):
    def __init__(self, S):
        # empirical mean and variance
        self._loc = ch.mean(S, dim=0)
        self._var = ch.var(S, dim=0)
        # apply gradient
        self.S = censored_sample_nll(S)

    def __len__(self): 
        return self.S.size(0)
    
    def __getitem__(self, idx):
        return [self.S[idx],]

    @property
    def loc(self):
        return self._loc.clone()

    @property
    def var(self):
        return self._var.clone()
    
    
class CensoredMultivariateNormal(ch.utils.data.Dataset):
    def __init__(self, S):
        # empirical mean and variance
        self._loc = S.mean(0)
        self._covariance_matrix = cov(S)
        # apply gradient to data
        self.S = censored_sample_nll(S) 

    def __len__(self): 
        return self.S.size(0)
    
    def __getitem__(self, idx):
        return [self.S[idx],]
    
    @property
    def loc(self): 
        return self._loc.clone()
    
    @property
    def covariance_matrix(self): 
        return self._covariance_matrix.clone()


class TruncatedNormal(ch.utils.data.Dataset):
    def __init__(self, S):
        self.S = S
        # samples 
        self._loc = ch.mean(S, dim=0)
        self._var = ch.var(S, dim=0)
        # compute gradients
        pdf = ch.exp(Normal(ch.zeros(1), ch.eye(1).flatten()).log_prob(self.S))
        self.loc_grad = pdf*(self._loc - self.S)
        self.var_grad = .5*pdf*(ch.bmm(self.S.unsqueeze(2), self.S.unsqueeze(1)) - self._var - self._loc.unsqueeze(0).matmul(self._loc.unsqueeze(1))).flatten(1)
        
    def __len__(self): 
        return self.S.size(0)
    
    def __getitem__(self, idx):
        """
        :returns: (sample, sample pdf, sample mean coeffcient, sample covariance matrix coeffcient)
        """
        return self.S[idx], self.loc_grad[idx], self.var_grad[idx]
    
    @property
    def loc(self): 
        return self._loc.clone()
    
    @property
    def var(self): 
        return self._var.clone()


class TruncatedMultivariateNormal(ch.utils.data.Dataset):
    def __init__(self, S):
        # samples 
        self.S = S
        self._loc = ch.mean(S, dim=0)
        self._covariance_matrix = cov(S)
        # compute gradients
        pdf = ch.exp(MultivariateNormal(ch.zeros(self.S.size(1)).double(), ch.eye(self.S.size(1)).double()).log_prob(self.S)).unsqueeze(1)
        # pdf of each sample
        self.loc_grad = pdf*(self._loc - self.S)
        self.cov_grad = (.5*pdf*((ch.bmm(S.unsqueeze(2), S.unsqueeze(1)) - self._covariance_matrix - self._loc.unsqueeze(0).matmul(self._loc.unsqueeze(1))).flatten(1))).unflatten(1, self._covariance_matrix.size())

    def __len__(self): 
        return self.S.size(0)
    
    def __getitem__(self, idx):
        """
        :returns: (sample, sample pdf, sample mean coeffcient, sample covariance matrix coeffcient)
        """
        return self.S[idx], self.loc_grad[idx], self.cov_grad[idx]
    
    @property
    def loc(self): 
        return self._loc.clone()
    
    @property
    def covariance_matrix(self): 
        return self._covariance_matrix.clone()


DATASETS = {
    'imagenet': ImageNet, 
    'cifar': CIFAR,
    'censored_normal': CensoredNormal, 
    'censored_multivariate_normal': CensoredMultivariateNormal, 
    'truncated_normal': TruncatedNormal, 
    'truncated_multivariate_normal': TruncatedMultivariateNormal, 
    'tensor': TensorDataset, 
}