"""
Module containing all the supported datasets, which are subclasses of the
abstract class :class:`robustness.datasets.DataSet`. 
Currently supported datasets:
- CIFAR-10 (:class:`delphi.datasets.CIFAR`)
"""

import torch as ch
import torch.linalg as LA
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions.multivariate_normal import MultivariateNormal
from torchvision import datasets

from .helpers import censored_sample_nll, cov
from .defaults import DATASET_DEFAULTS, check_and_fill_args
from . import data_augmentation as da
from .. import cifar_models
from .. import imagenet_models


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
        `model_utils.make_and_restore_model </source/delphi.utils.model_utils.html>`_.
        Args:
            arch (str) : name of architecture
            pretrained (bool): whether to try to load torchvision
                pretrained checkpoint
        Returns:
            A model with the given architecture that works for each
            dataset (e.g. with the right input/output dimensions).
        '''
        raise NotImplementedError

    def make_loaders(self, workers, batch_size, data_aug=True,
                    subset=None, subset_type='rand', subset_start=0, val_batch_size=None,
                    train=True, val=True, shuffle_train=True, shuffle_val=True, seed=1,
                    verbose=True):
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
        # check that at least a train loader or validation loader specified to be created
        if not train and not val:
            raise ValueError("Neither training loader nor validation loader specified")
        # initialize loader variables
        train_set, test_set = None, None
        train_loader, test_loader = None, None

        if not val_batch_size:
            val_batch_size = batch_size

        if not self.custom_class:
            if train:
                train_set = folder.ImageFolder(root=self.data_path, transform=self.transform_train if data_aug else self.transform_test,
                                            label_mapping=self.label_mapping)
            if val:
                test_set = folder.ImageFolder(root=self.data_path, transform=self.transform_test,
                                            label_mapping=self.label_mapping)

            if train:
                attrs = ["samples", "train_data", "data"]
                vals = {attr: hasattr(train_set, attr) for attr in attrs}
                assert any(vals.values()), f"dataset must expose one of {attrs}"
                train_sample_count = len(getattr(train_set, [k for k in vals if vals[k]][0]))

            if (train and not val) and (subset is not None) and (subset <= train_sample_count):
                assert train and not val
                if subset_type == 'rand':
                    rng = np.random.RandomState(seed)
                    subset = rng.choice(list(range(train_sample_count)), size=subset + subset_start, replace=False)
                    subset = subset[subset_start:]
                elif subset_type == 'first':
                    subset = np.arange(subset_start, subset_start + subset)
                else:
                    subset = np.arange(train_sample_count - subset, train_sample_count)

                train_set = Subset(train_set, subset)
        else:
            if self.custom_class_args is None: self.custom_class_args = {}
            if self.data_path is not None:
                if train:
                    train_set = self.custom_class(root=self.data_path, train=True, download=True,
                                        transform=self.transform_train, **self.custom_class_args)
                if val:
                    test_set = self.custom_class(root=self.data_path, train=False, download=True,
                                        transform=self.transform_test, **self.custom_class_args)
        if train_set is not None:
            train_loader = DataLoader(train_set, batch_size=batch_size,
                shuffle=shuffle_train, num_workers=workers, pin_memory=True)

        if test_set is not None:
            test_loader = DataLoader(test_set, batch_size=val_batch_size,
                    shuffle=shuffle_val, num_workers=workers, pin_memory=True)

        return train_loader, test_loader


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
        return imagenet_models.__dict__[arch](num_classes=self.num_classes, 
                                        pretrained=pretrained)


class Normalize:
    """
    Normalizes the input covariate features for truncated
    regression.
    """

    def __init__(self):
        '''
        Args:
            X (torch.Tensor): regression input features; shape expected to be n (number of samples) by d (number of dimensions)
        '''
        super(Normalize).__init__()
        self._l_inf, self._beta = None, None

    def fit_transform(self, X):
        '''
        Normalize input features truncated regression
        '''
        # normalize input features
        self._l_inf = LA.norm(X, dim=-1, ord=float('inf')).max()
        self._beta = self._l_inf * (X.size(1) ** .5)
        return self

    def transform(self, X):      
        return X / self._beta

    @property
    def beta(self):
        return self._beta

    @property
    def l_inf(self):
        return self._l_inf


def make_train_and_val(args, X, y): 
    # check arguments are correct
    args = check_and_fill_args(args, DATASET_DEFAULTS)
    # separate into training and validation set
    rand_indices = ch.randperm(X.size(0))
    val = int(args.val * X.size(0))
    train_indices, val_indices = rand_indices[val:], rand_indices[:val]
    X_train,y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]

    # normalize input covariates
    if args.normalize:
        train_norm = Normalize().fit_transform(X_train)
        X_train = train_norm.transform(X_train)
        val_norm = Normalize().fit_transform(X_val)
        X_val = val_norm.transform(X_val)

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.workers)

    return train_loader, val_loader


def make_train_and_val_distr(args, S, ds): 
    # check arguments are correct
    args = check_and_fill_args(args, DATASET_DEFAULTS)
    # separate into training and validation set
    rand_indices = ch.randperm(S.size(0))
    val = int(args.val * S.size(0))
    train_indices, val_indices = rand_indices[val:], rand_indices[:val]
    X_train, X_val = S[train_indices], S[val_indices]
    train_ds = ds(X_train)
    val_ds = ds(X_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size)
    val_loader = DataLoader(val_ds, batch_size=len(val_ds))

    return train_loader, val_loader


class CensoredNormalDataset(ch.utils.data.Dataset):
    def __init__(self, S):
        # empirical mean and variance
        self._loc = ch.mean(S, dim=0)
        self._covariance_matrix = cov(S)
        self.S = S 
        # apply gradient
        self.S_grad = censored_sample_nll(S)

    def __len__(self): 
        return self.S.size(0)
    
    def __getitem__(self, idx):
        return [self.S[idx], self.S_grad[idx],]

    @property
    def loc(self):
        return self._loc.clone()

    @property
    def covariance_matrix(self):
        return self._covariance_matrix.clone()


class TruncatedNormalDataset(ch.utils.data.Dataset):
    def __init__(self, S):
        self.S = S
        # samples 
        self._loc = self.S.mean(0)
        self._covariance_matrix = cov(self.S)
        # compute gradients
        M = MultivariateNormal(self._loc, self._covariance_matrix)
        self.pdf = ch.exp(MultivariateNormal(ch.zeros(self.S.size(1)), ch.eye(self.S.size(1))).log_prob(self.S))[...,None]
       # self.pdf = ch.exp(M.log_prob(self.S))[...,None]

        self.loc_grad =  self._loc - self.S
        self.cov_grad = .5 * (ch.bmm(self.S.unsqueeze(2), self.S.unsqueeze(1)) - self._covariance_matrix - self._loc[...,None] @ self._loc[None,...]).flatten(1)
        
    def __len__(self): 
        return self.S.size(0)
    
    def __getitem__(self, idx):
        """
        :returns: (sample, sample pdf, sample mean coeffcient, sample covariance matrix coeffcient)
        """
        return self.S[idx], self.pdf[idx], self.loc_grad[idx], self.cov_grad[idx]

    @property
    def loc(self): 
        return self._loc.clone()
    
    @property
    def covariance_matrix(self): 
        return self._covariance_matrix.clone()


DATASETS = {
    'imagenet': ImageNet, 
    'cifar': CIFAR,
    'tensor': TensorDataset, 
}
