from torch import Tensor

JUPYTER = 'jupyter'
TERMINAL = 'terminal'
IPYTHON = 'ipython'

LOGS_SCHEMA = {
    'epoch':int,
    'val_prec1':float,
    'val_loss':float,
    'train_prec1':float,
    'train_loss':float,
    'time':float
}

# scheduler constants
CYCLIC='cyclic'
COSINE='cosine'
LINEAR='linear'


LOGS_TABLE = 'logs'

CKPT_NAME = 'checkpoint.pt'
BEST_APPEND = '.best'
CKPT_NAME_LATEST = CKPT_NAME + '.latest'
CKPT_NAME_BEST = CKPT_NAME + BEST_APPEND

# CIFAR10 dataset info
CIFAR_LABELS = {
    0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
    5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck',
}


# CIFAR10 dataset information
_CIFAR10_STATS = {'mean': [0.4914, 0.4822, 0.4465],
                  'std': [0.2023, 0.1994, 0.2010]}

# IMAGENET dataset information
_IMAGENET_STATS = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}

_IMAGENET_PCA = {
    'eigval': Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}