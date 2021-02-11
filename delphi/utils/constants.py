LOGS_SCHEMA = {
    'epoch':int,
    'val_prec1':float,
    'val_loss':float,
    'train_prec1':float,
    'train_loss':float,
    'time':float
}

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