# CONSTANTS
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

EVAL_LOGS_SCHEMA = {
    'test_prec1':float,
    'test_loss':float,
    'time':float
}

# scheduler constants
CYCLIC='cyclic'
COSINE='cosine'
LINEAR='linear' 

LOGS_TABLE='logs'
EVAL_LOGS_TABLE='eval'

CKPT_NAME='checkpoint.pt'
BEST_APPEND='.best'
CKPT_NAME_LATEST=CKPT_NAME + '.latest'
CKPT_NAME_BEST=CKPT_NAME + BEST_APPEND