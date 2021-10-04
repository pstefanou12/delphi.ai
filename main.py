"""
CLI interface for using delphi.
"""

from argparse import ArgumentParser
import git
import torch as ch
import os
import cox
from cox.utils import Parameters
from cox.store import Store
import config

try: 
    from delphi import train
    from delphi.utils.model_utils import make_and_restore_model
    from delphi.utils.datasets import DATASETS
    from delphi.utils import constants as consts
    from delphi.utils import defaults
    from delphi.utils.defaults import check_and_fill_args
    from delphi.utils.helpers import setup_store_with_metadata, DataPrefetcher
    from delphi import grad
except: 
    raise ValueError("Error when importing packages.")
# set environment variable so that stores can create output files
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

parser = ArgumentParser()
parser = defaults.add_args_to_parser(defaults.CONFIG_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.MODEL_LOADER_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.TRAINING_ARGS, parser)

def main(args, store=None):
    '''Given arguments from `setup_args` and a store from `setup_store`,
    trains as a model. Check out the argparse object in this file for
    argument options.
    '''
    # MAKE DATASET AND LOADERS
    data_path = os.path.expandvars(args.data)
    dataset = DATASETS[args.dataset](data_path)

    train_loader, val_loader = dataset.make_loaders(args.workers,
                    args.batch_size, data_aug=bool(args.data_aug))

    train_loader = DataPrefetcher(train_loader)
    val_loader = DataPrefetcher(val_loader)
    loaders = (train_loader, val_loader)

    # MAKE MODEL
    model, checkpoint = make_and_restore_model(arch=args.arch,
            dataset=dataset, resume_path=args.resume)
    if 'module' in dir(model): model = model.module

    if args.eval_only:
        return eval_model(args, model, val_loader, store=store)

    if not args.resume_optimizer: checkpoint = None
    model = train.train_model(args, model, loaders, store=store,
                                    checkpoint=checkpoint, parallel=args.parallel)
    # if there is a store, close it at the end of the procedure
    if store is not None:
        store.close()
    return model

def setup_args(args):
    '''
    Fill the args object with reasonable defaults from
    :mod:`robustness.defaults`, and also perform a sanity check to make sure no
    args are missing.
    '''
    # override non-None values with optional config_path
    if args.config_path:
        args = cox.utils.override_json(args, args.config_path)

    ds_class = DATASETS[args.dataset]
    args = check_and_fill_args(args, defaults.CONFIG_ARGS, ds_class)

    if not args.eval_only:
        args = check_and_fill_args(args, defaults.TRAINING_ARGS, ds_class)

    if args.adv_train or args.adv_eval:
        args = check_and_fill_args(args, defaults.PGD_ARGS, ds_class)

    args = check_and_fill_args(args, defaults.MODEL_LOADER_ARGS, ds_class)
    if args.eval_only: assert args.resume is not None, \
            "Must provide a resume path if only evaluating"
    return args

def setup_store_with_metadata(args):
    '''
    Sets up a store for training according to the arguments object. See the
    argparse object above for options.
    '''
    # Add git commit to args
    try:
        repo = git.Repo(path=os.path.dirname(os.path.realpath(__file__)),
                            search_parent_directories=True)
        version = repo.head.object.hexsha
    except git.exc.InvalidGitRepositoryError:
        version = __version__
    args.version = version

    # Create the store
    store = Store(args.out_dir, args.exp_name)
    args_dict = args.__dict__
    schema = cox.store.schema_from_dict(args_dict)
    store.add_table('metadata', schema)
    store['metadata'].append_row(args_dict)

    return store

if __name__ == '__main__':
    # set environment variable so that stores can create output files
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

    args = parser.parse_args()
    args = cox.utils.Parameters(args.__dict__)

    args = setup_args(args)
    store = setup_store_with_metadata(args)

    args.__setattr__('custom_criterion', grad.GumbelCE.apply)
    args.__setattr__('num_samples', 1000)

    if ch.cuda.is_available(): 
        args.__setattr__('device', 'cuda')
    else: 
        args.__setattr__('device', 'cpu')

    # set global 
    config.args = args 
    final_model = main(config.args, store=store)







