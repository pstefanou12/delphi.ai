"""
The main file, which setups and performs stochastic processes for library.
"""

import os


try:
    from .train import train_model, eval_model
    from .defaults import check_and_fill_args

def main(args, model, loaders, update_params=None, store=None):
    '''Given arguments from `setup_args` and a store from `setup_store`,
    trains as a model. Check out the argparse object in this file for
    argument options.
    '''


    print(args)
    if args.eval_only:
        return eval_model(args, model, val_loader, store=store)

    if not args.resume_optimizer: checkpoint = None
    model = train_model(args, model, loaders, store=store,
                                    checkpoint=checkpoint, update_params=update_params)
    return model
