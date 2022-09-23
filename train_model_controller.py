#!/usr/bin/env python
# coding: utf-8

# Single model train and evaluate. Supports command line interface.
# Model specific hyperparameters and metric function defined in external .py module.
# General reusable module that is no model specific.

# Load libraries

import logging
import argparse
import importlib
import sys
from copy import deepcopy

from global_vars import HYPERPARAMETERS_DIR
from train import train_and_save_model
from train_model_controller_gyopt import update_hyperparameters, get_system_params


# Main block runs the hyperparameter optimization
if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train and evaluate a single model.')
    parser.add_argument('hyperparameters', type=str,
                        help='name of .py module defining hyperparameters')
    parser.add_argument('--beg_iter_num', metavar='N', type=int, default=1,
                        help='starting iteration number (default: 1)')
    parser.add_argument('--num_iter', metavar='N', type=int, default=1,
                        help='number of times to train and evaluate model (default: 1)')
    parser.add_argument('--model_name', metavar='name',
                        type=str, default=None, help='override model name')
    parser.add_argument('--max_prime', metavar='N',
                        type=int, default=None, help='override max prime defined in hyperparameters, also update reducing layers accordingly')
    parser.add_argument('--min_max_conductor', metavar='range',
                        type=str, default=None, help='override min_max_conductor')
    parser.add_argument('--min_max_test_conductor', metavar='range',
                        type=str, default=None, help='override min_max_test_conductor')
    parser.add_argument('--N_masks', metavar='list',
                        type=str, default=None, help='override N_masks')
    parser.add_argument('--use_last_cls', metavar='N',
                        type=int, default=1, help='number of last classes to group together for making binary classes in binary classification')
    parser.add_argument('--use_cache_file_name_prefix', action=argparse.BooleanOptionalAction,
                        default=True, help='force use of model name as a prefix for cache file name; alternatively will load from cache with or without prefix')
    parser.add_argument('--dump_test_set_classes', action=argparse.BooleanOptionalAction,
                        default=False, help='dump labels (classes) of test set in file true_test_classes.txt')
    parser.add_argument('--debug', action=argparse.BooleanOptionalAction,
                        default=False, help='print debug logs')
    parser.add_argument('--cuda_device_no', metavar='N', type=int, default=0,
                        help='cuda device number used for GPU computations; in single GPU system use default (default: 0)')
    parser.add_argument('--load_data_to_GPU', action=argparse.BooleanOptionalAction,
                        default=False, help='load dataset to GPU')
    parser.add_argument('--num_workers', metavar='N', type=int, default=4,
                        help='number of workers in case of using CPU dataloader (default: 4)')
    parser.add_argument('--lr_finder_usage', action=argparse.BooleanOptionalAction,
                        default=True, help='use learning rate finder and store results; if starting multiple parallel instances, only use for one instance')
    parser.add_argument('--only_load_dataset', action=argparse.BooleanOptionalAction,
                        default=False, help='only load dataset and quit, set num_iter=1, used for dataset caching')
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(
            format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG)
    else:
        logging.basicConfig(
            format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)

    sys.path.insert(0, str(HYPERPARAMETERS_DIR))
    hyperparameters_def = importlib.import_module(args.hyperparameters)
    hyperparameters = hyperparameters_def.hyperparameters

    if args.num_iter > 1:  # if training more then once, assume no specific model num if given - will be randomly generated
        assert hyperparameters['model_num'] is None

    # Update hyperparameters using command line parameters
    update_hyperparameters(hyperparameters, args)
    if args.only_load_dataset:
        logging.info("Only loading dataset. Overriding num_iter to 1.")
        args.num_iter = 1

    # System specific hyperparameters
    system_params = get_system_params(args)

    # Model training loop
    for iter_num in range(args.beg_iter_num, args.beg_iter_num + args.num_iter):
        hyperparameters_copy = deepcopy(hyperparameters)
        hyperparameters_copy['iter_num'] = iter_num
        logging.info(
            f"Starting iteration number: {hyperparameters_copy['iter_num']}")

        current_model_name, results = train_and_save_model(
            hyperparameters_copy, **system_params)

        logging.info(f"Results of {current_model_name}: {results}")

    logging.info(
        f"Finished after {iter_num} completed iterations. Logs saved.")
