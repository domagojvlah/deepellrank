#!/usr/bin/env python
# coding: utf-8

# Bayesian model hyperparameter optimizer. Supports command line interface.
# Model specific hyperparameters and metric function defined in external .py module.
# General reusable module that is no model specific.

# Load libraries

import os
import json
import time
import logging
import GPyOpt
import flatdict
import argparse
import importlib
import sys
import math
import primesieve as ps

from subprocess import Popen, PIPE, STDOUT
from threading import Thread
from queue import Queue, Empty

from global_vars import LOGS_DIR, HYPERPARAMETERS_DIR


# Optimizer black box function that calls the command line module
iter_num = 1  # global variable for optimizer black box function
all_metrics = {}


def train_call(param):
    # Set iteration number
    global iter_num, all_metrics
    hyperparameters['iter_num'] = iter_num
    logging.info(f"Iteration number: {hyperparameters['iter_num']}")

    # Set hyperparameters
    logging.info("Optimizing hyperparameters: ")
    for idx, (key, val) in enumerate(operation.items()):
        if ":" in key:
            parts = key.split(":")
            assert len(parts) == 2
            hyperparameters[parts[0]][parts[1]] = val(param[0][idx])
        else:
            hyperparameters[key] = val(param[0][idx])
        logging.info(
            f"{idx}, {key} : {val(param[0][idx])} : {type(val(param[0][idx]))}")

    # Calling model train module as separate process and handling its console output

    def enqueue_output(out, queue):
        for line in iter(out.readline, b''):
            queue.put(line.decode())
        out.close()
        return

    p = Popen(f"""python train_model_handler.py '{json.dumps(hyperparameters)}' '{json.dumps(system_params)}'""",
              shell=True, stdout=PIPE, stderr=STDOUT)
    q = Queue()
    t = Thread(target=enqueue_output, args=(p.stdout, q))
    t.daemon = True  # thread dies with the program
    t.start()

    lines = []
    while (p.poll() is None) or (not q.empty()):
        # read line without blocking
        try:
            line = q.get_nowait()  # or q.get(timeout=.1)
        except Empty:
            time.sleep(0.1)  # do nothing
        else:  # got line
            print(line, end="")
            lines.append(line)

    metric = compute_gyopt_metric(lines)
    logging.info(
        f"Finished computing {iter_num} iteration. Optimizer metric computed: {metric}. Saving metrics log...")
    all_metrics[iter_num] = metric
    with open(LOGS_DIR / f"{hyperparameters['model_name']}-metrics.json", "w") as fp:
        json.dump(all_metrics, fp)
    iter_num += 1
    return metric


def create_search_space_and_operations(hyperparameters):
    """Helper function for creating search space and operation dicts.
    Decodes which hyperparameters to optimize, optimization ranges
    and creates maps from optimization ranges to specific values of
    parameters.

    Args:
        hyperparameters (dict): data, model and training specific hyperparameters

    Raises:
        ValueError: invalid type of optimization, only allowed are 'discrete' and 'continuous'

    Returns:
        tuple of dict: search_space and operation dicts
    """
    flat_hyperparam = flatdict.FlatDict(hyperparameters)

    operation = {}
    opt_range = {}
    opt_type = {}
    for key, val in flat_hyperparam.items():
        if isinstance(val, str):
            if len(val) >= 9:
                if "OPTIMIZE:" == val[:9]:
                    parts = val[9:].split("#")
                    assert len(parts) == 3
                    operation[key] = eval(parts[0])
                    if "discrete" in parts[1]:
                        opt_type[key] = 'discrete'
                    elif "continuous" in parts[1]:
                        opt_type[key] = 'continuous'
                    else:
                        raise ValueError(
                            f"Unknown optimization type in {key} : {val}")
                    opt_range[key] = eval(parts[2])

    search_space = [{'name': key, 'type': opt_type[key], 'domain': val}
                    for key, val in opt_range.items()]
    logging.info(f"Search space: {search_space}")

    return search_space, operation


def update_hyperparameters(hyperparameters, args):
    # Possibly change model name
    if args.model_name is not None:
        hyperparameters['model_name'] = args.model_name
        logging.info(
            f"Overriding model name to {hyperparameters['model_name']}")

    # Possibly change max_prime and reducing_layers accordingly
    if args.max_prime is not None:
        hyperparameters['dataloader']['max_prime'] = args.max_prime
        logging.info(
            f"Overriding max prime to {hyperparameters['dataloader']['max_prime']}")
        if hyperparameters['model_type'] == "SimpleConvolutionalClassificationModel":
            data_length = len(
                ps.primes(hyperparameters['dataloader']['max_prime']))
            hyperparameters['model']['reducing_layers'] = int(math.ceil(
                math.log(data_length) / math.log(hyperparameters['model']['stride'])))
            logging.info(
                f"Overriding reducing layers to {hyperparameters['model']['reducing_layers']} - computed from max prime")

    # Possibly change min_max_conductor
    if args.min_max_conductor is not None:
        hyperparameters['dataloader']['min_max_conductor'] = eval(
            args.min_max_conductor)
        logging.info(
            f"Overriding min_max_conductor to {hyperparameters['dataloader']['min_max_conductor']}")

    # Possibly change min_max_test_conductor
    if args.min_max_test_conductor is not None:
        hyperparameters['dataloader']['min_max_test_conductor'] = eval(
            args.min_max_test_conductor)
        logging.info(
            f"Overriding min_max_test_conductor to {hyperparameters['dataloader']['min_max_test_conductor']}")

    # Possibly change N_masks
    if args.N_masks is not None:
        hyperparameters['dataloader']['N_masks'] = eval(args.N_masks)
        logging.info(
            f"Overriding N_masks to {hyperparameters['dataloader']['N_masks']}")
    
    # Possibly dump labels (classes) of test set in file
    if args.dump_test_set_classes:
        hyperparameters['dataloader']['dump_test_set_classes'] = args.dump_test_set_classes
        logging.info("Enabling dumping of labels (classes) of test set in file.")


def get_system_params(args):
    return dict(
        cuda_device=f"cuda:{args.cuda_device_no:d}",
        # Load dataset to GPU memory (possible speedup, if it fits the memory)
        data_load_to_GPU=args.load_data_to_GPU,
        num_workers=args.num_workers,
        lr_finder_usage=args.lr_finder_usage,
        only_load_dataset=args.only_load_dataset,
        use_cache_file_name_prefix=args.use_cache_file_name_prefix,
        use_last_cls=args.use_last_cls,
    )


# Main block runs the hyperparameter optimization
if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run optimization of selected hyperparameters.')
    parser.add_argument('hyperparameters', type=str,
                        help='name of .py module defining hyperparameters and metric')
    parser.add_argument('init_iter', type=int, default=10,
                        help='initial number of iterations')
    parser.add_argument('num_iter', type=int, default=100,
                        help='maximum number of iterations after initial')
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
    parser.add_argument('--gyopt_metric', metavar='name',
                        type=str, default=None, help='name of .py module defining gyopt metric')
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
                        default=False, help='only load dataset and quit, set init_iter=1 and num_iter=0, used for dataset caching')
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(
            format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG)
    else:
        logging.basicConfig(
            format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)

    # Global names
    global hyperparameters, compute_gyopt_metric
    sys.path.insert(0, str(HYPERPARAMETERS_DIR))
    hyperparameters_def = importlib.import_module(args.hyperparameters)
    hyperparameters = hyperparameters_def.hyperparameters
    if args.gyopt_metric:
        logging.info(f"Overriding gyopt metric function from {args.hyperparameters} and using definition from {args.gyopt_metric}.")
        metric_def = importlib.import_module(args.gyopt_metric)
        compute_gyopt_metric = metric_def.compute_gyopt_metric
    else:
        compute_gyopt_metric = hyperparameters_def.compute_gyopt_metric

    # Update hyperparameters using command line parameters
    update_hyperparameters(hyperparameters, args)
    if args.only_load_dataset:
        logging.info("Only loading dataset. Overriding init_iter to 1 and num_iter to 0.")
        args.init_iter = 1
        args.num_iter = 0

    # System specific hyperparameters
    global system_params
    system_params = get_system_params(args)

    # Gyopt parameters
    init_iterations = args.init_iter
    num_iterations = args.num_iter

    # Create GyOpt search space and operation
    global operation
    search_space, operation = create_search_space_and_operations(
        hyperparameters)

    logging.info("Initializing hyperparameter optimizer...")
    gpyopt_bo = GPyOpt.methods.BayesianOptimization(f=train_call,
                                                    domain=search_space,
                                                    model_type='GP',
                                                    initial_design_numdata=init_iterations,
                                                    initial_design_type='random',
                                                    acquisition_type='EI',
                                                    normalize_Y=True,
                                                    exact_feval=False,
                                                    acquisition_optimizer_type='lbfgs',
                                                    model_update_interval=1,
                                                    evaluator_type='sequential',
                                                    batch_size=1,
                                                    num_cores=os.cpu_count(),
                                                    verbosity=True,
                                                    verbosity_model=False,
                                                    maximize=True,
                                                    de_duplication=True,
                                                    )

    # text file with parameter info, gp info, best results
    rf = LOGS_DIR / f"{hyperparameters['model_name']}-report.txt"
    # text file with header: Iteration Y var_1 var_2 etc.
    ef = LOGS_DIR / f"{hyperparameters['model_name']}-evaluation.txt"
    # text file with gp model info
    mf = LOGS_DIR / f"{hyperparameters['model_name']}-models.txt"

    logging.info(
        f"Running hyperparameter optimization using a maximum of {num_iterations} iterations after {init_iterations} initial iterations...")
    gpyopt_bo.run_optimization(max_iter=num_iterations,
                               report_file=rf,
                               evaluations_file=ef,
                               models_file=mf)

    logging.info(
        f"Hyperparameter optimization finished after {iter_num - 1} completed iterations. Logs saved.")
