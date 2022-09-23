#!/usr/bin/env python
# coding: utf-8


# Load libraries

import numpy as np
import pandas as pd
import primesieve as ps
import torch
import json
import argparse

import logging

from tqdm import tqdm

from dataloader import load_data
from train import create_model
from file_ops import (
    model_exist,
    get_current_model_name,
    get_random_model_number,
    get_model_numbers,
    get_best_model_num,
    delete_temp_dataset,
    read_csv_pgbar,
)
from global_vars import MODEL_DIR, DATA_DIR


def parse_command_line_arguments():
    parser = argparse.ArgumentParser(
        description='Make elliptic curve rank prediction from pretrained neural network model.')
    parser.add_argument('model_name', type=str,
                        help=f"pretrained model name (in '{MODEL_DIR}' dir)")
    parser.add_argument('--model_num', metavar='num', type=str, default=None,
                        help='pretrained model number (if missing use model having the best MCC) (default: None)')
    parser.add_argument('--input_conductor_fname', metavar='name', type=str, default="conductors.txt",
                        help='input file having curve conductors (one conductor per line) (default: conductors.txt')
    parser.add_argument('--input_aps_fname', metavar='name', type=str, default="aps.txt",
                        help='input file having curve sequences of ap-s (sequence for one curve per line); in the same order of curves as for conductors' +
                        " (default: aps.txt)")
    parser.add_argument('--predicted_ranks_fname', metavar='name', type=str, default='predicted_ranks.txt',
                        help='output file having predicted curve ranks; in the same order of curves as for conductors (default: predicted_ranks.txt)')
    parser.add_argument('--test_batch_size', metavar='N', type=int, default=128,
                        help='batch size for evaluation neural network (default: 128); use less or more depending on GPU RAM;' +
                        ' small batch size might reduce processing speed')
    parser.add_argument('--debug', action=argparse.BooleanOptionalAction, default=False,
                        help='print debug logs')
    parser.add_argument('--cuda_device_no', metavar='N', type=int, default=0,
                        help='cuda device number used for GPU computations; in single GPU system use default (default: 0)')
    parser.add_argument('--load_data_to_GPU', action=argparse.BooleanOptionalAction, default=False,
                        help='load dataset to GPU')
    parser.add_argument('--num_workers', metavar='N', type=int, default=4,
                        help='number of workers in case of using CPU dataloader (default: 4)')
    return parser.parse_args()


def set_hyperparameters_for_loading_evaluation_dataset(hyperparameters, dataset_name, test_batch_size):
    # Select dataset files
    hyperparameters['dataloader']['data_files'] = [dataset_name]
    # DO NOT CHANGE - faster loading - do not need auxillary info for rank prediction
    hyperparameters['dataloader']['load_reduced_metadata'] = True
    # DO NOT CHANGE - we do not want to cache temporary dataset
    hyperparameters['dataloader']['load_save_cache'] = False
    hyperparameters['dataloader']['force_dataset_suffix'] = hyperparameters['dataloader']['max_prime']

    # Use the whole dataset:
    hyperparameters['dataloader']['test_size'] = 1.0
    hyperparameters['dataloader']['min_max_test_conductor'] = None

    # Set additional dataset options - DO NOT CHANGE - would break the expected order od curves in results
    hyperparameters['dataloader']['purge_repetitive_curves'] = False
    hyperparameters['dataloader']['sort'] = None
    hyperparameters['dataloader']['min_max_rank'] = None
    hyperparameters['dataloader']['only_prime_conductors'] = False
    hyperparameters['dataloader']['only_not_prime_conductors'] = False

    # Override necessary hyperparameters for evaluation purpose - DO NOT CHANGE
    hyperparameters['dataloader']['min_max_conductor'] = None
    hyperparameters['dataloader']['curves_count'] = None
    hyperparameters['dataloader']['test_curves_count'] = None
    hyperparameters['dataloader']['class_identifier'] = None
    # regulates required size of GPU RAM
    hyperparameters['dataloader']['test_batch_size'] = test_batch_size


# Main block
if __name__ == '__main__':

    args = parse_command_line_arguments()

    # Apply parsed command line arguments
    if args.debug:
        logging.basicConfig(
            format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG)
    else:
        logging.basicConfig(
            format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)

    model_name = args.model_name

    if args.model_num is None:
        # Find best model number
        model_nums = get_model_numbers(model_name)
        if len(model_nums) == 0:
            raise RuntimeError(
                f"There is no single model having name {model_name}.")
        elif len(model_nums) == 1:
            model_num = model_nums[0]
            logging.info(
                f"There is only a single model number: {model_num} for model name: {model_name}.")
        else:
            model_num, MCC = get_best_model_num(model_name, model_nums)
            logging.info(
                f"Using the best possible model num: {model_num}, having MCC: {MCC} out of {len(model_nums)} possible models.")
    else:
        model_num = args.model_num
        assert model_exist(model_num, model_name)

    # Load hyperparameters of trained model from .json file
    current_model_name = get_current_model_name(model_num, model_name)
    with open(MODEL_DIR / f"{current_model_name}.hyperparameters.json", "r") as fp:
        hyperparameters = json.load(fp)
    logging.info(f"Loaded model hyperparameters: {hyperparameters}")
    assert hyperparameters['model_name'] == model_name
    assert hyperparameters['model_num'] == model_num
    max_prime = hyperparameters['dataloader']['max_prime']

    # CUDA device number used for GPU computation. In single GPU systems should be set to 0
    cuda_device = f"cuda:{args.cuda_device_no:d}"

    # Load conductors
    input_conductor_fname = args.input_conductor_fname
    logging.info(f"Loading curve conductors from {input_conductor_fname} ...")
    data_cond_pd = pd.read_csv(input_conductor_fname, names=['conductor'])
    conductors = [{'conductor': int(elem)}
                  for elem in list(data_cond_pd['conductor'])]

    # Load ap-s
    input_aps_fname = args.input_aps_fname
    logging.info(f"Loading curve ap-s from {input_aps_fname} ...")
    primes = list(ps.primes(max_prime))
    data_qexp = read_csv_pgbar(input_aps_fname, chunksize=100, skip_first_row=False, names=[
        f'p{prime}' for prime in primes]).to_numpy()
    assert len(conductors) == data_qexp.shape[0]

    # Creating temp dataset files
    dataset_name = f"temp-{get_random_model_number()}"
    logging.info(
        f"Creating temporary dataset files {dataset_name} in directory {DATA_DIR} ...")
    logging.info("  Creating temporary .json.cached ...")
    with open(DATA_DIR / f"{dataset_name}.json.cached", "w") as fp:
        json.dump(conductors, fp)
    logging.info("  Creating temporary .npy ...")
    np.save(DATA_DIR / f"{dataset_name}.{max_prime}.npy", data_qexp)

    # Loading dataset
    test_batch_size = args.test_batch_size
    set_hyperparameters_for_loading_evaluation_dataset(
        hyperparameters, dataset_name, test_batch_size)
    logging.info("Loading dataset...")
    (data,
     dataset,
     dataset_size,
     data_channels,
     data_length,
     count_classes,
     big_classes,
     conductors,
     curves,
     labels,
     test_idx) = load_data(device=cuda_device,
                           data_load_to_GPU=args.load_data_to_GPU,
                           num_workers=args.num_workers,
                           cache_file_name_prefix=hyperparameters['model_name'],
                           **hyperparameters['dataloader']
                           )
    logging.info(f"  Dataset size (number of curves): {dataset_size}")
    logging.info(f"  Data channels: {data_channels}")
    logging.info(
        f"  Length of each dataset element (number of primes used): {data_length}")

    # Creating model and loading trained parameters
    original_num_of_classes = len(hyperparameters['loss_func_weights'])
    logging.info(
        f"Number of different ranks used when training used model: {original_num_of_classes}")
    logging.info(f"  Assuming lowest rank is 0 and highest rank is {original_num_of_classes - 1} in the model training dataset." +
                 " If the lowest rank is greater than 0 and/or some ranks were missing, output predicted ranks should be manually adjusted accordingly.")
    model = create_model(data_length,
                         data_channels,
                         # we have to use dummy list of big_classes having the same number of classes as in original data
                         ["DUMMY"] * original_num_of_classes,
                         hyperparameters,
                         cuda_device
                         )
    logging.info("Loading pretrained model weights...")
    model.load_state_dict(torch.load(
        MODEL_DIR / f"{current_model_name}.pth", map_location=cuda_device)['model'])
    model.eval()

    # Rank prediction evaluation
    logging.info(
        f"Evaluating model on loaded dataset having {len(test_idx)} curves...")
    test_Dataloader = data[2]
    classes = []
    for inputs, _ in tqdm(test_Dataloader, total=np.ceil(len(test_idx)/test_batch_size)):
        outputs = model(inputs)
        outputs_classes = (torch.max(torch.exp(outputs), 1)
                           [1]).data.cpu().numpy()
        classes.append(outputs_classes)

    classes = list(np.concatenate(classes))
    assert len(conductors) == len(classes)
    logging.info(
        f"Smallest predicted rank is {min(classes)} and biggest is {max(classes)}")

    predicted_ranks_fname = args.predicted_ranks_fname
    logging.info(f"Writing predicted ranks in {predicted_ranks_fname} ...")
    with open(predicted_ranks_fname, 'w') as out_file:
        file_content = "\n".join([str(elem) for elem in classes])
        out_file.write(file_content)

    # delete temp dataset files
    logging.info("Deleting temporary dataset files...")
    delete_temp_dataset(dataset_name, max_prime)

    logging.info("Finished. Exiting...")
