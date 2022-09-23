# Dataloader definition - load_data function

#! /usr/bin/env python

import numpy as np
import torch
import logging
import json
import itertools
import random
import math
import hashlib
import sys

from torch.utils.data import Subset
from fastai.basics import DataLoader
from fastai.basics import DataLoaders

from sklearn.model_selection import train_test_split
import primesieve as ps

from tqdm import tqdm
from sympy import isprime
from pathlib import Path
from ast import literal_eval
from numba import jit
from glob import glob
from parse import parse

from global_vars import CACHE_DIR, DATA_DIR
from datasets import EllipticCurveDataset, NSumDataset
from Nagao_sums import compute_Nagao_sums

# Helper functions


def get_conductors(data_all):
    return [elem["conductor"] for elem in data_all]


def get_curves(data_all):
    # if 'ainvs' in data_all[0].keys():
    #     curves = [tuple(elem['ainvs']) for elem in data_all]
    # else:
    #     curves = [(elem['a1'], elem['a2'], elem['a3'], elem['a4'], elem['a6'])
    #               for elem in data_all]

    curves = []
    for elem in data_all:
        if "ainvs" in elem.keys():
            curves.append(tuple(elem["ainvs"]))
        else:
            curves.append((elem["a1"], elem["a2"], elem["a3"], elem["a4"], elem["a6"]))

    return curves


def compute_log10_conductors(conductors):
    """ Compute log10 of sequence of conductors

    Args:
        conductors (int or str): sequence of conductors

    Returns:
        list: sequence of float log10 conductors
    """
    return [math.log10(int(cond)) for cond in conductors]


def get_idxs(data_all, min_max_conductor):
    """Returns list of indices of curves having conductor in segment defined with min_max_conductor.

    Args:
        data_all (list): list of dict of curve metadata
        min_max_conductor (list): list of two elements - min and max conductors

    Returns:
        list: indices
    """
    assert min_max_conductor is not None
    assert isinstance(min_max_conductor, list)
    assert len(min_max_conductor) == 2
    min_conductor = min_max_conductor[0]
    max_conductor = min_max_conductor[1]
    # assert min_conductor <= max_conductor  # without this we can return empty list of indices
    if min_conductor > max_conductor:
        logging.warning(
            f"Given min_conductor: {min_conductor} is greater than max_conductor: {max_conductor}"
        )

    conductors = get_conductors(data_all)
    return [
        idx
        for idx, cond in enumerate(conductors)
        if ((min_conductor <= cond) and (cond <= max_conductor))
    ]


def get_datafile_name_suffix(max_prime, limits):
    """Return appropriate minimal datafile name suffix depending on max_prime.

    Args:
        max_prime (int): upper boundary on used primes
        limits (list): precomputed dataset file sizes

    Raises:
        ValueError: If the number of primes requested is grater than supplied in precomputed dataset.

    Returns:
        int: minimal datafile name suffix
    """
    for limit in limits:
        if ps.count_primes(max_prime) <= ps.count_primes(limit):
            return limit
    raise ValueError(
        f"Number of primes smaller or equal to {max_prime} is greater than the size of ap-s in dataset."
    )


def read_dataset(
    data_files,
    max_prime,
    skip_first_primes=0,
    load_reduced_metadata=False,
    limits=[10 ** 5],
):
    """Read dataset from disk.

    Args:
        data_files (list): list of dataset file paths relative to DATA_DIR and without extension
        max_prime (int): upper boundary on used primes
        skip_first_primes (int): number of first primes to skip
        load_reduced_metadata (bool): faster loading reduced metadata in .json.cached files
        limits (list): precomputed dataset file sizes

    Returns:
        list, np.array: list of dict of curve metadata, array in each row having ap-s for a single curve
    """
    # Compute number of primes
    num_of_primes = ps.count_primes(max_prime)
    assert (0 <= skip_first_primes) and (skip_first_primes < num_of_primes)

    # Load data from disk and trim unused primes
    data_all = []
    for file in tqdm(data_files, leave=False):
        if load_reduced_metadata:
            fpath = DATA_DIR / f"{file}.json.cached"
        else:
            fpath = DATA_DIR / f"{file}.json"
        with open(fpath, "r") as fp:
            data_all.append(json.load(fp))
    data_all = list(itertools.chain(*data_all))

    data_qexp = []
    for file in tqdm(data_files, leave=False):
        loaded_data = np.load(
            DATA_DIR / f"{file}.{get_datafile_name_suffix(max_prime, limits)}.npy"
        )
        assert num_of_primes <= loaded_data.shape[1]
        data_qexp.append(loaded_data[:, skip_first_primes:num_of_primes])

    if len(data_qexp) == 1:
        data_qexp = data_qexp[0]
    else:
        logging.info("  Concatenating dataset...")
        data_qexp = np.concatenate(data_qexp)

    return data_all, data_qexp


def read_bad_primes(data_files, max_prime, skip_first_primes=0, limits=[10 ** 5]):
    # Compute number of primes
    num_of_primes = ps.count_primes(max_prime)
    assert (0 <= skip_first_primes) and (skip_first_primes < num_of_primes)

    bad_primes = []
    for file in tqdm(data_files, leave=False):
        if Path(
            DATA_DIR
            / f"{file}.{get_datafile_name_suffix(max_prime, limits)}.bad_primes.npy"
        ).is_file():
            loaded_data = np.load(
                DATA_DIR
                / f"{file}.{get_datafile_name_suffix(max_prime, limits)}.bad_primes.npy"
            )
        elif Path(
            DATA_DIR
            / f"{file}.{get_datafile_name_suffix(max_prime, limits)}.bad_primes.npz"
        ).is_file():
            loaded_data = np.load(
                DATA_DIR
                / f"{file}.{get_datafile_name_suffix(max_prime, limits)}.bad_primes.npz"
            )["bad_primes"]
        else:
            raise RuntimeError(
                f"Input file {file}.{get_datafile_name_suffix(max_prime, limits)}.bad_primes.* is missing"
            )
        assert num_of_primes <= loaded_data.shape[1]
        bad_primes.append(loaded_data[:, skip_first_primes:num_of_primes])

    if len(bad_primes) == 1:
        bad_primes = bad_primes[0]
    else:
        logging.info("  Concatenating dataset...")
        bad_primes = np.concatenate(bad_primes)

    return bad_primes


def purge_duplicate_curves(data_all, data_qexp):
    beg_len = len(data_all)
    logging.info("  Extract curve data from metadata...")
    curves = get_curves(data_all)
    logging.info("  Make unique curve data...")
    curves_unique = set(curves)
    logging.info("  Create unique curves indices...")
    idxs = []
    for idx, curve in enumerate(curves):
        if curve in curves_unique:
            curves_unique.remove(curve)
            idxs.append(idx)
    assert len(curves_unique) == 0
    logging.info("  Select meta data by unique idxs...")
    data_all = [data_all[i] for i in idxs]
    logging.info("  Select qexp by unique idxs...")
    data_qexp = data_qexp[idxs]
    logging.info(
        f"  Finished purging: Purged {beg_len - len(data_all)} out of {beg_len} curves."
    )

    return data_all, data_qexp


def sort_curves(data_all, data_qexp, sort):
    if sort == "conductor":
        logging.info("Sorting dataset by conductor...")
        logging.info("  Creating list of integer conductors...")
        conductors = get_conductors(data_all)
        logging.info("  Sort idxs of conductors list by conductor...")
        sort_idxs = sorted(range(len(conductors)), key=conductors.__getitem__)
        logging.info("  Reordering meta data by sorted idxs...")
        data_all = [data_all[i] for i in sort_idxs]
        logging.info("  Reordering qexp by sorted idxs...")
        data_qexp = data_qexp[sort_idxs]
        logging.info("  Finished sorting.")
    else:
        raise NotImplementedError(f"Sort by {sort}.")

    return data_all, data_qexp


def filter_by_rank(data_all, data_qexp, min_max_rank):
    idxs = [
        idx
        for idx, elem in enumerate(tqdm(data_all, leave=False))
        if (
            (min_max_rank[0] <= int(elem["rank"]))
            and (int(elem["rank"]) <= min_max_rank[1])
        )
    ]
    data_qexp = data_qexp[idxs]
    idxs = set(idxs)
    data_all = [
        elem for idx, elem in enumerate(tqdm(data_all, leave=False)) if idx in idxs
    ]

    return data_all, data_qexp


def filter_prime_conductors(data_all, data_qexp):
    idxs = [
        idx
        for idx, elem in enumerate(tqdm(data_all, leave=False))
        if isprime(int(elem["conductor"]))
    ]
    data_qexp = data_qexp[idxs]
    idxs = set(idxs)
    data_all = [
        elem for idx, elem in enumerate(tqdm(data_all, leave=False)) if idx in idxs
    ]

    return data_all, data_qexp


def filter_nonprime_conductors(data_all, data_qexp):
    idxs = [
        idx
        for idx, elem in enumerate(tqdm(data_all, leave=False))
        if not isprime(int(elem["conductor"]))
    ]
    data_qexp = data_qexp[idxs]
    idxs = set(idxs)
    data_all = [
        elem for idx, elem in enumerate(tqdm(data_all, leave=False)) if idx in idxs
    ]

    return data_all, data_qexp


def train_valid_test_split(
    data_all,
    data_qexp,
    valid_size,
    test_size,
    min_max_conductor,
    min_max_test_conductor,
    test_RAND_SEED,
    valid_RAND_SEED,
    shuffle_RAND_SEED,
    curves_count,
    test_curves_count,
    filter_test_set,
    filter_test_set_range,
):
    # Trim data if min and max conductors are supplied (possible separate min and max conductors supplied for test set)
    assert (0.0 < valid_size) and (valid_size < 1.0)
    logging.info(f"Whole dataset has {len(data_all)} curves.")
    random.seed(shuffle_RAND_SEED)
    if min_max_conductor is None:
        assert (curves_count is None) and (test_curves_count is None)
        # no min and max conductors supplied - use whole dataset
        logging.info("Using the whole dataset.")
        all_idx = list(range(len(data_all)))
        if min_max_test_conductor is None:
            # no min and max test conductors supplied - use randomly sampled test data in whole dataset
            assert (0.0 < test_size) and (test_size <= 1.0)
            if test_size < 1.0:
                logging.info(
                    f"  Test size is {test_size*100:.2f} percent and is randomly sampled."
                )
                train_valid_idx, test_idx = train_test_split(
                    all_idx, test_size=test_size, random_state=test_RAND_SEED
                )
                train_idx, valid_idx = train_test_split(
                    train_valid_idx, test_size=valid_size, random_state=valid_RAND_SEED
                )
            else:  # creating evaluation test dataloader
                logging.info(
                    f"  Creating test dataloader for evaluation. Test size is {test_size*100:.2f} percent."
                )
                train_valid_idx = []
                test_idx = all_idx
                train_idx = []
                valid_idx = []
        else:
            # test is defined by min and max conductors and the rest of the dataset is used for train/valid
            assert test_size is None  # assume no given test_size percentage
            test_idx = get_idxs(data_all, min_max_test_conductor)
            assert len(test_idx) > 0  # assume test set is non-empty
            logging.info(
                f"  Test size is {len(test_idx)} curves defined by min/max conductors { min_max_test_conductor}."
            )
            train_valid_idx = list(set(all_idx) - set(test_idx))
            train_idx, valid_idx = train_test_split(
                train_valid_idx, test_size=valid_size, random_state=valid_RAND_SEED
            )
    else:
        if min_max_test_conductor is None:
            assert test_curves_count is None
            # use only data having a range of conductors, test data is randomly sampled
            all_idx = get_idxs(data_all, min_max_conductor)
            if curves_count is not None:
                assert curves_count <= len(all_idx)
                random.shuffle(all_idx)
                all_idx = all_idx[:curves_count]
                logging.info(
                    f"Using randomly sampled {len(all_idx)} curves from min/max range of conductors {min_max_conductor}."
                )
            else:
                logging.info(
                    f"Using {len(all_idx)} curves defined by min/max range of conductors {min_max_conductor}."
                )
            data_all = [data_all[i] for i in tqdm(all_idx, leave=False)]
            data_qexp = data_qexp[all_idx]

            all_idx = list(range(len(data_all)))  # reindex data
            assert (0.0 < test_size) and (test_size <= 1.0)
            if test_size < 1.0:
                logging.info(
                    f"  Test size is {test_size*100:.2f} percent and is randomly sampled."
                )
                train_valid_idx, test_idx = train_test_split(
                    all_idx, test_size=test_size, random_state=test_RAND_SEED
                )
            else:  # creating evaluation test dataloader
                logging.info(
                    f"  Creating test dataloader for evaluation. Test size is {test_size*100:.2f} percent."
                )
                train_valid_idx = []
                test_idx = all_idx

            # additionally filter test set
            if filter_test_set is not None:
                assert filter_test_set_range is not None
                filtered_test_idx = get_idxs(data_all, filter_test_set_range)
                test_idx_intersection = list(set(test_idx) & set(filtered_test_idx))
                test_idx_difference = list(set(test_idx) - set(filtered_test_idx))
                if filter_test_set == "inclusive":
                    logging.info(
                        f"  Test set is restricted to inside of range {filter_test_set_range}."
                    )
                    train_valid_idx = train_valid_idx + test_idx_difference
                    test_idx = test_idx_intersection
                elif filter_test_set == "exclusive":
                    logging.info(
                        f"  Test set is restricted to outside of range {filter_test_set_range}."
                    )
                    train_valid_idx = train_valid_idx + test_idx_intersection
                    test_idx = test_idx_difference
                else:
                    raise ValueError(f"Unknown option: {filter_test_set=}")

            if (test_size < 1.0) or len(train_valid_idx) > 0:
                train_idx, valid_idx = train_test_split(
                    train_valid_idx, test_size=valid_size, random_state=valid_RAND_SEED
                )
            else:
                train_idx = []
                valid_idx = []
        else:
            # use separate ranges of conductors data for train/valid and test - compute idxs
            assert test_size is None  # assume no given test_size percentage
            train_valid_idx = get_idxs(data_all, min_max_conductor)
            test_idx = get_idxs(data_all, min_max_test_conductor)
            # assume train/valid is disjoint from test
            assert set(train_valid_idx).isdisjoint(set(test_idx))

            # additionally trim idxs if desired curve counts are supplied
            if curves_count is not None:
                assert curves_count <= len(train_valid_idx)
                random.shuffle(train_valid_idx)
                train_valid_idx = train_valid_idx[:curves_count]
                logging.info(
                    f"Using randomly sampled {len(train_valid_idx)} curves for train/valid from min/max range of conductors {min_max_conductor}."
                )
            else:
                logging.info(
                    f"Using {len(train_valid_idx)} curves for train/valid defined by min/max range of conductors {min_max_conductor}."
                )
            if test_curves_count is not None:
                assert test_curves_count <= len(test_idx)
                random.shuffle(test_idx)
                test_idx = test_idx[:test_curves_count]
                logging.info(
                    f"  Test is randomly sampled {len(test_idx)} curves from min/max range of conductors {min_max_test_conductor}."
                )
            else:
                logging.info(
                    f"  Test is {len(test_idx)} curves defined by min/max range of conductors {min_max_test_conductor}."
                )

            # trim dataset according to idxs
            train_valid_data_all = [
                data_all[i] for i in tqdm(train_valid_idx, leave=False)
            ]
            train_valid_data_qexp = data_qexp[train_valid_idx]
            test_data_all = [data_all[i] for i in tqdm(test_idx, leave=False)]
            test_data_qexp = data_qexp[test_idx]

            # reindex trimmed datasets
            train_valid_idx = list(range(len(train_valid_data_all)))
            test_idx = list(
                range(
                    len(train_valid_data_all),
                    len(train_valid_data_all) + len(test_data_all),
                )
            )
            data_all = train_valid_data_all + test_data_all
            data_qexp = np.concatenate([train_valid_data_qexp, test_data_qexp])

            train_idx, valid_idx = train_test_split(
                train_valid_idx, test_size=valid_size, random_state=valid_RAND_SEED
            )

    # sort all idxs - for debug log of first/last idx data
    train_idx.sort()
    valid_idx.sort()
    test_idx.sort()

    logging.info(
        f"Train dataset size is {len(train_idx)}, validation dataset size is {len(valid_idx)}, and test dataset size is {len(test_idx)} curves."
    )
    if len(train_idx) > 0:
        logging.debug(f"First train idx data: {data_all[train_idx[0]]}")
        logging.debug(f"Last train idx data: {data_all[train_idx[-1]]}")
    if len(valid_idx) > 0:
        logging.debug(f"First valid idx data: {data_all[valid_idx[0]]}")
        logging.debug(f"Last valid idx data: {data_all[valid_idx[-1]]}")
    if len(test_idx) > 0:
        logging.debug(f"First test idx data: {data_all[test_idx[0]]}")
        logging.debug(f"Last test idx data: {data_all[test_idx[-1]]}")

    return data_all, data_qexp, train_idx, valid_idx, test_idx


def create_classes_and_labels(
    data_all, class_identifier, min_class_size, remap_classes, train_idx, valid_idx
):

    # For classes not present in keys of remap_classes dict, adds identity map of class to padded str of class
    def add_new_classes(remap_classes, classes, str_PADDING=2):
        for c in classes:
            if c not in remap_classes.keys():
                remap_classes[c] = f"{c:{str_PADDING}}"
        return remap_classes

    # Making labels tensor. Taking into account min_class_size
    logging.info("Computing all classes...")
    # default index of the small classes to ignore, pytorch CrossEntropyLoss() default
    CLASS_IGNORE_INDEX = -100

    # all classes names are strings (converted to strings)
    # compute classes found only in train/valid dataset
    assert class_identifier in data_all[0].keys()
    classes = list(
        set(data_all[idx][class_identifier] for idx in (train_idx + valid_idx))
    )
    logging.debug(f"Classes in train/valid data of original type: {classes}")

    if remap_classes is None:
        logging.debug(
            "Remapping of classes not provided. Using empty dict for original remap."
        )
        remap_classes = {}
    else:
        assert isinstance(remap_classes, dict)
        assert set(map(type, remap_classes)) == {type(data_all[0][class_identifier])}
        assert set(map(type, remap_classes.values())) == {str}
    logging.debug(f"Original remap of classes: {remap_classes}")
    logging.debug(
        "Expanding remap on possible additional classes from train/valid set using identity map..."
    )
    remap_classes = add_new_classes(remap_classes, classes)
    logging.debug(f"Expanded remap of classes: {remap_classes}")

    classes = list(set([remap_classes[c] for c in classes]))
    classes.sort()
    logging.debug(f"Classes in train/valid data converted to str: {classes}")

    # compute all classes from whole dataset
    all_classes = list(set(elem[class_identifier] for elem in data_all))
    logging.debug(
        "Additionally expanding remap on possible additional classes from the whole dataset using identity map..."
    )
    remap_classes = add_new_classes(remap_classes, all_classes)
    logging.debug(f"Additionally expanded remap of classes: {remap_classes}")

    all_classes = list(set([remap_classes[c] for c in all_classes]))
    all_classes.sort()
    logging.debug(f"Classes in all data: {all_classes}")

    # count number of elements of each class from train/valid dataset
    logging.info("Counting all classes in train/valid data...")
    count_classes = {val: 0 for val in classes}
    for idx in train_idx + valid_idx:
        count_classes[remap_classes[data_all[idx][class_identifier]]] += 1
    logging.info(f"Sizes of classes in train/valid data: {count_classes}")

    # computing all classes found only in train/valid dataset that are bigger than a given threshold
    big_classes = [val for val in classes if count_classes[val] >= min_class_size]
    logging.debug(
        f"Classes in train/valid data bigger than {min_class_size} size: {big_classes}"
    )

    # ignore all classes not present in train/valid data and not big enough
    class_to_idx = {val: idx for idx, val in enumerate(big_classes)}
    for val in all_classes:
        if val not in big_classes:
            class_to_idx[val] = CLASS_IGNORE_INDEX  # ignore index
    logging.info(f"All classes to index map including ignore classes: {class_to_idx}")

    logging.info("Computing labels...")
    labels = [class_to_idx[remap_classes[elem[class_identifier]]] for elem in data_all]

    return labels, count_classes, big_classes


def load_S4_S5_sums_to_dict(fname):
    logging.info(f"Loading S4 and S5 Nagao sums from {fname} ...")
    fpath = DATA_DIR / fname
    with open(fpath, "r") as fp:
        data = json.load(fp)
    S4_S5_sums = {}
    for entry in tqdm(data, leave=False):
        S4_S5_sums[
            (entry["a1"], entry["a2"], entry["a3"], entry["a4"], entry["a6"])
        ] = [entry["N4"], entry["N5"]]

    return S4_S5_sums


def load_Bober_sums_to_dict(fname):

    # @jit(nopython=True)  # Numba is not working with tqdm and literal_eval
    def _process_lines(lines):
        Bsums = {}
        for line in tqdm(lines, leave=False):
            elems = line.split()
            assert len(elems) == 2
            curve = literal_eval(elems[0])
            Bsum = literal_eval(elems[1])
            Bsums[tuple(curve)] = Bsum
        return Bsums

    logging.info(f"Loading Bober sums from {fname} ...")
    fpath = DATA_DIR / f"{fname}"
    with open(fpath, "r") as fp:
        lines = fp.readlines()

    return _process_lines(lines)


def count_labeled_classes(labels, train_idx, valid_idx, test_idx):
    labels_set = list(set(labels))
    labels_set.sort()
    count_labels_train_valid = {}
    count_labels_test = {}
    for l in labels_set:
        count_labels_train_valid[l] = 0
        count_labels_test[l] = 0
    for idx in train_idx:
        count_labels_train_valid[labels[idx]] += 1
    for idx in valid_idx:
        count_labels_train_valid[labels[idx]] += 1
    for idx in test_idx:
        count_labels_test[labels[idx]] += 1

    return count_labels_train_valid, count_labels_test


# Load data


def load_data(
    data_files,  # list of dataset file paths relative to DATA_DIR without extensions
    # type of dataset to produce ("EllipticCurveDataset" or "NSumDataset")
    dataset="EllipticCurveDataset",
    # if supplied, force loading of dataset from specific file name suffix (max prime)
    force_dataset_suffix=None,
    # faster metadata load time - load .json.cached files having only rank and conductor info
    load_reduced_metadata=False,
    # save intermediate created dataset info to cache files for faster loading on subsequent calls
    load_save_cache=False,
    # if supplied, use prefix for naming cache file for later easier identification
    cache_file_name_prefix=None,
    # if supplied, writes classes of curves in test set in output txt file (one curve per line)
    dump_test_set_classes=False,
    max_prime=100000,  # upper bound on primes used
    skip_first_primes=0,  # number of first primes to skip
    #############################
    valid_size=0.2,  # relative percentage of data after removing test data
    test_size=0.2,  # percentage of whole data
    min_max_conductor=None,  # if supplied, interval of min/max conductors
    # if supplied, interval of min/max conductors for test set
    min_max_test_conductor=None,
    # if supplied, force specific size of dataset, to be used only with specified min_max_conductor
    curves_count=None,
    # if supplied, force specific size of test dataset, to be used only with specified min_max_test_conductor
    test_curves_count=None,
    ####
    # Filtering test set options - to be used only if min_max_conductor is given and min_max_test_conductor is None
    filter_test_set=None,  # if supplied, filter test set by a given criteria using list in filter_test_set_range, criteria: "inclusive", "exclusive"
    # if supplied (in case filter_test_set is not None), use this range to filter test set
    filter_test_set_range=None,
    #############################
    # remove duplicate curves identified by the same values of Weierstrass equation coefficients
    purge_repetitive_curves=True,
    sort=None,  # "conductor"
    min_max_rank=None,  # if supplied, use only curves having rank in this interval
    only_prime_conductors=False,  # use only curves having prime conductors
    only_not_prime_conductors=False,  # use only curves having non-prime conductors
    # target labels for training classifier, if None use dummy zero labels for all curves - only use for evaluating models!
    class_identifier="rank",
    min_class_size=50,  # minimal size of class for training classifier - ignore smaller classes
    remap_classes=None,  # if supplied, dict that maps real classes to meta classes, allows grouping of real classes, default is identity map
    # if supplied will override max_log10conductors computed from dataset - use only for evaluating models!
    max_log10conductors=None,
    batch_size=32,  # for valid/train
    test_batch_size=1024,  # for test
    test_RAND_SEED=42,
    valid_RAND_SEED=42,
    shuffle_RAND_SEED=42,
    #####################################
    # Dataset specific hyperparameters
    # file path to precomputed S4 and S5 nagao sums - used only for NSumDataset
    load_S4_S5_sums=False,
    load_Bober_sums=False,  # file path to precomputed Bober sums - used only for NSumDataset
    # divide aps by sqrt(p) - used only for EllipticCurveDataset
    normalize_aps=True,
    use_p=False,  # use additional data about normalized p - used only for EllipticCurveDataset
    # use additional data about normalized sqrt(p) - used only for EllipticCurveDataset
    use_sqrt_p=False,
    # use additional data about normalized log(p) - used only for EllipticCurveDataset
    use_log_p=False,
    # use additional data about normalized log10 of conductors - used for both datasets
    use_conductors=True,
    # use mask to select variants of Nagao-Mestre sums to use - used only for NSumDataset
    N_masks=[1, 1, 1, 1],
    #####################################
    # System parameters
    device="cuda",  # in multiple GPU systems change for using different GPU-s
    data_load_to_GPU=False,  # use to speed up training if data fits GPU RAM
    num_workers=4,  # num of dataloader threads - increase of GPU seems underutilized
):
    # Create dict of calling arguments which influence creation of cache files
    func_args = locals()  # dict of function arguments
    del func_args["cache_file_name_prefix"]
    del func_args["dump_test_set_classes"]
    del func_args["batch_size"]
    del func_args["test_batch_size"]
    del func_args["normalize_aps"]
    del func_args["use_p"]
    del func_args["use_sqrt_p"]
    del func_args["use_log_p"]
    del func_args["use_conductors"]
    del func_args["N_masks"]
    del func_args["device"]
    del func_args["data_load_to_GPU"]
    del func_args["num_workers"]

    # Load/save cache - preliminaries
    save_cache = False
    load_cache = False
    if load_save_cache:
        logging.info(f"Computing unique function call hash for load/save operations...")
        hash = hashlib.sha256()
        hash.update(json.dumps(func_args, sort_keys=True).encode("utf-8"))
        generated_hash = hash.hexdigest()
        if cache_file_name_prefix is not None:
            cache_file_name = f"{cache_file_name_prefix}.{generated_hash}"
        else:
            fnames = glob(str(CACHE_DIR / f"cache.*.{generated_hash}.json"))
            if len(fnames) > 0:
                logging.info(
                    f"  There is(are) {len(fnames)} possible cache file(s) present:"
                )
                for idx, fname in enumerate(fnames):
                    logging.info(f"    {idx+1} - {fname}")
                logging.info(f"  Using the first one {fnames[0]}")
                res = parse(
                    f"{{}}cache.{{cache_file_name_prefix}}.{generated_hash}.json",
                    fnames[0],
                )
                assert res is not None
                cache_file_name_prefix = res.named["cache_file_name_prefix"]
                logging.info(f"  Using cache file name prefix {cache_file_name_prefix}")
                cache_file_name = f"{cache_file_name_prefix}.{generated_hash}"
            else:
                cache_file_name = generated_hash

        if Path(CACHE_DIR / f"cache.{cache_file_name}.json").is_file():
            load_cache = True
        else:
            save_cache = True

    # Whole block is executed only if we are not going to load all data from cache
    if not load_cache:
        # Input takes either data file names or list of file names
        # Data is loaded in order which appears in list of file names
        if not isinstance(data_files, list):
            data_files = [data_files]

        # Read and trim dataset
        logging.info("Reading dataset from disk...")
        if force_dataset_suffix:
            assert isinstance(force_dataset_suffix, int)
            logging.info(
                f"  Force loading dataset from files having suffix (max prime) {force_dataset_suffix}"
            )
            data_all, data_qexp = read_dataset(
                data_files,
                max_prime,
                skip_first_primes,
                load_reduced_metadata,
                limits=[force_dataset_suffix],
            )
        else:
            data_all, data_qexp = read_dataset(
                data_files, max_prime, skip_first_primes, load_reduced_metadata
            )
        if dataset == "NSumDataset":
            logging.info("Reading divisions of conductors by primes from disk...")
            if force_dataset_suffix:
                assert isinstance(force_dataset_suffix, int)
                logging.info(
                    f"  Force loading bad primes from files having suffix (max prime) {force_dataset_suffix}"
                )
                bad_primes = read_bad_primes(
                    data_files,
                    max_prime,
                    skip_first_primes,
                    limits=[force_dataset_suffix],
                )
            else:
                bad_primes = read_bad_primes(data_files, max_prime, skip_first_primes)
            bad_primes = bad_primes.astype(np.int16)
            bad_primes = np.expand_dims(bad_primes, 1)
            data_qexp = np.expand_dims(data_qexp, 1)
            # encode ap-s and bad_primes in single tensor, adding intermediate dimension
            logging.info("  Concatenating input data...")
            data_qexp = np.concatenate([data_qexp, bad_primes], 1)

        # Optionally purge repetitive curves
        if purge_repetitive_curves:
            if not load_reduced_metadata:
                logging.info("Purging repetitive curves...")
                data_all, data_qexp = purge_duplicate_curves(data_all, data_qexp)
            else:
                raise ValueError(
                    "Can not purge duplicate curves. Weierstrass coefficients not present in meta-data. Turn off fast meta-data loading."
                )

        # Optionally sort dataset
        if sort is not None:
            data_all, data_qexp = sort_curves(data_all, data_qexp, sort)

        # Optionally filter curves by rank
        if min_max_rank is not None:
            assert isinstance(min_max_rank, list)
            assert len(min_max_rank) == 2
            logging.info(f"Use only curves having min/max rank: {min_max_rank}")
            data_all, data_qexp = filter_by_rank(data_all, data_qexp, min_max_rank)

        # Optionally select only dataset of curves having prime conductors
        if only_prime_conductors:
            logging.info(
                "Creating dataset using subset of curves having only prime conductors..."
            )
            data_all, data_qexp = filter_prime_conductors(data_all, data_qexp)

        # Optionally select only dataset of curves having non-prime conductors
        if only_not_prime_conductors:
            logging.info(
                "Creating dataset using subset of curves having only non-prime conductors..."
            )
            data_all, data_qexp = filter_nonprime_conductors(data_all, data_qexp)

        # Check that parameters are valid for creating dataset for either model evaluation or training
        if (
            class_identifier is None
        ):  # creating dataset for evaluation already trained model
            if test_size < 1.0:
                logging.info(
                    "Using only part of loaded dataset for test set."
                    + f" Test size is {test_size} and random seed for test selection is {test_RAND_SEED}."
                )
            assert (min_max_test_conductor is None) or (
                (test_size is None) and (min_max_test_conductor is not None)
            )
            assert max_log10conductors is not None
        else:  # creating dataset for training new model
            assert max_log10conductors is None

        # Split dataset in to train/valid/test
        data_all, data_qexp, train_idx, valid_idx, test_idx = train_valid_test_split(
            data_all,
            data_qexp,
            valid_size,
            test_size,
            min_max_conductor,
            min_max_test_conductor,
            test_RAND_SEED,
            valid_RAND_SEED,
            shuffle_RAND_SEED,
            curves_count,
            test_curves_count,
            filter_test_set,
            filter_test_set_range,
        )

        # Determine classes and labels
        if class_identifier is not None:  # creating dataset for training new model
            labels, count_classes, big_classes = create_classes_and_labels(
                data_all,
                class_identifier,
                min_class_size,
                remap_classes,
                train_idx,
                valid_idx,
            )
        else:  # creating dataset for evaluation - labels are not known
            logging.info(
                "Preparing dataset strictly for evaluation use. Labels assumed to be unknown. Using dummy labels."
            )
            labels = [0] * len(data_all)
            count_classes = {"DUMMY": len(labels)}
            big_classes = ["DUMMY"]

        # Making conductors list
        logging.info("Creating conductors list...")
        conductors = get_conductors(data_all)
        logging.info(
            f"  Min conductor is {min(conductors)}, max conductor is {max(conductors)}"
        )

        # Making log10 conductors list
        logging.info("Creating log10 conductors list...")
        log10conductors = compute_log10_conductors(conductors)
        log10conductors_hist = np.histogram(
            log10conductors, bins=range(int(np.ceil(max(log10conductors))) + 1)
        )
        logging.info(f"  Histogram of log10 conductors: {log10conductors_hist}")

        # Making curves coefficients list
        if not load_reduced_metadata:
            logging.info("Creating curves list...")
            curves = get_curves(data_all)
        else:
            # create dummy curves list
            curves = [(0, 0, 0, 0, 0)] * len(data_all)

        # Making data tensors
        logging.info("Creating data tensor...")
        data_t = torch.from_numpy(data_qexp)
        logging.info(f"  Data tensor is using type {data_t.dtype}")

        if dataset == "NSumDataset":
            logging.info("Compute Nagao-Mestre sums...")
            assert len(data_t.shape) == 3
            # ap-s and bad_primes are together in data_t tensor in type torch.int16

            data_t = compute_Nagao_sums(data_t[:, 0, :], max_prime, data_t[:, 1, :])

            # Possibly loading S4 and S5 nagao sums from file
            if load_S4_S5_sums:
                assert not load_reduced_metadata  # assure curves list is nontrivial
                S4_S5_sums_dict = load_S4_S5_sums_to_dict(load_S4_S5_sums)
                S4_S5_sums = [
                    S4_S5_sums_dict[curve] for curve in tqdm(curves, leave=False)
                ]
                assert data_t.shape[0] == len(S4_S5_sums)
                S4_S5_sums_t = torch.tensor(S4_S5_sums, dtype=data_t.dtype)
                data_t = torch.cat([data_t, S4_S5_sums_t], -1)

            # Possibly loading Bober sums from file
            if load_Bober_sums:
                assert not load_reduced_metadata  # assure curves list is nontrivial
                Bober_sums_dict = load_Bober_sums_to_dict(load_Bober_sums)
                Bober_sums = [
                    Bober_sums_dict[curve] for curve in tqdm(curves, leave=False)
                ]
                assert data_t.shape[0] == len(Bober_sums)
                Bober_sums_t = torch.tensor(Bober_sums, dtype=data_t.dtype).unsqueeze(
                    -1
                )
                data_t = torch.cat([data_t, Bober_sums_t], -1)

    ####################################################################################

    # Load/save cache
    if save_cache:
        logging.info(f"Saving all data to cache: {cache_file_name} ...")

        logging.info(f"  Saving to npy using type {data_t.numpy().dtype} ...")
        np.save(CACHE_DIR / f"cache.{cache_file_name}", data_t.numpy())

        logging.info(f"  Saving to json...")
        saved_data = {
            "train_idx": train_idx,
            "valid_idx": valid_idx,
            "test_idx": test_idx,
            "count_classes": count_classes,
            "big_classes": big_classes,
            "conductors": conductors,
            "log10conductors": log10conductors,
            "curves": curves,
            "labels": labels,
        }
        with open(CACHE_DIR / f"cache.{cache_file_name}.json", "w") as fp:
            json.dump(saved_data, fp)

    if load_cache:
        logging.info(f"Loading all data from cache: {cache_file_name} ...")

        logging.info(f"  loading from npy...")
        data_t = torch.from_numpy(np.load(CACHE_DIR / f"cache.{cache_file_name}.npy"))

        logging.info("  Loading from json...")
        with open(CACHE_DIR / f"cache.{cache_file_name}.json", "r") as fp:
            loaded_data = json.load(fp)
        train_idx = loaded_data["train_idx"]
        valid_idx = loaded_data["valid_idx"]
        test_idx = loaded_data["test_idx"]
        count_classes = loaded_data["count_classes"]
        big_classes = loaded_data["big_classes"]
        conductors = loaded_data["conductors"]
        log10conductors = loaded_data["log10conductors"]
        curves = loaded_data["curves"]
        labels = loaded_data["labels"]

    logging.info(
        f"Data tensor is using {(sys.getsizeof(data_t.storage()) / 1024**2):,.2f} MB of memory"
    )

    # Count the number of elements per label/class in both train/valid and test datasets
    logging.info("Counting labeled classes...")
    count_labels_train_valid, count_labels_test = count_labeled_classes(
        labels, train_idx, valid_idx, test_idx
    )
    logging.info(
        f"  Sizes of labeled classes in train/valid data: {count_labels_train_valid}"
    )
    logging.info(f"  Sizes of labeled classes in test data: {count_labels_test}")
    
    # Possibly dump labels (classes) of test set in file
    if dump_test_set_classes:
        logging.info("Dumping test set labels (classes) in true_test_classes.txt ...")
        labels_test = [labels[idx] for idx in tqdm(test_idx, leave=False)]
        with open('true_test_classes.txt', 'w') as out_file:
            file_content = "\n".join([str(l) for l in labels_test])
            out_file.write(file_content)        

    # Generate primes tensor
    # primes must fit in float32 to assure correct representation of primes in torch float32
    logging.info("Creating primes tensor...")
    assert max_prime <= 10 ** 9
    primes_t = torch.tensor(
        list(ps.primes(max_prime))[skip_first_primes:], dtype=torch.float
    )
    logging.info(
        f"  Number of primes used: {len(primes_t)}, from the first prime {int(primes_t[0])} to the last {int(primes_t[-1])}"
    )

    # Making labels tensors
    logging.info("Creating labels tensor...")
    labels_t = torch.from_numpy(np.array(labels)).long()

    # Get dataset size
    dataset_size = data_t.shape[0]
    assert (
        (dataset_size == labels_t.shape[0])
        and (dataset_size == len(conductors))
        and (dataset_size == len(curves))
    )

    # Possibly load all dataset to GPU - produces training speedup - all data must fit to GPU memory
    if data_load_to_GPU:
        logging.info("Loading data to GPU...")
        data_t = data_t.to(device)
        labels_t = labels_t.to(device)
        primes_t = primes_t.to(device)
        num_workers = 0
        # pin_memory = False
    else:
        logging.info("Dataset is in CPU RAM.")

    # Creating pytorch Datasets and computing data_length
    if dataset == "EllipticCurveDataset":
        logging.info("Creating elliptic curves dataset...")
        all_ds = EllipticCurveDataset(
            data_t,
            labels_t,
            primes_t,
            conductors,
            log10conductors,
            curves,
            normalize_aps=normalize_aps,
            use_p=use_p,
            use_sqrt_p=use_sqrt_p,
            use_log_p=use_log_p,
            use_conductors=use_conductors,
        )
        data_length = primes_t.shape[0]
        assert data_length == data_t.shape[-1]
    elif dataset == "NSumDataset":
        logging.info("Creating Nagao sums dataset...")
        all_ds = NSumDataset(
            data_t,
            labels_t,
            conductors,
            log10conductors,
            curves,
            N_masks=N_masks,
            use_conductors=use_conductors,
        )
        data_length = None  # not applicable for Nagao sums dataset
    else:
        raise ValueError(f"Unsupported dataset type: {dataset}")

    # Possibly patch dataset max_log10conductors in case of extrapolation
    if (
        class_identifier is None
    ):  # creating dataset for evaluation already trained model
        logging.info("Dataset is created for evaluation on already trained model.")
        if all_ds.max_log10conductors > max_log10conductors:
            logging.warning(
                f"Model is extrapolating: log10 of maximum conductor computed from from dataset {all_ds.max_log10conductors} is bigger than log10 of maximum conductor used for training model {max_log10conductors}"
            )
        all_ds.override_max_log10conductors(max_log10conductors)
    else:
        logging.info("Dataset is created for training new model.")

    train_ds = Subset(all_ds, train_idx)
    valid_ds = Subset(all_ds, valid_idx)
    test_ds = Subset(all_ds, test_idx)

    # Creating fastai Dataloaders
    logging.info(f"  Using torch device {torch.device(device)}")
    train_dl = DataLoader(
        train_ds,
        shuffle=True,
        batch_size=batch_size,
        drop_last=True,
        device=torch.device(device),
        num_workers=num_workers,
    )  # pin_memory=True
    valid_dl = DataLoader(
        valid_ds,
        shuffle=False,
        batch_size=batch_size,
        drop_last=False,
        device=torch.device(device),
        num_workers=num_workers,
    )
    test_dl = DataLoader(
        test_ds,
        shuffle=False,
        batch_size=test_batch_size,
        drop_last=False,
        device=torch.device(device),
        num_workers=num_workers,
    )

    logging.debug(f"Length of train_dl: {len(train_dl)}")
    logging.debug(f"Length of valid_dl: {len(valid_dl)}")
    logging.debug(f"Length of test_dl: {len(test_dl)}")

    return (
        DataLoaders(train_dl, valid_dl, test_dl, device=torch.device(device)),
        all_ds,
        dataset_size,
        all_ds.data_channels,
        data_length,
        count_classes,
        big_classes,
        conductors,
        curves,
        labels,
        test_idx,
    )


if __name__ == "__main__":
    pass
