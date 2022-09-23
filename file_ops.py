
####################################################################################
# File operations and different related helper functions
####################################################################################

import pathlib
import os  # for deleting temp files
import json
import glob
import uuid
import time
import flatdict
import pandas as pd
from tqdm import tqdm

from global_vars import MODEL_DIR, DATA_DIR


# Input dataset files operations


def get_beg_end_classes_list(first_class=0, last_class=2917287, split_size=100000):
    assert first_class % split_size == 0
    output_list = []
    for beg_class in range(first_class, last_class, split_size):
        end_class = beg_class + split_size
        end_class = min(end_class, last_class)
        output_list.append((beg_class, end_class))
    return output_list


def get_LMFDB_all_file_names(first_class=0, last_class=2917287, split_size=100000, min_cond=None, max_cond=None):
    thresholds = [11, 24274, 46786, 69390, 91683, 114235, 136752, 159525,
                  182280, 205175, 227910, 251055, 274260, 297450, 320768,
                  343990, 367236, 390313, 414050, 437570, 461440, 485056,
                  3810240, 27626273, 61683821, 101126843, 144400783,
                  190961443, 239839571, 291010021, 299996953]
    if min_cond is not None:
        for idx_low, el in enumerate(thresholds[1:]):
            if min_cond < el:
                break
    else:
        idx_low = 0
    if max_cond is not None:
        for idx_high, el in enumerate(thresholds):
            if max_cond < el:
                break
    else:
        idx_high = 30
    return [f"LMFDB_alg_curves_{beg_class:010}-{end_class:010}"
            for beg_class, end_class in get_beg_end_classes_list(first_class, last_class, split_size)[idx_low:idx_high]]


def delete_temp_dataset(dataset_name, max_prime):
    for fn in [DATA_DIR / f"{dataset_name}.json.cached",
               DATA_DIR / f"{dataset_name}.{max_prime}.npy"]:
        if fn.is_file():
            os.remove(fn)
    return


# Time measuring

def get_absolute_time():
    return [time.perf_counter(), time.process_time()]  # (total_time, CPU_time)


def get_relative_time(old_time):
    time = get_absolute_time()
    assert len(time) == len(old_time)
    # (total_time, CPU_time)
    return [time[i] - old_time[i] for i in range(len(old_time))]


# Model files operations

class ModelNotTrained(Exception):
    pass


def get_current_model_name(model_num, model_name):
    return f"{model_name}.{model_num}"


# Function to check if a model already exists on disk
def model_exist(model_num, model_name):
    current_model_name = get_current_model_name(model_num, model_name)
    current_model_name_path = MODEL_DIR / \
        f"{current_model_name}.hyperparameters.json"
    if current_model_name_path.is_file():
        return True  # This model is already calculated
    return False  # This model does not exist


def delete_temp_files(current_model_name):
    for fn in [MODEL_DIR / f"{current_model_name}-bestmodel.pth"]:
        if fn.is_file():
            os.remove(fn)
    return


def get_model_numbers(model_name):
    models_fpaths = glob.glob(
        f"{str(MODEL_DIR / pathlib.Path(model_name))}.*.hyperparameters.json")
    models = []
    for fpath in models_fpaths:
        with open(fpath, "r") as infile:
            models.append(json.load(infile)['model_num'])
    return models


def get_best_model_num(model_name, model_nums, criterion='MCC'):
    full_model_names = [get_current_model_name(
        num, model_name) for num in model_nums]

    list_of_dict = []
    for name in full_model_names:
        with open(MODEL_DIR / f"{name}.hyperparameters.json", "r") as fp:
            fd = flatdict.FlatDict(json.load(fp))
        for key in fd.keys():
            new_key = key.replace(r"model:", '').replace(
                r"dataloader:", '').replace(r"optimizer:", '')
            fd[new_key] = fd.pop(key)
        with open(MODEL_DIR / f"{name}.results.json", "r") as fp:
            res = json.load(fp)
        fd.update(res)
        list_of_dict.append(dict(fd))

    df = pd.DataFrame.from_records(list_of_dict)
    if criterion == 'MCC':
        df_best = df.sort_values('Matthews_corrcoef', ascending=False)[:1]
        MCC = float(df_best['Matthews_corrcoef'])
        model_num = str(list(df_best['model_num'])[0])
        return model_num, MCC
    else:
        raise NotImplementedError(
            f"Best model selection criterion {criterion} is not implemented")


def get_random_model_number():
    return str(uuid.uuid4())  # 128 bit random UUID in string form


# Pandas progress bar - adopted from: www.thiscodeworks.com
def read_csv_pgbar(csv_path, chunksize, names, usecols=None, dtype=None, skip_first_row=True):

    # print('Getting row count of csv file')

    rows = sum(1 for _ in open(csv_path, 'r'))
    if skip_first_row:
        rows = rows - 1  # minus the header

    # chunks = rows//chunksize + 1
    # print('Reading csv file')
    chunk_list = []

    with tqdm(total=rows, desc='Rows read: ') as bar:
        for chunk in pd.read_csv(csv_path, chunksize=chunksize, names=names, usecols=usecols, dtype=dtype):
            chunk_list.append(chunk)
            bar.update(len(chunk))

    df = pd.concat((f for f in chunk_list), axis=0)
    print('Finish reading csv file')

    return df
