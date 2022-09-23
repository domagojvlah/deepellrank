
# Defines global constants and variables

import pathlib

# GLOBALS #####################################
ROOT_DIR = pathlib.Path('.')
DATA_DIR = ROOT_DIR / pathlib.Path('data')  # datasets location
MODEL_DIR = ROOT_DIR / pathlib.Path("models")  # saved models and training/validation/test metadata
CACHE_DIR = ROOT_DIR / pathlib.Path("cache")  # different caches that speed up dataset loading - can be safely removed
LOGS_DIR = ROOT_DIR / pathlib.Path("logs")  # optimization logs
HYPERPARAMETERS_DIR = ROOT_DIR / pathlib.Path("hyperparameters")  # imports defining different models hyperparameters
###############################################