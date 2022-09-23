
# General training handler for calling from command line or from controller process.
# Model and problem specific training, validation and testing logic is implemented
# in imported function train_and_save_model

import sys
import json
import logging

from train import train_and_save_model

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)

if __name__ == '__main__':

    # MAIN BLOCK
    assert len(sys.argv) == 3
    hyperparams=json.loads(sys.argv[1])
    system_params = json.loads(sys.argv[2])
    current_model_name, results = train_and_save_model(hyperparams, **system_params)
    
    logging.info(f"Results of {current_model_name}: {results}")