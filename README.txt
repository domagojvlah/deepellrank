https://arxiv.org/abs/2207.06699

Contacts:
matija.kazalicki@math.hr
domagoj.vlah@fer.hr


Requires a Linux-like OS with CUDA, Bash, Miniconda (https://docs.conda.io/en/latest/miniconda.html).
Miniconda will download and install Python and all the required libraries in an isolated environment.

Tested on and configured for a single machine with:
- Nvidia Quadro RTX 5000 GPU with 16 GB RAM
- 256 GB RAM,
- 100 Gb free disk space for temporary cache files

A machine with lower specs might work but we did not test for it.
To use different GPU with less RAM, modify hyperparameters related to batch_size. This might influence the rank prediction result in the end. We did not test for it.
Less system RAM is used for smaller training datasets and smaller number od ap-s. Training might work with less RAM and big swap file but initialization will be slow.

models dir contains 24 best trained models from the paper in CNN and \Omega classifiers case.


Steps to reproduce our results ($ denotes terminal prompt):

0) set up conda environment

$ conda create --name NT --file spec-file.txt
$ conda activate NT


1) download dataset from https://ferhr-my.sharepoint.com/:f:/g/personal/dvlah_fer_hr/EhhUGxO9CU5GsDFNVdljxjEBaEEhsKry2FbX5uQr_Vuu4A?e=SmXD9j and save it to data dir


2) Example of training new CNN model:

$ python train_model_controller.py --model_name TEST hyp_LMFDB_uniform_Matthews_NN_p1000_FINISHED 


3) Example of making rank predictions:

$ jupyter notebook  # Start jupyter notebook

Open test_deepellrank.ipynb and run all cells up to "Run deepellrank"

$ python deepellrank.py TEST

Run remaining cells in test_deepellrank.ipynb


Notes:
    Some hyperparameter definitions in hyperparameters dir work only with script train_model_controller_gyopt.py