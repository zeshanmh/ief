# Intervention Effect Functions 

This repository contains code to recreate the results from the paper, "Attentive Pharmacodynamic State Space Modeling." It also provides the user with scripts to train the models PK-PD deep generative models detailed in the paper (arxiv link forthcoming). We have used PyTorch Lightning to build out our models (see https://pytorch-lightning.readthedocs.io/en/latest/) as well as the MMRF CoMMpass dataset (https://research.themmrf.org/).

To run this repo, 
1. **Set up the environment**: Run `bash requirements.sh`. Each time you use this repository, start with `conda activate disease_prog`.
2. **Set configuration variables**: 

## Running a Hyperparameter Sweep: 

To run a hyperparameter sweep, define a config file in ```ief_core/configs```. A test one has been provided. 

Then, run: ```python launch_run.py --config [NAME OF CONFIG FILE]```

Note that the folder containing the MMRF processing scripts (```data/ml_mmrf```) can be cloned from the clinicalML github. 

## Training a Model: 

If you would like to train a single model, you can define a ```train``` function similar to the examples shown in ```ief_core/tests/short_run.py```. These examples show how to instantiate and specify the hyperparameters of a model. Then, simply run ```python short_run.py``` and a checkpoint of the model will be saved in ```ief_core/tests/checkpoints```. 
