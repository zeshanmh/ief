# Intervention Effect Functions 

This repository contains code to recreate the results from the paper, "Attentive Pharmacodynamic State Space Modeling." It also provides the user with scripts to train the models PK-PD deep generative models detailed in the paper (arxiv link forthcoming). We have used PyTorch Lightning to build out our models (see https://pytorch-lightning.readthedocs.io/en/latest/) as well as the MMRF CoMMpass dataset (https://research.themmrf.org/).

To run this repo, 
1. **Set up the environment**: Run `bash requirements.sh`. Each time you use this repository, start with `conda activate disease_prog`.

To run a hyperparameter sweep on a new dataset: 
1. **Define a data.py file in a new folder**: In the ```data``` folder, define a new folder for the new dataset. Then, define a load function in a .py file, similar to load_mmrf in data.py under ml_mmrf.
2. **Adjust base.py file**: In ```ief_core/models/base.py```, add the loading function for the new dataset in the setup() function.
3. **Run hyperparameter sweep**: To run a hyperparameter sweep, define a config file in ```ief_core/configs```. A test one has been provided. Then, run: ```python launch_run.py --config [NAME OF CONFIG FILE]```. 

To train a model given a specific set of hyperparameters, 
1. **Define train function**: If you would like to train a single model, you can define a ```train``` function similar to the examples shown in ```ief_core/tests/short_run.py```. These examples show how to instantiate and specify the hyperparameters of a model.
2. **Run python file**:  Then, simply run ```python ief_core/tests/short_run.py``` and a checkpoint of the model will be saved in ```ief_core/tests/checkpoints```. 

Finally, to recreate the plots in the paper, see ```examples/model_analyses_aaai.ipynb```. 
