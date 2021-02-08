# Intervention Effect Functions in Deep Generative Models

This repository contains code to recreate the results from the paper, "Attentive Pharmacodynamic State Space Modeling." It also provides the user with scripts to train the models PK-PD deep generative models detailed in the paper (arxiv link forthcoming). We have used PyTorch Lightning to build out our models (see the [docs](https://pytorch-lightning.readthedocs.io/en/latest/)) as well as the [MMRF CoMMpass dataset](https://research.themmrf.org/) (see [repo](https://github.com/clinicalml/ml_mmrf), which you can clone and add in the ```data``` folder).

To run this repo, 
1. **Set up the environment**: Run `bash requirements.sh`. Each time you use this repository, start with `conda activate disease_prog`.

To run a hyperparameter sweep on a new dataset: 
1. **Define a data.py file in a new folder**: In the ```data``` folder, define a new folder for the new dataset. Then, define a load function in a .py file, similar to load_mmrf in data.py under ml_mmrf.
2. **Adjust base.py file**: In ```ief_core/models/base.py```, add the loading function for the new dataset in the setup() function.
3. **Run hyperparameter sweep**: To run a hyperparameter sweep, define a config file in ```ief_core/configs```. A test one has been provided. Then, run: ```python launch_run.py --config [NAME OF CONFIG FILE]```. 

To train a model given a specific set of hyperparameters, 
1. **Go into correct directory**: Go into ```ief/ief_core```, 
2. **Run command**: Run ```python main_trainer.py --model_name ssm --ttype lin --reg_type l2 --reg_all all --C 0.01 --dim_stochastic 48 --dim_hidden 300 --fold 1 --max_epochs 15000 --dataset mm --inf_noise 0.0 --data_dir /afs/csail.mit.edu/u/z/zeshanmh/research/ief/data/ml_mmrf/ml_mmrf/output/cleaned_mm_fold_2mos_comb3.pkl```
3. **Specify appropriate hyperparameters**: You can specify the hyperparameters of the model as shown above. Please see ```ief_core/main_trainer.py``` and ```ief_core/models/ssm/ssm.py``` for all options. To specify a path to save a checkpoint (corresponding to model with best validation loss), simply add ```--ckpt_path [PATH TO OUTPUT FILE]``` to the command in step 2.

Finally, to recreate the plots in the paper, see ```examples/model_analyses_final.ipynb```. 
