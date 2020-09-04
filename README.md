# IEF 

We have used PyTorch Lightning to build out our models (see https://pytorch-lightning.readthedocs.io/en/latest/).

## Instructions (EDIT): 

To run a hyperparameter sweep, define a config file in ```ief_core/configs```. A test one has been provided. 

Then, run: ```python launch_run.py --config [NAME OF CONFIG FILE]```

Note that the folder containing the MMRF processing scripts (```data/ml_mmrf```) can be cloned from the clinicalML github. To run the tests, first install pytest and then run ```py.test``` in ```ief_core/tests```. In the tests folder, you can also quickly test a model by running any one of the individual testing scripts, which contain examples on how to instantiate and change the hyperparameters of a model (e.g. see run_ssm_ss() in test_ssm.py). 
