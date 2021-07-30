conda create -n pkpd python=3.8.5
conda activate pkpd
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch 
conda install matplotlib scipy numpy pandas h5py ipython jupyterlab tensorflow-gpu keras cython seaborn scikit-learn
conda install -c conda-forge tslearn ipdb optuna pyro-ppl pytorch-lightning
conda install -c conda-forge lifelines wandb
conda install ecos CVXcanon # dependencies for fancyimpute
pip install tensorflow_probability jupyter-tensorboard fancyimpute anndata scanpy
pip install git+https://github.com/rtqichen/torchdiffeq
conda install -c conda-forge ncurses
conda install -c anaconda pytest
pip install imblearn
