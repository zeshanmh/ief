conda create -n disease_prog python=3.7.3
conda activate disease_prog
conda install pytorch cudatoolkit torchvision matplotlib scipy numpy pandas h5py ipython jupyterlab tensorflow-gpu keras cython seaborn scikit-learn r
conda install -c conda-forge tslearn ipdb
pip install pyro-ppl tensorflow_probability jupyter-tensorboard lifelines pytorch-lightning 
pip install git+https://github.com/rtqichen/torchdiffeq
conda install -c conda-forge ncurses
pip install pytorch-lightning
pip install pytest 
