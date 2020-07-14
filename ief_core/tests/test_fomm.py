import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
import sys 
import os
from lifelines.utils import concordance_index
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset
from torchcontrib.optim import SWA
from pytorch_lightning import Trainer, seed_everything
from argparse import ArgumentParser
sys.path.append('../')
sys.path.append('../../data/ml_mmrf')
sys.path.append('../../data/')
from ml_mmrf_v1.data import load_mmrf
from synthetic.synthetic_data import load_synthetic_data_trt, load_synthetic_data_noisy
from models.fomm import FOMM 

def test_fomm_load(): 
    checkpoint_path = '../tb_logs/test_model/version_0/checkpoints/epoch=403.ckpt'
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    hparams    = checkpoint['hyper_parameters']
    fomm = FOMM(**hparams); fomm.init_model() 
    fomm.load_state_dict(checkpoint['state_dict'])
    assert 'dim_data' in fomm.hparams
    assert 'dim_treat' in fomm.hparams
    assert 'dim_base' in fomm.hparams
    assert fomm.hparams['mtype'] == 'linear'

def test_fomm_linear(): 
    seed_everything(0)

    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default='fomm', help='fomm, ssm, or gru')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--anneal', type=float, default=1., help='annealing rate')
    parser.add_argument('--fname', type=str, help='name of save file')
    parser.add_argument('--imp_sampling', type=bool, default=False, help='importance sampling to estimate marginal likelihood')
    parser.add_argument('--nsamples', default=1, type=int)
    parser.add_argument('--optimizer_name', type=str, default='adam')
    parser.add_argument('--dataset', default='mm', type=str)
    parser.add_argument('--bs', default=600, type=int, help='batch size')
    parser.add_argument('--fold', default=1, type=int)

    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()

    # add rest of args from FOMM and base trainer
    parser = FOMM.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    # parse args and convert to dict
    args = parser.parse_args()
    dict_args = vars(args)

    # initialize FOMM w/ args and train 
    model = FOMM(**dict_args)
    trainer = Trainer.from_argparse_args(args, deterministic=True, logger=False, checkpoint_callback=False, max_epochs=100)
    trainer.fit(model)

    # evaluate on validation set; this should match what we were getting with the old codebase (after 100 epochs)
    valid_loader = model.val_dataloader()
    (nelbo, nll, kl, _), _ = model.forward(*valid_loader.dataset.tensors, anneal = 1.)
    assert (nelbo.item() - 1106.95) < 1e-1


if __name__ == '__main__':
    test_fomm_linear()

