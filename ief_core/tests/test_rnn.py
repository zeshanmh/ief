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
from distutils.util import strtobool
sys.path.append('../')
sys.path.append('../../data/ml_mmrf')
sys.path.append('../../data/')
from ml_mmrf_v1.data import load_mmrf
from synthetic.synthetic_data import load_synthetic_data_trt, load_synthetic_data_noisy
from models.rnn import GRU
from semi_synthetic.ss_data import *


def test_gru_load(): 
    checkpoint_path = '../tbp_logs/rnn_pkpd_semi_synthetic_subsample_best/version_0/checkpoints/epoch=969.ckpt'
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    hparams    = checkpoint['hyper_parameters']
    gru = GRU(**hparams); gru.setup(1) 
    gru.load_state_dict(checkpoint['state_dict'])
    assert 'dim_data' in gru.hparams
    assert 'dim_treat' in gru.hparams
    assert 'dim_base' in gru.hparams
    assert gru.hparams['mtype'] == 'pkpd_gru'

    valid_loader = gru.val_dataloader()
    (nelbo, nll, kl, _), _ = gru.forward(*valid_loader.dataset.tensors, anneal = 1.)
    print(nelbo)


def test_gru(): 
    seed_everything(0)

    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default='gru', help='fomm, ssm, or gru')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--anneal', type=float, default=1., help='annealing rate')
    parser.add_argument('--fname', type=str, help='name of save file')
    parser.add_argument('--imp_sampling', type=bool, default=False, help='importance sampling to estimate marginal likelihood')
    parser.add_argument('--nsamples', default=1, type=int)
    parser.add_argument('--nsamples_syn', default=100, type=int, help='number of training samples for synthetic data')
    parser.add_argument('--optimizer_name', type=str, default='adam')
    parser.add_argument('--dataset', default='mm', type=str)
    parser.add_argument('--eval_type', type=str, default='nelbo')
    parser.add_argument('--loss_type', type=str, default='semisup')
    parser.add_argument('--bs', default=600, type=int, help='batch size')
    parser.add_argument('--fold', default=1, type=int)

    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()

    # add rest of args from FOMM and base trainer
    parser = GRU.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    # parse args and convert to dict
    args = parser.parse_args()
    args.max_epochs = 100
    dict_args = vars(args)

    # initialize FOMM w/ args and train 
    model = GRU(**dict_args)
    trainer = Trainer.from_argparse_args(args, deterministic=True, logger=False, checkpoint_callback=False)
    trainer.fit(model)

    # evaluate on validation set; this should match what we were getting with the old codebase (after 100 epochs)
    valid_loader = model.val_dataloader()
    (nelbo, nll, kl, _), _ = model.forward(*valid_loader.dataset.tensors, anneal = 1.)
    assert (nelbo.item() - 146.43) < 1e-1

def run_gru_ss(): 
    seed_everything(0)
    model_configs = [ 
        (1000, 'gru', 250, 0.01, False, 'l1'), 
        (1000, 'pkpd_gru', 500, 0.01, True, 'l2'), 
        (1500, 'gru', 500, 0.01, False, 'l1'), 
        (1500, 'pkpd_gru', 500, 0.01, False, 'l2'), 
        (2000, 'gru', 250, 0.01, False, 'l1'),
        (2000, 'pkpd_gru', 500, 0.01, False, 'l1')
    ]
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default='gru', help='fomm, ssm, or gru')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--anneal', type=float, default=1., help='annealing rate')
    parser.add_argument('--fname', type=str, help='name of save file')
    parser.add_argument('--imp_sampling', type=bool, default=False, help='importance sampling to estimate marginal likelihood')
    parser.add_argument('--nsamples', default=1, type=int)
    parser.add_argument('--nsamples_syn', default=1000, type=int, help='number of training samples for synthetic data')
    parser.add_argument('--optimizer_name', type=str, default='adam')
    parser.add_argument('--dataset', default='semi_synthetic', type=str)
    parser.add_argument('--eval_type', type=str, default='nelbo')
    parser.add_argument('--loss_type', type=str, default='unsup')
    parser.add_argument('--bs', default=600, type=int, help='batch size')
    parser.add_argument('--fold', default=1, type=int)
    parser.add_argument('--ss_missing', type=strtobool, default=True, help='whether to add missing data in semi synthetic setup or not')
    parser.add_argument('--ss_in_sample_dist', type=strtobool, default=True, help='whether to use mm training patients to generate validation/test set in semi synthetic data')

    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()

    # add rest of args from FOMM and base trainer
    parser = GRU.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    # parse args and convert to dict
    args = parser.parse_args()

    for k,model_config in enumerate(model_configs): 
        nsamples_syn, mtype, dim_hidden, C, reg_all, reg_type = model_config
        args.max_epochs = 1000
        args.nsamples_syn = nsamples_syn
        args.mtype        = mtype
        args.dim_hidden   = dim_hidden
        args.alpha1_type  = 'linear'
        args.add_stochastic = False
        args.C = C; args.reg_all = reg_all; args.reg_type = reg_type
        dict_args = vars(args)

        # initialize FOMM w/ args and train 
        model = GRU(**dict_args)
        in_sample_dist = model.hparams.ss_in_sample_dist; add_missing = model.hparams.ss_missing
        print(f'[RUNNING] model config {k+1}: N = {args.nsamples_syn}, mtype = {args.mtype}, C = {args.C}, reg_all = {args.reg_all}, reg_type = {args.reg_type}, in_sample_dist = {in_sample_dist}, add_missing = {add_missing}')
        trainer = Trainer.from_argparse_args(args, deterministic=True, logger=False, checkpoint_callback=False, gpus=[0], check_val_every_n_epoch=10)
        trainer.fit(model)

        # evaluate on validation set; this should match what we were getting with the old codebase (after 100 epochs)
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device  = torch.device('cpu')
        ddata = load_ss_data(model.hparams['nsamples_syn'], gen_fly=True, eval_mult=200, in_sample_dist=in_sample_dist, add_missing=add_missing)
        print(f'eval set size: {ddata["valid"][0]["X"].shape}')
        nelbos = []
        for i in range(1,5):
            _, valid_loader = load_ss_helper(ddata, tvt='valid', bs=model.hparams['bs'], device=device, valid_fold=i)
            batch_nelbos = []
            for i_batch, valid_batch_loader in enumerate(valid_loader):
                (nelbo, nll, kl, _), _ = model.forward(*valid_batch_loader, anneal = 1.)
                nelbo, nll, kl         = nelbo.item(), nll.item(), kl.item()
                batch_nelbos.append(nelbo)
            # (nelbo, nll, kl, _), _ = model.forward(*valid_loader.dataset.tensors, anneal = 1.)
            nelbos.append(np.mean(batch_nelbos))
        # ddata = model.ddata 
        # nelbos = []
        # for i in range(5):
        #     import pdb; pdb.set_trace()
        #     _, valid_loader = load_ss_helper(ddata, tvt='valid', bs=model.hparams['bs'], device=device, valid_fold=i)
        #     (nelbo, nll, kl, _), _ = model.forward(*valid_loader.dataset.tensors, anneal = 1.)
        #     nelbos.append(nelbo.item())
        print(f'[COMPLETE] model config {k+1}: mean nelbo: {np.mean(nelbos)}, std nelbo: {np.std(nelbos)}')
        print()


def test_gru_pkpd(): 
    seed_everything(0)

    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default='gru', help='fomm, ssm, or gru')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--anneal', type=float, default=1., help='annealing rate')
    parser.add_argument('--fname', type=str, help='name of save file')
    parser.add_argument('--imp_sampling', type=bool, default=False, help='importance sampling to estimate marginal likelihood')
    parser.add_argument('--nsamples', default=1, type=int)
    parser.add_argument('--nsamples_syn', default=100, type=int, help='number of training samples for synthetic data')
    parser.add_argument('--optimizer_name', type=str, default='adam')
    parser.add_argument('--dataset', default='mm', type=str)
    parser.add_argument('--eval_type', type=str, default='nelbo')
    parser.add_argument('--loss_type', type=str, default='semisup')
    parser.add_argument('--bs', default=600, type=int, help='batch size')
    parser.add_argument('--fold', default=1, type=int)

    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()

    # add rest of args from FOMM and base trainer
    parser = GRU.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    # parse args and convert to dict
    args = parser.parse_args()
    args.max_epochs = 100
    args.mtype      = 'pkpd_gru'
    args.dim_hidden = 500
    args.alpha1_type = 'linear'
    args.add_stochastic = False
    dict_args = vars(args)

    # initialize FOMM w/ args and train 
    model = GRU(**dict_args)
    trainer = Trainer.from_argparse_args(args, deterministic=True, logger=False, checkpoint_callback=False)
    trainer.fit(model)

    # evaluate on validation set; this should match what we were getting with the old codebase (after 100 epochs)
    valid_loader = model.val_dataloader()
    (nelbo, nll, kl, _), _ = model.forward(*valid_loader.dataset.tensors, anneal = 1.)
    assert (nelbo.item() - 183.759) < 1e-1

if __name__ == '__main__':
    run_gru_ss()