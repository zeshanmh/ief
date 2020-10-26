import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
import sys 
import optuna 
import os
from lifelines.utils import concordance_index
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset
from torchcontrib.optim import SWA
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from argparse import ArgumentParser
from distutils.util import strtobool
sys.path.append('../')
sys.path.append('../../data/ml_mmrf')
sys.path.append('../../data/')
from ml_mmrf_v1.data import load_mmrf
from synthetic.synthetic_data import load_synthetic_data_trt, load_synthetic_data_noisy
from models.rnn import GRU
from main_trainer import *
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
    model_configs = [ 
        # (0, 1000, 'gru', 514, 0.198, False, 'l2', .0023), 
        # (0, 1500, 'gru', 578, 0.0445, False, 'l2', .000655), 
        # (0, 2000, 'gru', 677, 0.00585, True, 'l1', .000599),
        # (0, 10000, 'gru', 676, 0.002312, True, 'l1', 0.001280),
        # (0, 1000, 'pkpd_gru', 290, 0.09916, False, 'l2', .002916), 
        # (0, 1500, 'pkpd_gru', 502, 0.02635, False, 'l2', .001307), 
        # (0, 2000, 'pkpd_gru', 298, 0.031917, False, 'l2', .006825)
        (0, 1000, 'gru', 250, 0.01, False, 'l1', 1e-3), 
        (0, 1500, 'gru', 500, 0.01, False, 'l1', 1e-3), 
        (0, 2000, 'gru', 250, 0.01, False, 'l1', 1e-3),
        (0, 10000, 'gru', 500, 0.01, False, 'l1', 1e-3),
        (0, 1000, 'pkpd_gru', 500, 0.01, True, 'l2', 1e-3), 
        (0, 1500, 'pkpd_gru', 500, 0.01, False, 'l2', 1e-3), 
        (0, 2000, 'pkpd_gru', 500, 0.01, False, 'l1', 1e-3)
    ]
    fname  = './gru_ss_results_take2.txt'
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
    parser.add_argument('--bs', default=2000, type=int, help='batch size')
    parser.add_argument('--fold', default=1, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--ss_missing', type=strtobool, default=True, help='whether to add missing data in semi synthetic setup or not')
    parser.add_argument('--ss_in_sample_dist', type=strtobool, default=True, help='whether to use mm training patients to generate validation/test set in semi synthetic data')

    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()

    # add rest of args from FOMM and base trainer
    parser = GRU.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    # parse args and convert to dict
    args = parser.parse_args()
    fi   = open(fname, 'w')
    for k,model_config in enumerate(model_configs): 
        seed, nsamples_syn, mtype, dim_hidden, C, reg_all, reg_type, lr = model_config
        seed_everything(seed)
        args.lr = lr
        args.max_epochs = 1000
        args.nsamples_syn = nsamples_syn
        args.mtype        = mtype
        args.dim_hidden   = dim_hidden
        args.alpha1_type  = 'linear'
        args.add_stochastic = False
        args.C = C; args.reg_all = reg_all; args.reg_type = reg_type
        dict_args = vars(args)

        trial = optuna.trial.FixedTrial({'lr': args.lr, 'C': args.C, 'reg_all': args.reg_all, 'reg_type': args.reg_type, 'dim_hidden': args.dim_hidden})

        # initialize FOMM w/ args and train 
        model = GRU(trial, **dict_args)
        in_sample_dist = model.hparams.ss_in_sample_dist; add_missing = model.hparams.ss_missing
        print(f'[RUNNING] model config {k+1}: N = {args.nsamples_syn}, mtype = {args.mtype}, C = {args.C}, reg_all = {args.reg_all}, reg_type = {args.reg_type}, in_sample_dist = {in_sample_dist}, add_missing = {add_missing}')
        fi.write(f'[RUNNING] model config {k+1}: N = {args.nsamples_syn}, mtype = {args.mtype}, C = {args.C}, reg_all = {args.reg_all}, reg_type = {args.reg_type}, in_sample_dist = {in_sample_dist}, add_missing = {add_missing}\n')
        trainer = Trainer.from_argparse_args(args, deterministic=True, logger=False, checkpoint_callback=False, gpus=[2], check_val_every_n_epoch=10)
        trainer.fit(model)

        # evaluate on validation set; this should match what we were getting with the old codebase (after 100 epochs)
        if torch.cuda.is_available():
            device = torch.device('cuda:2')
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
        print(f'[COMPLETE] model config {k+1}: mean nelbo: {np.mean(nelbos)}, std nelbo: {np.std(nelbos)}')
        fi.write(f'[COMPLETE] model config {k+1}: mean nelbo: {np.mean(nelbos)}, std nelbo: {np.std(nelbos)}\n\n')
        print()


def test_gru_pkpd(): 
    seed_everything(0)
    
    configs = [
        (1000, 'pkpd_gru_att', 500, 0.01, True, 'l2')
#         (1000, 'gru', 250, 0.01, True, 'l2')
    ]
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
    parser.add_argument('--loss_type', type=str, default='unsup')
    parser.add_argument('--bs', default=600, type=int, help='batch size')
    parser.add_argument('--fold', default=1, type=int)
    parser.add_argument('--optuna', type=strtobool, default=True, help='whether to use optuna to optimize hyperparams')
    parser.add_argument('--ss_missing', type=strtobool, default=False, help='whether to add missing data in semi synthetic setup or not')
    parser.add_argument('--ss_in_sample_dist', type=strtobool, default=False, help='whether to use mm training patients to generate validation/test set in semi synthetic data')
    parser.add_argument('--att_mask', type=strtobool, default=False, help='set to True for SSMAtt and FOMMAtt')


    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()

    # add rest of args from GRU and base trainer
    parser = GRU.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    
    for k,config in enumerate(configs): 
        print(f'running config: {config}')
        max_epochs, mtype, dh, C, reg_all, reg_type = config
        # parse args and convert to dict
        args = parser.parse_args()
        args.max_epochs = max_epochs
        args.mtype      = mtype
        args.dim_hidden = dh
        args.reg_type   = reg_type
        args.C          = C
        args.reg_all    = reg_all
        args.alpha1_type = 'linear'
        args.add_stochastic = False
        dict_args = vars(args)

        # initialize FOMM w/ args and train 
        trial = optuna.trial.FixedTrial({'lr': args.lr, 'C': args.C, 'reg_all': args.reg_all, 'reg_type': args.reg_type, 'dim_hidden': args.dim_hidden}) 
        model = GRU(trial, **dict_args)
        # early_stop_callback = EarlyStopping(
        #    monitor='val_loss',
        #    min_delta=0.00,
        #    patience=10,
        #    verbose=False,
        #    mode='min'
        # )
        checkpoint_callback = ModelCheckpoint(filepath='./checkpoints/gru_att1{epoch:05d}-{val_loss:.2f}')
        trainer = Trainer.from_argparse_args(args, deterministic=True, logger=False, gpus=[2], \
            early_stop_callback=False, checkpoint_callback=checkpoint_callback)
        trainer.fit(model)

        # evaluate on validation set; this should match what we were getting with the old codebase (after 100 epochs)
        valid_loader = model.val_dataloader()
        nelbos = []
        for i in range(50): 
            (nelbo, nll, kl, _), _ = model.forward(*valid_loader.dataset.tensors, anneal = 1.)
            nelbos.append(nelbo.item())
        print(f'final nll for {config} (config {k+1}): mean: {np.mean(nelbos)}, std: {np.std(nelbos)}')
#     assert (nelbo.item() - 183.759) < 1e-1

if __name__ == '__main__':
    # run_gru_ss()
    test_gru_pkpd()