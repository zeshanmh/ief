import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
import sys 
import os
import optuna
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
from semi_synthetic.ss_data import *
from models.ssm.ssm import SSM, SSMAtt
from models.ssm.ssm_baseline import SSMBaseline
from models.rnn import GRU
from main_trainer import *
from models.fomm import FOMM, FOMMAtt
from models.sfomm import SFOMM
from distutils.util import strtobool

'''
Name: short_run.py 
Purpose: This set of functions is used to quickly train a model given a pre-determined
set of hyperparameters. In general, these are meant to be run after you have 
done a thorough hyperparameter sweep using launch_run.py. As in the other scripts, 
the args can be altered based on the user's needs (e.g. dataset, number of samples, 
using importance sampling or not for SSM, batch size, etc.).
Usage: If one wants to train an SSM model with a linear transition function on 20000 
samples of semi-synthetic, run -- train_ssm_gated_syn(ttype='lin', num_samples=20000).
'''

def train_ssm_gated_syn(ttype='attn_transition', num_samples=1000): 
    seed_everything(0)

    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default='ssm', help='fomm, ssm, or gru')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--anneal', type=float, default=1., help='annealing rate')
    parser.add_argument('--fname', type=str, help='name of save file')
    parser.add_argument('--imp_sampling', type=bool, default=False, help='importance sampling to estimate marginal likelihood')
    parser.add_argument('--nsamples', default=1, type=int)
    parser.add_argument('--nsamples_syn', default=1000, type=int, help='number of training samples for synthetic data')
    parser.add_argument('--optimizer_name', type=str, default='adam')
    parser.add_argument('--dataset', default='semi_synthetic', type=str)
    parser.add_argument('--eval_type', type=str, default='nelbo')
    parser.add_argument('--loss_type', type=str, default='semisup')
    parser.add_argument('--bs', default=1500, type=int, help='batch size')
    parser.add_argument('--fold', default=1, type=int)
    parser.add_argument('--ss_missing', type=strtobool, default=False, help='whether to add missing data in semi synthetic setup or not')
    parser.add_argument('--ss_in_sample_dist', type=strtobool, default=True, help='whether to use mm training patients to generate validation/test set in semi synthetic data')
    parser.add_argument('--att_mask', type=strtobool, default=False, help='set to True for SSMAtt and FOMMAtt')
    import pdb; pdb.set_trace()
    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()

    # add rest of args from SSM and base trainer
    parser = SSM.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    # parse args and convert to dict
    args = parser.parse_args()
    args.max_epochs = 1000
    args.nsamples_syn = num_samples
    args.ttype      = ttype
    args.alpha1_type = 'linear'
    args.add_stochastic = False
    dict_args = vars(args)
    args.C = 0.01; args.reg_all = True; args.reg_type = 'l2'
    args.dim_stochastic = 128

    # initialize SSM w/ args and train 
    trial = optuna.trial.FixedTrial({'bs': args.bs, 'lr': args.lr, 'C': args.C, 'reg_all': args.reg_all, 'reg_type': args.reg_type, 'dim_stochastic': args.dim_stochastic})
    model = SSM(trial, **dict_args)
    checkpoint_callback = ModelCheckpoint(filepath='./checkpoints/ssm_semi_syn_' + ttype + '_' + str(args.nsamples_syn) + 'sample_complexity{epoch:05d}-{val_loss:.2f}')
    trainer = Trainer.from_argparse_args(args, deterministic=True, logger=False, checkpoint_callback=False, gpus=[0])
    trainer.fit(model)

def train_ssm_gated_mm(fold=1, reg_all=True, C=0.01, reg_type='l2', ds=48, ttype='attn_transition'):
    print(f'[FOLD: {fold}, REG_ALL: {reg_all}]') 
    seed_everything(0)

    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default='ssm', help='fomm, ssm, or gru')
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
    parser.add_argument('--fold', default=fold, type=int)
    parser.add_argument('--ss_missing', type=strtobool, default=False, help='whether to add missing data in semi synthetic setup or not')
    parser.add_argument('--ss_in_sample_dist', type=strtobool, default=False, help='whether to use mm training patients to generate validation/test set in semi-synthetic data')
    parser.add_argument('--att_mask', type=strtobool, default=False, help='set to True for SSMAtt and FOMMAtt')

    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()

    # add rest of args from SSM and base trainer
    parser = SSM.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    # parse args and convert to dict
    args = parser.parse_args()
    args.max_epochs = 15000
    args.ttype      = ttype
    args.alpha1_type = 'linear'
    args.add_stochastic = False
    args.C = C; args.reg_all = reg_all; args.reg_type = reg_type
    args.dim_stochastic = ds
    dict_args = vars(args)

    # initialize FOMM w/ args and train 
    trial = optuna.trial.FixedTrial({'bs': args.bs, 'lr': args.lr, 'C': args.C, 'reg_all': args.reg_all, 'reg_type': args.reg_type, 'dim_stochastic': args.dim_stochastic})
    model = SSM(trial, **dict_args)
    checkpoint_callback = ModelCheckpoint(filepath='./checkpoints/mmfold' + str(fold) + str(ds) + '_' + ttype + '_ssm_dataaug_restrictedfeat_TR{epoch:05d}-{val_loss:.2f}')
    trainer = Trainer.from_argparse_args(args, deterministic=True, logger=False, checkpoint_callback=checkpoint_callback, gpus=[3])
    trainer.fit(model)
    
def train_sfomm_attn_mm(fold=1, reg_all=True, C=0.01, reg_type='l2', ds=16, mtype='linear'):
    print(f'[FOLD: {fold}, REG_ALL: {reg_all}]') 
    seed_everything(0)

    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default='sfomm', help='fomm, ssm, or gru')
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
    parser.add_argument('--fold', default=fold, type=int)
    parser.add_argument('--ss_missing', type=strtobool, default=False, help='whether to add missing data in semi synthetic setup or not')
    parser.add_argument('--ss_in_sample_dist', type=strtobool, default=False, help='whether to use mm training patients to generate validation/test set in semi-synthetic data')
    parser.add_argument('--att_mask', type=strtobool, default=False, help='set to True for SSMAtt and FOMMAtt')

    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()

    # add rest of args from SSM and base trainer
    parser = SFOMM.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    # parse args and convert to dict
    args = parser.parse_args()
    args.max_epochs = 10000
    args.mtype      = mtype
    args.alpha1_type = 'linear'
    args.add_stochastic = False
    args.C = C; args.reg_all = reg_all; args.reg_type = reg_type
    args.dim_stochastic = ds
    args.inftype = 'rnn'
    dict_args = vars(args)

    # initialize SFOMM w/ args and train 
    trial = optuna.trial.FixedTrial({'bs': args.bs, 'lr': args.lr, 'C': args.C, 'reg_all': args.reg_all, 'reg_type': args.reg_type, 'dim_stochastic': args.dim_stochastic, 'inftype': args.inftype})
    model = SFOMM(trial, **dict_args)
    checkpoint_callback = ModelCheckpoint(filepath='./checkpoints/mmfold' + str(fold) + str(ds) + '_sfomm_' + mtype + '_dataaug_fullfeat_test{epoch:05d}-{val_loss:.2f}')
    trainer = Trainer.from_argparse_args(args, deterministic=True, logger=False, checkpoint_callback=checkpoint_callback, gpus=[3])
    trainer.fit(model)
    
def train_fomm_gated(): 
    seed_everything(0)
    
    configs = [ 
        (1, 10000, 'attn_transition', .1, True, 'l1')
    ]
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default='fomm_att', help='fomm, ssm, or gru')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--anneal', type=float, default=1., help='annealing rate')
    parser.add_argument('--fname', type=str, help='name of save file')
    parser.add_argument('--imp_sampling', type=bool, default=False, help='importance sampling to estimate marginal likelihood')
    parser.add_argument('--nsamples', default=1, type=int)
    parser.add_argument('--nsamples_syn', default=1000, type=int, help='number of training samples for synthetic data')
    parser.add_argument('--optimizer_name', type=str, default='adam')
    parser.add_argument('--eval_type', type=str, default='nelbo')
    parser.add_argument('--loss_type', type=str, default='unsup')
    parser.add_argument('--dataset', default='mm', type=str)
    parser.add_argument('--bs', default=600, type=int, help='batch size')
    parser.add_argument('--fold', default=1, type=int)
    parser.add_argument('--ss_missing', type=strtobool, default=False, help='whether to add missing data in semi synthetic setup or not')
    parser.add_argument('--ss_in_sample_dist', type=strtobool, default=False, help='whether to use mm training patients to generate validation/test set in semi synthetic data')
    parser.add_argument('--att_mask', type=strtobool, default=False, help='set to True for SSMAtt and FOMMAtt')


    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()

    # add rest of args from FOMM and base trainer
    parser = FOMM.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    
    for k,config in enumerate(configs): 
        print(f'running config: {config}')
        fold, max_epochs, mtype, C, reg_all, reg_type = config
        
        # parse args and convert to dict
        args = parser.parse_args()
        args.fold       = fold
        args.max_epochs = max_epochs
        args.dim_hidden = 300
        args.mtype      = mtype
        args.C = C; args.reg_all = reg_all; args.reg_type = reg_type
        args.alpha1_type = 'linear'
        args.otype = 'identity'
        print(f'FOLD: {args.fold}')
        args.add_stochastic = False
        dict_args = vars(args)

        # initialize FOMM w/ args and train
        trial = optuna.trial.FixedTrial({'bs': args.bs, 'lr': args.lr, 'C': args.C, 'reg_all': args.reg_all, 'reg_type': args.reg_type, 'dim_hidden': args.dim_hidden}) 
        model = FOMM(trial, **dict_args)
        checkpoint_callback = ModelCheckpoint(filepath='./checkpoints/mmfold_' + str(fold) + '_bagoffunc_fomm_att1{epoch:05d}-{val_loss:.2f}')
        trainer = Trainer.from_argparse_args(args, deterministic=True, logger=False, gpus=[2], \
                        checkpoint_callback=checkpoint_callback, early_stop_callback=False)
        trainer.fit(model)


if __name__ == '__main__':
#     train_sfomm_attn_mm(fold=1, reg_all=True, C=0.01, reg_type='l2', ds=16)
    sizes = [16,48,64]
#     for ds in sizes: 
#         train_sfomm_attn_mm(fold=1, reg_all=True, C=0.01, reg_type='l2', ds=ds, mtype='linear')
#         train_sfomm_attn_mm(fold=1, reg_all=True, C=0.01, reg_type='l2', ds=ds, mtype='attn_transition')
    # no aug, restricted feature set 
    for ds in sizes: 
        train_ssm_gated_mm(fold=1, reg_all=True, C=0.01, reg_type='l1', ds=ds, ttype='attn_transition')
        train_ssm_gated_mm(fold=1, reg_all=True, C=0.01, reg_type='l2', ds=ds, ttype='lin')
