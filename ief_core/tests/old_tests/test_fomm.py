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
sys.path.append('../')
sys.path.append('../../data/ml_mmrf')
sys.path.append('../../data/')
from ml_mmrf_v1.data import load_mmrf
from synthetic.synthetic_data import load_synthetic_data_trt, load_synthetic_data_noisy
from models.fomm import FOMM, FOMMAtt
from semi_synthetic.ss_data import *
from distutils.util import strtobool

def test_fomm_load():
    # need to rewrite test by saving a model first and then loading it 
    checkpoint_path = '../tb_logs/test_model/version_0/checkpoints/epoch=403.ckpt'
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    hparams    = checkpoint['hyper_parameters']
    hparams['otype'] = 'linear'
    hparams['alpha1_type'] = 'linear'
    hparams['add_stochastic'] = False
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
    parser.add_argument('--nsamples_syn', default=1000, type=int, help='number of training samples for synthetic data')
    parser.add_argument('--optimizer_name', type=str, default='adam')
    parser.add_argument('--eval_type', type=str, default='nelbo')
    parser.add_argument('--loss_type', type=str, default='semisup')
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
    args.max_epochs = 100
    dict_args = vars(args)

    # initialize FOMM w/ args and train 
    model = FOMM(**dict_args)
    trainer = Trainer.from_argparse_args(args, deterministic=True, logger=False, checkpoint_callback=False)
    trainer.fit(model)

    # evaluate on validation set; this should match what we were getting with the old codebase (after 100 epochs)
    valid_loader = model.val_dataloader()
    (nelbo, nll, kl, _), _ = model.forward(*valid_loader.dataset.tensors, anneal = 1.)
    assert (nelbo.item() - 1106.95) < 1e-1

def test_fomm_gated(): 
    seed_everything(0)
    
    configs = [ 
        (1, 10000, 'attn_transition', .1, True, 'l1')
        # (3, 10000, 'attn_transition', 1, False, 'l2')
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
        # early_stop_callback = EarlyStopping(
        #    monitor='val_loss',
        #    min_delta=0.00,
        #    patience=10,
        #    verbose=False,
        #    mode='min'
        # )
        trainer = Trainer.from_argparse_args(args, deterministic=True, logger=False, gpus=[2], \
                        checkpoint_callback=checkpoint_callback, early_stop_callback=False)
        trainer.fit(model)

        # evaluate on validation set; this should match what we were getting with the old codebase (after 100 epochs)
        # valid_loader = model.val_dataloader()
        # nelbos = []
        # for i in range(50):
        #     (nelbo, nll, kl, _), _ = model.forward(*valid_loader.dataset.tensors, anneal = 1.)
        #     nelbos.append(nelbo.item())
        # print(f'final nelbo for {config} (config {k+1}): mean: {np.mean(nelbos)}, std: {np.std(nelbos)}')

#         assert (nelbo.item() - 175.237) < 1e-1

def run_fomm_ss(): 
    seed_everything(0)
    model_configs = [ 
        # (1000, 'linear', 0.01, True, 'l1'), # .01, True, 'l1'
        # (1500, 'linear', 0.01, True, 'l1'), #  .01, True, 'l1'
        (2000, 'nl', 0.01, True, 'l1') # .01, True, 'l1'
    ]
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default='fomm', help='fomm, ssm, or gru')
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
    parser.add_argument('--bs', default=600, type=int, help='batch size')
    parser.add_argument('--fold', default=1, type=int)
    parser.add_argument('--ss_missing', type=strtobool, default=False, help='whether to add missing data in semi synthetic setup or not')
    parser.add_argument('--ss_in_sample_dist', type=strtobool, default=True, help='whether to use mm training patients to generate validation/test set in semi synthetic data')
    parser.add_argument('--att_mask', type=strtobool, default=False, help='set to True for SSMAtt and FOMMAtt')

    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()

    # add rest of args from FOMM and base trainer
    parser = FOMM.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    # parse args and convert to dict
    args = parser.parse_args()

    for k,model_config in enumerate(model_configs): 
        nsamples_syn, mtype, C, reg_all, reg_type = model_config
        args.max_epochs = 10000
        args.nsamples_syn = nsamples_syn
        args.mtype        = mtype
        args.alpha1_type  = 'linear'
        args.add_stochastic = False
        args.C = C; args.reg_all = reg_all; args.reg_type = reg_type
        dict_args = vars(args)

        # initialize FOMM w/ args and train 
        model = FOMM(**dict_args)
        in_sample_dist = model.hparams.ss_in_sample_dist; add_missing = model.hparams.ss_missing
        print(f'[RUNNING] model config {k+1}: N = {args.nsamples_syn}, mtype = {args.mtype}, C = {args.C}, reg_all = {args.reg_all}, reg_type = {args.reg_type}, in_sample_dist = {in_sample_dist}, add_missing = {add_missing}')
        trainer = Trainer.from_argparse_args(args, deterministic=True, logger=False, checkpoint_callback=False, check_val_every_n_epoch=10, gpus=[0])
        trainer.fit(model)

        # evaluate on validation set; this should match what we were getting with the old codebase (after 100 epochs)
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device  = torch.device('cpu')
        ddata = load_ss_data(model.hparams['nsamples_syn'], gen_fly=True, eval_mult=100, in_sample_dist=in_sample_dist, add_missing=add_missing)
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
        # # ddata = load_ss_data(model.hparams['nsamples_syn'], gen_fly=True)
        # ddata = model.ddata 
        # nelbos = []
        # for i in range(5):
        #     _, valid_loader = load_ss_helper(ddata, tvt='valid', bs=model.hparams['bs'], device=device, valid_fold=i)
        #     (nelbo, nll, kl, _), _ = model.forward(*valid_loader.dataset.tensors, anneal = 1.)
        #     nelbos.append(nelbo.item())
        print(f'[COMPLETE] model config {k+1}: mean nelbo: {np.mean(nelbos)}, std nelbo: {np.std(nelbos)}')
        print()


if __name__ == '__main__':
    test_fomm_gated()