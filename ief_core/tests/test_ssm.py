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
from semi_synthetic.ss_data import *
from models.ssm.ssm import SSM

def test_ssm_sota(): 
    sys.path.append('../../../trvae')
    sys.path.append('../../../trvae/dmm')
    sys.path.append('../../../trvae/models')
    from dmm import DMM 
    ddata = load_ss_data(1000, gen_fly=True, eval_mult=500, in_sample_dist=False, add_missing=True)
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device  = torch.device('cpu')

    # _, valid_loader = load_ss_helper(ddata, tvt='valid', device=device, bs=600)
    dim_stochastic = 48; dim_hidden = 300 
    dim_base  = ddata['train']['B'].shape[-1]
    dim_data  = ddata['train']['X'].shape[-1]
    dim_treat = ddata['train']['A'].shape[-1]
    C = 0.01; ttype = 'gated'; etype = 'lin'
    model = DMM(dim_stochastic, dim_hidden, dim_base, dim_data, dim_treat, C = C, ttype = ttype, etype=etype, 
                            inftype = 'rnn_relu', combiner_type = 'pog', include_baseline = True, reg_type = 'l1', reg_all=True, augmented=False)
    model.to(device)
    fname = '../../../trvae/dmm/good_models/sota_ssm_mm.pt'
    print ('loading',fname)
    model.load_state_dict(torch.load(fname))

    print(f'eval set size: {ddata["valid"][0]["X"].shape}')
    nelbos = []
    for i in range(5):
        _, valid_loader = load_ss_helper(ddata, tvt='valid', bs=600, device=device, valid_fold=i)
        batch_nelbos = []
        for i_batch, valid_batch_loader in enumerate(valid_loader):
            (nelbo, nll, kl, _), _ = model.forward_unsupervised(*valid_batch_loader, anneal = 1.)
            nelbo, nll, kl         = nelbo.item(), nll.item(), kl.item()
            batch_nelbos.append(nelbo)
        # (nelbo, nll, kl, _), _ = model.forward(*valid_loader.dataset.tensors, anneal = 1.)
        nelbos.append(np.mean(batch_nelbos))
    print(f'NELBO (on ss data) of trained model from which semi-synthetic dataset was sampled: {np.mean(nelbos)}, std: {np.std(nelbos)}')    
    # batch_nelbos = []
    # for i_batch, valid_batch_loader in enumerate(valid_loader):
    #     (nelbo, nll, kl, _), _ = model.forward_unsupervised(*valid_batch_loader, anneal = 1.)
    #     nelbo, nll, kl         = nelbo.item(), nll.item(), kl.item()
    #     batch_nelbos.append(nelbo)
    # (nelbo, nll, kl, _), _ = model.forward_unsupervised(*valid_loader.dataset.tensors, anneal = 1.)
    # print(f'NELBO (on ss data) of trained model from which semi-synthetic dataset was sampled: {np.mean(batch_nelbos)}')

def test_ssm_load(): 
    checkpoint_path = '../tbp_logs/ssm_lin_semi_synthetic_subsample_best_epoch=03689-val_loss=-225.67.ckpt'
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    hparams    = checkpoint['hyper_parameters']
    ssm = SSM(**hparams); ssm.setup(1) 
    ssm.load_state_dict(checkpoint['state_dict'])
    assert 'dim_data' in ssm.hparams
    assert 'dim_treat' in ssm.hparams
    assert 'dim_base' in ssm.hparams
    assert ssm.hparams['ttype'] == 'lin'

    valid_loader = ssm.val_dataloader()
    (nelbo, nll, kl, _), _ = ssm.forward(*valid_loader.dataset.tensors, anneal = 1.)
    print(nelbo)

def run_ssm_ss(): 
    seed_everything(0)
    model_configs = [ 
        (1000, 'lin', 16, 0.01, True, 'l2'),
        (1000, 'lin', 48, 0.01, True, 'l2'),
        (1000, 'lin', 64, 0.01, True, 'l2'),
        (1000, 'lin', 128, 0.01, True, 'l2'),
        (1000, 'lin', 256, 0.01, True, 'l2')
    ]
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
    parser.add_argument('--loss_type', type=str, default='unsup')
    parser.add_argument('--bs', default=600, type=int, help='batch size')
    parser.add_argument('--fold', default=1, type=int)
    parser.add_argument('--ss_missing', type=strtobool, default=False, help='whether to add missing data in semi synthetic setup or not')
    parser.add_argument('--ss_in_sample_dist', type=strtobool, default=True, help='whether to use mm training patients to generate validation/test set in semi synthetic data')

    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()

    # add rest of args from FOMM and base trainer
    parser = SSM.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    # parse args and convert to dict
    args  = parser.parse_args()
    for k,model_config in enumerate(model_configs): 
        nsamples_syn, ttype, dim_stochastic, C, reg_all, reg_type = model_config
        args.max_epochs = 10000
        args.nsamples_syn = nsamples_syn
        args.ttype        = ttype
        args.dim_stochastic   = dim_stochastic
        args.dim_hidden   = 300
        args.alpha1_type  = 'linear'
        args.add_stochastic = False
        args.C = C; args.reg_all = reg_all; args.reg_type = reg_type
        dict_args = vars(args)

        # initialize FOMM w/ args and train 
        model = SSM(**dict_args)
        in_sample_dist = model.hparams.ss_in_sample_dist; add_missing = model.hparams.ss_missing
        print(f'[RUNNING] model config {k+1}: N = {args.nsamples_syn}, ttype = {args.ttype}, C = {args.C}, reg_all = {args.reg_all}, reg_type = {args.reg_type}, in_sample_dist = {in_sample_dist}, add_missing = {add_missing}')
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
        print()

def run_ssm_ss2(): 
    seed_everything(0)
    # model_configs = [ 
    #     (1000, 'moe', 48, 0.01, True, 'l2'), # 48.000, 0.010000, 1.0000, l2
    #     (1500, 'moe', 48, 0.01, True, 'l2'), # 48.000, 0.010000, 1.0000, l2
    #     (2000, 'moe', 48, 0.01, True, 'l2') # 48.000, 0.010000, 1.0000, l2
    # ]
    model_configs = [ 
        # (1000, 'lin', 48, 0.01, False, 'l2'), # 48.000, 0.010000, 1.0000, l2
        # (1500, 'lin', 48, 0.01, False, 'l2'), # 48.000, 0.010000, 0.0000, l2
        (2000, 'lin', 48, 0.01, True, 'l2') # 48.000, 0.010000, 0.0000, l2
        # (1000, 'gated', 48, 0.01, True, 'l2'), # 48.000, 0.010000, 1.0000, l2
        # (1500, 'gated', 48, 0.01, True, 'l2'), # 48.000, 0.010000, 1.0000, l2
        # (2000, 'gated', 48, 0.01, False, 'l2') # 48.000, 0.010000, 1.0000, l2
    ]
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
    parser.add_argument('--loss_type', type=str, default='unsup')
    parser.add_argument('--bs', default=600, type=int, help='batch size')
    parser.add_argument('--fold', default=1, type=int)
    parser.add_argument('--ss_missing', type=strtobool, default=True, help='whether to add missing data in semi synthetic setup or not')
    parser.add_argument('--ss_in_sample_dist', type=strtobool, default=True, help='whether to use mm training patients to generate validation/test set in semi synthetic data')

    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()

    # add rest of args from FOMM and base trainer
    parser = SSM.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    # parse args and convert to dict
    args  = parser.parse_args()
    for k,model_config in enumerate(model_configs): 
        nsamples_syn, ttype, dim_stochastic, C, reg_all, reg_type = model_config
        args.max_epochs = 10000
        args.nsamples_syn = nsamples_syn
        args.ttype        = ttype
        args.dim_stochastic   = dim_stochastic
        args.dim_hidden   = 300
        args.alpha1_type  = 'linear'
        args.add_stochastic = False
        args.C = C; args.reg_all = reg_all; args.reg_type = reg_type
        dict_args = vars(args)

        # initialize FOMM w/ args and train 
        model = SSM(**dict_args)
        in_sample_dist = model.hparams.ss_in_sample_dist; add_missing = model.hparams.ss_missing
        print(f'[RUNNING] model config {k+1}: N = {args.nsamples_syn}, ttype = {args.ttype}, C = {args.C}, reg_all = {args.reg_all}, reg_type = {args.reg_type}, in_sample_dist = {in_sample_dist}, add_missing = {add_missing}')
        trainer = Trainer.from_argparse_args(args, deterministic=True, logger=False, checkpoint_callback=False, gpus=[0], check_val_every_n_epoch=10)
        trainer.fit(model)

        # evaluate on validation set; this should match what we were getting with the old codebase (after 100 epochs)
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
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
        print()

def test_ssm_semi_synthetic(): 
    seed_everything(0)

    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default='ssm', help='fomm, ssm, or gru')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--anneal', type=float, default=1., help='annealing rate')
    parser.add_argument('--fname', type=str, help='name of save file')
    parser.add_argument('--imp_sampling', type=bool, default=False, help='importance sampling to estimate marginal likelihood')
    parser.add_argument('--nsamples', default=1, type=int)
    parser.add_argument('--eval_type', type=str, default='nelbo')
    parser.add_argument('--loss_type', type=str, default='semisup')
    parser.add_argument('--nsamples_syn', default=100, type=int, help='number of training samples for synthetic data')
    parser.add_argument('--optimizer_name', type=str, default='adam')
    parser.add_argument('--dataset', default='semi_synthetic', type=str)
    parser.add_argument('--bs', default=600, type=int, help='batch size')
    parser.add_argument('--fold', default=1, type=int)
    parser.add_argument('--ss_missing', type=strtobool, default=True, help='whether to add missing data in semi synthetic setup or not')
    parser.add_argument('--ss_in_sample_dist', type=strtobool, default=True, help='whether to use mm training patients to generate validation/test set in semi synthetic data')

    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()

    # add rest of args from SSM and base trainer
    parser = SSM.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    # parse args and convert to dict
    args = parser.parse_args()
    args.max_epochs = 100
    args.ttype      = 'gated'
    args.alpha1_type = 'linear'
    args.add_stochastic = False
    args.C = 0.1; args.reg_all = True; args.reg_type = 'l1'
    # args.C = 0.01; args.reg_all = False; args.reg_type = 'l2'
    dict_args = vars(args)

    # initialize FOMM w/ args and train 
    model = SSM(**dict_args)
    trainer = Trainer.from_argparse_args(args, deterministic=True, logger=False, checkpoint_callback=False, gpus=[3])
    trainer.fit(model)

    # evaluate on validation set; this should match what we were getting with the old codebase (after 100 epochs)
    if torch.cuda.is_available():
        device = torch.device('cuda:3')
    else:
        device  = torch.device('cpu')
    # ddata = load_ss_data(model.hparams['nsamples_syn'], gen_fly=True)
    ddata = model.ddata 
    _, valid_loader = load_ss_helper(ddata, tvt='valid', bs=model.hparams['bs'], device=device)
    (nelbo, nll, kl, _), _ = model.forward(*valid_loader.dataset.tensors, anneal = 1.)
    print(f'nelbo: {nelbo}')
    assert (nelbo.item() - 306) < 3e-1

def test_ssm_linear_mm(): 
    seed_everything(0)

    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default='ssm', help='fomm, ssm, or gru')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--anneal', type=float, default=1., help='annealing rate')
    parser.add_argument('--fname', type=str, help='name of save file')
    parser.add_argument('--imp_sampling', type=bool, default=False, help='importance sampling to estimate marginal likelihood')
    parser.add_argument('--nsamples', default=1, type=int)
    parser.add_argument('--eval_type', type=str, default='nelbo')
    parser.add_argument('--loss_type', type=str, default='unsup')
    parser.add_argument('--nsamples_syn', default=1000, type=int, help='number of training samples for synthetic data')
    parser.add_argument('--optimizer_name', type=str, default='adam')
    parser.add_argument('--dataset', default='mm', type=str)
    parser.add_argument('--bs', default=600, type=int, help='batch size')
    parser.add_argument('--fold', default=1, type=int)
    parser.add_argument('--ss_missing', type=strtobool, default=False, help='whether to add missing data in semi synthetic setup or not')
    parser.add_argument('--ss_in_sample_dist', type=strtobool, default=False, help='whether to use mm training patients to generate validation/test set in semi synthetic data')

    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()

    # add rest of args from SSM and base trainer
    parser = SSM.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    # parse args and convert to dict
    args = parser.parse_args()
    args.max_epochs = 100
    args.ttype = 'gated'
    args.dim_stochastic = 48
    args.dim_hidden = 300
    dict_args = vars(args)

    # initialize FOMM w/ args and train 
    model = SSM(**dict_args)
    trainer = Trainer.from_argparse_args(args, deterministic=True, logger=False, checkpoint_callback=False, gpus=[1])
    trainer.fit(model)

    # evaluate on validation set; this should match what we were getting with the old codebase (after 100 epochs)
    if torch.cuda.is_available():
        device = torch.device('cuda:1')
    else:
        device  = torch.device('cpu')
    _, valid_loader = model.load_helper('valid', device=device)
    (nelbo, nll, kl, _), _ = model.forward(*valid_loader.dataset.tensors, anneal = 1.)
    assert (nelbo.item() - 253.4) < 3e-1

def test_ssm_gated_mm(): 
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
    parser.add_argument('--loss_type', type=str, default='semisup')
    parser.add_argument('--bs', default=600, type=int, help='batch size')
    parser.add_argument('--fold', default=1, type=int)
    parser.add_argument('--ss_missing', type=strtobool, default=False, help='whether to add missing data in semi synthetic setup or not')
    parser.add_argument('--ss_in_sample_dist', type=strtobool, default=False, help='whether to use mm training patients to generate validation/test set in semi synthetic data')

    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()

    # add rest of args from SSM and base trainer
    parser = SSM.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    # parse args and convert to dict
    args = parser.parse_args()
    args.max_epochs = 500
    args.ttype      = 'gated'
    args.alpha1_type = 'linear'
    args.add_stochastic = False
    # args.C = 0.1; args.reg_all = True; args.reg_type = 'l1'
    args.C = 0.01; args.reg_all = False; args.reg_type = 'l2'
    dict_args = vars(args)

    # initialize FOMM w/ args and train 
    model = SSM(**dict_args)
    trainer = Trainer.from_argparse_args(args, deterministic=True, logger=False, checkpoint_callback=False, gpus=[3])
    trainer.fit(model)

    # evaluate on validation set; this should match what we were getting with the old codebase (after 100 epochs)
    if torch.cuda.is_available():
        device = torch.device('cuda:3')
    else:
        device  = torch.device('cpu')
    _, valid_loader = model.load_helper('valid', device=device)
    (nelbo, nll, kl, _), _ = model.forward(*valid_loader.dataset.tensors, anneal = 1.)
    print(f'nelbo: {nelbo}')
    assert (nelbo.item() - 230) < 3e-1

def test_ssm_linear_syn(): 
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
    parser.add_argument('--dataset', default='synthetic', type=str)
    parser.add_argument('--eval_type', type=str, default='nelbo')
    parser.add_argument('--loss_type', type=str, default='semisup')
    parser.add_argument('--bs', default=600, type=int, help='batch size')
    parser.add_argument('--fold', default=1, type=int)
    parser.add_argument('--ss_missing', type=strtobool, default=False, help='whether to add missing data in semi synthetic setup or not')
    parser.add_argument('--ss_in_sample_dist', type=strtobool, default=False, help='whether to use mm training patients to generate validation/test set in semi synthetic data')

    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()

    # add rest of args from SSM and base trainer
    parser = SSM.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    # parse args and convert to dict
    args = parser.parse_args()
    args.max_epochs = 100
    dict_args = vars(args)

    # initialize FOMM w/ args and train 
    model = SSM(**dict_args)
    trainer = Trainer.from_argparse_args(args, deterministic=True, logger=False, checkpoint_callback=False, gpus=[1])
    trainer.fit(model)

    # evaluate on validation set; this should match what we were getting with the old codebase (after 100 epochs)
    if torch.cuda.is_available():
        device = torch.device('cuda:1')
    else:
        device  = torch.device('cpu')
    _, valid_loader = model.load_helper('valid', device=device)
    (nelbo, nll, kl, _), _ = model.forward(*valid_loader.dataset.tensors, anneal = 1.)
    assert (nelbo.item() - 191) < 3e-1

def test_sota_mm_semi(): 
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
    parser.add_argument('--dataset', default='semi_synthetic', type=str)
    parser.add_argument('--eval_type', type=str, default='nelbo')
    parser.add_argument('--loss_type', type=str, default='unsup')
    parser.add_argument('--bs', default=600, type=int, help='batch size')
    parser.add_argument('--fold', default=1, type=int)
    parser.add_argument('--ss_missing', type=strtobool, default=False, help='whether to add missing data in semi synthetic setup or not')
    parser.add_argument('--ss_in_sample_dist', type=strtobool, default=False, help='whether to use mm training patients to generate validation/test set in semi synthetic data')

    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()

    # add rest of args from SSM and base trainer
    parser = SSM.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    # parse args and convert to dict
    args = parser.parse_args()
    args.max_epochs = 10000
    args.ttype      = 'gated'
    args.alpha1_type = 'linear'
    args.add_stochastic = False
    dict_args = vars(args)
    model = SSM(**dict_args)
    import pdb; pdb.set_trace()
    trainer = Trainer.from_argparse_args(args, deterministic=True, logger=False, \
        checkpoint_callback=False, gpus=[1], \
        resume_from_checkpoint='/afs/csail.mit.edu/u/z/zeshanmh/research/ief/ief_core/tbp_logs/checkpoints/ssm_mm_sota_fold1_epoch=13743-val_loss=66.07.ckpt')

    # automatically restores model, epoch, step, LR schedulers, apex, etc...
    trainer.fit(model)

def test_ssm_gated_syn(): 
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
    parser.add_argument('--dataset', default='synthetic', type=str)
    parser.add_argument('--eval_type', type=str, default='nelbo')
    parser.add_argument('--loss_type', type=str, default='semisup')
    parser.add_argument('--bs', default=600, type=int, help='batch size')
    parser.add_argument('--fold', default=1, type=int)
    parser.add_argument('--ss_missing', type=strtobool, default=False, help='whether to add missing data in semi synthetic setup or not')
    parser.add_argument('--ss_in_sample_dist', type=strtobool, default=False, help='whether to use mm training patients to generate validation/test set in semi synthetic data')

    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()

    # add rest of args from SSM and base trainer
    parser = SSM.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    # parse args and convert to dict
    args = parser.parse_args()
    args.max_epochs = 100
    args.ttype      = 'gated'
    args.alpha1_type = 'linear'
    args.add_stochastic = False
    dict_args = vars(args)

    # initialize FOMM w/ args and train 
    model = SSM(**dict_args)
    trainer = Trainer.from_argparse_args(args, deterministic=True, logger=False, checkpoint_callback=False, gpus=[3])
    trainer.fit(model)

    # evaluate on validation set; this should match what we were getting with the old codebase (after 100 epochs)
    if torch.cuda.is_available():
        device = torch.device('cuda:1')
    else:
        device  = torch.device('cpu')
    _, valid_loader = model.load_helper('valid', device=device)
    (nelbo, nll, kl, _), _ = model.forward(*valid_loader.dataset.tensors, anneal = 1.)
    assert (nelbo.item() - 166) < 3e-1


if __name__ == '__main__':
    run_ssm_ss2()