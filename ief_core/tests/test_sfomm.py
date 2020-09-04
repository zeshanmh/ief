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
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from argparse import ArgumentParser
sys.path.append('../')
sys.path.append('../../data/ml_mmrf')
sys.path.append('../../data/')
from ml_mmrf_v1.data import load_mmrf
from synthetic.synthetic_data import load_synthetic_data_trt, load_synthetic_data_noisy
from semi_synthetic.ss_data import *
from models.sfomm import SFOMM
from models.utils import *

# ave_diff test, linear, gated, rnn inftype
def return_per_class_acc(y_true, y_pred): 
    accs = []
    for i in range(max(y_true)+1): 
        idxs = np.where(y_true == i)
        t = y_true[idxs]
        p = y_pred[idxs]
        acc = sum(t == p) / len(t)
        accs.append(acc)
    return accs

def test_sfomm_texp_mm(): 
    seed_everything(0)
    
    configs = [
        (1000, 'gated', 'rnn', 'l2', 0.01, 48, True)
#         (1000, 'linear', 'rnn', 'l2', 0.01, 48, True),
#         (1000, 'gated', 'birnn', 'l2', 0.01, 48, True),
#         (1000, 'linear', 'birnn', 'l2', 0.01, 48, True)
    ]
    
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default='sfomm', help='fomm, ssm, or gru')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--anneal', type=float, default=-1., help='annealing rate')
    parser.add_argument('--fname', type=str, help='name of save file')
    parser.add_argument('--imp_sampling', type=bool, default=False, help='importance sampling to estimate marginal likelihood')
    parser.add_argument('--nsamples', default=1, type=int)
    parser.add_argument('--nsamples_syn', default=50, type=int, help='number of training samples for synthetic data')
    parser.add_argument('--optimizer_name', type=str, default='adam')
    parser.add_argument('--dataset', default='mm', type=str)
    parser.add_argument('--loss_type', type=str, default='unsup')
    parser.add_argument('--eval_type', type=str, default='nelbo')
    parser.add_argument('--bs', default=200, type=int, help='batch size')
    parser.add_argument('--fold', default=1, type=int)

    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()

    # add rest of args from SSM and base trainer
    parser = SFOMM.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    # parse args and convert to dict
    
    for k,config in enumerate(configs): 
        print(f'running config: {config}')
        max_epochs, mtype, inftype, reg_type, C, ds, reg_all = config
        
        args = parser.parse_args()
        args.max_epochs = max_epochs
        args.mtype      = mtype
        args.alpha1_type = 'linear'
        args.inftype     = inftype
        args.reg_type    = reg_type
        args.C           = C
        args.dim_stochastic= ds
        args.reg_all     = reg_all
        args.add_stochastic = False
        dict_args = vars(args)

        # initialize FOMM w/ args and train 
        model = SFOMM(**dict_args)
        checkpoint_callback = ModelCheckpoint(filepath='./checkpoints/sfomm_gated')
        early_stop_callback = EarlyStopping(
           monitor='val_loss',
           min_delta=0.00,
           patience=10,
           verbose=False,
           mode='min'
        )
        trainer = Trainer.from_argparse_args(args, deterministic=True, logger=False, checkpoint_callback=checkpoint_callback, gpus=[2], early_stop_callback=False)
        trainer.fit(model)

        # evaluate on validation set; this should match what we were getting with the old codebase (after 100 epochs)
        if torch.cuda.is_available():
            device = torch.device('cuda:2')
        else:
            device  = torch.device('cpu')
        _, valid_loader = model.load_helper('valid', device=device, oversample=False)

        nelbos = []
        for i in range(50): 
            (nelbo, nll, kl, _), _ = model.forward(*valid_loader.dataset.tensors, anneal = 1.)
            nelbos.append(nelbo.item())
        print(f'final nelbo for {config} (config {k+1}): mean: {np.mean(nelbos)}, std: {np.std(nelbos)}')
    # preds, _ = model.predict_ord(*valid_loader.dataset.tensors)

    # B, X, A, M, Y, CE = valid_loader.dataset.tensors
    # from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
    # preds  = pt_numpy(preds.argmax(dim=1))
    # print(preds); print(Y)
    # f1     = f1_score(pt_numpy(Y), preds, average='weighted')
    # precision = precision_score(pt_numpy(Y), preds, average='weighted')
    # recall = recall_score(pt_numpy(Y), preds, average='weighted')
    # auc    = roc_auc_score(pt_numpy(Y), preds, average='weighted')
    # # auc=0.
    # acc_class = return_per_class_acc(pt_numpy(Y), preds)

    # print(f'F1: {f1}, Precision: {precision}, Recall: {recall}, AUC: {auc}, per_class_acc: {acc_class}')
    # return preds
    

def test_sfomm_texp_syn(): 
    seed_everything(0)

    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default='sfomm', help='fomm, ssm, or gru')
    parser.add_argument('--lr', type=float, default=8e-3, help='learning rate')
    parser.add_argument('--anneal', type=float, default=-1., help='annealing rate')
    parser.add_argument('--fname', type=str, help='name of save file')
    parser.add_argument('--imp_sampling', type=bool, default=False, help='importance sampling to estimate marginal likelihood')
    parser.add_argument('--nsamples', default=1, type=int)
    parser.add_argument('--nsamples_syn', default=50, type=int, help='number of training samples for synthetic data')
    parser.add_argument('--optimizer_name', type=str, default='adam')
    parser.add_argument('--dataset', default='synthetic', type=str)
    parser.add_argument('--loss_type', type=str, default='unsup')
    parser.add_argument('--eval_type', type=str, default='nelbo')
    parser.add_argument('--bs', default=600, type=int, help='batch size')
    parser.add_argument('--fold', default=1, type=int)

    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()

    # add rest of args from SSM and base trainer
    parser = SFOMM.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    # parse args and convert to dict
    args = parser.parse_args()
    args.max_epochs = 5000
    args.mtype      = 'treatment_exp'
    args.alpha1_type = 'linear'
    args.inftype     = 'rnn'
    args.reg_type    = 'l2'
    args.C           = 0.
    args.dim_stochastic= 16
    args.reg_all     = True
    args.add_stochastic = False
    dict_args = vars(args)

    # initialize FOMM w/ args and train 
    model = SFOMM(**dict_args)
    trainer = Trainer.from_argparse_args(args, deterministic=True, logger=False, checkpoint_callback=False, gpus=[2])
    trainer.fit(model)

    # evaluate on validation set; this should match what we were getting with the old codebase (after 100 epochs)
    if torch.cuda.is_available():
        device = torch.device('cuda:2')
    else:
        device  = torch.device('cpu')
    _, valid_loader = model.load_helper('valid', device=device)
    preds, _ = model.predict(*valid_loader.dataset.tensors)
    mse, r2, ci = calc_stats(preds, valid_loader.dataset.tensors)
    assert abs(mse - 4.57) < 1e-1

if __name__ == '__main__':
    test_sfomm_texp_mm()

