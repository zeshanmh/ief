import torch
import torch.nn as nn
import numpy as np
import logging
import pytorch_lightning as pl
import sys 
from lifelines.utils import concordance_index
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset
from torchcontrib.optim import SWA

def get_masks(M):
    m_t    = ((torch.flip(torch.cumsum(torch.flip(M.sum(-1), (1,)), 1), (1,))>1.)*1)
    m_g_t  = (m_t.sum(-1)>1)*1.
    lens   = m_t.sum(-1)
    return m_t, m_g_t, lens
    
def masked_gaussian_nll_3d(x, mu, std, mask):
    nll        = 0.5*np.log(2*np.pi) + torch.log(std)+((mu-x)**2)/(2*std**2)
    masked_nll = (nll*mask)
    return masked_nll

def apply_reg(p, reg_type='l2'):
    if reg_type == 'l1':
        return torch.sum(torch.abs(p))
    elif reg_type=='l2':
        return torch.sum(p.pow(2))
    else:
        raise ValueError('bad reg')

def pt_numpy(tensor):
    return tensor.detach().cpu().numpy()

def calc_stats(preds, tensors):
    B, X, A, M, Y, CE = tensors
    if Y.shape[-1]>1:
        Y_oh      = Y.detach().cpu().numpy()
        bin_preds = self.prediction.detach().cpu().numpy()
        Y_np      = bin_preds[np.argmax(Y_oh,-1)]
    else:
        Y_np      = Y.detach().cpu().numpy().ravel()
    CE_np     = CE.detach().cpu().numpy().ravel()
    preds_np  = preds.detach().cpu().numpy().ravel()
    event_obs = (1.-CE_np).ravel()
    idx       = np.where(event_obs>0)[0]
    mse  = np.square(Y_np[idx]-preds_np[idx]).mean()
    r2   = r2_score(Y_np[idx], preds_np[idx])
    ci   = concordance_index(Y_np, preds_np, event_obs)
    return mse, r2, ci