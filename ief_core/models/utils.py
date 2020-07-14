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