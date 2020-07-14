import torch 
import sys, math
import numpy as np
import torch.jit as jit
import warnings 
import torch.nn.init as init
import torch.nn as nn

from models.base import Model
from models.utils import *
from pyro.distributions import Normal, Independent, Categorical, LogNormal
from typing import List, Tuple
from torch import Tensor
from collections import namedtuple
from typing import List, Tuple
from torch.autograd import Variable
from argparse import ArgumentParser

np.random.seed(0)

class FOMM(Model): 
    def __init__(self, 
                 dim_hidden: int = 300, 
                 mtype: str = 'linear', 
                 C: float = 0., 
                 reg_all: bool = True, 
                 reg_type: str = 'l1', 
                 **kwargs
                ): 
        super(FOMM, self).__init__()
        self.save_hyperparameters()

    def init_model(self): 
        mtype     = self.hparams['mtype']
        dim_data  = self.hparams['dim_data']
        dim_base  = self.hparams['dim_base']
        dim_treat = self.hparams['dim_treat']

        # define the transition function 
        if mtype == 'linear':
            self.model_mu   = nn.Linear(dim_data, dim_data)
            self.model_sig  = nn.Linear(dim_treat+dim_base+dim_data, dim_data) 

    def p_X(self, X, A, B):
        base_cat = B[:,None,:].repeat(1, max(1, X.shape[1]-1), 1)
        mtype    = self.hparams['mtype']
        if mtype =='carry_forward':
            p_x_mu   = X[:,:-1,:]
        elif mtype=='linear_prior':
            p_x_mu   = self.model_mu(X[:,:-1,:], A[:,:-1,:], base_cat)
        elif 'logcellkill' in mtype or 'treatment_exp' in mtype or 'gated' in mtype or 'moe' in mtype: 
            Aval     = A[:,:-1,:]
            cat      = torch.cat([X[:,:-1,:], A[:,:-1,:], base_cat], -1)
            p_x_mu   = self.model_mu(cat, torch.cat([Aval[...,[0]], base_cat, Aval[...,1:]],-1))
        elif 'nl' in mtype:
            cat      = torch.cat([X[:,:-1,:], A[:,:-1,:], base_cat],-1)
            p_x_mu   = self.model_mu(cat)
        else: 
            p_x_mu   = self.model_mu(X[:,:-1,:])
        cat      = torch.cat([X[:,:-1,:], A[:,:-1,:], base_cat],-1)
        p_x_sig  = torch.nn.functional.softplus(self.model_sig(cat))
        return p_x_mu, p_x_sig 
    
    def get_loss(self, B, X, A, M, Y, CE, anneal = 1., return_reconstruction = False):
        _, _, lens         = get_masks(M)
        B, X, A, M, Y, CE  = B[lens>1], X[lens>1], A[lens>1], M[lens>1], Y[lens>1], CE[lens>1]
        p_x_mu, p_x_std    = self.p_X(X, A, B)
        masked_nll = masked_gaussian_nll_3d(X[:,1:,:], p_x_mu, p_x_std, M[:,1:,:])
        nll        = masked_nll.sum(-1).sum(-1)
        if return_reconstruction:
            return (nll, p_x_mu*M[:,1:,:], p_x_std*M[:,1:,:])
        else:
            return (nll,)
    
    def forward(self, B, X, A, M, Y, CE, anneal = 1.):
        (nll,)     = self.get_loss(B, X, A, M, Y, CE, anneal = anneal)
        reg_loss   = torch.mean(nll)
        for name,param in self.named_parameters():
            if self.hparams['reg_all']:
                reg_loss += self.hparams['C']*apply_reg(param, reg_type=self.hparams['reg_type'])
            else:
                if 'weight' in name:
                    reg_loss += self.hparams['C']*apply_reg(param, reg_type=self.hparams['reg_type'])
        return (torch.mean(nll), torch.mean(nll), torch.tensor(0.), torch.tensor(0.)), torch.mean(reg_loss) 
    
    def sample(self, T_forward, X, A, B):
        with torch.no_grad():
            base           = B[:,None,:]
            obs_list       = [X[:,[0],:]]
            for t in range(1, T_forward):
                x_prev     = obs_list[-1]
                if self.mtype =='carry_forward':
                    p_x_mu   = x_prev
                elif self.mtype=='linear_prior':
                    p_x_mu   = self.model_mu(x_prev, A[:,[t-1],:], base)
                elif 'logcellkill' in self.mtype or 'treatment_exp' in self.mtype or 'gated' in self.mtype:
                    Aval     = A[:,[t-1],:]
                    cat      = torch.cat([x_prev, A[:,[t-1],:], base], -1)
                    p_x_mu   = self.model_mu(cat, torch.cat([Aval[...,[0]], base, Aval[...,1:]],-1))
                else:
                    p_x_mu   = self.model_mu(torch.cat([x_prev, A[:,[t-1],:], base], -1))
                obs_list.append(p_x_mu) 
        return torch.cat(obs_list, 1)
    
    def inspect(self, T_forward, T_condition, B, X, A, M, Y, CE, restrict_lens = False):
        self.eval()
        if restrict_lens: 
            m_t, m_g_t, lens         = get_masks(M[:,1:,:])
            B, X, A, M, Y, CE  = B[lens>1], X[lens>1], A[lens>1], M[lens>1], Y[lens>1], CE[lens>1]
        p_x_mu, p_x_std = self.p_X(X, A, B)
        masked_nll      = masked_gaussian_nll_3d(X[:,1:,:], p_x_mu, p_x_std, M[:,1:,:])
        masked_nll_s    = masked_nll.sum(-1).sum(-1)
        nll             = torch.mean(masked_nll_s)
        per_feat_nll    = masked_nll.sum(1).mean(0)
        
        # calculate MSE instead
        mse = (((p_x_mu-X[:,1:])**2)*M[:,1:]).sum(0).sum(0)
        vals=M[:,1:].sum(0).sum(0)
        per_feat_nll   = mse/vals
        
        # Sample forward unconditionally
        inp_x      = self.sample(T_forward, X, A, B)
        inp_x_post = self.sample(T_forward+1, X[:,T_condition-1:], A[:,T_condition-1:], B)
        inp_x_post = torch.cat([X[:,:T_condition], inp_x_post[:,1:]], 1) 
        empty      = torch.ones(X.shape[0], 3)
        return nll, per_feat_nll, empty, empty, inp_x_post, inp_x
    
    def predict(self, **kwargs):
        raise NotImplemented()

    @staticmethod
    def add_model_specific_args(parent_parser): 
        parser = ArgumentParser(parents = [parent_parser], add_help=False)

        parser.add_argument('--dim_hidden', type=int, default=300, help='hidden dimension for nonlinear model')
        parser.add_argument('--mtype', type=str, default='linear', help='transition function in FOMM')
        parser.add_argument('--C', type=float, default=.1, help='regularization strength')
        parser.add_argument('--reg_all', type=bool, default=True, help='regularize all weights or only subset')    
        parser.add_argument('--reg_type', type=str, default='l1', help='regularization type')

        return parser 



