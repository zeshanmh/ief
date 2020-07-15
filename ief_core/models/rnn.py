import torch 
import sys, math
import numpy as np
import torch.jit as jit
import warnings 
import torch.nn.init as init
import torch.nn as nn

from models.base import Model
from models.utils import *
from models.ssm.inference import RNN_STInf, Attention_STInf
from models.iefs.gated import GatedTransition
from models.iefs.moe import MofE
from pyro.distributions import Normal, Independent, Categorical, LogNormal
from typing import List, Tuple
from torch import Tensor
from collections import namedtuple
from typing import List, Tuple
from torch.autograd import Variable
from argparse import ArgumentParser

class GRU(Model): 
    def __init__(self, 
                 dim_hidden: int = 300, 
                 mtype: str = 'rnn', 
                 utype: str = 'linear',
                 dropout: float = 0.,  
                 C: float = 0., 
                 reg_all: bool = True, 
                 reg_type: str = 'l1', 
                 **kwargs
                ): 
        super(GRU, self).__init__()
        self.save_hyperparameters()

    def init_model(self): 
        mtype = self.hparams['mtype']; utype = self.hparams['utype']; 
        dropout = self.hparams['dropout']
        dim_hidden  = self.hparams['dim_hidden']
        dim_data    = self.hparams['dim_data']
        dim_base    = self.hparams['dim_base']
        dim_treat   = self.hparams['dim_treat']

        if mtype   == 'rnn':
            self.model   = nn.RNN(dim_data+dim_treat+dim_base, dim_hidden, 1, batch_first = True, dropout = dropout)
        elif mtype == 'gru':
            self.model   = nn.GRU(dim_data+dim_treat+dim_base, dim_hidden, 1, batch_first = True, dropout = dropout)
        elif mtype == 'pkpd_gru':
            self.model   = GRULayer(PKPD_GRU, dim_data, dim_treat, dim_base, dim_hidden)
        else:
            raise ValueError('Bad RNN model')

        if utype == 'linear':  
            self.model_mu   = nn.Linear(dim_hidden, dim_data)
            self.model_sig  = nn.Linear(dim_hidden, dim_data)
        elif utype == 'fomm': 
            self.model_mu    = nn.Linear(dim_data+dim_treat+dim_base, dim_data)
            self.model_sig   = nn.Linear(dim_data+dim_treat+dim_base, dim_data)
            self.hidden_te   = nn.Linear(dim_hidden, dim_data)
            self.eps         = torch.randn(dim_data)*0.1
        else: 
            raise ValueError('Bad utype')

        # treatment effect (for synthetic data)
        self.te_W1       = nn.Linear(dim_treat, dim_data)
        self.te_W2       = nn.Linear(dim_treat, dim_data*2)

    def treatment_effects(self, A): 
        A_0T   = A[:,:-1,:]
        ddata  = self.hparams['dim_data']
        scale  = (torch.sigmoid(self.te_W1(A_0T)) - 0.5)[...,:ddata]
        result = self.te_W2(A_0T)
        shift  = result[...,:ddata]
        sigma  = result[...,ddata:(ddata*2)]
        return scale, shift, sigma 
        
    def p_X(self, X, A, B, lens):
        base_cat  = B[:,None,:].repeat(1, max(1, X.shape[1]-1), 1)
        cat       = torch.cat([X[:,:-1,:], A[:,:-1,:], base_cat],-1)
        """
        pdseq     = torch.nn.utils.rnn.pack_padded_sequence(cat, lens, batch_first=True, enforce_sorted = False)
        out_pd, _ = self.model(pdseq)
        out, lens_unpacked = torch.nn.utils.rnn.pad_packed_sequence(out_pd, batch_first=True)
        """
        out, _ = self.model(cat)
        if self.hparams.utype == 'linear': 
            p_x_mu    = self.model_mu(out)
            p_x_sig   = torch.nn.functional.softplus(self.model_sig(out))
        elif self.hparams.utype == 'fomm': 
            scaleA, shiftA, _ = self.treatment_effects(A)
            p_x_mu    = self.model_mu(cat)+scaleA*self.hidden_te(out)+shiftA        
            p_x_sig   = torch.nn.functional.softplus(self.model_sig(cat))
        return p_x_mu, p_x_sig 
    
    def get_loss(self,B, X, A, M, Y, CE, anneal = 1., return_reconstruction = False):
        _, _, lens         = get_masks(M)
        B, X, A, M, Y, CE  = B[lens>1], X[lens>1], A[lens>1], M[lens>1], Y[lens>1], CE[lens>1]
        _, _, lens_1T      = get_masks(M[:,1:,:])
        p_x_mu, p_x_std    = self.p_X(X, A, B, lens_1T)
        masked_nll = masked_gaussian_nll_3d(X[:,1:,:], p_x_mu, p_x_std, M[:,1:,:])
        full_masked_nll = masked_nll
        nll        = masked_nll.sum(-1).sum(-1)
        if return_reconstruction:
            return (nll, p_x_mu*M[:,1:,:], p_x_std*M[:,1:,:])
        else:
            return (nll,)
    
    def forward(self, B, X, A, M, Y, CE, anneal = 1.):
        (nll,)     = self.get_loss(B, X, A, M, Y, CE, anneal = anneal)
        reg_loss   = torch.mean(nll)
        for name,param in self.named_parameters():
            if self.hparams.reg_all:
                reg_loss += self.hparams.C*apply_reg(param, reg_type=self.hparams.reg_type)
            else:
                if 'weight' in name:
                    reg_loss += self.hparams.C*apply_reg(param, reg_type=self.hparams.reg_type)
        return (torch.mean(nll), torch.mean(nll), torch.tensor(0.), torch.tensor(0.)), torch.mean(reg_loss) 
    
    def get_hidden(self, X, A, B):
        base_cat       = B[:,None,:].repeat(1, max(1, A.shape[1]), 1)
        inp            = torch.cat([X, A, base_cat],-1)
        _, hidden      = self.model(inp)
        return hidden
        
    def sample(self, T_forward, X, A, B, hidden = None):
        if hidden is None:
            hidden = Variable(torch.zeros(1, X.shape[0], self.dh)).to(X.device)
        inp        = torch.cat([X[:,[0],:], A[:,[0],:], B[:,None,:]],-1)
        mtype      = self.hparams['mtype']
        with torch.no_grad():
            olist = []
            for t in range(1, T_forward+1):
                if mtype == 'rnn' or mtype == 'gru': 
                    output, hidden = self.model(inp, hidden)
                else: 
                    output, hidden = self.model(inp, hidden.squeeze())
                x_mu           = self.model_mu(output)
                inp            = torch.cat([x_mu, A[:,[t],:], B[:,None,:]],-1)
                olist.append(x_mu)
            olist = torch.cat(olist, 1)
        return olist
    
    def inspect(self, T_forward, T_condition, B, X, A, M, Y, CE, restrict_lens = False):
        self.eval()
        if restrict_lens: 
            m_t, m_g_t, lens   = get_masks(M[:,1:,:])
            B, X, A, M, Y, CE  = B[lens>1], X[lens>1], A[lens>1], M[lens>1], Y[lens>1], CE[lens>1]
        _, _, lens_0TM1 = get_masks(M[:,:-1,:])
        p_x_mu, p_x_std = self.p_X(X, A, B, lens_0TM1)
        Tmax       = p_x_mu.shape[1]
        masked_nll = masked_gaussian_nll_3d(X[:,1:Tmax+1,:], p_x_mu, p_x_std, M[:,1:Tmax+1,:])
        masked_nll_s    = masked_nll.sum(-1).sum(-1)
        nll             = torch.mean(masked_nll_s)
        per_feat_nll    = masked_nll.sum(1).mean(0)
        # calculate MSE instead
        mse = (((p_x_mu-X[:,1:Tmax+1])**2)*M[:,1:Tmax+1]).sum(0).sum(0)
        vals= M[:,1:Tmax+1].sum(0).sum(0)
        per_feat_nll = mse/vals
        # Sample conditionally
        hidden     = self.get_hidden(X[:,:T_condition-1], A[:,:T_condition-1], B)
        inp_x_post = self.sample(T_forward+1, X[:,T_condition-1:], A[:,T_condition-1:], B, hidden = hidden)
        inp_x_post = torch.cat([X[:,:T_condition], inp_x_post[:,1:]], 1)
        # Sample forward unconditionally
        inp_x = self.sample(T_forward, X, A, B, hidden = None)
        inp_x = torch.cat([X[:,[0]], inp_x[:,1:]], 1)
        empty = torch.ones(X.shape[0], 3)
        return nll, per_feat_nll, empty, empty, inp_x_post, inp_x
    
    def predict(self,**kwargs):
        raise NotImplemented()

    @staticmethod
    def add_model_specific_args(parent_parser): 
        parser = ArgumentParser(parents = [parent_parser], add_help=False)

        parser.add_argument('--dim_hidden', type=int, default=250, help='hidden dimension for nonlinear model')
        parser.add_argument('--mtype', type=str, default='gru', help='transition function in RNN')
        parser.add_argument('--utype', type=str, default='linear', help='effect of treatments for RNN')
        parser.add_argument('--dropout', type=float, default=0.)
        parser.add_argument('--C', type=float, default=.1, help='regularization strength')
        parser.add_argument('--reg_all', type=bool, default=True, help='regularize all weights or only subset')    
        parser.add_argument('--reg_type', type=str, default='l1', help='regularization type')
        parser.add_argument('--alpha1_type', type=str, default='linear', help='alpha1 parameterization in TreatExp IEF')
        parser.add_argument('--otype', type=str, default='linear', help='final layer of GroMOdE IEF (linear, identity, nl)')

        return parser 

class GRULayer(nn.Module):#jit.ScriptModule):
    def __init__(self, cell, *cell_args):
        super(GRULayer, self).__init__()
        
        self.cell = cell(*cell_args)
    #@jit.script_method
    def forward(self, input, state=torch.tensor(-1)):
        # type: (Tensor) -> Tuple[Tensor, Tensor]
        input   = input.permute(1,0,2)
        inputs  = input.unbind(0)
        if len(state.size()) == 0: 
            state   = self.cell.get_hidden(input.shape[1])
        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            state = self.cell(inputs[i], state)
            outputs += [state]
        res = torch.stack(outputs)
        return res.permute(1,0,2), state

class PKPD_GRU(nn.Module):#jit.ScriptModule):
    __constants__ = ['dim_data','dim_treat','dim_base','dim_hidden', 'dim_subtype']
    def __init__(self, dim_data, dim_treat, dim_base, dim_hidden, dim_subtype=-1):
        super(PKPD_GRU, self).__init__()
        
        self.dim_data    = dim_data
        self.dim_treat   = dim_treat
        self.dim_base    = dim_base
        self.dim_hidden  = dim_hidden
        self.dim_subtype = dim_subtype
        if self.dim_subtype == -1: 
            self.x2h  = nn.Linear(dim_data+dim_treat+dim_base,  2 * dim_hidden, bias=True)
            self.te2h = nn.Linear(dim_data+dim_treat+dim_base, 3 * dim_hidden, bias=True)
            self.te   = GatedTransition(dim_data+dim_treat+dim_base, dim_treat+dim_base, response_only = True)
        else: 
            self.x2h  = nn.Linear(dim_data+dim_treat+dim_base+dim_subtype,  2 * dim_hidden, bias=True)
            self.te2h = nn.Linear(dim_data+dim_treat+dim_base+dim_subtype, 3 * dim_hidden, bias=True)
            self.te   = GatedTransition(dim_data+dim_treat+dim_base+dim_subtype, dim_treat+dim_base, dim_subtype=dim_subtype, response_only = True)

        self.h2h  = nn.Linear(dim_hidden,  3 * dim_hidden, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.dim_hidden)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    
    def get_hidden(self, bs):
        return torch.zeros(bs, self.dim_hidden).to(self.x2h.weight.device)
    
    #@jit.script_method
    def forward(self, x, hidden):
        # type: (Tensor, Tensor) -> Tensor 
        # x now consists of [X, A, B, Z]
        X = x[...,:self.dim_data]
        A = x[...,self.dim_data:self.dim_data+self.dim_treat]
        B = x[...,self.dim_data+self.dim_treat:self.dim_data+self.dim_treat+self.dim_base]
        if self.dim_subtype != -1: 
            Z = x[...,self.dim_data+self.dim_treat+self.dim_base:]
            assert Z.shape[-1] == self.dim_subtype
            x = torch.cat([X,A,B,Z], -1)
        else: 
            x = torch.cat([X,A,B], -1)
        x    = x.view(-1, x.size(1))
        gate_h = self.h2h(hidden)
        gate_h = gate_h.squeeze()
        h_r, h_i, h_n = gate_h.chunk(3, 1)
        con  = torch.cat([A[...,[0]],B,A[...,1:]],-1)
        te   = self.te(x, con)
        te_h   = self.te2h(te)
        te_h   = te_h.view(-1, te_h.size(1))
        te_r, te_i, te_n = te_h.chunk(3,1)
        resetgate = torch.sigmoid(te_r + h_r)
        inputgate = torch.sigmoid(te_i + h_i)
        newgate   = torch.tanh(te_n + (resetgate * h_n))
        hy = newgate + inputgate * (hidden - newgate)
        
        return hy