import torch
import math
import numpy as np
import pickle 
from torch import nn
import torch.functional as F
from pyro.distributions import Normal, Independent, Categorical, LogNormal
import sys, os
from torch.autograd import grad
from models.iefs.logcellkill import LogCellKill
from models.iefs.treatexp import TreatmentExponential

def te_matrix():
    te_matrix = np.array([[-1, 0, 1, -1, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], 
                          [-1, 0, 1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], 
                          [-1, 0, 1, -1, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], 
                          [-1, 0, 0, -1, 0, 1,  1,  0,  0,  1,  0, -1,  0,  0,  0,  0], 
                          [-1, 0, 1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])
    return te_matrix

   
class GatedTransition(nn.Module):
    def __init__(self, dim_stochastic, dim_treat, dim_hidden = 300, dim_output = -1, dim_subtype = -1, dim_input = -1, use_te = False, \
                 response_only = False, avoid_init = False, dataset = 'mm', otype = 'linear', alpha1_type = 'linear', add_stochastic=False):
        super(GatedTransition, self).__init__()
        self.response_only    = response_only
        if dim_input == -1: 
            dim_input = dim_stochastic
        if dim_output == -1: 
            dim_output = dim_stochastic
        self.dim_output       = dim_output
        self.dim_subtype      = dim_subtype
        self.dim_input        = dim_input
        self.dim_treat        = dim_treat
        if not self.response_only:
            self.linear_layer = nn.Parameter(torch.eye(dim_stochastic))#nn.Linear(dim_stochastic, dim_stochastic)
            if otype == 'linear': 
                self.out_layer    = nn.Linear(dim_stochastic, dim_output) 
            elif otype == 'nl': 
                omodel            = nn.Sequential(nn.Linear(dim_stochastic, dim_hidden),nn.ReLU(True))
                self.out_layer    = nn.Sequential(omodel, nn.Linear(dim_hidden, dim_output)) 
            elif otype == 'identity': # useful for FOMM
                self.out_layer    = nn.Sequential()
        self.control_layer    = nn.Linear(dim_treat, dim_stochastic)
        self.inp_layer        = nn.Linear(dim_input, dim_stochastic)
        if not avoid_init:
            if use_te: # only use
                prior_weight_15   = te_matrix()
                self.control_layer.weight.data[:,-8:-3] = self.control_layer.weight.data[:,-8:-3]*0.+torch.tensor(prior_weight_15).T
            else:
                te = te_matrix()
                prior_weight      = torch.tensor(te.astype('float32')) 
                rand_w            = torch.randn(16, dim_stochastic)*0.1
                pw                = torch.matmul(prior_weight, rand_w)
                self.control_layer.weight.data[:,-8:-3] = self.control_layer.weight.data[:,-8:-3]*0.+pw.T
        if dim_subtype != -1: 
            self.l_alphas       = nn.Linear(dim_subtype, dim_stochastic)
            self.lc_alphas      = nn.Linear(dim_subtype, dim_stochastic)
            self.te_alphas      = nn.Linear(dim_subtype, dim_stochastic)
        else: 
            self.alphas       = nn.Parameter(torch.Tensor(dim_stochastic,3))
        self.logcell      = LogCellKill(dim_stochastic, dim_treat, mtype='logcellkill_1', response_only = True, alpha1_type=alpha1_type)
        self.treatment_exp= TreatmentExponential(dim_stochastic, dim_treat, response_only = True, alpha1_type=alpha1_type, add_stochastic=add_stochastic)
        self.reset_parameters()
    
    def reset_parameters(self):
        if self.dim_subtype == -1:  
            # alphas not conditioned on subtype in this case
            nn.init.constant_(self.alphas, 1)
        
    def forward(self, inpx, con, eps=0.):
        inp        = self.inp_layer(inpx)
        out_linear = inp*torch.tanh(self.control_layer(con))
        out_te     = self.treatment_exp(inp, con, eps=eps)
        out_logcell= self.logcell(inp, con)
        
        if self.dim_subtype == -1: 
            probs      = torch.softmax(self.alphas, 1)
        else: 
            if inpx.dim() == 3: 
                a_inp  = inpx[:,[-1],-self.dim_subtype:]
            elif inpx.dim() == 2: 
                a_inp  = inpx[:,-self.dim_subtype:]     
            alphas_cat = torch.cat([self.l_alphas(a_inp)[...,None], self.lc_alphas(a_inp)[...,None], \
                self.te_alphas(a_inp)[...,None]], -1)
            probs      = torch.softmax(alphas_cat,-1)
        
        if self.dim_subtype != -1:
            pass 
        elif out_linear.dim()==3:
            probs  = probs[None,None,...]
        elif out_linear.dim()==2:
            probs  = probs[None,...]
        out        = probs[...,0]*out_linear + probs[...,1]*out_logcell+ probs[...,2]*out_te
        if self.response_only:
            return out
        else: 
            return self.out_layer(torch.matmul(inp, self.linear_layer)+out)