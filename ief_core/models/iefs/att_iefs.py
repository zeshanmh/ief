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
from models.multi_head_att import MultiHeadedAttention

def te_matrix():
    te_matrix = np.array([[-1, 0, 1, -1, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], 
                          [-1, 0, 1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], 
                          [-1, 0, 1, -1, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], 
                          [-1, 0, 0, -1, 0, 1,  1,  0,  0,  1,  0, -1,  0,  0,  0,  0], 
                          [-1, 0, 1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])
    return te_matrix

class AttentionIEFTransition(nn.Module):
    def __init__(self, dim_stochastic, 
                       dim_treat, 
                       dim_hidden = 300, 
                       dim_output = -1, 
                       dim_subtype = -1, 
                       dim_input = -1, 
                       use_te = False, 
                       response_only = False, 
                       avoid_init = False, 
                       dataset = 'mm', 
                       otype = 'linear', 
                       alpha1_type = 'linear', 
                       add_stochastic = False, 
                       num_heads = 1):
        super(AttentionIEFTransition, self).__init__()
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
            self.linear_layer = nn.Parameter(torch.eye(dim_stochastic))#nn.Linear(dim_stochastic, dim_stochastic) # take this out
            if otype == 'linear': 
                self.out_layer    = nn.Linear(dim_stochastic, dim_output) 
            elif otype == 'nl': 
                omodel            = nn.Sequential(nn.Linear(dim_stochastic, dim_hidden),nn.ReLU(True))
                self.out_layer    = nn.Sequential(omodel, nn.Linear(dim_hidden, dim_output)) 
            elif otype == 'identity': # useful for FOMM
                self.out_layer    = nn.Sequential()
        # self.control_layers   = nn.ModuleList([nn.Linear(dim_treat, dim_stochastic) for _ in range(2)]) 
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
        self.attn = MultiHeadedAttention(num_heads, dim_stochastic)
        self.logcell      = LogCellKill(dim_stochastic, dim_treat, mtype='logcellkill_1', response_only = True, alpha1_type=alpha1_type)
        self.treatment_exp= TreatmentExponential(dim_stochastic, dim_treat, response_only = True, alpha1_type=alpha1_type, add_stochastic=add_stochastic)
        
    def forward(self, inpx, con, eps=0.):
        inp        = self.inp_layer(inpx)
        # out_linears= [torch.tanh(l(con))[...,None] for l in self.control_layers]
        # out_te     = [t(inp,con,eps=eps)[...,None] for t in self.treatment_exps]
        out_linear = inp*torch.tanh(self.control_layer(con))
        out_te     = self.treatment_exp(inp, con, eps=eps)
        out_logcell= self.logcell(inp, con)
        # f   = tuple(out_linears + [out_te, out_logcell])
        value = torch.cat((out_linear[...,None], out_te[...,None], out_logcell[...,None]), dim=-1).transpose(-2,-1)
        key   = torch.cat((out_linear[...,None], out_te[...,None], out_logcell[...,None]), dim=-1).transpose(-2,-1)
        query = inp[...,None,:]
        out   = self.attn.forward(query, key, value, use_matmul=False).squeeze()
#         if len(out.shape) == 2: 
#             out = out[:,None,:]
        if self.response_only:
            return out
        else: 
            return self.out_layer(torch.matmul(inp, self.linear_layer)+out)

