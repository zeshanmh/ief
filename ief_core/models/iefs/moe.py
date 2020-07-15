import torch
import math
import numpy as np
import pickle 
from torch import nn
import torch.functional as F
from pyro.distributions import Normal, Independent, Categorical, LogNormal
import sys, os
from torch.autograd import grad

class MofE(nn.Module): 
    def __init__(self, dim_stochastic, dim_treat, dim_hidden = 300, dim_output = -1, dim_subtype = -1, dim_input = -1, use_te = False, \
                 response_only = False, avoid_init = False, otype = 'lin', num_experts=10, eclass='lin'): 
        super(MofE, self).__init__()
        self.response_only = response_only
        if dim_input == -1: 
            dim_input  = dim_stochastic
        if dim_output == -1: 
            dim_output = dim_stochastic
        self.num_experts      = num_experts
        self.dim_output       = dim_output
        self.dim_subtype      = dim_subtype
        self.dim_input        = dim_input
        self.dim_treat        = dim_treat

        if not self.response_only:
            self.linear_layer = nn.Parameter(torch.eye(dim_stochastic))#nn.Linear(dim_stochastic, dim_stochastic)
            if otype == 'lin': 
                self.out_layer    = nn.Linear(dim_stochastic, dim_output) 
            elif otype == 'nl': 
                omodel            = nn.Sequential(nn.Linear(dim_stochastic, dim_hidden),nn.ReLU(True))
                self.out_layer    = nn.Sequential(omodel, nn.Linear(dim_hidden, dim_output)) 

        if eclass == 'lin':
            self.experts  = nn.ModuleList([nn.Linear(dim_treat, dim_output) for _ in range(self.num_experts)])
        elif eclass == 'nl': 
            el = []
            for _ in range(self.num_experts): 
                omodel = nn.Sequential(nn.Linear(dim_treat, dim_hidden), nn.ReLU(True))
                mlp    = nn.Sequential(omodel, nn.Linear(dim_hidden, dim_output))
                el.append(mlp)
            self.experts  = nn.ModuleList(el)

        self.inp_layer     = nn.Linear(dim_input, dim_stochastic)
        if self.dim_subtype == -1: 
            self.alphas       = nn.Parameter(torch.Tensor(dim_stochastic, self.num_experts))
        self.reset_parameters()

    def reset_parameters(self):
        if self.dim_subtype == -1:  
            # alphas not conditioned on subtype in this case
            nn.init.constant_(self.alphas, 1)

    def forward(self, inpx, con, eps=0.):
        inp        = self.inp_layer(inpx)
        outs = [self.experts[i](con) for i in range(self.num_experts)]
        if self.dim_subtype == -1: 
            probs      = torch.softmax(self.alphas, 1)
        else: 
            pass
        if self.dim_subtype != -1:
            pass
        else:
            probs      = probs[None,None,...]
        out = torch.sum(torch.stack([probs[...,i]*outs[i] for i in range(self.num_experts)],dim=-1),dim=-1)
        if self.response_only:
            return out
        else: 
            return self.out_layer(torch.matmul(inp, self.linear_layer)+out)