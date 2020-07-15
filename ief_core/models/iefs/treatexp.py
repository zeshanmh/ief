import torch
import math
import numpy as np
import pickle 
from torch import nn
import torch.functional as F
from pyro.distributions import Normal, Independent, Categorical, LogNormal
import sys, os
from torch.autograd import grad

class TreatmentExponential(nn.Module): 
    def __init__(self, dim_stochastic, dim_treat, dim_hidden=300, response_only=False, alpha1_type='linear'):
        super(TreatmentExponential, self).__init__() 
        self.dim_stochastic   = dim_stochastic 
        self.dim_treat        = dim_treat 
        self.b0    = nn.Parameter(torch.Tensor(dim_stochastic,))
        self.b1    = nn.Parameter(torch.Tensor(dim_stochastic,))
        self.pred_prms = nn.Linear(3, 3)
        
        if alpha1_type == 'linear': 
            self.alpha_1_layer    = nn.Linear(self.dim_treat, self.dim_stochastic)
        elif alpha1_type == 'nl': 
            omodel             = nn.Sequential(nn.Linear(dim_treat, dim_hidden),nn.ReLU(True))
            self.alpha_1_layer = nn.Sequential(omodel, nn.Linear(dim_hidden, dim_stochastic)) 
        elif alpha1_type == 'linear_fix': 
            self.alpha_1_layer = nn.Linear(self.dim_treat, 1)
            
        self.response_only    = response_only
        if not response_only:
            self.inpfxn           = nn.Linear(dim_stochastic, dim_stochastic)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.constant_(self.b0, 1)
        nn.init.constant_(self.b1, 1)
    
    def forward(self, inp, con, eps=0.):
        tvals = con[...,[0]]
        a     = con[...,1:]
        tmax_lot                = ((tvals*con[...,-3:]).max(1, keepdims=True)[0]*con[...,-3:]).sum(-1, keepdims=True)
        pred  = self.pred_prms(con[...,-3:])
        alpha_2, alpha_3, gamma = torch.sigmoid(pred[...,[0]]), torch.sigmoid(pred[...,[1]]), torch.sigmoid(pred[...,[2]])*tmax_lot
        
        b_0, b_1= self.b0, self.b1
        alpha_1 = self.alpha_1_layer(con)+eps
        alpha_0 = (alpha_1+2*b_0-b_1)/(1+torch.exp(-alpha_3*0.5*gamma))
        
        mask    = (tvals-gamma).detach()
        mask[mask<=0]= 0
        mask[mask>0] = 1
        g       = (1-mask)*(b_0+alpha_1/(1+torch.exp(-alpha_2*(tvals-0.5*gamma))))+mask*(b_1 + alpha_0/(1+torch.exp(alpha_3*(tvals-1.5*gamma))))
        if self.response_only:
            return g
        else:
            return self.inpfxn(inp) + g