import torch 
import sys, math
import numpy as np
import torch.jit as jit
import warnings 
import torch.nn.init as init
import torch.nn as nn

from distutils.util import strtobool
from models.base import Model
from models.utils import *
from models.multi_head_att import MultiHeadedAttention
from models.ssm.inference import RNN_STInf, Attention_STInf
from models.iefs.gated import GatedTransition
from models.iefs.att_iefs import AttentionIEFTransition
from models.iefs.moe import MofE
from models.ssm.ssm import *
from pyro.distributions import Normal, Independent, Categorical, LogNormal
from typing import List, Tuple
from torch import Tensor
from collections import namedtuple
from typing import List, Tuple
from torch.autograd import Variable
from argparse import ArgumentParser


class SSMBaseline(SSM): 
    ''' 
        Implementation of Ahmed Alaa and Mihaela's paper 
        "Attentive State Space Modeling Disease Progression"
    ''' 

    def __init__(self, trial, **kwargs): 
        super(SSMBaseline, self).__init__(trial)
        self.save_hyperparameters()


    def init_model(self): 
        ttype       = self.hparams['ttype']; etype = self.hparams['etype']
        dim_hidden  = self.hparams['dim_hidden']
        # dim_stochastic = self.hparams['dim_stochastic']
        dim_stochastic = self.trial.suggest_categorical('dim_stochastic',[16,48])
        num_heads   = self.hparams['nheads']
        dim_data    = self.hparams['dim_data']
        dim_base    = self.hparams['dim_base']
        dim_treat   = self.hparams['dim_treat']
        post_approx = self.hparams['post_approx']
        inftype     = self.hparams['inftype']; etype = self.hparams['etype']; ttype = self.hparams['ttype']
        augmented   = self.hparams['augmented']; alpha1_type = self.hparams['alpha1_type']
        rank        = self.hparams['rank']; combiner_type = self.hparams['combiner_type']; nheads = self.hparams['nheads']
        add_stochastic = self.hparams['add_stochastic']

        # Inference Network
        if inftype == 'rnn':
            self.inf_network    = RNN_STInf(self.trial, dim_base, dim_data, dim_treat, dim_hidden, dim_stochastic, post_approx = post_approx, rank = rank, combiner_type = combiner_type)
        elif inftype == 'rnn_bn':
            self.inf_network    = RNN_STInf(self.trial, dim_base, dim_data, dim_treat, dim_hidden, dim_stochastic, post_approx = post_approx, rank = rank, use_bn=True, combiner_type = combiner_type)
        elif inftype == 'rnn_relu':
            self.inf_network    = RNN_STInf(self.trial, dim_base, dim_data, dim_treat, dim_hidden, dim_stochastic, post_approx = post_approx, rank = rank, nl='relu', combiner_type = combiner_type)
        elif inftype == 'att':
            self.inf_network    = Attention_STInf(dim_base, dim_data, dim_treat, dim_hidden, dim_stochastic, nheads = num_heads, post_approx = post_approx, rank = rank)
        else:
            raise ValueError('Bad inference type')

        # Emission Function
        if etype == 'lin':
            self.e_mu    = nn.Linear(dim_stochastic, dim_data)
            self.e_sigma = nn.Linear(dim_stochastic, dim_data)
        elif etype  == 'nl':
            dim_hidden   = self.trial.suggest_int('dim_hidden',100,500)
            emodel       = nn.Sequential(nn.Linear(dim_stochastic, dim_hidden), nn.ReLU(True))
            self.e_mu    = nn.Sequential(emodel, nn.Linear(dim_hidden, dim_data))
            self.e_sigma = nn.Sequential(emodel, nn.Linear(dim_hidden, dim_data))
        else:
            raise ValueError('bad etype')

        # Transition Function
        if self.hparams['include_baseline']:
            self.transition_fxn = TransitionFunction(dim_stochastic, dim_data, dim_treat+dim_base, dim_hidden, ttype, \
                augmented=augmented, alpha1_type=alpha1_type, add_stochastic=add_stochastic, num_heads=num_heads)                
        else:
            self.transition_fxn = TransitionFunction(dim_stochastic, dim_data, dim_treat, dim_hidden, ttype, \
                augmented=augmented, alpha1_type=alpha1_type, add_stochastic=add_stochastic, num_heads=num_heads)
        
        # Prior over Z1
        self.prior_W        = nn.Linear(dim_treat+dim_data+dim_base, dim_stochastic)
        self.prior_sigma    = nn.Linear(dim_treat+dim_data+dim_base, dim_stochastic)

        # Attention 
        self.attn     = MultiHeadedAttention(num_heads, dim_stochastic)
        self.attn_lin = nn.Linear(dim_data+dim_treat, dim_stochastic)

    def p_Zt_Ztm1(self, Zt, A, B, X, A0, Am, eps = 0.):
        X0 = X[:,0,:]; Xt = X[:,1:,:]
        inp_cat  = torch.cat([B, X0, A0], -1)
        mu1      = self.prior_W(inp_cat)[:,None,:]
        sig1     = torch.nn.functional.softplus(self.prior_sigma(inp_cat))[:,None,:]
        
        Tmax     = Zt.shape[1]
        if self.hparams['augmented']: 
            Zinp = torch.cat([Zt[:,:-1,:], Xt[:,:-1,:]], -1)
        else: 
            Zinp = Zt[:,:-1,:]
        Aval = A[:,1:Tmax,:]
        sub_mask = np.triu(np.ones((Aval.shape[0],Aval.shape[1],Aval.shape[1])), k=1).astype('uint8')
        Zm = (torch.from_numpy(sub_mask) == 0).to(Am.device)
        res  = self.attn(self.attn_lin(torch.cat([Xt[:,:-1,:],Aval],-1)), Zinp, Zinp, mask=Zm, use_matmul=True)
        if self.hparams['include_baseline']:
            Acat = torch.cat([Aval[...,[0]],B[:,None,:].repeat(1,Aval.shape[1],1), Aval[...,1:]],-1)
            mu2T, sig2T = self.transition_fxn(res, Acat, eps = eps)
        else:
            mu2T, sig2T = self.transition_fxn(res, A[:,1:Tmax,:], eps = eps)
        mu, sig     = torch.cat([mu1,mu2T],1), torch.cat([sig1,sig2T],1)
        return Independent(Normal(mu, sig), 1)

    def get_loss(self, B, X, A, M, Y, CE, Am, anneal = 1., return_reconstruction = False, with_pred = False):
        _, _, lens         = get_masks(M)
        B, X, A, M, Y, CE, Am = B[lens>1], X[lens>1], A[lens>1], M[lens>1], Y[lens>1], CE[lens>1], Am[lens>1]
        m_t, m_g_t, _      = get_masks(M[:,1:,:])
        Z_t, q_zt          = self.inf_network(X, A, M, B)
        Tmax               = Z_t.shape[1]
        p_x_mu, p_x_std    = self.p_X_Z(Z_t, A[:,1:Tmax+1,[0]])
        p_zt               = self.p_Zt_Ztm1(Z_t, A, B, X, A[:,0,:], Am)
        masked_nll         = masked_gaussian_nll_3d(X[:,1:Tmax+1,:], p_x_mu, p_x_std, M[:,1:Tmax+1,:])
        full_masked_nll    = masked_nll
        masked_nll         = masked_nll.sum(-1).sum(-1)
    
        if with_pred:
            p_x_mu_pred, p_x_std_pred = self.p_X_Z(p_zt.mean, A[:,:Z_t.shape[1],[0]])
            masked_nll_pred           = masked_gaussian_nll_3d(X[:,1:Tmax+1,:], p_x_mu_pred, p_x_std_pred, M[:,1:Tmax+1,:])
            masked_nll_pred           = masked_nll_pred.sum(-1).sum(-1)
            masked_nll = (masked_nll+masked_nll_pred)*0.5
        kl_t       = q_zt.log_prob(Z_t)-p_zt.log_prob(Z_t)
        masked_kl_t= (m_t[:,:Tmax]*kl_t).sum(-1)
        neg_elbo   = masked_nll + anneal*masked_kl_t
    
        if return_reconstruction:
            return (neg_elbo, masked_nll, masked_kl_t, torch.ones_like(masked_kl_t), p_x_mu*M[:,1:,:], p_x_std*M[:,1:,:])
        else:
            return (neg_elbo, masked_nll, masked_kl_t, torch.ones_like(masked_kl_t))

    def forward(self, B, X, A, M, Y, CE, Am, anneal = 1.):
        if self.training:
            if self.hparams['elbo_samples']>1:
                B, X = torch.repeat_interleave(B, repeats=self.elbo_samples, dim=0), torch.repeat_interleave(X, repeats=self.elbo_samples, dim=0)
                A, M = torch.repeat_interleave(A, repeats=self.elbo_samples, dim=0), torch.repeat_interleave(M, repeats=self.elbo_samples, dim=0)
                Y, CE= torch.repeat_interleave(Y, repeats=self.elbo_samples, dim=0), torch.repeat_interleave(CE, repeats=self.elbo_samples, dim=0)
            neg_elbo, masked_nll, kl, _  = self.get_loss(B, X, A, M, Y, CE, Am, anneal = anneal, with_pred = True)
        else:
            neg_elbo, masked_nll, kl, _  = self.get_loss(B, X, A, M, Y, CE, Am, anneal = anneal, with_pred = False)
        reg_loss   = torch.mean(neg_elbo)
        
        for name,param in self.named_parameters():
            if self.reg_all:
                # reg_loss += self.hparams['C']*apply_reg(param, reg_type=self.hparams['reg_type'])
                reg_loss += self.C*apply_reg(param, reg_type=self.reg_type)
            else:
                if 'weight' in name:
                    reg_loss += self.C*apply_reg(param, reg_type=self.reg_type)
        loss = torch.mean(reg_loss)
        return (torch.mean(neg_elbo), torch.mean(masked_nll), torch.mean(kl), torch.ones_like(kl)), loss

    def forward_sample(self, A, T_forward, Z_start = None, B=None, X0=None, A0=None, eps = 0.):
        pass 

    def inspect(self, T_forward, T_condition, B, X, A, M, Y, CE, Am, restrict_lens = False, nsamples = 1, eps = 0.):
        pass

    def inspect_trt(self, B, X, A, M, Y, CE, Am, nsamples=3): 
        pass 


    #     # Transition Function
    #     self.P = nn.Parameter(torch.Tensor(dim_stochastic,dim_stochastic))
    #     self.attn_encoder = nn.LSTM(dim_data+dim_treat+dim_base, dim_hidden, batch_first=True, bidirectional=False)
    #     self.attn_decoder = nn.LSTMCell(dim_hidden, dim_hidden, batch_first=True, bidirectional=False)
    #     self.attn_lin     = nn.Linear(dim_hidden, dim_stochastic)

    # def p_Zt_Ztm1(self, Zt, A, B, X, A0, Am, M, eps = 0.):
    #     X0 = X[:,0,:]
    #     pz0 = torch.ones((Zt.shape[0],1,self.ds)) / self.ds
    #     Tmax   = Zt.shape[1]
        
    #     k  = torch.ones((Zt.shape[0],Tmax,self.ds))
    #     for t in range(1,Tmax+1): 
    #         base_cat = B[:,None,:].repeat(1, max(1, t), 1)
    #         Xt = X[:,1:t,:]; At = A[:,1:t,:]
    #         cat       = torch.cat([Xt, At, base_cat]) 
    #         _, _, lens= get_masks(M[:,1:t,:])
            
    #         pdseq      = torch.nn.utils.rnn.pack_padded_sequence(cat, lens, batch_first=True, enforce_sorted = False)
    #         out_pd, (h,c)  = self.attn_encoder(pdseq)
    #         out, out_lens  = torch.nn.utils.rnn.pad_packed_sequence(out_pd, batch_first=True)
            
    #         inp  = out[:,out_lens-1,:]

    #         attn = torch.ones((Zt.shape[0],t,self.ds))
    #         for tt in range(t): 
    #             h,c = self.attn_decoder(inp, (h,c))
    #             inp = h 
    #             attn[:,tt,:] = torch.nn.functional.softmax(self.attn_lin(h), dim = -1)

    #     p_zt = Independent(Categorical(probs))


