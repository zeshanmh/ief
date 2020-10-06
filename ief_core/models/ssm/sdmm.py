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
from models.ssm.ssm import TransitionFunction
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

class SDMM_InferenceNetwork(nn.Module):
    def __init__(self, dim_base, dim_data, dim_treat, dim_hidden, dim_stochastic, dim_subtype, post_approx = 'diag', \
                rank = 5, use_bn = False, nl = 'tanh', combiner_type = 'standard', bidirectional = False):
        super(SDMM_InferenceNetwork, self).__init__()
        self.dim_base   = dim_base
        self.dim_data   = dim_data
        self.dim_treat  = dim_treat
        self.dim_hidden = dim_hidden
        self.dim_stochastic = dim_stochastic
        self.dim_subtype    = dim_subtype 
        self.use_bn     = use_bn
        if self.use_bn:
            print ('using bn in inf. network')
            self.bn     = nn.LayerNorm(dim_hidden, elementwise_affine=False)
        if nl == 'relu':
            print ('using relu in inf. network')
            self.nonlinearity = torch.relu
        else:
            print ('using tanh in inf. network')
            self.nonlinearity = torch.tanh

        self.inf_rnn    = nn.GRU(dim_data+dim_treat, dim_hidden, 1, batch_first = True, bidirectional=bidirectional)
        self.hid_zg_b   = nn.Linear(dim_base+dim_data+dim_treat, dim_hidden)
        self.base_h1    = nn.Linear(dim_base+dim_data+dim_treat, dim_hidden)

        # combiner type, posterior approximation type, and rank
        self.combiner_type = combiner_type
        self.post_approx= post_approx
        self.rank       = rank
        
        if bidirectional:
            self.hid_zg = nn.Linear(dim_hidden*2, dim_hidden)
            self.hid_rnn_zt = nn.Linear(dim_hidden*2, dim_hidden)
        else:
            self.hid_zg = nn.Linear(dim_hidden, dim_hidden)
            self.hid_rnn_zt = nn.Linear(dim_hidden, dim_hidden)
        self.mu_zg      = nn.Linear(dim_hidden, dim_subtype)
        self.sigma_zg   = nn.Linear(dim_hidden, dim_subtype)
        
        self.hid_zg_zt  = nn.Linear(dim_subtype, dim_hidden)
        self.hid_ztm1_zt= nn.Linear(dim_stochastic, dim_hidden)
        
        if self.combiner_type == 'standard' or self.combiner_type == 'masked': 
            self.mu_z1      = nn.Linear(dim_hidden, dim_stochastic)
            self.mu_zt      = nn.Linear(dim_hidden, dim_stochastic) 
            if self.post_approx == 'diag': 
                self.sigma_z1   = nn.Linear(dim_hidden, dim_stochastic)
                self.sigma_zt   = nn.Linear(dim_hidden, dim_stochastic)
            elif self.post_approx == 'low_rank':
                self.sigma_z1   = nn.Linear(dim_hidden, (dim_stochastic*rank)+dim_stochastic)
                self.sigma_zt   = nn.Linear(dim_hidden, (dim_stochastic*rank)+dim_stochastic)
            else: 
                raise ValueError('bad setting for post_approx:'+str(post_approx))
        elif self.combiner_type == 'pog': 
            assert self.post_approx == 'diag','bad post_approx'
            self.mu_zt       = nn.Linear(dim_hidden, dim_stochastic)
            self.sigma_zt    = nn.Linear(dim_hidden, dim_stochastic)
            self.mu_zt2      = nn.Linear(dim_hidden, dim_stochastic)
            self.sigma_zt2   = nn.Linear(dim_hidden, dim_stochastic)
            self.mu_zt3      = nn.Linear(dim_hidden, dim_stochastic)
            self.sigma_zt3   = nn.Linear(dim_hidden, dim_stochastic)
        else:
            raise ValueError('Bad assignment to inference_type')

    def reparam_dist(self, mu, sigma):
        if self.post_approx == 'diag':
            dist = Independent(Normal(mu, sigma), 1)
        elif self.post_approx == 'low_rank':
            if sigma.dim()==2:
                W = sigma[...,self.dim_stochastic:].view(sigma.shape[0], self.dim_stochastic, self.rank)
            elif sigma.dim()==3:
                W = sigma[...,self.dim_stochastic:].view(sigma.shape[0], sigma.shape[1], self.dim_stochastic, self.rank)
            else:
                raise NotImplemented()
            D = sigma[...,:self.dim_stochastic]
            dist = LowRankMultivariateNormal(mu, W, D)
        else:
            raise ValueError('should not be here')
        return torch.squeeze(dist.rsample((1,))), dist

    def pogN(self, muN, sigN): 
        sigsqN   = [sig.pow(2)+1e-8 for sig in sigN]
        sigsqNm1 = [np.prod(sigsqN[:i] + sigsqN[i+1:]) for i,sig in enumerate(sigsqN)]
        sigmasq  = np.prod(sigsqN) / np.sum(sigsqNm1)
        muDsigsq = [muN[i] / sigsqN[i] for i in range(len(muN))]
        mu       = (np.sum(muDsigsq))*sigmasq
        sigma    = sigmasq.pow(0.5)
        return mu, sigma

    def combiner_fxn(self, prev_hid, current_hid, rnn_mask, mu1fxn, sig1fxn, mu2fxn = None, \
                        sig2fxn = None, mu3fxn = None, sig3fxn = None, global_hid = None):
        if self.combiner_type   =='standard' or self.combiner_type == 'masked':
            if self.combiner_type == 'standard':
                if global_hid is not None: 
                    out = 1/3.*(prev_hid+current_hid+global_hid)
                else: 
                    out = 0.5*(prev_hid+current_hid)
            else:
                if global_hid is not None: 
                    out = rnn_mask*(1/3.*(prev_hid+current_hid+global_hid)) + (1-rnn_mask)*prev_hid
                else: 
                    out = rnn_mask*(0.5*(prev_hid+current_hid)) + (1-rnn_mask)*prev_hid
            if self.use_bn:
                h1         = self.nonlinearity(self.bn(out))
            else:
                h1         = self.nonlinearity(out)
            mu, sigma  = mu1fxn(h1), torch.nn.functional.softplus(sig1fxn(h1))
        elif self.combiner_type == 'pog':
            if self.use_bn:
                h1         = self.nonlinearity(self.bn(prev_hid))
                h2         = self.nonlinearity(self.bn(current_hid))
                if global_hid is not None: 
                    h3         = self.nonlinearity(self.bn(global_hid))
            else:
                h1         = self.nonlinearity(prev_hid)
                h2         = self.nonlinearity(current_hid)
                if global_hid is not None: 
                    h3         = self.nonlinearity(global_hid)
            mu1, sig1  = mu1fxn(h1), torch.nn.functional.softplus(sig1fxn(h1))
            mu2, sig2  = mu2fxn(h2), torch.nn.functional.softplus(sig2fxn(h2))
            muN = [mu1, mu2]; sigN = [sig1, sig2]
            if global_hid is not None: 
                mu3, sig3  = mu3fxn(h3), torch.nn.functional.softplus(sig3fxn(h3))
                muN.append(mu3); sigN.append(sig3)
            mu, sigma = self.pogN(muN, sigN)
        else:
            raise ValueError('bad combiner type')
        return mu, sigma
            
    def forward(self, x, a, m, b):
        rnn_mask        = (m[:,1:].sum(-1)>1)*1.
        inp             = torch.cat([x[:,1:,:], a[:,:-1,:]], -1)
        m_t, _, lens    = get_masks(m[:,1:,:])
        pdseq     = torch.nn.utils.rnn.pack_padded_sequence(inp, lens, batch_first=True, enforce_sorted = False)
        out_pd, _ = self.inf_rnn(pdseq)
        out, _    = torch.nn.utils.rnn.pad_packed_sequence(out_pd, batch_first=True)
        
        # Infer global latent variable 
        hid_zg   = torch.tanh(self.hid_zg(out).sum(1)/lens[...,None] + self.hid_zg_b(torch.cat([b, x[:,0,:], a[:,0,:]],-1)))
        zg_mu    = self.mu_zg(hid_zg)
        zg_sigma = torch.nn.functional.softplus(self.sigma_zg(hid_zg))
        q_zg     = Independent(Normal(zg_mu, zg_sigma), 1)
        Z_g      = torch.squeeze(q_zg.rsample((1,)))
        
        # Infer per-time-step variables in the DMM
        hid_zg_zt  = self.hid_zg_zt(Z_g)
        hid_rnn_zt = self.hid_rnn_zt(out)
        hid_base   = self.base_h1(torch.cat([x[:,0,:], b, a[:,0,:]],-1)) ## test this out
        if self.combiner_type == 'standard' or self.combiner_type == 'masked': 
            mu, sigma = self.combiner_fxn(hid_base, hid_rnn_zt[:,0,:], rnn_mask[:,[0]], self.mu_z1, self.sigma_z1, global_hid=hid_zg_zt) # change to self.mu_zt, self.sigma_zt if necessary 
        else:
            mu, sigma = self.combiner_fxn(hid_base, hid_rnn_zt[:,0,:], rnn_mask[:,[0]], self.mu_zt, self.sigma_zt, \
                self.mu_zt2, self.sigma_zt2, self.mu_zt3, self.sigma_zt3, global_hid=hid_zg_zt)
        z, _     = self.reparam_dist(mu, sigma)

        meanlist = [mu[:,None,:]]
        sigmalist= [sigma[:,None,:]]
        zlist    = [z[:,None,:]]
        for t in range(1, out.shape[1]):
            ztm1       = torch.squeeze(zlist[t-1])
            hid_ztm1_zt= self.hid_ztm1_zt(ztm1)

            if self.combiner_type == 'standard' or self.combiner_type == 'masked': 
                mu, sigma = self.combiner_fxn(hid_ztm1_zt, hid_rnn_zt[:,t,:], rnn_mask[:,[t]], self.mu_zt, self.sigma_zt, global_hid = hid_zg_zt)
            else: 
                mu, sigma = self.combiner_fxn(hid_ztm1_zt, hid_rnn_zt[:,t,:], rnn_mask[:,[t]], self.mu_zt, self.sigma_zt, \
                    self.mu_zt2, self.sigma_zt2, self.mu_zt3, self.sigma_zt3, global_hid = hid_zg_zt)
            z, _       = self.reparam_dist(mu, sigma)

            meanlist  += [mu[:,None,:]]
            sigmalist += [sigma[:,None,:]]
            zlist     += [z[:,None,:]]
        # q_zt     = Independent(Normal(torch.cat(meanlist, 1), torch.cat(sigmalist, 1)), 1)
        _,q_zt     = self.reparam_dist(torch.cat(meanlist, 1), torch.cat(sigmalist, 1))
        Z_t      = torch.cat(zlist, 1)
        return Z_g, q_zg, Z_t, q_zt

class TransitionFxnSDMM(TransitionFunction): 
    def __init__(self, dim_stochastic, dim_subtype, dim_data, dim_treat, dim_hidden, ttype, dim_base=0, augmented=False, alpha1_type='linear', otype='linear', add_stochastic=False):
        super(TransitionFxnSDMM, self).__init__(dim_stochastic, dim_data, dim_treat, dim_hidden, ttype, augmented)
        self.augmented = augmented
        dim_treat  = dim_treat + dim_base

        if self.augmented: 
            dim_input = dim_stochastic+dim_data
        else: 
            dim_input = dim_stochastic
        pre_mu_sig = ['lin', 'contractive', 'monotonic', 'logcell', 'logcellkill', 'treatment_exp', 'gated', 'syn_trt']
        if self.ttype in pre_mu_sig: 
            self.pre_t_mu           = nn.Linear(dim_subtype, dim_stochastic)
            self.pre_t_sigma        = nn.Linear(dim_subtype, dim_stochastic)

        if self.ttype   == 'lin':
            self.t_mu               = nn.Linear(dim_input+dim_subtype+dim_treat, dim_stochastic) 
            self.t_sigma            = nn.Linear(dim_input+dim_subtype+dim_treat, dim_stochastic)
        elif self.ttype == 'nl':
            self.pre_t_mu           = nn.Sequential(nn.Linear(dim_subtype, dim_hidden), nn.ReLU(True), nn.Linear(dim_hidden, dim_stochastic))
            self.pre_t_sigma        = nn.Sequential(nn.Linear(dim_subtype, dim_hidden), nn.ReLU(True), nn.Linear(dim_hidden, dim_stochastic))
            tmodel                  = nn.Sequential(nn.Linear(dim_input+dim_subtype+dim_treat, dim_hidden),nn.ReLU(True))
            self.t_mu               = nn.Sequential(tmodel, nn.Linear(dim_hidden, dim_stochastic)) 
            self.t_sigma            = nn.Sequential(tmodel, nn.Linear(dim_hidden, dim_stochastic))
        elif self.ttype == 'monotonic':
            self.t_mu               = MonotonicLayer(dim_stochastic, dim_treat, sign='positive')
            self.t_sigma            = nn.Linear(dim_stochastic+dim_treat, dim_stochastic)
        elif self.ttype == 'logcell':
            self.t_mu               = LogCellTransition(dim_stochastic, dim_treat)
            self.t_sigma            = nn.Linear(dim_stochastic+dim_treat, dim_stochastic)
        elif 'logcellkill' in self.ttype:
            self.t_mu               = LogCellKill(dim_stochastic, dim_treat, mtype=self.ttype)
            self.t_sigma            = nn.Linear(dim_stochastic+dim_treat, dim_stochastic)
        elif self.ttype=='treatment_exp':
            self.t_mu               = TreatmentExponential(dim_stochastic, dim_treat)
            self.t_sigma            = nn.Linear(dim_stochastic+dim_treat, dim_stochastic)
        elif self.ttype=='gated':
            avoid_init = False
            if dim_data != 16:
                avoid_init = True
            self.t_mu               = GatedTransition(dim_stochastic, dim_treat, dim_hidden=dim_hidden, dim_subtype=dim_subtype, \
                                        dim_input=dim_input+dim_subtype+dim_treat, avoid_init = avoid_init, otype=otype, alpha1_type=alpha1_type, add_stochastic=add_stochastic)
            tmodel                  = nn.Sequential(nn.Linear(dim_input+dim_subtype+dim_treat, dim_hidden),nn.ReLU(True))
            self.t_sigma            = nn.Sequential(tmodel, nn.Linear(dim_hidden, dim_stochastic))
            # self.t_sigma            = nn.Linear(dim_subtype+dim_input+dim_treat, dim_stochastic)
        elif self.ttype=='debug':
            self.t_mu               = Debug(dim_stochastic, dim_treat, dim_base)
            self.t_sigma            = nn.Linear(dim_stochastic+dim_treat, dim_stochastic)
        elif self.ttype=='syn_trt': 
            self.t_mu               = SyntheticTrtTransition(dim_stochastic, dim_treat)
            self.t_sigma            = nn.Linear(dim_stochastic+dim_treat, dim_stochastic)
        else:
            raise ValueError('bad ttype')
    
    def apply(self, fxn, z, u, eps=0.):
        if 'Monotonic' in fxn.__class__.__name__ or 'LogCellTransition' in fxn.__class__.__name__ or 'LogCellKill' in fxn.__class__.__name__ or 'TreatmentExp' in fxn.__class__.__name__ or 'GatedTransition' in fxn.__class__.__name__ or 'Synthetic' in fxn.__class__.__name__:
            return fxn(z, u)
        else:
            return fxn(z)
           
    def get_prior_global(self): 
        return self.pre_t_mu, self.pre_t_sigma

class SDMM(Model):
    def __init__(self, trial, **kwargs): 
        super(SDMM, self).__init__(trial)
        self.save_hyperparameters() 

    def init_model(self): 
        dim_subtype     = self.trial.suggest_categorical('dim_subtype',[4,16,48])
        dim_stochastic  = self.trial.suggest_int('dim_stochastic',16,128)
        dim_hidden      = self.trial.suggest_int('dim_hidden',100,500)
        dim_base        = self.hparams.dim_base
        dim_data        = self.hparams.dim_data
        dim_treat       = self.hparams.dim_treat
        ttype           = self.hparams.ttype 
        etype           = self.hparams.etype 
        inftype         = self.hparams.inftype 
        post_approx     = self.hparams.post_approx
        self.include_baseline= self.hparams.include_baseline
        self.elbo_samples    = self.hparams.elbo_samples
        self.augmented       = self.hparams.augmented
        self.fixed_var = None
        alpha1_type = self.hparams.alpha1_type
        otype       = self.hparams.otype
        add_stochastic = self.hparams.add_stochastic
        rank           = self.hparams.rank; combiner_type = self.hparams.combiner_type


        # Inference network
        if inftype == 'rnn':
            self.inf_network    = SDMM_InferenceNetwork(dim_base, dim_data, dim_treat, dim_hidden, dim_stochastic, dim_subtype, \
                                        post_approx = post_approx, rank = rank, combiner_type = combiner_type)
        elif inftype=='rnn_bn':
            self.inf_network    = SDMM_InferenceNetwork(dim_base, dim_data, dim_treat, dim_hidden, dim_stochastic, dim_subtype, \
                                        post_approx = post_approx, rank = rank, use_bn = True, combiner_type = combiner_type)
        elif inftype=='rnn_relu':
            self.inf_network    = SDMM_InferenceNetwork(dim_base, dim_data, dim_treat, dim_hidden, dim_stochastic, dim_subtype, \
                                        post_approx = post_approx, rank = rank, nl = 'relu', combiner_type = combiner_type)
        elif inftype=='rnn_relu_bi':
            self.inf_network    = SDMM_InferenceNetwork(dim_base, dim_data, dim_treat, dim_hidden, dim_stochastic, dim_subtype, \
                                        post_approx = post_approx, rank = rank, nl = 'relu', combiner_type = combiner_type, bidirectional=True)
        else:
            raise ValueError('Bad inference type')

        # Emission function
        if etype   == 'lin':
            self.e_mu   = nn.Linear(dim_stochastic+dim_subtype, dim_data)
            self.e_sigma= nn.Linear(dim_stochastic+dim_subtype, dim_data)
        elif etype == 'nl':
            emodel      = nn.Sequential(nn.Linear(dim_stochastic+dim_subtype, dim_hidden), nn.ReLU(True))
            self.e_mu   = nn.Sequential(emodel, nn.Linear(dim_hidden, dim_data))
            self.e_sigma= nn.Sequential(emodel, nn.Linear(dim_hidden, dim_data))
        elif etype == 'identity': 
            self.e_mu    = nn.Sequential()
            self.e_sigma = nn.Sequential() 
        else:
            raise ValueError('bad etype')
        
        if self.include_baseline:
            self.transition_fxn = TransitionFxnSDMM(dim_stochastic, dim_subtype, dim_data, dim_treat, dim_hidden, ttype, dim_base=dim_base, augmented=self.augmented, alpha1_type=alpha1_type, otype=otype, add_stochastic=add_stochastic)
        else:
            self.transition_fxn = TransitionFxnSDMM(dim_stochastic, dim_subtype, dim_data, dim_treat, dim_hidden, ttype, augmented=self.augmented, alpha1_type=alpha1_type, otype=otype, add_stochastic=add_stochastic)
        self.pre_t_mu, self.pre_t_sigma = self.transition_fxn.get_prior_global()

        # Prior over Zg
        self.prior_W        = nn.Linear(dim_treat+dim_data+dim_base, dim_subtype)
        self.prior_sigma    = nn.Linear(dim_treat+dim_data+dim_base, dim_subtype)

    def p_Zg(self, B, X0, A0):
        inp_cat = torch.cat([B, X0, A0], -1)
        mu      = self.prior_W(inp_cat)
        sigma   = torch.nn.functional.softplus(self.prior_sigma(inp_cat))
        p_z_bxa = Independent(Normal(mu, sigma), 1)
        return p_z_bxa
    
    def p_X_Z(self, Zt, Zg):
        Zg_rep      = Zg[:,None,:].repeat(1, Zt.shape[1], 1)
        inp         = torch.cat([Zg_rep, Zt],-1)
        mu          = self.e_mu(inp)
        if self.fixed_var is None:
            sigma       = torch.nn.functional.softplus(self.e_sigma(inp))#*0. + 0.3
        else:
            sigma       = mu*0. + self.fixed_var
        return mu, sigma
    
    def p_Zt_Ztm1(self, Zg, Zt_1T, A, B, Xt):
        mu0         = self.pre_t_mu(Zg)[:,None,:]
        sig0        = torch.nn.functional.softplus(self.pre_t_sigma(Zg))[:,None,:]        
        Tmax        = Zt_1T.shape[1]
        Z_rep       = Zg[:,None,:].repeat(1,Tmax-1,1)
        if self.augmented: 
            Zinp = torch.cat([Zt_1T, Xt], -1)
        else: 
            Zinp = Zt_1T
        inp         = torch.cat([Zinp[:,:-1,:], A[:,1:Tmax,:], Z_rep], -1)
        
        if self.include_baseline: 
            Aval = A[:,1:Tmax,:]
            # include baseline in both control and input signals
            Acat = torch.cat([Aval[...,[0]],B[:,None,:].repeat(1,Aval.shape[1],1), Aval[...,1:]],-1)
            inp  = torch.cat([B[:,None,:].repeat(1,Aval.shape[1],1), inp],-1)
            mu1T, sig1T = self.transition_fxn(inp, Acat)
        else: 
            mu1T, sig1T = self.transition_fxn(inp, A[:,1:Tmax,:])
        
        mu, sig     = torch.cat([mu0,mu1T],1), torch.cat([sig0,sig1T],1)
        return Independent(Normal(mu, sig), 1)
        
    def get_loss(self, B, X, A, M, Y, CE, anneal = 1., return_reconstruction = False):
        _, _, lens         = get_masks(M)
        B, X, A, M, Y, CE  = B[lens>1], X[lens>1], A[lens>1], M[lens>1], Y[lens>1], CE[lens>1]
        m_t, m_g_t, _          = get_masks(M[:,1:,:])
        Z_g, q_zg, Z_t, q_zt = self.inf_network(X, A, M, B)
        p_x_mu, p_x_std      = self.p_X_Z(Z_t, Z_g)
        p_zg   = self.p_Zg(B, X[:,0,:], A[:,0,:])
        p_zt   = self.p_Zt_Ztm1(Z_g, Z_t, A, B, X[:,1:,:])
        
        masked_nll = masked_gaussian_nll_3d(X[:,1:,:], p_x_mu, p_x_std, M[:,1:,:])
        masked_nll = masked_nll.sum(-1).sum(-1)
        kl_g       = q_zg.log_prob(Z_g)-p_zg.log_prob(Z_g)
        kl_t       = q_zt.log_prob(Z_t)-p_zt.log_prob(Z_t)
        masked_kl_t= (m_t*kl_t).sum(-1)
        kl         = masked_kl_t+(kl_g*m_g_t)
        neg_elbo   = masked_nll + anneal*kl
        if return_reconstruction:
            return (neg_elbo,masked_nll, kl, Z_g, p_x_mu*M[:,1:,:], p_x_std*M[:,1:,:])
        else:
            return (neg_elbo,masked_nll, kl, Z_g)
    
    def forward(self, B, X, A, M, Y, CE, anneal = 1.):
        neg_elbo, masked_nll, kl, Z_g = self.get_loss(B, X, A, M, Y, CE, anneal = anneal)
        reg_loss   = torch.mean(neg_elbo)
        for name,param in self.named_parameters():
            if self.reg_all:
                reg_loss += self.C*apply_reg(param, reg_type=self.reg_type)
            else:
                if 'weight' in name:
                    reg_loss += self.C*apply_reg(param, reg_type=self.reg_type)
        return (torch.mean(neg_elbo), torch.mean(masked_nll), torch.mean(kl), Z_g), torch.mean(reg_loss)

    def forward_sample(self, Z_g, A, T_forward, Z_start = None):
        if Z_start is None:
            mu0         = self.pre_t_mu(Z_g)
            sig0        = torch.nn.functional.softplus(self.pre_t_sigma(Z_g))
            Z_start     = torch.squeeze(Independent(Normal(mu0, sig0), 1).sample((1,)))
        Zlist = [Z_start]
        for t in range(1, T_forward):
            Ztm1       = Zlist[t-1]
            inp        = torch.cat([Z_g, Ztm1, A[:,t-1,:]],-1)
            # try this below as well 
            # if self.include_baseline:
            #     Aval = A[:,1:Tmax,:]
            #     Acat = torch.cat([Aval[...,[0]],B[:,None,:].repeat(1,Aval.shape[1],1), Aval[...,1:]],-1)
            #     mu2T, sig2T = self.transition_fxn(Zt[:,:-1,:], Acat)
            # else:
            #     mu2T, sig2T = self.transition_fxn(Zt[:,:-1,:], A[:,1:Tmax,:])
            mut        = self.t_mu(inp)
            sigmat     = torch.nn.functional.softplus(self.t_sigma(inp))
            Zlist.append(torch.squeeze(Independent(Normal(mut, sigmat), 1).sample((1,))))
        Z_t               = torch.cat([k[:,None,:] for k in Zlist], 1)
        p_x_mu, p_x_sigma = self.p_X_Z(Z_t, Z_g)
        sample = torch.squeeze(Independent(Normal(p_x_mu, p_x_sigma), 1).sample((1,)))
        return sample, (Z_t, p_x_mu, p_x_sigma)
    
    def inspect(self, T_forward, T_condition, B, X, A, M, Y, CE, restrict_lens = False):
        self.eval()
        m_t, m_g_t, lens         = get_masks(M[:,1:,:])
        if restrict_lens: 
            B, X, A, M, Y, CE  = B[lens>1], X[lens>1], A[lens>1], M[lens>1], Y[lens>1], CE[lens>1]
            m_t, m_g_t, lens          = get_masks(M[:,1:,:])
        
        Z_g, q_zg, Z_t, q_zt = self.inf_network(X, A, M, B)
        p_x_mu, p_x_std      = self.p_X_Z(Z_t, Z_g)
        p_zg       = self.p_Zg(B, X[:,0,:], A[:,0,:])
        p_zt       = self.p_Zt_Ztm1(Z_g, Z_t, A)
        Tmax       = Z_t.shape[1]
        masked_nll = masked_gaussian_nll_3d(X[:,1:Tmax+1,:], p_x_mu, p_x_std, M[:,:Tmax,:])
        kl_g       = q_zg.log_prob(Z_g)-p_zg.log_prob(Z_g)
        kl_t       = q_zt.log_prob(Z_t)-p_zt.log_prob(Z_t)
        masked_kl_t= (m_t[:,:Tmax]*kl_t).sum(-1)
        kl         = masked_kl_t+(kl_g*m_g_t)
        
        per_feat_nelbo = (masked_nll.sum(1) + kl[...,None]).mean(0)
        neg_elbo   = torch.mean(masked_nll.sum(-1).sum(-1)+kl)
        
        # Sample T_forward while conditioning on T = T_condition -- sample from the inference network
        Z_g_cond, _, Z_t_cond, _       = self.inf_network(X[:,:T_condition,:], A[:,:T_condition,:], M[:,:T_condition,:], B)
        _,(_,x_forward_conditional,_)  = self.forward_sample(Z_g_cond, A[:,T_condition:,:], T_forward, Z_start = Z_t_cond[:,-1,:])
        x_sample_conditional           = torch.cat([X[:,:T_condition,:], x_forward_conditional],1)
        _,(_,x_forward,_)              = self.forward_sample(torch.squeeze(p_zg.sample((1,))), A, T_forward-1)
        x_forward                      = torch.cat([X[:,[0],:],x_forward],1)
        
        return neg_elbo, per_feat_nelbo, p_zg.mean, q_zg.mean, x_sample_conditional, x_forward

    @staticmethod
    def add_model_specific_args(parent_parser): 
        parser = ArgumentParser(parents = [parent_parser], add_help=False)
        parser.add_argument('--dim_subtype', type=int, default=16, help='dimension of global latent variable of state space model')
        parser.add_argument('--dim_stochastic', type=int, default=48, help='stochastic dimension of state space model')
        parser.add_argument('--dim_hidden', type=int, default=300, help='hidden dimension for nonlinear model')
        parser.add_argument('--etype', type=str, default='lin', help='SSM emission function')
        parser.add_argument('--ttype', type=str, default='lin', help='SSM transition function')
        parser.add_argument('--inftype', type=str, default='rnn_relu', help='inference network type')
        parser.add_argument('--post_approx', type=str, default='diag', help='inference of approximate posterior distribution')
        parser.add_argument('--include_baseline', type=strtobool, default=True, help='whether or not to condition on baseline data in gen model')            
        parser.add_argument('--elbo_samples', type=int, default=1, help='number of samples to run through inference network')        
        parser.add_argument('--augmented', type=strtobool, default=False, help='SSM augmented')        
        parser.add_argument('--C', type=float, default=.01, help='regularization strength')
        parser.add_argument('--nheads', type=int, default=1, help='number of heads for attention inference network')        
        parser.add_argument('--rank', type=int, default=5, help='rank of matrix for low_rank posterior approximation')
        parser.add_argument('--combiner_type', type=str, default='pog', help='combiner function used in inference network')
        parser.add_argument('--reg_all', type=strtobool, default=False, help='regularize all weights or only subset')    
        parser.add_argument('--reg_type', type=str, default='l2', help='regularization type (l1 or l2)')
        parser.add_argument('--alpha1_type', type=str, default='linear', help='alpha1 parameterization in TreatExp IEF')
        parser.add_argument('--otype', type=str, default='linear', help='final layer of GroMOdE IEF (linear, identity, nl)')
        parser.add_argument('--add_stochastic', type=strtobool, default=False, help='conditioning alpha-1 of TEXP on S_[t-1]')

        return parser 
