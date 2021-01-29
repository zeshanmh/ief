import torch 
import sys, math
import numpy as np
import torch.jit as jit
import warnings 
import copy
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
from pyro.distributions import Normal, Independent, Categorical, LogNormal
from typing import List, Tuple
from torch import Tensor
from collections import namedtuple
from typing import List, Tuple
from torch.autograd import Variable
from argparse import ArgumentParser

class SSM(Model): 
    def __init__(self, trial, **kwargs): 
        super(SSM, self).__init__(trial)
        self.save_hyperparameters()

    def init_model(self): 
        ttype       = self.hparams['ttype']; etype = self.hparams['etype']
        dim_hidden  = self.hparams['dim_hidden']
        num_heads   = self.hparams['nheads']
        dim_stochastic = self.hparams['dim_stochastic']
        #dim_stochastic = self.trial.suggest_int('dim_stochastic',8,256)
        dim_data    = self.hparams['dim_data']
        dim_base    = self.hparams['dim_base']
        dim_treat   = self.hparams['dim_treat']
        post_approx = self.hparams['post_approx']
        inftype     = self.hparams['inftype']; etype = self.hparams['etype']; ttype = self.hparams['ttype']
        augmented   = self.hparams['augmented']; alpha1_type = self.hparams['alpha1_type']
        rank        = self.hparams['rank']; combiner_type = self.hparams['combiner_type']; nheads = self.hparams['nheads']
        add_stochastic = self.hparams['add_stochastic']
        zmatrix     = self.hparams['zmatrix']

        # Inference Network
        if inftype == 'rnn':
            self.inf_network    = RNN_STInf(dim_base, dim_data, dim_treat, dim_hidden, dim_stochastic, post_approx = post_approx, rank = rank, combiner_type = combiner_type)
        elif inftype == 'rnn_bn':
            self.inf_network    = RNN_STInf(dim_base, dim_data, dim_treat, dim_hidden, dim_stochastic, post_approx = post_approx, rank = rank, use_bn=True, combiner_type = combiner_type)
        elif inftype == 'rnn_relu':
            self.inf_network    = RNN_STInf(dim_base, dim_data, dim_treat, dim_hidden, dim_stochastic, post_approx = post_approx, rank = rank, nl='relu', combiner_type = combiner_type)
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
        if self.hparams['include_baseline'] != 'none':
            self.transition_fxn = TransitionFunction(dim_stochastic, dim_data, dim_treat+dim_base, dim_hidden, ttype, \
                augmented=augmented, alpha1_type=alpha1_type, add_stochastic=add_stochastic, num_heads=num_heads, zmatrix=zmatrix)
        else: 
            self.transition_fxn = TransitionFunction(dim_stochastic, dim_data, dim_treat, dim_hidden, ttype, \
                augmented=augmented, alpha1_type=alpha1_type, add_stochastic=add_stochastic, num_heads=num_heads, zmatrix=zmatrix)   
        
        # Prior over Z1
        self.prior_W        = nn.Linear(dim_treat+dim_data+dim_base, dim_stochastic)
        self.prior_sigma    = nn.Linear(dim_treat+dim_data+dim_base, dim_stochastic)

    def p_Z1(self, B, X0, A0):
        inp_cat = torch.cat([B, X0, A0], -1)
        mu      = self.prior_W(inp_cat)
        sigma   = torch.nn.functional.softplus(self.prior_sigma(inp_cat))
        p_z_bxa = Independent(Normal(mu, sigma), 1)
        return p_z_bxa
    
    def p_X_Z(self, Zt, Tval):
        if 'spiral' in self.hparams['etype']:
            mu          = self.e_mu(Zt, Tval)
            if 'Spiral' in self.e_sigma.__class__.__name__:
                sigma       = torch.nn.functional.softplus(self.e_sigma(Zt, Tval))
            else:
                sigma       = torch.nn.functional.softplus(self.e_sigma(Zt))
        else:
            mu          = self.e_mu(Zt)
            sigma       = torch.nn.functional.softplus(self.e_sigma(Zt))
            
        return mu, sigma
    
    def p_Zt_Ztm1(self, Zt, A, B, X, A0, eps = 0.):
        X0 = X[:,0,:]; Xt = X[:,1:,:]
        inp_cat  = torch.cat([B, X0, A0], -1)
        mu1      = self.prior_W(inp_cat)[:,None,:]
        sig1     = torch.nn.functional.softplus(self.prior_sigma(inp_cat))[:,None,:]
#         mu1      = torch.zeros_like(sig1).to(sig1.device)
        
        Tmax     = Zt.shape[1]
        if self.hparams['augmented']: 
            Zinp = torch.cat([Zt[:,:-1,:], Xt[:,:-1,:]], -1)
        else: 
            Zinp = Zt[:,:-1,:]
        if self.hparams['include_baseline'] != 'none':
            Aval = A[:,1:Tmax,:]
            Acat = torch.cat([Aval[...,[0]],B[:,None,:].repeat(1,Aval.shape[1],1), Aval[...,1:]],-1)
            mu2T, sig2T = self.transition_fxn(Zinp, Acat, eps = eps)
        else:
            mu2T, sig2T = self.transition_fxn(Zinp, A[:,1:Tmax,:], eps = eps)
        mu, sig     = torch.cat([mu1,mu2T],1), torch.cat([sig1,sig2T],1)
        return Independent(Normal(mu, sig), 1)
    
    def get_loss(self, B, X, A, M, Y, CE, anneal = 1., return_reconstruction = False, with_pred = False):
        _, _, lens         = get_masks(M)
        B, X, A, M, Y, CE  = B[lens>1], X[lens>1], A[lens>1], M[lens>1], Y[lens>1], CE[lens>1]
        m_t, m_g_t, _      = get_masks(M[:,1:,:])
        Z_t, q_zt          = self.inf_network(X, A, M, B)
        Tmax               = Z_t.shape[1]
        p_x_mu, p_x_std    = self.p_X_Z(Z_t, A[:,1:Tmax+1,[0]])
        p_zt               = self.p_Zt_Ztm1(Z_t, A, B, X, A[:,0,:])
        masked_nll         = masked_gaussian_nll_3d(X[:,1:Tmax+1,:], p_x_mu, p_x_std, M[:,1:Tmax+1,:])
        full_masked_nll    = masked_nll
        masked_nll         = masked_nll.sum(-1).sum(-1)
#         full_masked_kl_t = m_t[:,:Tmax]*kl_t
#         full_nelbo = full_masked_nll + (m_t[:,:Tmax]*kl_t)[...,None]
    
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
        
        '''
        full_masked_kl_t = m_t[:,:Tmax]*kl_t
        full_nelbo = full_masked_nll + (m_t[:,:Tmax]*kl_t)[...,None]
        return (neg_elbo, masked_nll, masked_kl_t, torch.ones_like(masked_kl_t), full_masked_nll, full_masked_kl_t)
        '''
            
    def imp_sampling(self, B, X, A, M, Y, CE, anneal = 1., imp_samples=100, idx = -1, mask = None):
        _, _, lens = get_masks(M)
        B, X, A, M, Y, CE  = B[lens>1], X[lens>1], A[lens>1], M[lens>1], Y[lens>1], CE[lens>1]
        m_t, m_g_t, _      = get_masks(M[:,1:,:])

        ll_estimates   = torch.zeros((imp_samples,X.shape[0])).to(X.device)
        ll_priors      = torch.zeros((imp_samples,X.shape[0])).to(X.device)
        ll_posteriors  = torch.zeros((imp_samples,X.shape[0])).to(X.device)

        X0 = X[:,0,:]; Xt = X[:,1:,:]; A0 = A[:,0,:]
        inp_cat  = torch.cat([B, X0, A0], -1)
        mu1      = self.prior_W(inp_cat)[:,None,:]
        sig1     = torch.nn.functional.softplus(self.prior_sigma(inp_cat))[:,None,:]
        
        for sample in range(imp_samples): 
            Z_s, q_zt = self.inf_network(X, A, M, B)
            Tmax = Z_s.shape[-2]
            p_x_mu, p_x_std    = self.p_X_Z(Z_s, A[:,1:Tmax+1,[0]])
            masked_nll         = masked_gaussian_nll_3d(X[:,1:Tmax+1,:], p_x_mu, p_x_std, M[:,1:Tmax+1,:]) # (bs,T,D)

            if idx != -1: 
                masked_ll = -1*masked_nll[...,[idx]].sum(-1).sum(-1)
            elif mask is not None: 
                mask = mask[...,:Tmax]
                masked_ll = -1*(masked_nll.sum(-1)*mask).sum(-1)
            else: 
                masked_ll          = -1*masked_nll.sum(-1).sum(-1)

            # prior
            if self.hparams['augmented']: 
                Zinp = torch.cat([Z_s[:,:-1,:], Xt[:,:-1,:]], -1)
            else: 
                Zinp = Z_s[:,:-1,:]
            if self.hparams['include_baseline']:
                Aval = A[:,1:Tmax,:]
                Acat = torch.cat([Aval[...,[0]],B[:,None,:].repeat(1,Aval.shape[1],1), Aval[...,1:]],-1)
                mu2T, sig2T = self.transition_fxn(Zinp, Acat)
            else:
                mu2T, sig2T = self.transition_fxn(Zinp, A[:,1:Tmax,:])
            mu_prior, std_prior     = torch.cat([mu1,mu2T],1), torch.cat([sig1,sig2T],1)
            ll_prior     = -1*masked_gaussian_nll_3d(Z_s, mu_prior, std_prior, m_t[:,:Tmax,None])

            # posterior 
            ll_posterior = -1*masked_gaussian_nll_3d(Z_s, q_zt.mean, q_zt.stddev, m_t[:,:Tmax,None])

            # store values 
            ll_estimates[sample]  = masked_ll
            if idx != -1: 
                ll_priors[sample] = ll_prior[...,[idx]].sum(-1).sum(-1)
                ll_posteriors[sample] = ll_posterior[...,[idx]].sum(-1).sum(-1)
            elif mask is not None: 
                mask = mask[...,:Tmax]
                ll_priors[sample] = (ll_prior.sum(-1)*mask).sum(-1)
                ll_posteriors[sample] = (ll_posterior.sum(-1)*mask).sum(-1)
            else: 
                ll_priors[sample] = ll_prior.sum(-1).sum(-1)
                ll_posteriors[sample] = ll_posterior.sum(-1).sum(-1)

        nll_estimate = -1*(torch.logsumexp(ll_estimates + ll_priors - ll_posteriors, dim=0) - np.log(imp_samples))
        return nll_estimate, torch.mean(nll_estimate)
    
    def forward(self, B, X, A, M, Y, CE, anneal = 1., full_ret_loss=False):
        if self.hparams.clock_ablation: 
            A[...,0] = torch.ones(A.shape[1])
        if self.training:
            if self.hparams['elbo_samples']>1:
                B, X = torch.repeat_interleave(B, repeats=self.elbo_samples, dim=0), torch.repeat_interleave(X, repeats=self.elbo_samples, dim=0)
                A, M = torch.repeat_interleave(A, repeats=self.elbo_samples, dim=0), torch.repeat_interleave(M, repeats=self.elbo_samples, dim=0)
                Y, CE= torch.repeat_interleave(Y, repeats=self.elbo_samples, dim=0), torch.repeat_interleave(CE, repeats=self.elbo_samples, dim=0)
            neg_elbo, masked_nll, kl, _  = self.get_loss(B, X, A, M, Y, CE, anneal = anneal, with_pred = True)
        else:
            neg_elbo, masked_nll, kl, _  = self.get_loss(B, X, A, M, Y, CE, anneal = anneal, with_pred = False)
        reg_loss   = torch.mean(neg_elbo)
        
        for name,param in self.named_parameters():
            if self.reg_all:
                # reg_loss += self.hparams['C']*apply_reg(param, reg_type=self.hparams['reg_type'])
                reg_loss += self.C*apply_reg(param, reg_type=self.reg_type)
            else:
                if 'attn' not in name:
                    reg_loss += self.C*apply_reg(param, reg_type=self.reg_type)
        loss = torch.mean(reg_loss)
#         if full_ret_loss: 
#             return (full_nelbo, torch.mean(neg_elbo), torch.mean(masked_nll), torch.mean(kl), torch.ones_like(kl)), loss
        
        return (torch.mean(neg_elbo), torch.mean(masked_nll), torch.mean(kl), torch.ones_like(kl)), loss
    
    def forward_sample(self, A, T_forward, Z_start = None, B=None, X0=None, A0=None, eps = 0.):
        if Z_start is None:
            inp_cat  = torch.cat([B, X0, A0], -1)
            mu1      = self.prior_W(inp_cat)
            sig1     = torch.nn.functional.softplus(self.prior_sigma(inp_cat))
            Z_start  = torch.squeeze(Independent(Normal(mu1, sig1), 1).sample((1,)))
        Zlist = [Z_start]
        for t in range(1, T_forward):
            Ztm1       = Zlist[t-1]
            if self.hparams.include_baseline != 'none':
                Aval = A[:,t-1,:]
                Acat = torch.cat([Aval[...,[0]], B, Aval[...,1:]], -1)
                mut, sigmat= self.transition_fxn(Ztm1, Acat, eps = eps)
            else:
                mut, sigmat= self.transition_fxn(Ztm1, A[:,t-1,:], eps = eps)
            sample = torch.squeeze(Independent(Normal(mut, sigmat), 1).sample((1,)))
            if len(sample.shape) == 1: 
                sample = sample[None,...]
            Zlist.append(sample)
        Z_t               = torch.cat([k[:,None,:] for k in Zlist], 1)
        p_x_mu, p_x_sigma = self.p_X_Z(Z_t, A[:,:Z_t.shape[1],[0]])
        sample = torch.squeeze(Independent(Normal(p_x_mu, p_x_sigma), 1).sample((1,)))
        return sample, (Z_t, p_x_mu, p_x_sigma)
    
    def inspect(self, T_forward, T_condition, B, X, A, M, Y, CE, restrict_lens = False, nsamples = 1, eps = 0.):
        self.eval()
        m_t, _, lens           = get_masks(M)
        idx_select = lens>1
        B, X, A, M, Y, CE  = B[lens>1], X[lens>1], A[lens>1], M[lens>1], Y[lens>1], CE[lens>1]
        m_t, m_g_t, lens   = get_masks(M[:,1:,:])
        Z_t, q_zt            = self.inf_network(X, A, M, B)
        p_x_mu, p_x_std      = self.p_X_Z(Z_t, A[:,:Z_t.shape[1],[0]])
        p_zt                 = self.p_Zt_Ztm1(Z_t, A, B, X, A[:,0,:], eps = eps)
        Tmax                 = Z_t.shape[1]
        masked_nll     = masked_gaussian_nll_3d(X[:,1:Tmax+1,:], p_x_mu, p_x_std, M[:,1:Tmax+1,:])
        kl_t           = q_zt.log_prob(Z_t)-p_zt.log_prob(Z_t)
        masked_kl_t    = (m_t[:,:Tmax]*kl_t).sum(-1)
        per_feat_nelbo = (masked_nll.sum(1) + masked_kl_t[...,None]).mean(0)
        
        # calculate MSE instead
        mse = (((p_x_mu-X[:,1:Tmax+1])**2)*M[:,1:Tmax+1]).sum(0).sum(0)
        vals= M[:,1:Tmax+1].sum(0).sum(0)
        per_feat_nelbo = mse/vals
        
        neg_elbo       = torch.mean(masked_nll.sum(-1).sum(-1)+masked_kl_t)
        if restrict_lens: 
            _, _, lens         = get_masks(M)
            idx_select         = lens>(T_forward+T_condition)
            B, X, A, M, Y, CE  = B[idx_select], X[idx_select], A[idx_select], M[idx_select], Y[idx_select], CE[idx_select]
        
        x_forward_list = []
        for n in range(nsamples):
            _,(_,x_forward,_)          = self.forward_sample(A[:,1:T_forward+1,:], T_forward-1, B = B, X0=X[:,0,:], A0=A[:,0,:], eps = eps)
            x_forward_list.append(x_forward[...,None])
        x_forward                      = torch.cat(x_forward_list,-1).mean(-1)
        x_forward                      = torch.cat([X[:,[0],:], x_forward], 1)
        
        if T_condition != -1: 
            x_forward_conditional_list = []
            for n in range(nsamples):
                Z_t_cond, _                    = self.inf_network(X[:,:T_condition,:], A[:,:T_condition,:], M[:,:T_condition,:], B)
                _,(_,x_forward_conditional,_)  = self.forward_sample(A[:,T_condition:,:], T_forward, Z_start = Z_t_cond[:,-1,:], B = B, eps = eps)
                x_forward_conditional_list.append(x_forward_conditional[...,None])
            
            x_forward_conditional = torch.cat(x_forward_conditional_list, -1).mean(-1)
            x_sample_conditional  = torch.cat([X[:,:T_condition,:], x_forward_conditional],1)
        
            return neg_elbo, per_feat_nelbo, torch.ones_like(masked_kl_t), torch.ones_like(masked_kl_t), x_sample_conditional, x_forward, (B,X,A,M,Y,CE), idx_select

        return neg_elbo, per_feat_nelbo, torch.ones_like(masked_kl_t), torch.ones_like(masked_kl_t), x_forward, (B,X,A,M,Y,CE), idx_select

    def inspect_trt(self, B, X, A, M, Y, CE, nsamples=3): 
        self.eval()
        m_t, _, lens           = get_masks(M)
        idx_select = lens>1
        B, X, A, M, Y, CE  = B[lens>1], X[lens>1], A[lens>1], M[lens>1], Y[lens>1], CE[lens>1]
        
        x_conditional_list = []
        for n in range(nsamples): 
            x_conditionals_per_pt = []
            for i in range(X.shape[0]): 
                # np.unique(np.where(pt_numpy(A)[...,-2:] == 1.)[0])
                T_condition = np.max(np.where(pt_numpy(A[i,:,-3]) == 1.)[0])+1
                print(i)
                if i == 38: 
                    import pdb; pdb.set_trace()
                l           = np.where(pt_numpy(A[i,:,-1]) == 1.)[0]
                if len(l) == 0: 
                    T_total = np.max(np.where(pt_numpy(A[i,:-2] == 1.))[0])+1
                else: 
                    T_total = np.max(l)+1
                T_forward   = T_total - T_condition
                Z_t_cond, _ = self.inf_network(X[[i],:T_condition,:], A[[i],:T_condition,:], M[[i],:T_condition,:], B[[i]])
                _, (_, x_forward_conditional, _) = self.forward_sample(A[[i],T_condition:,:], T_forward, Z_start=Z_t_cond[:,-1,:], B = B[[i]])
                x_conditional = torch.cat((X[[i],:T_condition], x_forward_conditional, X[[i],T_total:]),1)
                x_conditionals_per_pt.append(x_conditional)
            x_conditional_list.append(torch.cat(x_conditionals_per_pt,0)[...,None])
        x_final_conditional  = torch.cat(x_conditional_list, -1).mean(-1)

        return x_final_conditional, (B,X,A,M,Y,CE), idx_select
    
    def predict(self, **kwargs):
        raise NotImplemented()

    @staticmethod
    def add_model_specific_args(parent_parser): 
        parser = ArgumentParser(parents = [parent_parser], add_help=False)
        parser.add_argument('--dim_stochastic', type=int, default=48, help='stochastic dimension of state space model')
        parser.add_argument('--dim_hidden', type=int, default=300, help='hidden dimension for nonlinear model')
        parser.add_argument('--etype', type=str, default='lin', help='SSM emission function')
        parser.add_argument('--ttype', type=str, default='lin', help='SSM transition function')
        parser.add_argument('--inftype', type=str, default='rnn_relu', help='inference network type')
        parser.add_argument('--post_approx', type=str, default='diag', help='inference of approximate posterior distribution')
        parser.add_argument('--elbo_samples', type=int, default=1, help='number of samples to run through inference network')        
        parser.add_argument('--augmented', type=strtobool, default=False, help='SSM augmented')        
        parser.add_argument('--C', type=float, default=.01, help='regularization strength')
        parser.add_argument('--nheads', type=int, default=1, help='number of heads for attention inference network and generative model')        
        parser.add_argument('--rank', type=int, default=5, help='rank of matrix for low_rank posterior approximation')
        parser.add_argument('--combiner_type', type=str, default='pog', help='combiner function used in inference network')
        parser.add_argument('--reg_all', type=strtobool, default=False, help='regularize all weights or only subset')    
        parser.add_argument('--reg_type', type=str, default='l2', help='regularization type (l1 or l2)')
        parser.add_argument('--alpha1_type', type=str, default='linear', help='alpha1 parameterization in TreatExp IEF')
        parser.add_argument('--zmatrix', type=str, default='identity')
        parser.add_argument('--otype', type=str, default='linear', help='final layer of GroMOdE IEF (linear, identity, nl)')
        parser.add_argument('--add_stochastic', type=strtobool, default=False, help='conditioning alpha-1 of TEXP on S_[t-1]')
        parser.add_argument('--clock_ablation', type=strtobool, default=False, help='set to true to run without local clock')

        return parser 

class SSMAtt(SSM): 
    def __init__(self, trial, **kwargs): 
        super(SSMAtt, self).__init__(trial)
        self.save_hyperparameters()

    def init_model(self): 
        ttype       = 'attn_transition'; etype = self.hparams['etype']
        dim_hidden  = self.hparams['dim_hidden']
        # dim_stochastic = self.hparams['dim_stochastic']
        dim_stochastic = self.trial.suggest_int('dim_stochastic',16,64)
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
            self.inf_network    = RNN_STInf(dim_base, dim_data, dim_treat, dim_hidden, dim_stochastic, post_approx = post_approx, rank = rank, combiner_type = combiner_type)
        elif inftype == 'rnn_bn':
            self.inf_network    = RNN_STInf(dim_base, dim_data, dim_treat, dim_hidden, dim_stochastic, post_approx = post_approx, rank = rank, use_bn=True, combiner_type = combiner_type)
        elif inftype == 'rnn_relu':
            self.inf_network    = RNN_STInf(dim_base, dim_data, dim_treat, dim_hidden, dim_stochastic, post_approx = post_approx, rank = rank, nl='relu', combiner_type = combiner_type)
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
        if self.hparams['include_baseline'] == 'all':
            self.transition_fxn = TransitionFunction(dim_stochastic, dim_data, dim_treat+dim_base, dim_hidden, ttype, \
                augmented=augmented, alpha1_type=alpha1_type, add_stochastic=add_stochastic, num_heads=num_heads)                
        elif self.hparams['include_baseline'] == 'none': 
            self.transition_fxn = TransitionFunction(dim_stochastic, dim_data, dim_treat, dim_hidden, ttype, \
                augmented=augmented, alpha1_type=alpha1_type, add_stochastic=add_stochastic, num_heads=num_heads)
        else: 
            pass
        
        # Prior over Z1
        self.prior_W        = nn.Linear(dim_treat+dim_data+dim_base, dim_stochastic)
        self.prior_sigma    = nn.Linear(dim_treat+dim_data+dim_base, dim_stochastic)

        # Attention 
        self.attn     = MultiHeadedAttention(num_heads, dim_treat+dim_base)
        self.attn_lin = nn.Linear(dim_stochastic, dim_treat+dim_base)

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
        Aval = A[:,1:Tmax,:]; Am_res = Am[:,1:Tmax,1:Tmax]
        if self.hparams['include_baseline']:
            Acat = torch.cat([Aval[...,[0]],B[:,None,:].repeat(1,Aval.shape[1],1), Aval[...,1:]],-1)
            res  = self.attn(self.attn_lin(Zinp), Acat, Acat, mask=Am_res, use_matmul=True)
            mu2T, sig2T = self.transition_fxn(Zinp, res, eps = eps)
        else:
            res  = self.attn(self.attn_lin(Zinp), Aval, Aval, mask=Am_res, use_matmul=True) # res
            mu2T, sig2T = self.transition_fxn(Zinp, res, eps = eps)
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
                if 'attn' not in name:
                    reg_loss += self.C*apply_reg(param, reg_type=self.reg_type)
        loss = torch.mean(reg_loss)
        return (torch.mean(neg_elbo), torch.mean(masked_nll), torch.mean(kl), torch.ones_like(kl)), loss

    def forward_sample(self, A, T_forward, Z_start = None, B=None, X0=None, A0=None, eps = 0.):
        pass 

    def inspect(self, T_forward, T_condition, B, X, A, M, Y, CE, Am, restrict_lens = False, nsamples = 1, eps = 0.):
        pass

    def inspect_trt(self, B, X, A, M, Y, CE, Am, nsamples=3): 
        pass 


class TransitionFunction(nn.Module):
    def __init__(self, 
                 dim_stochastic, 
                 dim_data, 
                 dim_treat, 
                 dim_hidden, 
                 ttype, 
                 augmented: bool = False, 
                 alpha1_type: str = 'linear', 
                 otype: str = 'linear', 
                 add_stochastic: bool = False, 
                 num_heads: int = 1,
                 zmatrix: str = 'identity'):
        super(TransitionFunction, self).__init__()
        self.dim_stochastic  = dim_stochastic
        self.dim_treat       = dim_treat
        self.dim_hidden      = dim_hidden
        self.dim_data        = dim_data
        # Number of different lines of therapy to multiplex on (only for heterogenous models)
        self.K               = 3
        self.ttype           = ttype
        dim_treat_mK         = dim_treat-self.K
        if augmented: # augmented does not completely work for transition function other than ('gated','lin'), ('lin','lin') 
            dim_input = dim_stochastic+dim_data
        else: 
            dim_input = dim_stochastic
        if self.ttype   == 'lin':
            self.t_mu               = nn.Linear(dim_input+dim_treat, dim_stochastic) 
            self.t_sigma            = nn.Linear(dim_input+dim_treat, dim_stochastic)
        elif self.ttype == 'nl':
            tmodel                  = nn.Sequential(nn.Linear(dim_input+dim_treat, dim_hidden),nn.ReLU(True))
            self.t_mu               = nn.Sequential(tmodel, nn.Linear(dim_hidden, dim_stochastic)) 
            self.t_sigma            = nn.Sequential(tmodel, nn.Linear(dim_hidden, dim_stochastic))
        elif self.ttype == 'het_lin':
            self.t_mu               = nn.ModuleList([nn.Linear(dim_input+dim_treat_mK, dim_stochastic) for k in range(self.K)])
            self.t_sigma            = nn.ModuleList([nn.Linear(dim_input+dim_treat_mK, dim_stochastic) for k in range(self.K)])
        elif self.ttype == 'het_nl':
            t_mu, t_sigma = [],[]
            for k in range(self.K):
                tmodel              = nn.Sequential(nn.Linear(dim_input+dim_treat_mK, dim_hidden), nn.ReLU(True))
                t_mu.append(nn.Sequential(tmodel, nn.Linear(dim_hidden, dim_stochastic)))
                t_sigma.append(nn.Sequential(tmodel, nn.Linear(dim_hidden, dim_stochastic)))
            self.t_mu           = nn.ModuleList(t_mu)
            self.t_sigma        = nn.ModuleList(t_sigma)
        elif self.ttype == 'gated':
            avoid_init = False
            if self.dim_data != 16 or self.dim_treat != 9:
                avoid_init = True
            self.t_mu               = GatedTransition(dim_input, dim_treat, avoid_init = avoid_init, dim_output=dim_stochastic, alpha1_type=alpha1_type, otype=otype, add_stochastic=add_stochastic)
            self.t_sigma            = nn.Linear(dim_input+dim_treat, dim_stochastic)
        elif self.ttype == 'attn_transition': 
            avoid_init = False
            if self.dim_data != 16 or self.dim_treat != 9:
                avoid_init = True
            self.t_mu               = AttentionIEFTransition(dim_input, dim_treat, avoid_init = avoid_init, dim_output=dim_stochastic, alpha1_type=alpha1_type, otype=otype, add_stochastic=add_stochastic, num_heads=num_heads, zmatrix=zmatrix)
            self.t_sigma            = nn.Linear(dim_input+dim_treat, dim_stochastic)
        elif self.ttype == 'moe': 
            self.t_mu               = MofE(dim_input, dim_treat, dim_output=dim_stochastic, eclass='nl', num_experts=3) 
            self.t_sigma            = nn.Linear(dim_input+dim_treat, dim_stochastic)
        else:
            raise ValueError('bad ttype')
    
    def apply(self, fxn, z, u, eps=0.):
        if 'Monotonic' in fxn.__class__.__name__ or 'LogCellTransition' in fxn.__class__.__name__ or 'LogCellKill' in fxn.__class__.__name__ \
            or 'TreatmentExp' in fxn.__class__.__name__ or 'GatedTransition' in fxn.__class__.__name__ or 'Synthetic' in fxn.__class__.__name__ \
            or 'MofE' in fxn.__class__.__name__ or 'Ablation1' in fxn.__class__.__name__ or 'Ablation2' in fxn.__class__.__name__ \
            or 'AttentionIEFTransition' in fxn.__class__.__name__:
            return fxn(z, u, eps)
        else:
            return fxn(torch.cat([z, u],-1))

    def forward(self, z, u, eps=0.):
        if 'het_' in self.ttype:
            treat     = u[...,:-self.K]
            lot_oh    = u[...,-self.K:]
            mul, sigl = [], []
            for t_mu, t_sigma in zip(self.t_mu, self.t_sigma):
                mu  = self.apply(t_mu, z, treat)[...,None] 
                sig = torch.nn.functional.softplus(self.apply(t_sigma, z, treat))[...,None]
                mul.append(mu)
                sigl.append(sig)
            mu = torch.cat(mul,-1)
            sig= torch.cat(sigl,-1)
            mu = torch.sum(mu*lot_oh.unsqueeze(-2),-1)
            sig= torch.sum(sig*lot_oh.unsqueeze(-2),-1)+0.05
        else:
            mu  = self.apply(self.t_mu, z, u, eps)
            sig = torch.nn.functional.softplus(self.apply(self.t_sigma, z, u))
        return mu, sig

