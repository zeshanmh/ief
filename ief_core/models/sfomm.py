import torch 
import sys, math
import numpy as np
import torch.jit as jit
import warnings 
import torch.nn.init as init
import torch.nn as nn

from models.base import Model
from distutils.util import strtobool
from models.utils import *
from models.ssm.inference import RNN_STInf, Attention_STInf
from models.iefs.gated import GatedTransition
from models.iefs.moe import MofE
from models.iefs.treatexp import TreatmentExponential
from models.iefs.att_iefs import AttentionIEFTransition
from pyro.distributions import Normal, Independent, Categorical, LogNormal
from typing import List, Tuple
from torch import Tensor
from torch.nn import functional as F
from collections import namedtuple
from typing import List, Tuple
from torch.autograd import Variable
from argparse import ArgumentParser
# from spacecutter.models import OrdinalLogisticModel
# # from spacecutter.losses import CumulativeLinkLoss
# from spacecutter.losses import cumulative_link_loss
# from spacecutter.callbacks import AscensionCallback

class SFOMM(Model): 
    def __init__(self, trial, **kwargs): 
        super(SFOMM, self).__init__(trial)
        self.save_hyperparameters()

    def init_model(self):         
        mtype      = self.hparams['mtype']; otype = self.hparams['otype']
        alpha1_type = self.hparams['alpha1_type']
        dim_stochastic = self.trial.suggest_categorical('dim_stochastic',[4,16,48,64,128])
#         dim_hidden   = self.trial.suggest_categorical('dim_hidden',[100,300,500])
        dim_hidden = 300
        dim_data   = self.hparams['dim_data']
        dim_base   = self.hparams['dim_base']
        dim_treat  = self.hparams['dim_treat']
        num_heads  = self.hparams['nheads']
        # dim_stochastic = self.hparams['dim_stochastic']
        add_stochastic = self.hparams['add_stochastic']
        # inftype    = self.hparams['inftype']
        inftype = self.trial.suggest_categorical('inftype', ['rnn', 'birnn'])

        if inftype == 'rnn': 
            self.inf_network= nn.GRU(dim_data+dim_treat+dim_base+1, dim_hidden, 1, batch_first = True)
            self.post_W     = nn.Linear(dim_hidden, dim_stochastic)
        elif inftype == 'birnn': 
            self.inf_network= nn.GRU(dim_data+dim_treat+dim_base+1, dim_hidden, 1, batch_first = True, bidirectional=True)
            self.post_W     = nn.Linear(dim_hidden*2, dim_stochastic)
        elif inftype == 'ave_diff': 
            self.post_W      = nn.Linear(dim_data*2, dim_stochastic)
        else: 
            raise ValueError('bad inftype...')
        self.post_sigma  = torch.nn.Parameter(torch.ones(1,)*0.1, requires_grad = True)

        # define the transition function 
        if mtype == 'linear':
            self.model_mu   = nn.Linear(dim_treat+dim_base+dim_data*2, dim_data)
            self.model_sig  = nn.Linear(dim_treat+dim_base+dim_data, dim_data) 
        elif mtype == 'gated':
            avoid_init = False
            if dim_data !=16:
                avoid_init = True
            self.model_mu   = GatedTransition(dim_stochastic, dim_treat+dim_base, dim_hidden=dim_hidden, dim_output=dim_data, dim_input=dim_data*2, \
                                use_te = False, avoid_init = avoid_init, alpha1_type=alpha1_type, otype=otype, add_stochastic=add_stochastic)
            self.model_sig  = nn.Linear(dim_treat+dim_base+dim_data, dim_data)
        elif mtype == 'attn_transition': 
            avoid_init = False
            if dim_data != 16:
                avoid_init = True
            self.model_mu   = AttentionIEFTransition(dim_stochastic, dim_treat+dim_base, avoid_init = avoid_init, dim_output=dim_data,\
                dim_input=dim_data*2, alpha1_type=alpha1_type, otype=otype, add_stochastic=add_stochastic, num_heads=num_heads)
            self.model_sig  = nn.Linear(dim_treat+dim_base+dim_data, dim_data)    
        elif mtype == 'treatment_exp':
            self.model_mu   = TreatmentExponential(dim_stochastic, dim_treat, add_stochastic=add_stochastic, alpha1_type=alpha1_type, response_only=True)
            self.model_sig  = nn.Linear(dim_treat+dim_base+dim_data, dim_data)
            self.out_layer = nn.Linear(dim_stochastic, dim_data)
        elif mtype == 'nl': 
            self.model_mu   = nn.Sequential(nn.Linear(dim_treat+dim_base+dim_data*2, dim_hidden), nn.ReLU(True), nn.Linear(dim_hidden, dim_data))
            self.model_sig  = nn.Sequential(nn.Linear(dim_treat+dim_base+dim_data, dim_hidden), nn.ReLU(True), nn.Linear(dim_hidden, dim_data))
        else: 
            raise ValueError('Bad model type.')

        self.data_W         = nn.Linear(dim_stochastic, dim_data)
        # prior over Z 
        self.prior_W        = nn.Linear(dim_treat+dim_data+dim_base, dim_stochastic)
        self.prior_sigma    = torch.nn.Parameter(torch.ones(dim_stochastic)*0.1, requires_grad = True)

        # prediction 
        self.num_classes = 2
        self.pred_W1     = nn.Linear(dim_stochastic, dim_hidden, bias = False)
        self.pred_W2     = nn.Linear(dim_hidden, self.num_classes, bias = False)
        self.pred_sigma  = torch.nn.Parameter(torch.ones(self.num_classes,)*0.1, requires_grad = True)
        if self.num_classes >= 3: 
            self.m_pred      = nn.Sequential(
                                nn.Linear(dim_stochastic, dim_hidden),
                                # nn.ReLU(True),
                                # nn.Linear(dim_hidden, dim_hidden), 
                                nn.ReLU(True),
                                nn.Linear(dim_hidden, 1, bias=False)
                            )
#             self.mord        = OrdinalLogisticModel(self.m_pred, self.num_classes)
            # self.link_loss   = CumulativeLinkLoss()

        # treatment effect (for synthetic data)
        self.te_W1       = nn.Linear(dim_treat, dim_data)
        self.te_W2       = nn.Linear(dim_treat, dim_data*2)

    def p_Y_Z(self, Z, C): 
        if 'ord' not in self.hparams.loss_type: 
            mu = self.pred_W2(torch.sigmoid(self.pred_W1(Z)))
        else: 
            mu = self.m_pred(Z)
        sigma = mu*0.+torch.nn.functional.softplus(self.pred_sigma)
        p_y_z = Independent(Normal(mu, sigma), 1)
        return p_y_z

    def p_Z_BXA(self, B, X0, A0): 
        inp_cat = torch.cat([B,X0,A0],-1)
        mu      = self.prior_W(inp_cat)
        sigma   = torch.nn.functional.softplus(self.prior_sigma)
        p_z_bxa = Independent(Normal(mu, sigma), 1)
        return p_z_bxa 

    def treatment_effects(self, A): 
        A_0T   = A[:,:-1,:]
        scale  = (torch.sigmoid(self.te_W1(A_0T)) - 0.5)[...,:self.hparams.dim_data]
        result = self.te_W2(A_0T)
        shift  = result[...,:self.hparams.dim_data]
        sigma  = result[...,self.hparams.dim_data:(self.hparams.dim_data*2)]
        return scale, shift, sigma 

    def q_Z_XA(self, X, A, B, M):         
        if self.hparams.inftype == 'rnn' or self.hparams.inftype == 'birnn': 
            rnn_mask  = (M[:,1:].sum(-1)>1)*1.
            inp       = torch.cat([X[:,1:,:], rnn_mask[...,None], A[:,1:,:], B[:,None,:].repeat(1,A.shape[1]-1,1)], -1)
            m_t, _, lens = get_masks(M[:,1:,:])
            pdseq     = torch.nn.utils.rnn.pack_padded_sequence(inp, lens, batch_first=True, enforce_sorted=False)
            out_pd, _ = self.inf_network(pdseq)
            out, _    = torch.nn.utils.rnn.pad_packed_sequence(out_pd, batch_first=True) 
            mu        = self.post_W(out[:,-1,:])    
        elif self.hparams.inftype == 'ave_diff': 
            sc, sh, _ = self.treatment_effects(A)
            A_inv = 1 - A # for M_post 
            A_ = A[...,1,None]; A_inv_ = A_inv[...,1,None]
            pre = A_inv_*X; post = A_*X 
            diffs_pre = pre[:,1:,:] - pre[:,:-1,:]; diffs_post = post[:,1:,:] - post[:,:-1,:]
            M_pre = diffs_pre.sum(1) / A_inv_.sum(1)
            diffs_post = ((diffs_post - sh) / sc)*A_[:,:-1,:]
            M_post = diffs_post.sum(1) / (A_.sum(1)-1)
            M_ = torch.cat((M_pre, M_post), 1)
            mu = self.post_W(M_)
        sigma  = mu*0.+torch.nn.functional.softplus(self.post_sigma)
        q_dist = Independent(Normal(mu, sigma), 1) 
        return q_dist

    def p_X_Z(self, Z, X, A, B):
        base_cat = B[:,None,:].repeat(1, max(1, X.shape[1]-1), 1)
        patterns   = self.data_W(Z)
        patternsT  = patterns[:,None,:].repeat(1, max(1,X.shape[1]-1), 1)
        mtype      = self.hparams['mtype']

        if mtype == 'gated' or mtype == 'attn_transition':
            Aval     = A[:,:-1,:]
            cat      = torch.cat([X[:,:-1,:], patternsT], -1)
            p_x_mu   = self.model_mu(cat, torch.cat([Aval[...,[0]], base_cat, Aval[...,1:]],-1))
        elif mtype == 'treatment_exp': 
            scaleA, shiftA, sigmaA = self.treatment_effects(A)
            o        = self.out_layer(self.model_mu(Z, A))[:,1:,:]
            p_x_mu   = X[:,:-1,:]+scaleA*patternsT+shiftA+o 
        else:
            cat      = torch.cat([X[:,:-1,:], A[:,:-1,:], base_cat, patternsT],-1)
            p_x_mu   = self.model_mu(cat)
        cat      = torch.cat([X[:,:-1,:], A[:,:-1,:], base_cat],-1)
        if 'treatment_exp' in mtype: 
            p_x_sig   = torch.nn.functional.softplus(sigmaA)
        else:
            p_x_sig  = torch.nn.functional.softplus(self.model_sig(cat))

        return p_x_mu, p_x_sig 
    
    def get_loss(self, B, X, A, M, Y, CE, anneal = 1., return_reconstruction = False):
        _, _, lens         = get_masks(M)
        B, X, A, M, Y, CE  = B[lens>1], X[lens>1], A[lens>1], M[lens>1], Y[lens>1], CE[lens>1]
        m_t, m_g_t, _      = get_masks(M[:,1:,:])
        q_Z   = self.q_Z_XA(X, A, B, M)
        Z     = torch.squeeze(q_Z.rsample((1,)))
        p_Z   = self.p_Z_BXA(B, X[:,0,:], A[:,0,:])
        p_x_mu, p_x_std = self.p_X_Z(Z, X, A, B)
        masked_nll = masked_gaussian_nll_3d(X[:,1:,:], p_x_mu, p_x_std, M[:,1:,:])
        nll        = masked_nll.sum(-1).sum(-1)
        kl         = (q_Z.log_prob(Z)-p_Z.log_prob(Z))
        neg_elbo   = nll+anneal*kl

        if return_reconstruction:
            return (neg_elbo, nll, kl, p_x_mu*M[:,1:,:], p_x_std*M[:,1:,:])
        else:
            return (neg_elbo, nll, kl, Z)

    
    def forward(self, B, X, A, M, Y, CE, anneal = 1.):
        if self.hparams['loss_type'] == 'unsup' or self.hparams['loss_type'] == 'ord_unsup': 
            neg_elbo, nll, kl, Z = self.get_loss(B, X, A, M, Y, CE, anneal = anneal)
            reg_loss          = torch.mean(neg_elbo)
            ret_loss          = neg_elbo
        elif self.hparams['loss_type'] == 'semisup' or self.hparams['loss_type'] == 'ord_semisup': 
            neg_elbo, nll, kl, Z = self.get_loss(B, X, A, M, Y, CE, anneal = anneal)
            # supervised loss
            _, _, lens         = get_masks(M)
            B, X, A, M, Y, CE  = B[lens>1], X[lens>1], A[lens>1], M[lens>1], Y[lens>1], CE[lens>1]
            pred_dist  = self.p_Y_Z(Z, CE)
            weights    = self.get_weights(Y)
            if self.hparams['loss_type'] == 'semisup': 
                sup_loss   = F.cross_entropy(pred_dist.mean, Y, weight=weights)
            else: 
                sup_loss   = cumulative_link_loss(pred_dist.mean, Y[:,None], class_weights=weights)
            # sup_loss   = -pred_dist.log_prob(Y)
            reg_loss   = neg_elbo + sup_loss
            ret_loss   = reg_loss
        elif self.hparams['loss_type'] == 'ord_sup': 
            _, reg_loss = self.predict_ord(B, X, A, M, Y, CE)
            ret_loss    = reg_loss 
            nll = torch.ones_like(X); kl = torch.ones_like(X)
        elif self.hparams['loss_type'] == 'sup': 
            _, reg_loss = self.predict(B, X, A, M, Y, CE)
            ret_loss    = reg_loss
            nll = torch.ones_like(X); kl = torch.ones_like(X)

        for name,param in self.named_parameters():
            if self.hparams.reg_all:
                reg_loss += self.hparams.C*apply_reg(param, reg_type=self.hparams.reg_type)
            else:
                if 'weight' in name:
                    reg_loss += self.hparams.C*apply_reg(param, reg_type=self.hparams.reg_type)
        loss = torch.mean(reg_loss) 
        return (torch.mean(ret_loss), torch.mean(nll), torch.mean(kl), torch.ones_like(kl)), loss 

    
    def sample(self, T_forward, X, A, B, Z=None):
        with torch.no_grad():
            if Z is None: 
                p_Z = self.p_Z_BXA(B, X[:,0,:], A[:,0,:])
                Z      = p_Z.mean
            patterns   = self.data_W(Z)
            patternsT  = patterns[:,None,:].repeat(1, max(1,X.shape[1]-1), 1)
            base       = B[:,None,:]
            obs_list   = [X[:,[0],:]]
            mtype      = self.hparams.mtype

            for t in range(1, T_forward):
                x_prev     = obs_list[-1]
                if mtype == 'gated' or mtype == 'attn_transition':
                    Aval     = A[:,[t-1],:]
                    cat      = torch.cat([x_prev, patternsT[:,[t-1],:]], -1)
#                     p_x_mu   = self.model_mu(x_prev, torch.cat([Aval[...,[0]], base, Aval[...,1:]],-1))
                    p_x_mu   = self.model_mu(cat, torch.cat([Aval[...,[0]], base, Aval[...,1:]],-1))
                elif 'treatment_exp' in mtype: 
                    cat      = torch.cat([x_prev, A[:,[t-1],:], base, patternsT[:,[t-1],:]], -1)
                    p_x_mu   = self.model_mu(cat, A[:,[t-1],:])
                else:
                    p_x_mu     = self.model_mu(torch.cat([x_prev, A[:,[t-1],:], base, patternsT[:,[t-1],:]], -1))
                obs_list.append(p_x_mu) 
        return torch.cat(obs_list, 1)
    
    def inspect(self, T_forward, T_condition, B, X, A, M, Y, CE, restrict_lens = False):
        self.eval()
        # nelbo
        if restrict_lens: 
            m_t, m_g_t, lens         = get_masks(M[:,1:,:])
            B, X, A, M, Y, CE  = B[lens>1], X[lens>1], A[lens>1], M[lens>1], Y[lens>1], CE[lens>1]
        q_Z             = self.q_Z_XA(X, A, B, M)
        Z               = torch.squeeze(q_Z.rsample((1,)))
        p_Z             = self.p_Z_BXA(B, X[:,0,:], A[:,0,:])
        p_x_mu, p_x_std = self.p_X_Z(Z, X, A, B)
        masked_nll      = masked_gaussian_nll_3d(X[:,1:,:], p_x_mu, p_x_std, M[:,1:,:])
        nll             = masked_nll.sum(-1).sum(-1)
        kl              = (q_Z.log_prob(Z)-p_Z.log_prob(Z))
        nelbo        = torch.mean(nll + kl)

        # per_feat_nelbo 
        mse = (((p_x_mu-X[:,1:])**2)*M[:,1:]).sum(0).sum(0)
        vals= M[:,1:].sum(0).sum(0)
        per_feat_nelbo = mse/vals
        
        # Sample forward unconditionally and conditionally 
        inp_x      = self.sample(T_forward, X, A, B)
        q_Z        = self.q_Z_XA(X[:,:T_condition], A[:,:T_condition], B, M[:,:T_condition])
        Z_cond     = torch.squeeze(q_Z.rsample((1,))) 
        inp_x_post = self.sample(T_forward+1, X[:,T_condition-1:], A[:,T_condition-1:], B, Z_cond)
        inp_x_post = torch.cat([X[:,:T_condition], inp_x_post[:,1:]], 1) 
        empty      = torch.ones(X.shape[0], 3)
        return nelbo, per_feat_nelbo, empty, empty, inp_x_post, inp_x, Z, torch.squeeze(p_Z.rsample((1,)))

    def get_weights(self, Y): 
        # Y1 = torch.ones_like(Y); Y0 = torch.zeros_like(Y)     
        # N  = torch.zeros((int(torch.max(Y))+1,))
        # for c in range(int(torch.max(Y))+1): 
        #     N[c] = torch.sum(torch.where(Y == c, Y1, Y0))
        weights = torch.zeros((self.num_classes,)).to(Y.device)
        N = np.array([102,302,29])
        for i in range(N.shape[0]): 
            weights[i] = 1 - N[i] / sum(N)
        return weights 

    def predict(self, B, X, A, M, Y, CE):
        weights    = self.get_weights(Y)
        prior_dist = self.p_Z_BXA(B, X[:,0,:], A[:,0,:])
        prior_mean = prior_dist.mean
        pred_dist  = self.p_Y_Z(prior_mean, CE)
        sup_loss   = F.cross_entropy(pred_dist.mean, Y, weight=weights)

        return pred_dist.mean, torch.mean(sup_loss)

    def predict_ord(self, B, X, A, M, Y, CE): 
        weights    = self.get_weights(Y) 
        prior_dist = self.p_Z_BXA(B, X[:,0,:], A[:,0,:])
        prior_mean = prior_dist.mean 
        pred_dist  = self.p_Y_Z(prior_mean, CE)
        sup_loss   = cumulative_link_loss(pred_dist.mean, Y[:,None], class_weights=weights)
        return pred_dist.mean, torch.mean(sup_loss)


    @staticmethod
    def add_model_specific_args(parent_parser): 
        parser = ArgumentParser(parents = [parent_parser], add_help=False)
        parser.add_argument('--dim_stochastic', type=int, default=16, help='dimension of latent variable')
        parser.add_argument('--dim_hidden', type=int, default=300, help='hidden dimension for nonlinear model')
        parser.add_argument('--mtype', type=str, default='linear', help='transition function in FOMM')
        parser.add_argument('--C', type=float, default=.1, help='regularization strength')
        parser.add_argument('--reg_all', type=strtobool, default=True, help='regularize all weights or only subset')    
        parser.add_argument('--reg_type', type=str, default='l1', help='regularization type')
        parser.add_argument('--alpha1_type', type=str, default='linear', help='alpha1 parameterization in TreatExp IEF')
        parser.add_argument('--inftype', type=str, default='rnn', help='type of inference network')
        parser.add_argument('--otype', type=str, default='linear', help='final layer of GroMOdE IEF (linear, identity, nl)')
        parser.add_argument('--add_stochastic', type=strtobool, default=False, help='conditioning alpha-1 of TEXP on S_[t-1]')
        parser.add_argument('--nheads', type=int, default=1, help='number of heads for attention inference network and generative model')        

        return parser 



