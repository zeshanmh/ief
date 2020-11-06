import torch
import math
import sys
import torch.functional as F
from torch import nn
from pyro.distributions import Normal, Independent, Categorical, LogNormal, LowRankMultivariateNormal
# from models.base import Model
from models.utils import *

class RNN_STInf(nn.Module):
    def __init__(self, dim_base, dim_data, dim_treat, dim_hidden, dim_stochastic, post_approx='diag', rank = 5, use_bn = False, nl = 'tanh', combiner_type = 'standard'):
        super(RNN_STInf, self).__init__()
        self.dim_base   = dim_base
        self.dim_data   = dim_data
        self.dim_treat  = dim_treat
        self.dim_stochastic = dim_stochastic
        self.dim_hidden = dim_hidden
        self.use_bn     = use_bn
        if self.use_bn:
            print ('using bn in inf. network')
            self.bn     = nn.LayerNorm(dim_hidden, elementwise_affine=False)
        if nl == 'relu':
            print ('using relu in inf. network')
            self.nonlinearity = torch.relu
        else:
            self.nonlinearity = torch.tanh
        self.inf_rnn    = nn.GRU(dim_data+1+dim_treat+dim_base, dim_hidden, 1, batch_first = True, bidirectional=True)
        self.hid_rnn_zt = nn.Linear(dim_hidden*2, dim_hidden)
        self.combiner_type = combiner_type
        self.post_approx= post_approx
        self.rank       = rank
        self.base_h1    = nn.Linear(dim_data+dim_base+dim_treat, dim_hidden)
        self.hid_ztm1_zt= nn.Linear(dim_stochastic, dim_hidden)
        
        if self.combiner_type == 'standard' or self.combiner_type == 'masked':
            if self.post_approx in 'diag':
                self.mu_z1      = nn.Linear(dim_hidden, dim_stochastic)
                self.mu_zt      = nn.Linear(dim_hidden, dim_stochastic)
                self.sigma_z1   = nn.Linear(dim_hidden, dim_stochastic)
                self.sigma_zt   = nn.Linear(dim_hidden, dim_stochastic)
            elif self.post_approx == 'low_rank':
                self.mu_z1      = nn.Linear(dim_hidden, dim_stochastic)
                self.mu_zt      = nn.Linear(dim_hidden, dim_stochastic)
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
        sample = torch.squeeze(dist.rsample((1,)))
        if len(sample.shape) == 1: 
            sample = sample[None,...]
        return sample, dist
    
    def pog(self, mu1, sig1, mu2, sig2):
        sigsq1 = sig1.pow(2) + 1e-8
        sigsq2 = sig2.pow(2) + 1e-8
        sigmasq= (sigsq1*sigsq2)/(sigsq1+sigsq2)
        mu     = (mu1/sigsq1 + mu2/sigsq2)*sigmasq
        sigma  = sigmasq.pow(0.5)
        return mu, sigma

    def combiner_fxn(self, prev_hid, current_hid, rnn_mask, mu1fxn, sig1fxn, mu2fxn = None, sig2fxn = None):
        if self.combiner_type   =='standard' or self.combiner_type == 'masked':
            if self.combiner_type == 'standard':
                out        = 0.5*(prev_hid+current_hid)
            else:
                out        = rnn_mask*(0.5*(prev_hid+current_hid)) + (1-rnn_mask)*prev_hid
            if self.use_bn:
                h1         = self.nonlinearity(self.bn(out))
            else:
                h1         = self.nonlinearity(out)
            mu, sigma  = mu1fxn(h1), torch.nn.functional.softplus(sig1fxn(h1))
        elif self.combiner_type == 'pog':
            if self.use_bn:
                h1         = self.nonlinearity(self.bn(prev_hid))
                h2         = self.nonlinearity(self.bn(current_hid))
            else:
                h1         = self.nonlinearity(prev_hid)
                h2         = self.nonlinearity(current_hid)
            mu1, sig1  = mu1fxn(h1), torch.nn.functional.softplus(sig1fxn(h1))
            mu2, sig2  = mu2fxn(h2), torch.nn.functional.softplus(sig2fxn(h2))
            mu, sigma= self.pog(mu1, sig1, mu2, sig2)
        else:
            raise ValueError('bad combiner type')
        return mu, sigma
    
    def forward(self, x, a, m, b):
        rnn_mask        = (m[:,1:].sum(-1)>1)*1.
        inp             = torch.cat([x[:,1:,:], rnn_mask[...,None], a[:,1:,:], b[:,None,:].repeat(1,a.shape[1]-1,1)], -1)
        m_t, _, lens    = get_masks(m[:,1:,:])
        pdseq      = torch.nn.utils.rnn.pack_padded_sequence(inp, lens, batch_first=True, enforce_sorted = False)
        out_pd, _  = self.inf_rnn(pdseq)
        out, _     = torch.nn.utils.rnn.pad_packed_sequence(out_pd, batch_first=True)
        hid_rnn_zt = self.hid_rnn_zt(out)
        hid_base   = self.base_h1(torch.cat([x[:,0,:], b, a[:,0,:]],-1))
        if self.combiner_type == 'standard' or self.combiner_type == 'masked':
            mu, sigma  = self.combiner_fxn(hid_base, hid_rnn_zt[:,0,:], rnn_mask[:,[0]], self.mu_z1, self.sigma_z1)
        else:
            mu, sigma  = self.combiner_fxn(hid_base, hid_rnn_zt[:,0,:], rnn_mask[:,[0]], self.mu_zt, self.sigma_zt, self.mu_zt2, self.sigma_zt2)
        z,_        = self.reparam_dist(mu, sigma)

        meanlist = [mu[:,None,:]]
        sigmalist= [sigma[:,None,:]]
        zlist    = [z[:,None,:]]
        for t in range(1, hid_rnn_zt.shape[1]):
            ztm1       = torch.squeeze(zlist[t-1])
            hid_ztm1_zt= self.hid_ztm1_zt(ztm1)
            if self.combiner_type == 'standard' or self.combiner_type == 'masked':
                mu, sigma  = self.combiner_fxn(hid_ztm1_zt, hid_rnn_zt[:,t,:], rnn_mask[:,[t]], self.mu_zt, self.sigma_zt)
            else:
                mu, sigma  = self.combiner_fxn(hid_ztm1_zt, hid_rnn_zt[:,t,:], rnn_mask[:,[t]], self.mu_zt, self.sigma_zt, self.mu_zt2, self.sigma_zt2)
            z,_        = self.reparam_dist(mu, sigma)
            meanlist  += [mu[:,None,:]]
            sigmalist += [sigma[:,None,:]]
            zlist     += [z[:,None,:]]
        #q_zt     = Independent(Normal(torch.cat(meanlist, 1), torch.cat(sigmalist, 1)), 1)
        _,q_zt     = self.reparam_dist(torch.cat(meanlist, 1), torch.cat(sigmalist, 1))
        Z_t      = torch.cat(zlist, 1)
        return Z_t, q_zt
    
    
class Attention_STInf(nn.Module):
    def __init__(self, dim_base, dim_data, dim_treat, dim_hidden, dim_stochastic, nheads = 1, post_approx='diag', rank = 5):
        super(Attention_STInf, self).__init__()
        self.dim_base   = dim_base
        self.dim_data   = dim_data
        self.dim_treat  = dim_treat
        self.dim_hidden = dim_hidden
        self.dim_stochastic = dim_stochastic
        self.nheads     = nheads
        self.base_h1_k  = nn.Linear(dim_data+dim_base, dim_hidden)
        self.base_h1_v  = nn.Linear(dim_data+dim_base, dim_hidden*nheads)
        
        self.query_inp  = nn.Linear(dim_data+dim_treat, dim_hidden*nheads)
        self.value_inp  = nn.Linear(dim_data+dim_treat, dim_hidden*nheads)
        self.hid_ztm1_zt_k= nn.Linear(dim_stochastic, dim_hidden)
        self.hid_ztm1_zt_v= nn.Linear(dim_stochastic, dim_hidden*nheads)
        self.mu_z1      = nn.Linear(dim_hidden*nheads, dim_stochastic)
        self.sigma_z1   = nn.Linear(dim_hidden*nheads, dim_stochastic)
        self.mu_zt      = nn.Linear(dim_hidden*nheads, dim_stochastic)
        self.sigma_zt   = nn.Linear(dim_hidden*nheads, dim_stochastic)
        self.post_approx= post_approx
        self.rank       = rank
        if self.post_approx in 'diag':
            self.mu_z1      = nn.Linear(dim_hidden*nheads, dim_stochastic)
            self.sigma_z1   = nn.Linear(dim_hidden*nheads, dim_stochastic)
            self.mu_zt      = nn.Linear(dim_hidden*nheads, dim_stochastic)
            self.sigma_zt   = nn.Linear(dim_hidden*nheads, dim_stochastic)
        elif self.post_approx == 'low_rank':
            self.mu_z1      = nn.Linear(dim_hidden*nheads, dim_stochastic)
            self.mu_zt      = nn.Linear(dim_hidden*nheads, dim_stochastic)
            self.sigma_z1   = nn.Linear(dim_hidden*nheads, (dim_stochastic*rank)+dim_stochastic)
            self.sigma_zt   = nn.Linear(dim_hidden*nheads, (dim_stochastic*rank)+dim_stochastic)
        else:
            raise ValueError('bad setting for post_approx:'+str(post_approx))
    
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

    def multiple_attention(self, query, key, value, mask):
        """
        query: bs x T x dimq x nheads
        key:   bs x 1 x dimq
        value: bs x T x dimv x nheads
        mask:  bs x T
        """
        dim        = query.size(-2)
        scores     = (torch.matmul(key[:,None,None,:].repeat(1,query.shape[1],1,1), query)/ math.sqrt(dim)).squeeze(2)
        scores     = scores.masked_fill(mask[...,None] == 0, -1e9)
        p_attn     = torch.nn.functional.softmax(scores, dim = 1).unsqueeze(-2)
        return (value*p_attn).sum(1).view(p_attn.shape[0],-1)#.mean(-1)
    
    def forward(self, x, a, m, b):
        inp             = torch.cat([x[:,1:,:], a[:,:-1,:]], -1)
        m_t, _, _       = get_masks(m[:,1:,:])
        mask            = (m[:,1:,:].sum(-1)>0.)*1.
        q_inp           = torch.relu(self.query_inp(inp).view(inp.shape[0], inp.shape[1], self.dim_hidden, self.nheads))
        v_inp           = self.value_inp(inp).view(inp.shape[0], inp.shape[1], self.dim_hidden, self.nheads)
        
        key1            = torch.relu(self.base_h1_k(torch.cat([x[:,0,:],b],-1)))
        val1            = self.base_h1_v(torch.cat([x[:,0,:],b],-1))
        output1         = self.multiple_attention(q_inp, key1, v_inp, mask)
        h1              = torch.relu(0.5*(output1+val1))
        mu, sigma  = self.mu_z1(h1), torch.nn.functional.softplus(self.sigma_z1(h1))
        #z         = torch.squeeze(Independent(Normal(mu, sigma), 1).rsample((1,)))
        z,_     = self.reparam_dist(mu, sigma)
        meanlist = [mu[:,None,:]]
        sigmalist= [sigma[:,None,:]]
        zlist    = [z[:,None,:]]
        for t in range(1, x.shape[1]-1):
            ztm1       = torch.squeeze(zlist[t-1])
            keyt       = torch.relu(self.hid_ztm1_zt_k(ztm1))
            valt       = self.hid_ztm1_zt_v(ztm1)
            outputt    = self.multiple_attention(q_inp, keyt, v_inp, mask)
            ht         = torch.relu(0.5*(outputt + valt))
            mu, sigma  = self.mu_zt(ht), torch.nn.functional.softplus(self.sigma_zt(ht))
            #z          = torch.squeeze(Independent(Normal(mu, sigma), 1).rsample((1,)))
            z,_        = self.reparam_dist(mu, sigma)
            meanlist  += [mu[:,None,:]]
            sigmalist += [sigma[:,None,:]]
            zlist     += [z[:,None,:]]
        #q_zt     = Independent(Normal(torch.cat(meanlist, 1), torch.cat(sigmalist, 1)), 1)
        _,q_zt     = self.reparam_dist(torch.cat(meanlist, 1), torch.cat(sigmalist, 1))
        Z_t      = torch.cat(zlist, 1)
        #self.debug = [Z_t, key1, val1, keyt, valt]
        return Z_t, q_zt