import torch
import torch.nn as nn
import numpy as np
import logging
import pytorch_lightning as pl
import sys 
from lifelines.utils import concordance_index
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset
from torchcontrib.optim import SWA
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN


def resample(data, device=None, strategy='random'): 
    ''' Good effort, lol....
     # B, X, A, M, Y, CE = data.tensors 
        # y = pt_numpy(Y)
        # mclass_idxs     = np.where(y == np.bincount(y).argmin())[0]
        # num_lfreq_class   = len(mclass_idxs)
        # num_mfreq_class  = len(np.where(y == np.bincount(y).argmax())[0]) # find most frequent element and number of times it occurs
        # num_iters = int(np.ceil(num_mfreq_class / num_lfreq_class))
        # num_classes = len(np.bincount(y))
        # if device is not None: 
        #     Bn = torch.zeros((num_iters*num_lfreq_class*num_classes, B.shape[1])).to(device)
        #     Xn = torch.zeros((num_iters*num_lfreq_class*num_classes, X.shape[1], X.shape[2])).to(device)
        #     Mn = torch.zeros((num_iters*num_lfreq_class*num_classes, M.shape[1], M.shape[2])).to(device)
        #     An = torch.zeros((num_iters*num_lfreq_class*num_classes, A.shape[1], A.shape[2])).to(device)
        #     Yn = torch.zeros((num_iters*num_lfreq_class*num_classes,),dtype=torch.long).to(device)
        #     CEn = torch.zeros((num_iters*num_lfreq_class*num_classes, CE.shape[1])).to(device)
        # else: 
        #     Bn = torch.zeros((num_iters*num_lfreq_class*num_classes, B.shape[1]))
        #     Xn = torch.zeros((num_iters*num_lfreq_class*num_classes, X.shape[1], X.shape[2]))
        #     Mn = torch.zeros((num_iters*num_lfreq_class*num_classes, M.shape[1], M.shape[2]))
        #     An = torch.zeros((num_iters*num_lfreq_class*num_classes, A.shape[1], A.shape[2]))
        #     Yn = torch.zeros((num_iters*num_lfreq_class*num_classes,),dtype=torch.long)
        #     CEn = torch.zeros((num_iters*num_lfreq_class*num_classes, CE.shape[1]))

        # zidxs = list(np.where(y == 0.)[0])
        # oidxs = list(np.where(y == 1.)[0])
        # tidxs = list(np.where(y == 2.)[0])

        # def get_ids(idxs, l, u): 
        #     if u < l: 
        #         ids = idxs[l:] + idxs[:u]
        #     else: 
        #         ids = idxs[l:u]
        #     return ids
        
        # for it in range(num_iters): 
        #     mi = it*num_classes*num_lfreq_class; miP1 = ((it*num_classes)+num_classes)*num_lfreq_class
        #     zl = np.mod(it*num_lfreq_class,len(zidxs)); zu = np.mod((it+1)*num_lfreq_class-1,len(zidxs))
        #     ol = np.mod(it*num_lfreq_class,len(oidxs)); ou = np.mod((it+1)*num_lfreq_class-1,len(oidxs))
        #     tl = np.mod(it*num_lfreq_class,len(tidxs)); tu = np.mod((it+1)*num_lfreq_class-1,len(tidxs))

        #     zids = get_ids(zidxs, zl, zu+1)
        #     oids = get_ids(oidxs, ol, ou+1)
        #     tids = get_ids(tidxs, tl, tu+1)
        #     idxs = zids + oids + tids
        #     print(idxs)
        #     Bn[mi:miP1]  = B[idxs]; Xn[mi:miP1] = X[idxs]; Mn[mi:miP1] = M[idxs]; An[mi:miP1] = A[idxs]; Yn[mi:miP1] = Y[idxs]
        #     CEn[mi:miP1] = CE[idxs] 

        # if oversample: 
        #     y_vals_sort = self.ddata[fold][tvt]['ys_seq'][:,0][idx_sort]
        #     class_count = np.unique(y_vals_sort, return_counts=True)[1]
        #     print(class_count)
        #     weight = 1. / class_count 
        #     samples_weight = weight[y_vals_sort]
        #     if device is not None: 
        #         samples_weight = torch.from_numpy(samples_weight).to(device)
        #     else: 
        #         samples_weight = torch.from_numpy(samples_weight)
        #     sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    '''
    B, X, A, M, Y, CE = data.tensors 
    Bnp = pt_numpy(B); Ynp = pt_numpy(Y)
    if strategy == 'random': 
        sampler = RandomOverSampler(random_state=0)

        Bnp, Ynp = sampler.fit_resample(Bnp, Ynp)
        new_idxs = sampler.sample_indices_ 
        num_examples = len(new_idxs)

        if device is not None: 
            Bn = torch.from_numpy(Bnp.astype('float32')).to(device)
            Xn = torch.zeros((num_examples, X.shape[1])).to(device)
            Mn = torch.zeros((num_examples, M.shape[1], M.shape[2])).to(device)
            An = torch.zeros((num_examples, A.shape[1], A.shape[2])).to(device)
            Yn = torch.from_numpy(Ynp.astype('int64')).to(device)
            CEn = torch.zeros((num_examples, CE.shape[1])).to(device)
        else: 
            Bn = torch.from_numpy(Bnp.astype('float32'))
            Xn = torch.zeros((num_examples, X.shape[1]))
            Mn = torch.zeros((num_examples, M.shape[1], M.shape[2]))
            An = torch.zeros((num_examples, A.shape[1], A.shape[2]))
            Yn = torch.from_numpy(Ynp.astype('int64'))
            CEn = torch.zeros((num_examples, CE.shape[1]))

        Xn = X[new_idxs]; Mn = M[new_idxs]; An = A[new_idxs]; CEn = CE[new_idxs]
        shuffle_idxs = [x for x in range(num_examples)]
        np.random.shuffle(shuffle_idxs)
        Bn = Bn[shuffle_idxs]; Xn = Xn[shuffle_idxs]; An = An[shuffle_idxs]
        Mn = Mn[shuffle_idxs]; Yn = Yn[shuffle_idxs]; CEn = CEn[shuffle_idxs]
    elif strategy == 'smote': 
        sampler = SMOTE() 
        raise NotImplementedError()
    elif strategy == 'adasyn': 
        sampler = ADASYN()
        raise NotImplementedError()
    
    return TensorDataset(Bn, Xn, An, Mn, Yn, CEn)

def get_masks(M):
    m_t    = ((torch.flip(torch.cumsum(torch.flip(M.sum(-1), (1,)), 1), (1,))>1.)*1)
    m_g_t  = (m_t.sum(-1)>1)*1.
    lens   = m_t.sum(-1)
    return m_t, m_g_t, lens
    
def masked_gaussian_nll_3d(x, mu, std, mask):
    nll        = 0.5*np.log(2*np.pi) + torch.log(std)+((mu-x)**2)/(2*std**2)
    masked_nll = (nll*mask)
    return masked_nll

def apply_reg(p, reg_type='l2'):
    if reg_type == 'l1':
        return torch.sum(torch.abs(p))
    elif reg_type=='l2':
        return torch.sum(p.pow(2))
    else:
        raise ValueError('bad reg')

def pt_numpy(tensor):
    return tensor.detach().cpu().numpy()

def calc_stats(preds, tensors):
    B, X, A, M, Y, CE = tensors
    if Y.shape[-1]>1:
        Y_oh      = Y.detach().cpu().numpy()
        bin_preds = self.prediction.detach().cpu().numpy()
        Y_np      = bin_preds[np.argmax(Y_oh,-1)]
    else:
        Y_np      = Y.detach().cpu().numpy().ravel()
    CE_np     = CE.detach().cpu().numpy().ravel()
    preds_np  = preds.detach().cpu().numpy().ravel()
    event_obs = (1.-CE_np).ravel()
    idx       = np.where(event_obs>0)[0]
    mse  = np.square(Y_np[idx]-preds_np[idx]).mean()
    r2   = r2_score(Y_np[idx], preds_np[idx])
    ci   = concordance_index(Y_np, preds_np, event_obs)
    return mse, r2, ci