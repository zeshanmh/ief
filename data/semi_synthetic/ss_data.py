import pickle 
import torch
import torch.nn as nn
import numpy as np
import sys
from numpy.random import choice 
import random
sys.path.append('/afs/csail.mit.edu/u/z/zeshanmh/research/trvae')
sys.path.append('/afs/csail.mit.edu/u/z/zeshanmh/research/trvae/dmm')
sys.path.append('/afs/csail.mit.edu/u/z/zeshanmh/research/trvae/models')
from base import setup_torch_dataset, pt_numpy
sys.path.append('../data/ml_mmrf')
sys.path.append('../data/')
from ml_mmrf_v1.data import load_mmrf
from dmm import DMM 
from torch.utils.data import DataLoader, TensorDataset

def gen_ss_helper(model, B, X, A, M, Y, CE, k='train', add_missing=False, eval_mult=30): 
    _, _, lens = model.get_masks(M)
    B, X, A, M, Y, CE = B[lens>1], X[lens>1], A[lens>1], M[lens>1], Y[lens>1], CE[lens>1]
    base_cat   = B[:,None,:].repeat(1, max(1, X.shape[1]-1), 1)
    T_forward  = 20
    if k == 'train':
        mult = 150
    else: 
        mult = eval_mult
    nsamples = mult*B.shape[0]
    
    Bs = np.zeros((nsamples,B.shape[1]))
    Xs = np.zeros((nsamples,T_forward,X.shape[2]))
    As = np.zeros((nsamples,T_forward,A.shape[2]))
    Ms = np.zeros((nsamples,T_forward,M.shape[2]))
    Ys = np.zeros((nsamples,Y.shape[1]))
    CEs = np.zeros((nsamples,CE.shape[1]))

    for i in range(mult): 
        _, _, _, _, sample = model.inspect_ss(T_forward, B, X, A, M, Y, CE)
        Bs[i*B.shape[0]:(i+1)*B.shape[0]] = pt_numpy(B)
        if add_missing: 
            # retain original MM missingness pattern 
#             Ms[i*M.shape[0]:(i+1)*M.shape[0]] = pt_numpy(M[:,:T_forward,:])
#             Xs[i*X.shape[0]:(i+1)*X.shape[0]] = pt_numpy(sample)

            # sample randomly from all indices
            Mtemp = np.ones((M.shape[0],T_forward,M.shape[2]))
            sample, Mnew = add_missingness(pt_numpy(sample), Mtemp)
            Ms[i*M.shape[0]:(i+1)*M.shape[0]] = Mnew
            Xs[i*X.shape[0]:(i+1)*X.shape[0]] = sample            
        else: 
            Ms[i*M.shape[0]:(i+1)*M.shape[0]] = np.ones((M.shape[0],T_forward,M.shape[2]))
            Xs[i*X.shape[0]:(i+1)*X.shape[0]] = pt_numpy(sample)
        As[i*A.shape[0]:(i+1)*A.shape[0]] = pt_numpy(A[:,:T_forward,:])      

        Ys[i*Y.shape[0]:(i+1)*Y.shape[0]] = pt_numpy(Y)
        CEs[i*CE.shape[0]:(i+1)*CE.shape[0]] = pt_numpy(CE)    
    
    return Bs, Xs, As, Ms, Ys, CEs

def add_missingness(x, m, per_missing=0.4):
    possible_idxs = []
    for i in range(1,m.shape[0]): 
        for j in range(1,m.shape[1]): 
            possible_idxs.append((i,j))
    missing_idxs = random.sample(possible_idxs, int(per_missing*m.shape[0]*m.shape[1]))
    missing_arrs = list(zip(*missing_idxs)) 

    new_m = np.ones_like(m)
    new_m[missing_arrs[0],missing_arrs[1]] = 0. 

    # forward fill 
    new_x = np.copy(x)
    new_x[np.where(1-new_m)] = np.nan
    for feat in range(new_x.shape[2]): 
        x_feat = new_x[...,feat]
        mask = np.isnan(x_feat)
        idx = np.where(~mask,np.arange(mask.shape[1]),0)
        np.maximum.accumulate(idx,axis=1, out=idx)
        out = x_feat[np.arange(idx.shape[0])[:,None], idx]
        new_x[...,feat] = out
        # new_x[mask] = new_x[np.nonzero(mask)[0], idx[mask]]

    return new_x, new_m 

def gen_ss_data(in_sample_dist=True, add_missing=False, eval_mult=30): 
    dim_stochastic = 48; dim_hidden = 300 
    dim_base  = 16; dim_data  = 16; dim_treat = 9
    C = 0.01; ttype = 'gated'; etype = 'lin'
    model = DMM(dim_stochastic, dim_hidden, dim_base, dim_data, dim_treat, C = C, ttype = ttype, etype=etype, 
                            inftype = 'rnn_relu', combiner_type = 'pog', include_baseline = True, reg_type = 'l1', reg_all=True, augmented=False)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device  = torch.device('cpu')
    model.to(device)
    fname = '/afs/csail.mit.edu/u/z/zeshanmh/research/trvae/dmm/good_models/sota_ssm_mm.pt'
    print ('loading',fname)
    model.load_state_dict(torch.load(fname))

    semi_syn = {
        'train': {}, 
        'valid': {
                0: {},
                1: {},
                2: {}, 
                3: {},
                4: {}
            }, 
        'test': {}
    }
    fold = 1
    mmdata  = load_mmrf(fold_span = [fold], digitize_K = 20, digitize_method = 'uniform', suffix='_2mos')
    dim_base, dim_data, dim_treat = mmdata[fold]['train']['b'].shape[-1], mmdata[fold]['train']['x'].shape[-1], mmdata[fold]['train']['a'].shape[-1]
    train, train_loader = setup_torch_dataset(mmdata, fold, 'train', device)
    valid, valid_loader = setup_torch_dataset(mmdata, fold, 'valid', device)
    test, test_loader   = setup_torch_dataset(mmdata, fold, 'test', device)
    print('sampling from SSM model trained on MM data...')
    Btr, Xtr, Atr, Mtr, Ytr, CEtr = gen_ss_helper(model, *train_loader.dataset.tensors, add_missing=add_missing, eval_mult=eval_mult)
    semi_syn['train'] = {
        'B': Btr, 
        'X': Xtr,
        'A': Atr, 
        'M': Mtr, 
        'Y': Ytr,
        'CE': CEtr
    }

    for fold in range(5):
        if in_sample_dist:  
            Bv, Xv, Av, Mv, Yv, CEv = gen_ss_helper(model, *train_loader.dataset.tensors, k='valid', add_missing=add_missing, eval_mult=eval_mult)
        else: 
            Bv, Xv, Av, Mv, Yv, CEv = gen_ss_helper(model, *valid_loader.dataset.tensors, k='valid', add_missing=add_missing, eval_mult=eval_mult)
        semi_syn['valid'][fold] = {
            'B': Bv,
            'X': Xv,
            'A': Av,
            'M': Mv,
            'Y': Yv,
            'CE': CEv
        }

    if in_sample_dist: 
        Bte, Xte, Ate, Mte, Yte, CEte = gen_ss_helper(model, *train_loader.dataset.tensors, k='test', add_missing=add_missing, eval_mult=eval_mult)
    else: 
        Bte, Xte, Ate, Mte, Yte, CEte = gen_ss_helper(model, *test_loader.dataset.tensors, k='test', add_missing=add_missing, eval_mult=eval_mult)
    semi_syn['test'] = {
        'B': Bte, 
        'X': Xte,
        'A': Ate, 
        'M': Mte, 
        'Y': Yte,
        'CE': CEte
    }

    return semi_syn

def load_ss_data(nsamples, add_missing=False, gen_fly=False, in_sample_dist=True, eval_mult=30):
    if not gen_fly: 
        if add_missing: 
            with open('/afs/csail.mit.edu/group/clinicalml/datasets/multiple_myeloma/semi_synthetic/semi_syn_orig_missing.pkl', 'rb') as f: 
                dataset = pickle.load(f)
        else:     
            with open('/afs/csail.mit.edu/u/z/zeshanmh/research/ief/data/semi_synthetic/semi_syn_dataset.pkl','rb') as f:
                dataset = pickle.load(f)
    else: 
        print('generating semi synthetic data...')
        dataset = gen_ss_data(in_sample_dist=in_sample_dist, add_missing=add_missing, eval_mult=eval_mult)

    T = np.arange(dataset['train']['B'].shape[0])
    idxs = np.random.choice(T,size=nsamples)
    for k,v in dataset['train'].items(): 
        dataset['train'][k] = v[idxs]
    return dataset

def load_ss_helper(sdata, tvt='train', bs=600, device=None, valid_fold=0):
    if tvt == 'train' or tvt == 'test': 
        ddata = sdata[tvt]
    elif tvt == 'valid': 
        ddata = sdata[tvt][valid_fold]

    if device is not None: 
        B  = torch.from_numpy(ddata['B'].astype('float32')).to(device)
        X  = torch.from_numpy(ddata['X'].astype('float32')).to(device)
        A  = torch.from_numpy(ddata['A'].astype('float32')).to(device)
        M  = torch.from_numpy(ddata['M'].astype('float32')).to(device)
    else: 
        B  = torch.from_numpy(ddata['B'].astype('float32'))
        X  = torch.from_numpy(ddata['X'].astype('float32'))
        A  = torch.from_numpy(ddata['A'].astype('float32'))
        M  = torch.from_numpy(ddata['M'].astype('float32'))
    Y = torch.from_numpy(ddata['Y'].astype('float32'))
    if device is not None: 
        Y = Y.to(device)
        CE = torch.from_numpy(ddata['CE'].astype('float32')).to(device)
    else: 
        CE = torch.from_numpy(ddata['CE'].astype('float32'))

    data        = TensorDataset(B, X, A, M, Y, CE)
    data_loader = DataLoader(data, batch_size=bs, shuffle=False)
    return data, data_loader

if __name__ == '__main__':
    np.random.seed(0)
    ddata = load_ss_data(1000, add_missing=False, gen_fly=True)
    import pdb; pdb.set_trace()
