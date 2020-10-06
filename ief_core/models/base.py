import torch
import torch.nn as nn
import numpy as np
import logging
import pytorch_lightning as pl
import sys 
from lifelines.utils import concordance_index
from sklearn.metrics import r2_score
from pytorch_lightning.metrics.functional import f1_score, precision_recall, auroc
from pytorch_lightning.metrics.sklearns import F1, Precision, Recall
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torchcontrib.optim import SWA
sys.path.append('../data/ml_mmrf')
sys.path.append('../data/')
from ml_mmrf_v1.data import load_mmrf
from synthetic.synthetic_data import load_synthetic_data_trt, load_synthetic_data_noisy
from semi_synthetic.ss_data import *
from models.utils import *

class Model(pl.LightningModule): 

    def __init__(self, trial, **kwargs): 
        super().__init__()
        torch.manual_seed(0)
        np.random.seed(0)
        if torch.cuda.is_available(): # don't need 
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        self.trial = trial
        # self.lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
        self.bs = trial.suggest_categorical('bs', [600,1500])
        self.lr = 1e-3
        self.C  = trial.suggest_categorical('C', [.01,.1,1])
        self.reg_all  = trial.suggest_categorical('reg_all', [True, False])
        self.reg_type = trial.suggest_categorical('reg_type', ['l1', 'l2'])
    
    def forward(self,**kwargs):
        raise ValueError('Should be overriden')
    def predict(self,**kwargs):
        raise ValueError('Should be overriden')
    def init_model(self,**kwargs):
        raise ValueError('Should be overriden')

    def training_step(self, batch, batch_idx): 
        nsamples = self.hparams['nsamples']
        if nsamples>1:
            dt = [k.repeat(nsamples,1) if k.dim()==2 else k.repeat(nsamples,1,1) for k in batch]
        else:
            dt = batch
        # use KL annealing for SSM and SFOMM
        # print('Batch {}, classes {}, count {}'.format(batch_idx, *np.unique(pt_numpy(dt[-2]), return_counts=True)))
        if self.hparams['anneal'] != -1.: 
            anneal = min(1, self.current_epoch/(self.hparams['max_epochs']*0.5))
            self.hparams['anneal'] = anneal
        else: 
            anneal = 1.
        _, loss  = self.forward(*dt, anneal = anneal) 
        return {'loss': loss}

    def training_epoch_end(self, outputs):  
        reg_losses       = [x['loss'] for x in outputs]
        avg_loss         = torch.stack(reg_losses).mean()
        tensorboard_logs = {'train_loss': avg_loss}
        # only report anneal param in progress bar for SSM
        dict_ = {}
        if self.hparams['model_name'] == 'ssm' or self.hparams['model_name'] == 'sfomm': 
            dict_ = {'anneal': self.hparams['anneal']}
        return {'loss': avg_loss, 'log': tensorboard_logs, 'progress_bar': dict_}

    def validation_step(self, batch, batch_idx):
        # import pdb; pdb.set_trace()
        (nelbo, nll, kl, _), _ = self.forward(*batch, anneal = 1.)
        if self.hparams['imp_sampling']: 
            pass
            '''
            batch_nll      = []
            for i, valid_batch_loader in enumerate(valid_loader): 
                nll_estimate   = self.imp_sampling(*valid_batch_loader, nelbo, anneal = 1.)
                nll_estimate   = nll_estimate.item()
                batch_nll.append(nll_estimate)
            nll_estimate = np.mean(batch_nll)
            '''
        if self.hparams['eval_type'] != 'nelbo': 
            if 'ord' not in self.hparams['loss_type']: 
                preds, _ = self.predict(*batch)
            else: 
                preds, _ = self.predict_ord(*batch)
            return self.compute_metrics(preds, batch, (nelbo, nll, kl))
            
        return {'val_loss': nelbo, 'nll': nll, 'kl': kl}

    def validation_epoch_end(self, outputs): 
        nelbos = [x['val_loss'] for x in outputs]; nlls = [x['nll'] for x in outputs] 
        kls    = [x['kl'] for x in outputs]
        avg_nelbo = torch.stack(nelbos).mean()
        avg_nll   = torch.stack(nlls).mean()
        avg_kl    = torch.stack(kls).mean()
        tensorboard_logs = {'val_NELBO': avg_nelbo, 'val_nll': avg_nll, 'val_kl': avg_kl}

        if self.hparams['eval_type'] != 'nelbo': 
            return self.average_metrics(tensorboard_logs, outputs)

        # self.logger.log_metrics(tensorboard_logs)
        return {'val_loss': avg_nelbo, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}

    def average_metrics(self, tensorboard_logs, outputs):         
        avg_nelbo = tensorboard_logs['val_NELBO']
        if self.hparams['eval_type'] == 'mse': 
            avg_mse = np.mean([x['mse'] for x in outputs])
            tensorboard_logs['mse'] = avg_mse
            return {'val_loss': avg_nelbo, 'mse': avg_mse, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}
        elif self.hparams['eval_type'] == 'f1': 
            avg_f1 = torch.stack([x['F1'] for x in outputs]).mean()
            avg_pr = torch.stack([x['precision'] for x in outputs]).mean()
            avg_re = torch.stack([x['recall'] for x in outputs]).mean()
            tensorboard_logs['f1'] = avg_f1; tensorboard_logs['precision'] = avg_pr; tensorboard_logs['recall'] = avg_re
            return {'val_loss': avg_nelbo, 'F1': avg_f1, 'Pr': avg_pr, 'Re': avg_re, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}
        elif self.hparams['eval_type'] == 'auc': 
            avg_auc = torch.stack([x['auc'] for x in outputs]).mean()
            tensorboard_logs['auc'] = avg_auc
            return {'val_loss': avg_nelbo, 'auc': avg_auc, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}

    def compute_metrics(self, preds, tensors, metrics):
        nelbo, nll, kl    = metrics 
        B, X, A, M, Y, CE = tensors
        if self.hparams['eval_type'] == 'mse': 
            mse, r2, ci = calc_stats(preds, tensors)
            return {'val_loss': nelbo, 'nll': nll, 'kl': kl, 'mse': mse, 'r2': r2, 'ci': ci}
        elif self.hparams['eval_type'] == 'f1': 
            f1 = self.f1(preds.argmax(dim=1), Y)
            p  = self.precision(preds.argmax(dim=1), Y)
            r  = self.recall(preds.argmax(dim=1), Y)
            return {'val_loss': nelbo, 'nll': nll, 'kl': kl, 'F1': f1, 'precision': p, 'recall': r}
        elif self.hparams['eval_type'] == 'auc': 
            auc = auroc(preds.argmax(dim=1), Y)
            return {'val_loss': nelbo, 'nll': nll, 'kl': kl, 'auc': auc}
        else: 
            raise ValueError('bad metric specified...')

    def configure_optimizers(self): 
        if self.hparams['optimizer_name'] == 'adam': 
            # opt = torch.optim.Adam(self.parameters(), lr=self.hparams['lr']) 
            opt = torch.optim.Adam(self.parameters(), lr=self.lr) 
            return opt
        elif self.hparams['optimizer_name'] == 'rmsprop': 
            opt = torch.optim.RMSprop(self.parameters(), lr=self.hparams['lr'], momentum=.001)
        elif self.hparams['optimizer_name'] == 'swa': 
            opt = torch.optim.Adam(self.parameters(), lr=self.hparams['lr']) 
            return SWA(opt, swa_start=100, swa_freq=50, swa_lr=self.hparams['lr'])

    def setup(self, stage): 
        fold = self.hparams['fold']
        if self.hparams['dataset'] == 'mm': 
            # ddata = load_mmrf(fold_span = [fold], \
            #                   digitize_K = 0, \
            #                   digitize_method = 'uniform', \
            #                   suffix='_2mos_tr', \
            #                   restrict_markers=True, \
            #                   add_syn_marker=True, \
            #                   window='first_second')
            ddata = load_mmrf(fold_span = [fold], digitize_K = 20, digitize_method = 'uniform', suffix='_2mos')

        elif self.hparams['dataset'] == 'synthetic': 
            nsamples        = {'train':self.hparams['nsamples_syn'], 'valid':1000, 'test':200}
            print(f'training on {nsamples["train"]} samples')
            alpha_1_complex = False; per_missing = 0.; add_feat = 0; num_trt = 1
            ddata = load_synthetic_data_trt(fold_span = [fold], \
                                            nsamples = nsamples, \
                                            distractor_dims_b=4, \
                                            sigma_ys=0.7, \
                                            include_line=True, \
                                            alpha_1_complex=alpha_1_complex, \
                                            per_missing=per_missing, \
                                            add_feats=add_feat, \
                                            num_trt=num_trt, \
                                            sub=True)

        if self.hparams['dataset'] == 'mm' or self.hparams['dataset'] == 'synthetic': 
            self.hparams['dim_base']  = ddata[fold]['train']['b'].shape[-1]
            self.hparams['dim_data']  = ddata[fold]['train']['x'].shape[-1]
            self.hparams['dim_treat'] = ddata[fold]['train']['a'].shape[-1]

        if self.hparams['dataset'] == 'semi_synthetic': 
            ddata = load_ss_data(self.hparams['nsamples_syn'], \
                                 add_missing=self.hparams['ss_missing'], \
                                 gen_fly=True, \
                                 in_sample_dist=self.hparams['ss_in_sample_dist'])
            self.hparams['dim_base']  = ddata['train']['B'].shape[-1]
            self.hparams['dim_data']  = ddata['train']['X'].shape[-1]
            self.hparams['dim_treat'] = ddata['train']['A'].shape[-1]
            print(f'shape of training data:{ddata["train"]["X"].shape}')
        if self.hparams['eval_type'] == 'f1': 
            self.f1 = F1(average='weighted')
            self.precision = Precision(average='weighted')
            self.recall = Recall(average='weighted')

        self.ddata = ddata 
        self.init_model()
        
    def load_helper(self, tvt, device=None, oversample=True, att_mask=False):
        fold = self.hparams['fold']; batch_size = self.bs

        if device is not None: 
            B  = torch.from_numpy(self.ddata[fold][tvt]['b'].astype('float32')).to(device)
            X  = torch.from_numpy(self.ddata[fold][tvt]['x'].astype('float32')).to(device)
            A  = torch.from_numpy(self.ddata[fold][tvt]['a'].astype('float32')).to(device)
            M  = torch.from_numpy(self.ddata[fold][tvt]['m'].astype('float32')).to(device)
        else: 
            B  = torch.from_numpy(self.ddata[fold][tvt]['b'].astype('float32'))
            X  = torch.from_numpy(self.ddata[fold][tvt]['x'].astype('float32'))
            A  = torch.from_numpy(self.ddata[fold][tvt]['a'].astype('float32'))
            M  = torch.from_numpy(self.ddata[fold][tvt]['m'].astype('float32'))

        y_vals   = self.ddata[fold][tvt]['ys_seq'][:,0].astype('float32')
        idx_sort = np.argsort(y_vals)

        if 'digitized_y' in self.ddata[fold][tvt]:
            print ('using digitized y')
            Y  = torch.from_numpy(self.ddata[fold][tvt]['digitized_y'].astype('float32'))
        else:
            Y  = torch.from_numpy(self.ddata[fold][tvt]['ys_seq'][:,[0]]).squeeze()

        if device is not None: 
            Y = Y.to(device)
            CE = torch.from_numpy(self.ddata[fold][tvt]['ce'].astype('float32')).to(device)
        else: 
            CE = torch.from_numpy(self.ddata[fold][tvt]['ce'].astype('float32'))

        if att_mask: 
            attn_shape  = (A.shape[0],A.shape[1],A.shape[1])
            Am   = get_attn_mask(attn_shape, self.ddata[fold][tvt]['a'].astype('float32'), device)
            data = TensorDataset(B[idx_sort], X[idx_sort], A[idx_sort], M[idx_sort], Y[idx_sort], CE[idx_sort], Am[idx_sort])
        else: 
            data = TensorDataset(B[idx_sort], X[idx_sort], A[idx_sort], M[idx_sort], Y[idx_sort], CE[idx_sort])
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
        # if tvt == 'train': 
        #     data        = resample(data, device)
        #     data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
        # elif tvt == 'valid' and not oversample:
        #     data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
        # else: 
        #     data        = resample(data, device)
        #     data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
        return data, data_loader

    @pl.data_loader
    def train_dataloader(self):
        if self.hparams['dataset'] == 'mm' or self.hparams['dataset'] == 'synthetic':
            _, train_loader = self.load_helper(tvt='train', att_mask=self.hparams['att_mask'])
        elif self.hparams['dataset'] == 'semi_synthetic': 
            _, train_loader = load_ss_helper(self.ddata, tvt='train', bs=self.bs)
        return train_loader

    @pl.data_loader
    def val_dataloader(self):
        if self.hparams['dataset'] == 'mm' or self.hparams['dataset'] == 'synthetic':
            _, valid_loader = self.load_helper(tvt='valid', att_mask=self.hparams['att_mask'])
        elif self.hparams['dataset'] == 'semi_synthetic': 
            _, valid_loader = load_ss_helper(self.ddata, tvt='valid', bs=self.bs)
        return valid_loader

