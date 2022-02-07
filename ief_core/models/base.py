import torch, os
import torch.nn as nn
import numpy as np
import logging
import pytorch_lightning as pl
import sys 
from lifelines.utils import concordance_index
from sklearn.metrics import r2_score
from torchmetrics.functional import f1, precision_recall, auroc
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torchcontrib.optim import SWA
fpath= os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(fpath,'../../data/ml_mmrf'))
sys.path.append(os.path.join(fpath,'../../data/'))
print (sys.path)
from ml_mmrf.data import load_mmrf
from synthetic.synthetic_data import load_synthetic_data_trt, load_synthetic_data_noisy
from semi_synthetic.ss_data import *
from models.utils import *

class DataModule(pl.LightningDataModule): 
    def __init__(self, hparams, model): 
        super().__init__()
        self.hparams = hparams
        self.model   = model

    def setup(self, stage): 
        ''' 
        When adding a dataset (e.g. VA MM dataset), the data loading function should return 
        the following structure: 
            key: fold number, val: dictionary ==> {key: 'train', 'test', 'valid', \
                val: dictionary ==> {key: data type ('x','m'), val: np matrix}}
        See load_mmrf() function in data.py file in ml_mmrf folder.
        '''
        fold = self.hparams['fold'] 
        if self.hparams['dataset'] == 'mm': 
            data_dir = self.hparams['data_dir']
            if self.hparams['data_dir'] == 'cluster':
                data_dir = os.path.join(os.environ['PT_DATA_DIR'],'ml_mmrf','ml_mmrf','output','cleaned_mm_fold_2mos.pkl')
            elif self.hparams['data_dir'] == 'cluster_comb3':
                data_dir = os.path.join(os.environ['PT_DATA_DIR'],'ml_mmrf','ml_mmrf','output','cleaned_mm_fold_2mos_comb3.pkl')
            elif self.hparams['data_dir'] == 'cluster_comb4':
                data_dir = os.path.join(os.environ['PT_DATA_DIR'],'ml_mmrf','ml_mmrf','output','cleaned_mm_fold_2mos_comb4.pkl')
            ddata = load_mmrf(fold_span = [fold], \
                              data_dir  = data_dir, \
                              digitize_K = 20, \
                              digitize_method = 'uniform', \
                              restrict_markers=[], \
                              add_syn_marker=False, \
                              window='all', \
                              data_aug=False, \
                              ablation=True, \
                              feats=[self.hparams['include_baseline'], self.hparams['include_treatment']])
            # restrict baseline to only first six features  
            for t in ['train', 'valid', 'test']: 
                ddata[fold][t]['b'] = np.concatenate((ddata[fold][t]['b'][:,:11],ddata[fold][t]['b'][:,56:]),axis=-1)
                ddata[fold][t]['feature_names'] = np.concatenate((ddata[fold][t]['feature_names'][:11],ddata[fold][t]['feature_names'][56:]),axis=-1)
            # restrict longitudinal features to only four core features
            if self.hparams['restrict_feats']: 
                feats = list(ddata[fold][t]['feature_names_x'])
                kappa_idx = feats.index('serum_kappa'); iga_idx = feats.index('serum_iga')
                igg_idx   = feats.index('serum_igg');   lambda_idx = feats.index('serum_lambda')
                for t in ['train', 'valid', 'test']: 
                    ddata[fold][t]['x'] = ddata[fold][t]['x'][...,[kappa_idx, iga_idx, igg_idx, lambda_idx]]
                    ddata[fold][t]['m'] = ddata[fold][t]['m'][...,[kappa_idx, iga_idx, igg_idx, lambda_idx]]
                    ddata[fold][t]['feature_names_x'] = ddata[fold][t]['feature_names_x'][[kappa_idx,iga_idx,igg_idx,lambda_idx]]

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
        self.model.hparams.update(self.hparams)
        self.model.init_model()
        
    def load_helper(self, tvt, device=None, oversample=True, att_mask=False):
        fold = self.hparams['fold']; batch_size = self.hparams['bs']
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
        return data, data_loader
    
    def load_resplit(self, tvt, device=None, att_mask=False, include_test=True): 
        fold = self.hparams['fold']; batch_size = self.hparams['bs']
        
        # concatenate all tensors 
        if include_test: 
            Xnp = np.concatenate((self.ddata[fold]['train']['x'],self.ddata[fold]['valid']['x'],self.ddata[fold]['test']['x']),axis=0)
            Bnp = np.concatenate((self.ddata[fold]['train']['b'],self.ddata[fold]['valid']['b'],self.ddata[fold]['test']['b']),axis=0)
            Ynp = np.concatenate((self.ddata[fold]['train']['ys_seq'],self.ddata[fold]['valid']['ys_seq'],\
                                  self.ddata[fold]['test']['ys_seq']),axis=0)
            Anp = np.concatenate((self.ddata[fold]['train']['a'],self.ddata[fold]['valid']['a'],self.ddata[fold]['test']['a']),axis=0)
            Mnp = np.concatenate((self.ddata[fold]['train']['m'],self.ddata[fold]['valid']['m'],self.ddata[fold]['test']['m']),axis=0)
            CEnp = np.concatenate((self.ddata[fold]['train']['ce'],self.ddata[fold]['valid']['ce'],self.ddata[fold]['test']['ce']),axis=0)
            pids_np = np.concatenate((self.ddata[fold]['train']['pids'],self.ddata[fold]['valid']['pids'],\
                                      self.ddata[fold]['test']['pids']),axis=0)
        else: 
            Xnp = np.concatenate((self.ddata[fold]['train']['x'],self.ddata[fold]['valid']['x']),axis=0)
            Bnp = np.concatenate((self.ddata[fold]['train']['b'],self.ddata[fold]['valid']['b']),axis=0)
            Ynp = np.concatenate((self.ddata[fold]['train']['ys_seq'],self.ddata[fold]['valid']['ys_seq']),axis=0)
            Anp = np.concatenate((self.ddata[fold]['train']['a'],self.ddata[fold]['valid']['a']),axis=0)
            Mnp = np.concatenate((self.ddata[fold]['train']['m'],self.ddata[fold]['valid']['m']),axis=0)
            CEnp = np.concatenate((self.ddata[fold]['train']['ce'],self.ddata[fold]['valid']['ce']),axis=0)
            pids_np = np.concatenate((self.ddata[fold]['train']['pids'],self.ddata[fold]['valid']['pids']),axis=0)
        
        ##
        if device is not None: 
            B  = torch.from_numpy(Bnp.astype('float32')).to(device)
            X  = torch.from_numpy(Xnp.astype('float32')).to(device)
            A  = torch.from_numpy(Anp.astype('float32')).to(device)
            M  = torch.from_numpy(Mnp.astype('float32')).to(device)
        else: 
            B  = torch.from_numpy(Bnp.astype('float32'))
            X  = torch.from_numpy(Xnp.astype('float32'))
            A  = torch.from_numpy(Anp.astype('float32'))
            M  = torch.from_numpy(Mnp.astype('float32'))
            
        if 'digitized_y' in self.ddata[fold]['train']:
            print ('using digitized y')
            if include_test: 
                Ynp = np.concatenate((self.ddata[fold]['train']['digitized_y'],self.ddata[fold]['valid']['digitized_y'],\
                              self.ddata[fold]['test']['digitized_y']),axis=0)
            else: 
                Ynp = np.concatenate((self.ddata[fold]['train']['digitized_y'],\
                                      self.ddata[fold]['valid']['digitized_y']),axis=0)
            Y  = torch.from_numpy(Ynp.astype('float32'))
        else:
            Y  = torch.from_numpy(Ynp.astype('float32')[:,[0]]).squeeze()

        if device is not None: 
            Y = Y.to(device)
            CE = torch.from_numpy(CEnp.astype('float32')).to(device)
        else: 
            CE = torch.from_numpy(CEnp.astype('float32'))

        from sklearn.model_selection import train_test_split
        train_idxs, test_idxs = train_test_split(np.arange(Bnp.shape[0]),test_size=0.2,stratify=CEnp,random_state=42)
        
        if att_mask: 
            attn_shape = (A.shape[0],A.shape[1],A.shape[1])
            Am   = get_attn_mask(attn_shape, Anp, device)
        if tvt == 'train': 
            B  = B[train_idxs]; X = X[train_idxs]; A = A[train_idxs]; M = M[train_idxs]; Y = Y[train_idxs]; CE = CE[train_idxs]
            if att_mask: 
                Am = Am[train_idxs]
        elif tvt == 'valid' or tvt == 'test': 
            B = B[test_idxs]; X = X[test_idxs]; A = A[test_idxs]; M = M[test_idxs]; Y = Y[test_idxs]; CE = CE[test_idxs]
            if att_mask: 
                Am = Am[test_idxs]
                
        if att_mask: 
            data = TensorDataset(B, X, A, M, Y, CE, Am)
        else: 
            data = TensorDataset(B, X, A, M, Y, CE)
        
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
        return data, data_loader

    def train_dataloader(self):
        if self.hparams['dataset'] == 'mm':
            _, train_loader = self.load_resplit(tvt='train', att_mask=self.hparams['att_mask'], include_test=False)
        elif self.hparams['dataset'] == 'synthetic':
            _, train_loader = self.load_helper(tvt='train', att_mask=self.hparams['att_mask'])
        elif self.hparams['dataset'] == 'semi_synthetic': 
            _, train_loader = load_ss_helper(self.ddata, tvt='train', bs=self.hparams['bs'])
        return train_loader

    def val_dataloader(self):
        if self.hparams['dataset'] == 'mm':
            _, valid_loader = self.load_resplit(tvt='valid', att_mask=self.hparams['att_mask'], include_test=False)
        elif self.hparams['dataset'] == 'synthetic':
            _, valid_loader = self.load_helper(tvt='valid', att_mask=self.hparams['att_mask'])
        elif self.hparams['dataset'] == 'semi_synthetic': 
            _, valid_loader = load_ss_helper(self.ddata, tvt='valid', bs=self.hparams['bs'])
        return valid_loader
    

class Model(pl.LightningModule): 

    def __init__(self, trial, **kwargs): 
        super().__init__()
        torch.manual_seed(0)
        np.random.seed(0)
        if torch.cuda.is_available(): # don't need 
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        self.trial = trial
        self.bs = trial.suggest_categorical('bs', [600,1500])
        self.lr = 1e-3
        self.C  = trial.suggest_categorical('C', [0.0,.001,.01,.1,1,10])
        self.reg_all  = trial.suggest_categorical('reg_all', ['all', 'except_multi_head', 'except_multi_head_ief', True, False])
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
        self.log('loss', avg_loss)
        self.log('log', tensorboard_logs)
        self.log('progress_bar', dict_)

    def validation_step(self, batch, batch_idx):
        (nelbo, nll, kl, _), _ = self.forward(*batch, anneal = 1.)
        if self.hparams['imp_sampling']: 
            batch_nll      = []
            for i, valid_batch_loader in enumerate(valid_loader): 
                nll_estimate   = self.imp_sampling(*valid_batch_loader, nelbo, anneal = 1.)
                nll_estimate   = nll_estimate.item()
                batch_nll.append(nll_estimate)
            nll_estimate = np.mean(batch_nll)
            
        if self.hparams['eval_type'] != 'nelbo': 
            if 'ord' not in self.hparams['loss_type']: 
                preds, _ = self.predict(*batch)
            else: 
                preds, _ = self.predict_ord(*batch)
            return self.compute_metrics(preds, batch, (nelbo, nll, kl))
        self.log('val_loss', nelbo, prog_bar=True)
        self.log('nll', nll, prog_bar=True)
        self.log('kl',kl, prog_bar=True)
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
            opt = torch.optim.Adam(self.parameters(), lr=self.lr) 
            return opt
        elif self.hparams['optimizer_name'] == 'rmsprop': 
            opt = torch.optim.RMSprop(self.parameters(), lr=self.hparams['lr'], momentum=.001)
        elif self.hparams['optimizer_name'] == 'swa': 
            opt = torch.optim.Adam(self.parameters(), lr=self.hparams['lr']) 
            return SWA(opt, swa_start=100, swa_freq=50, swa_lr=self.hparams['lr'])
