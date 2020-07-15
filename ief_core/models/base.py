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
sys.path.append('../data/ml_mmrf')
sys.path.append('../data/')
from ml_mmrf_v1.data import load_mmrf
from synthetic.synthetic_data import load_synthetic_data_trt, load_synthetic_data_noisy

class Model(pl.LightningModule): 

    def __init__(self, 
                 lr: float = 1e-3, 
                 anneal: float = 1., 
                 imp_sampling: bool = False, 
                 optimizer_name: str = 'adam',
                 fname = None, 
                 **kwargs
                ): 
        super().__init__()
        torch.manual_seed(0)
        np.random.seed(0)
        if torch.cuda.is_available(): # don't need 
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
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
        # use KL annealing for DMM 
        self.hparams['anneal'] = min(1, self.current_epoch/(self.hparams['max_epochs']*0.5))
        _, loss  = self.forward(*dt, anneal = self.hparams['anneal']) 
        return {'loss': loss}

    def training_epoch_end(self, outputs):  
        reg_losses       = [x['loss'] for x in outputs]
        avg_loss         = torch.stack(reg_losses).mean()
        tensorboard_logs = {'train_loss': avg_loss}
        # only report anneal param in progress bar for SSM
        dict_ = {}
        if self.hparams['model_name'] == 'ssm': 
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
        return {'val_loss': nelbo, 'nll': nll, 'kl': kl}

    def validation_epoch_end(self, outputs): 
        nelbos = [x['val_loss'] for x in outputs]; nlls = [x['nll'] for x in outputs] 
        kls    = [x['kl'] for x in outputs]
        avg_nelbo = torch.stack(nelbos).mean()
        avg_nll   = torch.stack(nlls).mean()
        avg_kl    = torch.stack(kls).mean()
        tensorboard_logs = {'val_NELBO': avg_nelbo, 'val_nll': avg_nll, 'val_kl': avg_kl}
        return {'val_loss': avg_nelbo, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self): 
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams['lr']) 
        if self.hparams['optimizer_name'] == 'adam': 
            return opt
        elif self.hparams['optimizer_name'] == 'swa': 
            return SWA(opt, swa_start=100, swa_freq=50, swa_lr=self.hparams['lr'])

    def setup(self, stage): 
        fold = self.hparams['fold']
        if self.hparams['dataset'] == 'mm': 
            ddata = load_mmrf(fold_span = [fold], \
                              digitize_K = 20, \
                              digitize_method = 'uniform', \
                              suffix='_2mos')

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

        self.hparams['dim_base']  = ddata[fold]['train']['b'].shape[-1]
        self.hparams['dim_data']  = ddata[fold]['train']['x'].shape[-1]
        self.hparams['dim_treat'] = ddata[fold]['train']['a'].shape[-1]
        self.ddata = ddata 
        self.init_model()
        
    def load_helper(self, tvt):
        fold = self.hparams['fold']; batch_size = self.hparams['bs']
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
            Y  = torch.from_numpy(self.ddata[fold][tvt]['ys_seq'][:,[0]].astype('float32'))
        CE = torch.from_numpy(self.ddata[fold][tvt]['ce'].astype('float32'))

        data        = TensorDataset(B[idx_sort], X[idx_sort], A[idx_sort], M[idx_sort], Y[idx_sort], CE[idx_sort])
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
        return data, data_loader

    @pl.data_loader
    def train_dataloader(self):
        _, train_loader = self.load_helper(tvt='train')
        return train_loader

    @pl.data_loader
    def val_dataloader(self):
        _, valid_loader = self.load_helper(tvt='valid')
        return valid_loader

