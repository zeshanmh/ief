import pytorch_lightning as pl
import sys
sys.path.append('./models')
import os
import optuna 
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning import Trainer, seed_everything
from argparse import ArgumentParser
from pytorch_lightning.loggers import TensorBoardLogger, TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Callback
from models.fomm import FOMM, FOMMAtt
from models.ssm.ssm import SSM, SSMAtt
from models.ssm.sdmm import SDMM 
from models.ssm.ssm_baseline import SSMBaseline 
from models.rnn import GRU 
from models.sfomm import SFOMM
from distutils.util import strtobool


'''
Name: main_trainer.py 
Purpose: High-level training script 
Usage: Not meant to use directly, but rather will be called by launch_run.py.
'''

class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics['val_loss'].item())

def objective(trial, args): 
    dict_args = vars(args)

    # pick model
    if args.model_name == 'fomm':
        model = FOMM(trial, **dict_args)
    elif args.model_name == 'fomm_att':
        model = FOMMAtt(trial, **dict_args)
    elif args.model_name == 'ssm': 
        model = SSM(trial, **dict_args)
    elif args.model_name == 'ssm_att': 
        model = SSMAtt(trial, **dict_args)
    elif args.model_name == 'ssm_baseline': 
        model = SSMBaseline(trial, **dict_args)
    elif args.model_name == 'gru':
        model = GRU(trial, **dict_args)
    elif args.model_name == 'sfomm': 
        model = SFOMM(trial, **dict_args)
    elif args.model_name == 'sdmm': 
        model = SDMM(trial, **dict_args)

    metrics_callback = MetricsCallback()
    if args.ckpt_path != 'none': 
#         checkpoint_callback = ModelCheckpoint(filepath=args.ckpt_path + str(args.fold) + str(args.dim_stochastic) + '_' + args.ttype + '_' + args.include_baseline + args.include_treatment + args.zmatrix + '_ssm_baseablation{epoch:05d}-{val_loss:.2f}')
        checkpoint_callback = ModelCheckpoint(filepath=args.ckpt_path + str(args.nsamples_syn) + str(args.fold) + str(args.dim_stochastic) + '_' + args.ttype + '_ssm_{epoch:05d}-{val_loss:.2f}')
    else: 
        checkpoint_callback = False
    trainer = Trainer.from_argparse_args(args, 
        deterministic=True, 
        logger=False, 
        gpus=[args.gpu_id], 
        checkpoint_callback=checkpoint_callback, 
        callbacks=[metrics_callback], 
#         early_stop_callback=PyTorchLightningPruningCallback(trial, monitor='val_loss')
        early_stop_callback=False,
        progress_bar_refresh_rate=1
    )
    trainer.fit(model)
    return min([x for x in metrics_callback.metrics])

if __name__ == '__main__':
    parser = ArgumentParser()

    # figure out which model to use and other basic params
    parser.add_argument('--model_name', type=str, default='sdmm', help='fomm, ssm, gru or sdmm')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--anneal', type=float, default=1., help='annealing rate')
    parser.add_argument('--fname', type=str, help='name of save file')
    parser.add_argument('--imp_sampling', type=strtobool, default=False, help='importance sampling to estimate marginal likelihood')
    parser.add_argument('--nsamples', default=1, type=int)
    parser.add_argument('--gpu_id', default=1, type=int)
    parser.add_argument('--optimizer_name', type=str, default='adam')
    parser.add_argument('--dataset', default='mm', type=str)
    parser.add_argument('--loss_type', type=str, default='unsup')
    parser.add_argument('--eval_type', type=str, default='nelbo')
    parser.add_argument('--nsamples_syn', default=1000, type=int, help='number of training samples for synthetic data')
    parser.add_argument('--ss_missing', type=strtobool, default=False, help='whether to add missing data in semi synthetic setup or not')
    parser.add_argument('--ss_in_sample_dist', type=strtobool, default=False, help='whether to use mm training patients to generate validation/test set in semi synthetic data')
    parser.add_argument('--att_mask', type=strtobool, default=False, help='set to True for SSMAtt and FOMMAtt')
    parser.add_argument('--bs', default=600, type=int, help='batch size')
    parser.add_argument('--fold', default=1, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--optuna', type=strtobool, default=False, help='whether to use optuna for optimization')
    parser.add_argument('--num_optuna_trials', default=100, type=int)
    parser.add_argument('--include_baseline', type=str, default='all')
    parser.add_argument('--include_treatment', type=str, default='lines')
    parser.add_argument('--ckpt_path', type=str, default='none')
    parser.add_argument('--cluster_run', type=strtobool, default=True)
    parser.add_argument('--data_dir', type=str, \
            default='/afs/csail.mit.edu/u/z/zeshanmh/research/ief/data/ml_mmrf/ml_mmrf/output/cleaned_mm_fold_2mos.pkl')

    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()

    seed_everything(temp_args.seed)

    # let the model add what it wants
    if temp_args.model_name == 'fomm': 
        parser = FOMM.add_model_specific_args(parser)
    elif temp_args.model_name == 'fomm_att': 
        parser = FOMMAtt.add_model_specific_args(parser)
    elif temp_args.model_name == 'ssm': 
        parser = SSM.add_model_specific_args(parser)
    elif temp_args.model_name == 'ssm_att': 
        parser = SSMAtt.add_model_specific_args(parser)
    elif temp_args.model_name == 'ssm_baseline': 
        parser = SSMBaseline.add_model_specific_args(parser)
    elif temp_args.model_name == 'gru': 
        parser = GRU.add_model_specific_args(parser)
    elif temp_args.model_name == 'sfomm': 
        parser = SFOMM.add_model_specific_args(parser)
    elif temp_args.model_name == 'sdmm': 
        parser = SDMM.add_model_specific_args(parser)

    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    
    # train
    if args.optuna: 
        pruner = optuna.pruners.MedianPruner()
        study  = optuna.create_study(direction='minimize', pruner=pruner, study_name=args.fname)
        study.optimize(lambda trial: objective(trial, args), n_trials=args.num_optuna_trials)
        print('Number of finished trials: {}'.format(len(study.trials)))
        print('Best trial:')
        trial = study.best_trial
        print('  Value: {}'.format(trial.value))
        print('  Params: ')
        for key, value in trial.params.items():
            print('    {}: {}'.format(key, value))
        if not os.path.exists('./tbp_logs/optuna_logs'):
            os.mkdir('./tbp_logs/optuna_logs')
        with open(os.path.join('./tbp_logs/optuna_logs/', args.fname + '.txt'), 'a') as fi: 
            fi.write('[OPTUNA STUDY]\n')
            fi.write('command line args: ' + ' '.join(sys.argv[1:]) + '\n\n')
            fi.write(f'\tNumber of finished trials: {len(study.trials)}\n')
            fi.write(f'\tBest trial:\n')
            fi.write(f'\t\tValue: {trial.value}\n')
            for k,v in trial.params.items(): 
                fi.write(f'\t\t{k}: {v}\n')
    else: 
        trial = optuna.trial.FixedTrial({'bs': args.bs, 'lr': args.lr, 'C': args.C, 'reg_all': args.reg_all, 'reg_type': args.reg_type, 'dim_stochastic': args.dim_stochastic})    
        best_nelbo = objective(trial, args)
        print(f'BEST_NELBO: {best_nelbo}')
