import pytorch_lightning as pl
import sys
import os
from pytorch_lightning import Trainer, seed_everything
from argparse import ArgumentParser
from pytorch_lightning.loggers import TensorBoardLogger, TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from models.fomm import FOMM
from models.ssm.ssm import SSM
from models.rnn import GRU 
from models.sfomm import SFOMM
from distutils.util import strtobool


def main(args):
    dict_args = vars(args)

    # pick model
    if args.model_name == 'fomm':
        model = FOMM(**dict_args)
    elif args.model_name == 'ssm': 
        model = SSM(**dict_args)
    elif args.model_name == 'gru':
        model = GRU(**dict_args)
    elif args.model_name == 'sfomm': 
        model = SFOMM(**dict_args)

    if not args.logger: 
        logger = False
    elif args.fname is not None: 
        logger = TensorBoardLogger('tbp_logs', name=os.path.join('tbp_logs',args.fname))
    else: 
        logger = TensorBoardLogger('tbp_logs')

    if args.checkpoint_callback: 
        checkpoint_callback = ModelCheckpoint(filepath='./tbp_logs/checkpoints/' + args.fname + '_{epoch:05d}-{val_loss:.2f}')
    else: 
        checkpoint_callback = False
    trainer = Trainer.from_argparse_args(args, deterministic=True, logger=logger, gpus=[args.gpu_id], checkpoint_callback=checkpoint_callback)
    trainer.fit(model)

if __name__ == '__main__':
    parser = ArgumentParser()

    # figure out which model to use and other basic params
    parser.add_argument('--model_name', type=str, default='ssm', help='fomm, ssm, or gru')
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
    parser.add_argument('--bs', default=600, type=int, help='batch size')
    parser.add_argument('--fold', default=1, type=int)
    parser.add_argument('--seed', default=0, type=int)

    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()

    seed_everything(temp_args.seed)

    # let the model add what it wants
    if temp_args.model_name == 'fomm': 
        parser = FOMM.add_model_specific_args(parser)
    elif temp_args.model_name == 'ssm': 
        parser = SSM.add_model_specific_args(parser)
    elif temp_args.model_name == 'gru': 
        parser = GRU.add_model_specific_args(parser)
    elif temp_args.model_name == 'sfomm': 
        parser = SFOMM.add_model_specific_args(parser)

    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    
    # train
    main(args)
