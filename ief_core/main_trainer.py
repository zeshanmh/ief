import pytorch_lightning as pl
import sys
import os
from pytorch_lightning import Trainer, seed_everything
from argparse import ArgumentParser
from pytorch_lightning.loggers import TensorBoardLogger, TestTubeLogger
from models.fomm import FOMM

def main(args):
    dict_args = vars(args)

    # pick model
    if args.model_name == 'fomm':
        model = FOMM(**dict_args)

    if args.fname is not None: 
        logger = TensorBoardLogger('tb_logs', name=os.path.join('tb_logs',args.fname))
    else: 
        logger = TensorBoardLogger('tb_logs')
    trainer = Trainer.from_argparse_args(args)
    trainer.deterministic = True
    trainer.logger        = logger
    trainer.fit(model)

if __name__ == '__main__':
    seed_everything(0)

    parser = ArgumentParser()

    # figure out which model to use and other basic params
    parser.add_argument('--model_name', type=str, default='fomm', help='fomm, ssm, or gru')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--anneal', type=float, default=1., help='annealing rate')
    parser.add_argument('--fname', type=str, help='name of save file')
    parser.add_argument('--imp_sampling', type=bool, default=False, help='importance sampling to estimate marginal likelihood')
    parser.add_argument('--nsamples', default=1, type=int)
    parser.add_argument('--optimizer_name', type=str, default='adam')
    parser.add_argument('--dataset', default='mm', type=str)
    parser.add_argument('--bs', default=600, type=int, help='batch size')
    parser.add_argument('--fold', default=1, type=int)

    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()

    # let the model add what it wants
    if temp_args.model_name == 'fomm': 
        parser = FOMM.add_model_specific_args(parser)

    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # train
    main(args)
