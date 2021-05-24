import os
import logging
import argparse

import torch
import wandb

from core.data_loader import get_structure_loader
from core.solver import Solver
# from core.solver


def main(args):
    logging.info(args)
    torch.manual_seed(args.seed)

    # MAIN
    solver = Solver(args)

    if args.mode == 'rstruc':
        logging.info('*--Structure reconstruction task--*')
        

        # Data Loader
        structure_data_loader = get_structure_loader(task=args.task, 
                                                     batch_size=args.rs_bs
                                                     )

        # Train
        solver.train_rstruc(structure_data_loader)


    # elif args.mode == 'train':
    #     print('*--TRAIN--*')

    # elif args.mode == 'test':
    #     print('--*TEST*--')

    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=2021,
                        help='Seed')
    
    # mode
    parser.add_argument('--mode', type=str, default='rstruc',
                        choices=['rstruc', 'rdynam', 'train', 'eval'],
                        help='')
    # parser.add_argument('--recon_source', type=str, default='train',
    #                     choices=['train', 'load'],
    #                     help='')
    # parser.add_argument()

    parser.add_argument('--task', type=str, default='Reacher',
                        # choices=['Reacher'],
                        help='')
    parser.add_argument('--rs_epoch', type=int, default=200,
                        help='Recontruction task, structure, epoch.')
    parser.add_argument('--rs_bs', type=int, default=16,
                        help='Recontruction task, structure, batch_size.')
    parser.add_argument('--rs_lr', type=float, default=0.01,
                        help='Recontruction task, structure, learning_rate.')
    parser.add_argument('--rs_latent', type=int, default=16,
                        help='Recontruction task, structure, latent_size.')
    parser.add_argument('--rs_conv', type=str, default='gcn',
                        choices=['gcn','tree'],
                        help='Select conv')
    parser.add_argument('--rs_sweep', action='store_true', default=False,
                        help='Enable W&B hyperparam sweep')

    
    
    
    
    # parser.add_argument('--test_mode', action='store_true', default=False,
    #                     help='')

    # misc
    parser.add_argument('-d', '--debug',
                        action='store_const', dest='loglevel', const=logging.DEBUG,
                        default=logging.WARNING,
                        help="Debugging")

    parser.add_argument('-i', '--info',
                        action='store_const', dest='loglevel', const=logging.INFO,
                        help="Info")

    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel)

    # W&B
    if args.rs_sweep:
        sweep_config = {
            # 'method': 'random', #grid, random
            'method': 'grid', #grid, random
            'metric': {
                'name': 'loss',
                'goal': 'minimize'   
            },
            'parameters': {
                'learning_rate': {
                    'values': [0.1, 0.03, 0.01, 0.001]
                },
                'latent_size':{
                    'values': [4, 8, 16, 32]
                },
                # 'optimizer': {
                #     'values': ['adam', 'sgd']
                # }
            }
        }
        sweep_id = wandb.sweep(sweep_config, 
                               project="rstruc-sweep"
                               )
        wandb.agent(sweep_id=sweep_id, function=lambda:main(args), entity="jtk")
        # wandb.agent(sweep_id=sweep_id, function=lambda:main(args), entity="jtk", count=16)

    else:
        main(args)


# TODO: data generator
