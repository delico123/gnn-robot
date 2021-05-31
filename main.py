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

    # init Solver
    solver = Solver(args)

    if args.mode == 'rstruc':
        logging.info('*--Structure reconstruction task--*')
        

        # Data Loader
        structure_data_loader = get_structure_loader(task=args.task, 
                                                     batch_size=args.rs_bs,
                                                     node_padding=args.node_padding,
                                                     data_simple=args.data_simple
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
    
    # For debugging/testing
    parser.add_argument('--data_simple', action='store_true', default=False,
                        help='Joint==3')

    parser.add_argument('--seed', type=int, default=2021,
                        help='Seed')

    # data
    parser.add_argument('--node_padding', type=int, default=8,
                        help='Fix the node input size. Set to 0 to turn off.')

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
                        choices=['gcn','tree','test_decoder','test_simple_decoder','test_encoder'],
                        help='Select conv')
    parser.add_argument('--rs_sweep', action='store_true', default=False,
                        help='Enable W&B hyperparam sweep')
    parser.add_argument('--rs_sweep_short', action='store_true', default=False,
                        help='Enable W&B hyperparam sweep, short')


    # parser.add_argument('--test_mode', action='store_true', default=False,
    #                     help='')

        
    # W&B
    parser.add_argument('--wnb', action='store_true', default=False,
                        help='Turn on W&B')

    # log
    parser.add_argument('--log_per', type=int, default=10,
                        help='Log interval')

    parser.add_argument('-d', '--debug',
                        action='store_const', dest='loglevel', const=logging.DEBUG,
                        default=logging.WARNING,
                        help="Debugging")

    parser.add_argument('-i', '--info',
                        action='store_const', dest='loglevel', const=logging.INFO,
                        help="Info")

    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel)


    # W&B hyperparam sweep
    if args.rs_sweep:
        logging.info("W&B SWEEP")
        if args.wnb:
            logging.warning("W&B Sweep turned on, regular W&B igonored.")
            
        if args.rs_sweep_short:
            logging.info("SHORT SWEEP")
            param_dict = {
                'learning_rate': {
                    'values': [0.1, 0.03]
                },
                'latent_size':{
                    'values': [4, 8]
                },
            }

        else:
            param_dict = {
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

        sweep_config = {
            'name': args.rs_conv,
            # 'method': 'random', #grid, random
            'method': 'grid', #grid, random
            'metric': {
                'name': 'loss',
                'goal': 'minimize'   
            },
            'parameters': param_dict
        }
        sweep_id = wandb.sweep(sweep_config, 
                               project=f"rstruc-{args.rs_conv}-sweep"
                            #    project="rstruc-sweep"
                               )
        wandb.agent(sweep_id=sweep_id, function=lambda:main(args), entity="jtk")
        # wandb.agent(sweep_id=sweep_id, function=lambda:main(args), entity="jtk", count=16)

    else:
        main(args)


# TODO: data generator
