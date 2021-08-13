import os
import logging
import argparse
import datetime
from pathlib import Path

import torch
import wandb

from core.data_loader import get_structure_loader, get_motion_loader
from core.solver import Solver
# from core.solver

# TODO: dataloader
def main(args):
    logging.info(args)
    torch.manual_seed(args.seed)

    # init Solver
    solver = Solver(args)

    if args.mode == 'rstruc':
        logging.info('*--Structure reconstruction task--*')
        
        # Data Loader
        structure_data_loader = get_structure_loader(task=args.task, 
                                                     eval_ratio=args.eval_ratio,
                                                     batch_size=args.rs_bs,
                                                     node_padding=args.node_padding,
                                                     data_simple=args.data_simple,
                                                     temp_flag=args.temp_flag
                                                     )
        # Train
        solver.train_reconstruc(structure_data_loader)

    elif args.mode == 'rmotion':
        logging.info('*--Motion reconstruction task--*')
        
        # Data Loader
        structure_data_loader = get_motion_loader(task=args.task, 
                                                     eval_ratio=args.eval_ratio,
                                                     batch_size=args.rs_bs,
                                                     node_padding=args.node_padding,
                                                     data_simple=args.data_simple,
                                                     temp_flag=args.temp_flag
                                                     )
        # Train
        solver.train_reconstruc(structure_data_loader)


    elif args.mode == 'train':
        logging.info('*--TRAIN with pretrained models--*')
        data_loader = get_motion_loader(task=args.task, 
                                        eval_ratio=args.eval_ratio,
                                        batch_size=args.rs_bs,
                                        node_padding=args.node_padding,
                                        data_simple=args.data_simple,
                                        temp_flag=args.temp_flag
                                        )
        solver.train(data_loader)
        

    # elif args.mode == 'test':
    #     print('--*TEST*--')

    else:
        logging.warning(f"{args.mode}")
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
                        choices=['rstruc', 'rmotion', 'train', 'eval'],
                        help='')
    # parser.add_argument('--recon_source', type=str, default='train',
    #                     choices=['train', 'load'],
    #                     help='')
    # parser.add_argument()
    parser.add_argument('--freeze', action='store_true', default=False,
                        help='')

    parser.add_argument('--task', type=str, default='Reacher',
                        # choices=['Reacher'],
                        help='')
    parser.add_argument('--subtask', type=str, default='forward',
                        choices=['forward','inverse','multi'],
                        help='')

    parser.add_argument('--eval', type=str, default='val',
                        choices=['train', 'val', 'partial'],
                        help='train: train loss, val: val loss, partial: joint')
    parser.add_argument('--eval_ratio', type=float, default=0.2,
                        help='val: ratio, partial: ')

    # Phase1: Reconstruct structure
    parser.add_argument('--rs_epoch', type=int, default=200,
                        help='Recontruction task, structure, epoch.')
    parser.add_argument('--rs_bs', type=int, default=1,
                        help='Recontruction task, structure, batch_size.')
    parser.add_argument('--opt', type=str, default='adam',
                        choices=['adam', 'sgd'],
                        help='Optimizer')
    parser.add_argument('--opt_eps', type=float, default=0.001, # ddefault: 1e-08
                        help='epsilon')
    parser.add_argument('--rs_lr', type=float, default=0.001,
                        help='Recontruction task, structure, learning_rate.')
    parser.add_argument('--rs_latent', type=int, default=16,
                        help='Recontruction task, structure, latent_size.')
    parser.add_argument('--rs_conv', type=str, default='gcn',
                        choices=['gcn','tree',
                                'test_decoder','test_simple_decoder',
                                'tree_enc_simple_dec'],
                        help='Select conv')
    parser.add_argument('--rs_loss_single', action='store_true', default=False,
                        help='Default (False): separated/multi-task loss')
    parser.add_argument('--rs_sweep', action='store_true', default=False,
                        help='Enable W&B hyperparam sweep')
    parser.add_argument('--rs_sweep_short', action='store_true', default=False,
                        help='Enable W&B hyperparam sweep, short')
    parser.add_argument('--rs_ckpt', type=str, default='log/rs',
                        help='checkpoint dir for rs')
    
    parser.add_argument('--rm_ckpt', type=str, default='log/rm',
                        help='checkpoint dir for rm')

    parser.add_argument('--use_pre_rm', action='store_true', default=False,
                        help='Set True, then load pretrained rmotion.')
    parser.add_argument('--train_ckpt', type=str, default='log/train',
                        help='checkpoint dir for train')

    # TEMP
    parser.add_argument('--rs_dnorm', action='store_true', default=False,
                        help='min 0.1, max0.4 norm')
    parser.add_argument('--gru_update', action='store_true', default=False,
                        help='GGNN-like update, rnn')
    parser.add_argument('--gru_readout', action='store_true', default=False,
                        help='gru readout')

    parser.add_argument('--temp_flag', action='store_true', default=False,
                        help='temp flag')
    


    # parser.add_argument('--test_mode', action='store_true', default=False,
    #                     help='')

        
    # W&B
    parser.add_argument('--wnb', action='store_true', default=False,
                        help='Turn on W&B')
    parser.add_argument('--wnb_note', type=str, default=None,
                        help='Run notes, saved in config')

    # log
    parser.add_argument('--log_per', type=int, default=10,
                        help='Log interval')
    parser.add_argument('--save_latent', action='store_true', default=False,
                        help="Save latent (per 10 epoch)")
    parser.add_argument('-d', '--debug',
                        action='store_const', dest='loglevel', const=logging.DEBUG,
                        default=logging.WARNING,
                        help="Debugging")
    parser.add_argument('-i', '--info',
                        action='store_const', dest='loglevel', const=logging.INFO,
                        help="Info")

    args = parser.parse_args()

    Path(args.rs_ckpt).mkdir(parents=True, exist_ok=True)
    Path(args.rm_ckpt).mkdir(parents=True, exist_ok=True)
    Path(args.train_ckpt).mkdir(parents=True, exist_ok=True)

    if args.save_latent:
        Path("./log/logging/").mkdir(parents=True, exist_ok=True)
        Path("./log/latent/").mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.now()
    now = f"{now.hour}-{now.minute}"
    log_file = f"./log/logging/{args.mode}-{args.rs_conv}-ls_{args.rs_latent}-{now}"
    # logging.basicConfig(level=args.loglevel, filename=log_file)
    logging.basicConfig(level=args.loglevel, 
                        handlers=[
                                logging.FileHandler(log_file),
                                logging.StreamHandler()
                        ])
                    

    # W&B hyperparam sweep
    if args.rs_sweep or args.rs_sweep_short:
        logging.info("W&B SWEEP")
        args.save_latent = False
        if args.wnb:
            logging.warning("W&B Sweep turned on, regular W&B igonored.")
            
        # Short sweep
        if args.rs_sweep_short:
            logging.info("SHORT SWEEP")
            args.rs_sweep = True

            param_dict = {
                'learning_rate': {
                    'values': [0.1, 0.03]
                },
                # 'latent_size': {
                #     'values': [4, 8]
                # },
                'optimizer': {
                    'distribution': 'categorical',
                    'values': ['adam', 'sgd']
                }
            }
        
        # Full sweep
        else:
            param_dict = {
                'learning_rate': {
                    'values': [0.01, 0.001, 0.0001, 0.00001]
                    # 'values': [0.0001]
                },
                'latent_size': {
                    'values': [4, 8, 16, 64]
                    # 'values': [4, 8, 64]
                    # 'values': [4, 8, 64, 1024]
                },
                # 'optimizer': {
                    # 'distribution': 'categorical',
                    # 'values': ['adam', 'sgd']
                # }
                'opt_epsilon': {
                    # 'values': [1.0, 1e-1, 1e-3, 1e-8]
                    'values': [1e-3, 1e-8]
                }
            }

        sweep_config = {
            'name': args.rs_conv,
            'method': 'grid', #grid, random
            # 'method': 'random', #grid, random
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
        # wandb.agent(sweep_id=sweep_id, function=lambda:main(args), entity="jtk", count=10) # limit max num run (for random sweep)

    else:
        main(args)
