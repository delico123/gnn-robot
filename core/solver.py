import logging

import torch
import torch.nn as nn
import wandb

from core.model import build_rstruc_model
from core.optimizer import get_optimizer


# TODO: batch size error (dimension)

class Solver(nn.Module):
    # TODO: checkpoints
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # W&B Sweep for rs
        if args.rs_sweep:
            config_defaults = {
                'optimizer': args.opt,
                'learning_rate': args.rs_lr,
                'latent_size': args.rs_latent
            }
            # Init each wandb run
            wandb.init(config=config_defaults)

            # Define config
            config = wandb.config

            self.net, config = build_rstruc_model(args, config)
        
        else:
            self.net, config = build_rstruc_model(args)
            
            if args.wnb:        
                # Init wandb run
                # name, project, run
                wandb.init(config=config,
                            tags=[args.rs_conv],
                            name=args.rs_conv # Run name
                            # project= # Project name. Default: gnn-robot
                            )

                wandb.config.update({'data_simple':args.data_simple,  # True: joint3 only
                                        'wnb_note':args.wnb_note})

                # wandb magic
                wandb.watch(self.net, log_freq=100)
            
        # Optimizer
        self.optimizer = get_optimizer(config, self.net)

        self.to(self.device)

    def train_rstruc(self, data_loader):
        """ train structure reconstruction model (rs)
        """
        args = self.args
        net = self.net
        optimizer = self.optimizer

        for epoch in range(args.rs_epoch):
            net.train()

            total_loss = 0
            for ii, data in enumerate(data_loader):
                optimizer.zero_grad()

                z = net.encode(data.x, data.y, data.edge_index)
                # output = net.decode(z, data.y, data.edge_index)
                output = net.decode(z, args.node_padding, data.y, data.edge_index) # for debug

                loss = net.recon_loss(predict=output[:data.y], target=data.x[:data.y])
                # loss = net.recon_loss(predict=output, target=data.x)
                total_loss += loss.item()

                loss.backward()
                optimizer.step()


            if args.rs_sweep or args.wnb:
                wandb.log({"loss": total_loss/len(data_loader)})

            if epoch % args.log_per == 0:
                logging.info(f"Epoch: {epoch}, Loss: {total_loss/len(data_loader)}")
                logging.debug(f"Z: {z}, O: {output}, T:{data.x}")