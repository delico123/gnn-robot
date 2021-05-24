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
        
        # W&B Sweep
        if args.rs_sweep:
            config_defaults = {
                'learning_rate': 0.1,
                'latent_size': 4
            }

            # Init each wandb run
            wandb.init(config=config_defaults)

            # Define config
            config = wandb.config

            self.net, config = build_rstruc_model(args, config)
        
        else: 
            self.net, _ = build_rstruc_model(args)
        
        # Optimizer
        self.optimizer = get_optimizer(args, self.net)

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
                output = net.decode(z, data.y, data.edge_index)

                loss = net.recon_loss(predict=output, target=data.x[:data.y])
                total_loss += loss.item()

                loss.backward()
                optimizer.step()


            if args.rs_sweep:
                wandb.log({"loss": total_loss/len(data_loader)})

            if epoch % 10 == 0:
                logging.info(f"Epoch {epoch}: , Loss: {total_loss/len(data_loader)}")