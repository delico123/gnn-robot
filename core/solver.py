import logging

import torch
import torch.nn as nn

from core.model import build_rstruc_model
from core.optimizer import get_optimizer


# TODO: batch size error (dimension)

class Solver(nn.Module):
    # TODO: checkpoints
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.net = build_rstruc_model(args)
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

            if epoch % 10 == 0:
                logging.info(f"Epoch {epoch}: , Loss: {total_loss/len(data_loader)}")