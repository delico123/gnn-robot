# import random
# import logging
# from numpy import int

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.nn.inits import reset


# TODO: generalize reacher only prop (spread, collect), aggr mean
# TODO: num node, y

class STAE(nn.Module):
    """ Structure tree auto encoder
    """
    def __init__(self, encoder, decoder):
        super(STAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        STAE.reset_parameters(self)

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)

    def encode(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

    def recon_loss(self, predict, target):
        """ Structure recontruction loss
            - MSE Loss from node features
        """
        criterion = nn.MSELoss()
        
        loss = criterion(predict, target)

        return loss


class StrucTreeEncoder(nn.Module):
    def __init__(self, in_=2, latent=16, out_=16, conv='tree'):
        super(StrucTreeEncoder, self).__init__()
        # conv = GCNConv if conv == 'gcn' else TreeConv
        # self.conv1 = conv(in_, latent, cached=True)
        # self.conv2 = conv(latent, out_, cached=True)
        
        """
        in_: int
            len(data.x)
        """
        self.conv_spread = TreeConv(latent, latent)
        # self.conv_spread = GCNConv(latent, latent)
        # self.conv_collect = GCNConv(latent, out_)
        self.conv_collect = TreeConv(latent, out_)

        self.in_ = in_
        self.latent = latent
        self.out_ = out_

    def forward(self, x, num_node, edge_index):
        # logging.info("Encoding..")

        # SPREAD
        # logging.debug("Encoding-1/2- Spread")
        pad = nn.ZeroPad2d((0, self.latent - x.size()[1], 0, 0))
        # print(pad(x))
        x = pad(x)
        for ii, _ in enumerate(x):
            # logging.debug(f"-------- Spread-{ii}")
            if ii + 1 == len(x): # last item
                continue # or break
            next_edge_index = torch.tensor([[ii], [ii+1]]) # TODO generalize (thru data util?)
            x = self.conv_spread(x, next_edge_index)
        # logging.debug(x)

        # COLLECT
        # logging.debug("Encoding-2/2- Collect")
        for ii, x_i in enumerate(reversed(x)):
            # logging.debug(f"-------- Collect-{ii}")
            if ii == 0: # last item (0 -> -1)
                continue # or break
            next_edge_index = torch.tensor([[ii], [ii-1]]) # TODO generalize
            x = self.conv_collect(x, next_edge_index)

        return x[0]


class StrucTreeDecoder(nn.Module):
    def __init__(self, in_=16, latent=16, out_=2, conv='tree'):
        # out_: number of features for each node
        super(StrucTreeDecoder, self).__init__()

        # conv = GCNConv if conv == 'gcn' else TreeConv
        # self.conv1 = conv(in_, latent, cached=True)
        # self.conv2 = conv(latent, out_, cached=True)
        
        """
        in_: int
            len(data.x)
        """
        self.conv_spread = TreeConv(in_, latent)
        # self.conv_spread = GCNConv(in_, latent)
        # self.conv_collect = GCNConv(latent, out_)
        self.conv_collect = TreeConv(latent, latent)
        self.decode_fc = nn.Linear(latent, out_)

        self.latent = latent
        self.out_ = out_

    def forward(self, z, num_node, edge_index):
        # logging.info("Decoding..")

        # init each node feat w/ z
        x = z.repeat(num_node, 1)

        # SPREAD
        for ii, _ in enumerate(x):
            if ii + 1 == len(x): # last item
                continue # or break
            next_edge_index = torch.tensor([[ii], [ii+1]]) # TODO generalize (thru data util?)
            x = self.conv_spread(x, next_edge_index)
        
        # COLLECT
        pad = nn.ConstantPad1d((0, self.latent - self.out_), 0)
        for ii, _ in enumerate(reversed(x)):
            if ii == 0: # last item (0 -> -1)
                continue # or break
            next_edge_index = torch.tensor([[ii], [ii-1]]) # TODO generalize
            x = self.conv_collect(x, next_edge_index)
            # x_i = self.conv_collect(x, next_edge_index)
            # print("!@#!@#")
            # print(ii)
            # print(x_i)
            # x[ii] = pad(x_i[ii])
        
        # PREDICT
        # 각 state을 한 linear에 통과하는 것 반복-> 예측하도록
        output = torch.tensor([])
        for ii, x_i in enumerate(x):
            d_i = self.decode_fc(x_i)
            output = torch.cat([output, d_i.unsqueeze(0)], dim=0)

        return output


"""Conv"""
class TreeConv(MessagePassing): # TODO: Generalize
    def __init__(self, in_=2, out_=16):
        """
        hidden size = out_channel * 2
        """
        super(TreeConv, self).__init__(aggr='add')
        self.lin1 = nn.Linear(in_, out_*2)
        self.lin2 = nn.Linear(out_*2, out_)
        # message 만들때 relu가는걸로

    def forward(self, x, edge_index):
        x = self.lin1(x)
        x = x.relu()
        x = self.lin2(x)
        return self.propagate(edge_index, x=x)



def build_rstruc_model(args, sweep_config=None):
    if args.rs_sweep:
        config = sweep_config
    else:
        config = {
            'learning_rate': args.rs_lr,
            'latent_size': args.rs_latent
        }

    struc_tree_encoder = StrucTreeEncoder(latent=config.latent_size,
                                          out_=config.latent_size)
    struc_tree_decoder = StrucTreeDecoder(in_=config.latent_size,
                                          latent=config.latent_size)

    net = STAE(encoder=struc_tree_encoder,
               decoder=struc_tree_decoder)

    return net, config
