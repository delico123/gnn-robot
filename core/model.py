# import random
import logging
# from numpy import int

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.nn.inits import reset

# temp
from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.typing import Adj, Size

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
        # criterion = nn.MSELoss()
        criterion = nn.L1Loss()

        loss = criterion(predict, target)

        return loss


class StrucTreeEncoder(nn.Module): # N/A
    def __init__(self, in_=2, latent=16, out_=16, conv='tree'):
        super(StrucTreeEncoder, self).__init__()
        """
        in_: int
            len(data.x)
        """
        self.tree_encoder = TreeConv(in_, latent, latent) # TODO: split latent and out_

    def forward(self, x, num_node, edge_index):
        # logging.info("Encoding..")

        x = self.tree_encoder(x, edge_index, num_node)

        return x[0]


class StrucTreeDecoder(nn.Module):
    def __init__(self, in_=16, latent=16, out_=2):
        # out_: number of features for each node
        super(StrucTreeDecoder, self).__init__()

        """
        in_: int
            len(z)
        """
        self.tree_decoder = TreeConv(in_, latent, latent)
        self.readout = nn.Linear(latent, out_)

    def forward(self, z, node_max, num_node, edge_index):
        # logging.debug("Decoding..")

        # init each node feat w/ z
        x = z.repeat(node_max, 1)

        # Decoder part 1: message passing tree
        x = self.tree_decoder(x, edge_index, num_node)
        
        # Decoder part 2: linear, predict.
        output = torch.tensor([])
        for x_i in x:
            d_i = self.readout(x_i)
            output = torch.cat([output, d_i.unsqueeze(0)], dim=0)

        return output


"""Conv"""
class TreeConv(MessagePassing): # TODO: Generalize
    def __init__(self, in_=2, out_=16, latent=16, loc_msg=False):
        """
        (down==spread==root->leaf, up==collect==l->r)

        in == |node|
        latent == |message| == |hidden_down|
        out_ == |hidden_down|
        """
        super(TreeConv, self).__init__(aggr='add')

        # latent size
        _msg_down = _msg_up = _lnt_down = latent
        _lnt_up = out_
        _edge = 0

        self.empty_msg = torch.zeros(_msg_down)
        # self.loc_msg = loc_msg
        self.latent = latent

        # self.lin_root = nn.Linear(in_, latent) # separate fn for initial state

        # message fn
        self.message_down = nn.Sequential(nn.Linear(_lnt_down+_edge, _msg_down),
                                    #   nn.BatchNorm1d(latent),
                                    #   nn.ReLU())
                                      nn.Sigmoid())
        self.message_up = nn.Sequential(nn.Linear(_lnt_up+_edge, _msg_up),
                                    # nn.BatchNorm1d(out_),
                                    # nn.ReLU())
                                    nn.Sigmoid())
    
        # update fn
        self.update_down = nn.Sequential(nn.Linear(in_+_msg_down, _lnt_down),
        # self.update_down = nn.Sequential(nn.Linear(in_+_msg_down+8, _lnt_down),
                                    nn.Sigmoid())
        self.update_up = nn.Sequential(nn.Linear(_lnt_down+_msg_up, _lnt_up),
                                    nn.Sigmoid())


    def _organize_edges(self, edge_index, num_node):
        # TODO: generalize to tree (mltr), use edge_index, remove num_node
        # direction[step[ins[], outs[]]]
        
        prt = range(0, num_node-1)
        chd = range(1, num_node)

        down_ei = torch.tensor([[[p],[c]] for p, c in zip(prt, chd)])
        up_ei = down_ei.flip([0,1])

        return down_ei, up_ei

    def forward(self, x, edge_index, num_node=8):
        # Edges
        down_ei, up_ei = self._organize_edges(edge_index, num_node)
       
        # Node vectors
        # x = self.lin_root(x) ###
        # x = x.relu()

        # SPREAD
        x = self.propagate_oneway(down_ei, x, self.message_down, self.update_down)

        # COLLECT
        self.loc_msg = False
        x = self.propagate_oneway(up_ei, x, self.message_up, self.update_up)
        # raise NotImplementedError

        return x

    def propagate_oneway(self, edge_indices, x, fn_m, fn_u):
        """ x -> h
        """
        # 0에 있는 node에 대해 update 먼저 진행. msg는 0으로. (OR, use separate fn)
        m_e = self.empty_msg.repeat(x.shape[0], 1)
        h = torch.cat([x, m_e], dim=1)
        # if self.loc_msg:
        #     mmm = torch.zeros(8)
        #     mmm = mmm.repeat(x.shape[0], 1)
        #     h = torch.cat([h, mmm], dim=1)
        h = fn_u(h)

        for edge_index in edge_indices:
            # logging.debug(edge_index)
            # logging.debug(x)
            h = self.propagate(edge_index=edge_index, x=x, h=h.clone(), fn_m=fn_m, fn_u=fn_u)
            # h[edge_index[1]] = h_updated[edge_index[1]]
        return h
        
    def message(self, h_j, fn_m, edge_index_j):
        """ i: receiver, j: sender (j -> i, will be aggregated at i)
        - Currently, only receives node latent vector.
        Need to be updated to receive edge features also.
        - Message does not depend on the receiver node.
        - (Sender's vec, edge vec) -> (message vec)
        """
        # m = torch.cat([h_j, e_j], dim=1) # include edge feat
        m = fn_m(h_j)
        # if self.loc_msg:
        #     m = torch.cat([m, torch.tensor(np.eye(8)[edge_index_j], dtype=m.dtype)], dim=1)
            # m = torch.tensor(np.eye(self.latent)[edge_index_j])
        # logging.debug(f"\t- message:\t{m}")
        return m

    def update(self, m, x, h, fn_u, edge_index_i):
        """ pop_first == True """
        h_updated = fn_u(torch.cat([x, m], dim=1))
        h[edge_index_i] = h_updated[edge_index_i]
        return h

""" For debugging """
class ConcatEncoder(nn.Module):
    def __init__(self, in_=2, latent=16, out_=16):
        super(ConcatEncoder, self).__init__()

    def forward(self, x, num_node, edge_index):
        
        return torch.flatten(x)


class SimpleDecoder(nn.Module):
    def __init__(self, in_=16, latent=16, out_=16):
        super(SimpleDecoder, self).__init__()
        
        """
        in_: int
            len(data.x)
        """

        self.in_ = in_
        self.latent = latent
        self.out_ = out_

        self.lin1 = nn.Linear(in_, latent)
        self.lin2 = nn.Linear(latent, out_)

    def forward(self, z, node_max, num_node, edge_index):
        d = self.lin1(z)
        d = d.relu()
        d = self.lin2(d)

        d = torch.reshape(d, (-1,2))

        return d
""""""

def build_rstruc_model(args, sweep_config=None):
    logging.info("build model..")
    if args.rs_sweep:
        config = dict(sweep_config)
    else:
        config = {
            'learning_rate': args.rs_lr,
            'latent_size': args.rs_latent
        }

    latent_size = config['latent_size']

    if args.rs_conv == "tree":
        struc_tree_encoder = StrucTreeEncoder(latent=latent_size,
                                            out_=latent_size)
        struc_tree_decoder = StrucTreeDecoder(in_=latent_size,
                                            latent=latent_size)
    elif args.rs_conv == "test_decoder":
        struc_tree_encoder = ConcatEncoder()
        struc_tree_decoder = StrucTreeDecoder(in_=16, # concated == 16
                                        latent=latent_size,
                                        out_=2) # node feat to recon

    elif args.rs_conv == "test_simple_decoder":
        struc_tree_encoder = ConcatEncoder()
        struc_tree_decoder = SimpleDecoder(in_=16, # concated == 16
                                        latent=latent_size)

    elif args.rs_conv == "tree_enc_simple_dec":
        struc_tree_encoder = StrucTreeEncoder(latent=latent_size,
                                            out_=latent_size)
        struc_tree_decoder = SimpleDecoder(in_=latent_size,
                                        latent=latent_size)
    
    else:
        print(args.rs_conv)
        raise NotImplementedError
    

    net = STAE(encoder=struc_tree_encoder,
               decoder=struc_tree_decoder)

    return net, config
