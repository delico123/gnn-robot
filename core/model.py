# import random
import logging
# from numpy import int

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.nn.inits import reset, uniform

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

    def recon_loss_cls(self, predict, target):
        """ Structure recontruction loss (Classification)
        - BCE for end-effector prediction
        """
        # loss binary
        criterion = nn.BCEWithLogitsLoss()

        loss = criterion(predict, target)

        return loss

    def recon_loss_rgr(self, predict, target):
        """ Structure recontruction loss (Regression)
        - L1
        """
        # loss binary
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
        # self.readout_sig = nn.Linear(latent, out_)
        # self.readout_lin = nn.Linear(latent, out_)

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

# uniform(size, tensor):
#     bound = 1.0 / math.sqrt(size)
#     tensor.data.uniform_(-bound, bound)
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
        # Handling x: (Currently, v3)
        # v1) Zero padding, 
        # v2) Separate linear layer, 
        # v3) Message zero w/ update fn
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

class FNET(nn.Module):
    def __init__(self, in_s=16, in_m=16, ee_=2, subtask='forward'):
        super(FNET, self).__init__()
        
        """
        in_: int
            len(data.x)
        """
        self.subtask = subtask
        
        f_in_ = in_s + in_m
        f_out_ = ee_

        i_in_ = in_s + ee_
        i_out_ = in_m

        # c_in_ = in_s + in_m
        # c_out_ = 1

        self.t_forward = nn.Sequential(
            nn.Linear(f_in_, f_in_*2),
            nn.ReLU(),
            nn.Linear(f_in_*2, f_in_*2),
            nn.ReLU(),
            nn.Linear(f_in_*2, f_out_)
        )

        self.t_inverse = nn.Sequential(
            nn.Linear(i_in_, i_in_*2),
            nn.ReLU(),
            nn.Linear(i_in_*2, i_in_*2),
            nn.ReLU(),
            nn.Linear(i_in_*2, i_out_)
        )

        # self.t_collision = nn.Sequential(
        #     nn.Linear(self.in_, self.latent),
        #     nn.ReLU(),
        #     nn.Linear(self.latent, self.latent),
        #     nn.ReLU(),
        #     nn.Linear(self.latent, out_)
        # )


    def forward(self, rs, rm, ee=2):
        rs = torch.flatten(rs)
        rm = torch.flatten(rm)
        
        if self.subtask in ['forward', 'multi']:
            rsm = torch.cat([rs, rm], dim=0) #.unsqeeuze
            out_f = self.t_forward(rsm)
        
        if self.subtask in ['inverse', 'multi']:
            ee = torch.flatten(ee)
            rse = torch.cat([rs, ee], dim=0) #.unsqeeuze
            out_i = self.t_inverse(rse)

        # TODO: collision task

        return out_f, out_i

    def loss(self, predict, target):
        # criterion = nn.MSELoss()
        criterion = nn.L1Loss()

        loss = criterion(predict.view(-1,1), target.view(-1,1))

        return loss

    def loss_multi(self, predict_f, predict_i, target_f, target_i):
        criterion = nn.L1Loss()

        loss_forward = criterion(predict_f.view(-1,1), target_f.view(-1,1))
        loss_inverse = criterion(predict_i.view(-1,1), target_i.view(-1,1))

        loss = loss_forward + loss_inverse

        # BCEWithLogitsLoss for c

        return loss

def build_rstruc_model(args, feat=2, sweep_config=None):
    logging.info("build model..")
    if args.rs_sweep:
        config = dict(sweep_config)
    else:
        config = {
            'optimizer': args.opt,
            'learning_rate': args.rs_lr,
            'latent_size': args.rs_latent,
            'opt_epsilon': args.opt_eps
        }

    latent_size = config['latent_size']

    if args.rs_conv == "tree":
        struc_tree_encoder = StrucTreeEncoder(in_=feat, #Hard coded
                                            latent=latent_size,
                                            out_=latent_size)
        struc_tree_decoder = StrucTreeDecoder(in_=latent_size,
                                            latent=latent_size,
                                            out_=feat)
    elif args.rs_conv == "test_decoder":
        struc_tree_encoder = ConcatEncoder()
        struc_tree_decoder = StrucTreeDecoder(in_=16, # concated == 16
                                        latent=latent_size,
                                        out_=2) # node feat to recon

    elif args.rs_conv == "test_simple_decoder": #ground truth
        struc_tree_encoder = ConcatEncoder()
        struc_tree_decoder = SimpleDecoder(in_=feat*args.node_padding, # concated == 16
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


def build_full_model(args): # TODO: merge nets
    config = {
        'optimizer': args.opt,
        'learning_rate': args.rs_lr,
        'latent_size': args.rs_latent,
        'opt_epsilon': args.opt_eps
    }

    if args.rs_conv == 'test_simple_decoder': # (gt)
        rs_size = 16 # TODO # Hard coded (gt) concate encoder
        # if args.subtask == 'forward':
        #     rm_size = 16 # TODO # Hard coded (gt) concate encoder forward
        # elif args.subtask == 'inverse':
        rm_size = 8 # TODO # Hard coded (gt) concate encoder inverse
    elif args.rs_conv == 'tree':
        rs_size = args.rs_latent # (ours)
        rm_size = args.rs_latent # TODO # Hard coded (ours)
    else:
        raise NotImplementedError

    net = FNET(rs_size, rm_size, ee_=2, subtask=args.subtask)

    return net, config

# class TreeConv(MessagePassing): # TODO: Generalize
#     def __init__(self, in_=2, out_=16, latent=16, loc_msg=False):
#         """
#         (down==spread==root->leaf, up==collect==l->r)

#         in == |node|
#         latent == |message| == |hidden_down|
#         out_ == |hidden_down|
#         """
#         super(TreeConv, self).__init__(aggr='add')

#         # latent size
#         _msg_down = _msg_up = _lnt_down = latent
#         _lnt_up = out_
#         _edge = 0

#         self.empty_msg = torch.zeros(_msg_down)
#         # self.loc_msg = loc_msg
#         self.latent = latent

#         # self.lin_root = nn.Linear(in_, latent) # separate fn for initial state

#         # message fn
#         self.message_down = nn.Sequential(nn.Linear(_lnt_down+_edge, _msg_down),
#                                     #   nn.BatchNorm1d(latent),
#                                     #   nn.ReLU())
#                                       nn.Sigmoid())
#         self.message_up = nn.Sequential(nn.Linear(_lnt_up+_edge, _msg_up),
#                                     # nn.BatchNorm1d(out_),
#                                     # nn.ReLU())
#                                     nn.Sigmoid())
    
#         # update fn
#         self.update_down = nn.Sequential(nn.Linear(in_+_msg_down, _lnt_down),
#         # self.update_down = nn.Sequential(nn.Linear(in_+_msg_down+8, _lnt_down),
#                                     nn.Sigmoid())
#         self.update_up = nn.Sequential(nn.Linear(_lnt_down+_msg_up, _lnt_up),
#                                     nn.Sigmoid())


#     def _organize_edges(self, edge_index, num_node):
#         # TODO: generalize to tree (mltr), use edge_index, remove num_node
#         # direction[step[ins[], outs[]]]
        
#         prt = range(0, num_node-1)
#         chd = range(1, num_node)

#         down_ei = torch.tensor([[[p],[c]] for p, c in zip(prt, chd)])
#         up_ei = down_ei.flip([0,1])

#         return down_ei, up_ei

#     def forward(self, x, edge_index, num_node=8):
#         # Edges
#         down_ei, up_ei = self._organize_edges(edge_index, num_node)
#         # Node vectors
#         # x = self.lin_root(x) ###
#         # x = x.relu()

#         # SPREAD
#         x = self.propagate_oneway(down_ei, x, self.message_down, self.update_down)

#         # COLLECT
#         self.loc_msg = False
#         x = self.propagate_oneway(up_ei, x, self.message_up, self.update_up)
#         # raise NotImplementedError

#         return x

#     def propagate_oneway(self, edge_indices, x, fn_m, fn_u):
#         """ x -> h
#         """
#         # Handling x: (Currently, v3)
#         # v1) Zero padding, 
#         # v2) Separate linear layer, 
#         # v3) Message zero w/ update fn
#         # 0에 있는 node에 대해 update 먼저 진행. msg는 0으로. (OR, use separate fn)
#         m_e = self.empty_msg.repeat(x.shape[0], 1)
#         h = torch.cat([x, m_e], dim=1)
#         # if self.loc_msg:
#         #     mmm = torch.zeros(8)
#         #     mmm = mmm.repeat(x.shape[0], 1)
#         #     h = torch.cat([h, mmm], dim=1)
#         h = fn_u(h)

#         for edge_index in edge_indices:
#             # logging.debug(edge_index)
#             # logging.debug(x)
#             h = self.propagate(edge_index=edge_index, x=x, h=h.clone(), fn_m=fn_m, fn_u=fn_u)
#             # h[edge_index[1]] = h_updated[edge_index[1]]
#         return h
        
#     def message(self, h_j, fn_m, edge_index_j):
#         """ i: receiver, j: sender (j -> i, will be aggregated at i)
#         - Currently, only receives node latent vector.
#         Need to be updated to receive edge features also.
#         - Message does not depend on the receiver node.
#         - (Sender's vec, edge vec) -> (message vec)
#         """
#         # m = torch.cat([h_j, e_j], dim=1) # include edge feat
#         m = fn_m(h_j)
#         # if self.loc_msg:
#         #     m = torch.cat([m, torch.tensor(np.eye(8)[edge_index_j], dtype=m.dtype)], dim=1)
#             # m = torch.tensor(np.eye(self.latent)[edge_index_j])
#         # logging.debug(f"\t- message:\t{m}")
#         return m

#     def update(self, m, x, h, fn_u, edge_index_i):
#         """ pop_first == True """
#         h_updated = fn_u(torch.cat([x, m], dim=1))
#         h[edge_index_i] = h_updated[edge_index_i]
#         return h