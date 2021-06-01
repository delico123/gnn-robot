# import random
import logging
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
        # criterion = nn.MSELoss()
        criterion = nn.L1Loss()

        loss = criterion(predict, target)

        return loss


class StrucTreeEncoder(nn.Module): # N/A
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
    def __init__(self, in_=2, out_=16, latent=16):
        """
        in_
        out_
        direction: up or down (down==spread==root->leaf, up==collect==l->r)
        """
        super(TreeConv, self).__init__(aggr='add')
        self.lin_root = nn.Linear(in_, latent)
        self.mlp_down = nn.Sequential(nn.Linear(2*latent, latent),
                                    #   nn.BatchNorm1d(latent),
                                    #   nn.ReLU())
                                      nn.Sigmoid())
        # self.mlp_up = nn.Sequential(nn.Linear(2*latent, latent),
        self.mlp_up = nn.Sequential(nn.Linear(2*latent, out_),
                                    # nn.BatchNorm1d(out_),
                                    # nn.ReLU())
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
        # print(x)
        # Node vectors
        x = self.lin_root(x) ###
        # x = x.relu()
        # print(x)

        # SPREAD
        x = self.propagate_down(down_ei, x)
        # print(x)

        # COLLECT
        x = self.propagate_up(up_ei, x)
        # print(x)
        # raise NotImplementedError

        return x

    def propagate_down(self, down_ei, x):
        # print('---pd')
        for edges in down_ei:
            x_new = self.propagate(edge_index=edges, x=x, mlp=self.mlp_down)
            i_r = edges[1] # receiver
            x[i_r] = x_new[i_r]
            # print(x)
            # print(x_new)
        return x

    def propagate_up(self, up_ei, x):
        # print('---pu')
        for edges in up_ei:
            x_new = self.propagate(edge_index=edges, x=x, mlp=self.mlp_up)
            i_r = edges[1] # receiver
            x[i_r] = x_new[i_r]
            # print(x)
            # print(x_new)
        return x        

    def message(self, x_i, x_j, mlp):
        m = torch.cat([x_i, x_j], dim=1)
        m = mlp(m)
        return m

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
        struc_tree_decoder = StrucTreeDecoder(in_=16,
                                        latent=latent_size)

    elif args.rs_conv == "test_simple_decoder":
        struc_tree_encoder = ConcatEncoder()
        struc_tree_decoder = SimpleDecoder(in_=16,
                                        latent=latent_size)
    
    else:
        print(args.rs_conv)
        raise NotImplementedError
    

    net = STAE(encoder=struc_tree_encoder,
               decoder=struc_tree_decoder)

    return net, config
