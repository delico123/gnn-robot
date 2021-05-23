import torch

# TODO: self loop option
def adj_matrix_to_list(adj_matrix, info_matrix=None, info_und=False, self_loop=True, tensor=False): # TODO: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/to_dense_adj.html
    adj_list = []
    info_list = []
    for ii, row in enumerate(adj_matrix):
        for jj, item in enumerate(row):
            if item == 1:
                adj_list += [[ii, jj]]
                if info_matrix:
                    if (info_und is False) or (ii < jj): # out edge
                        info_list += [[info_matrix[ii][jj]]]
            elif item == 0:
                pass
            else:
                raise ValueError("Value other than 0 or 1. Set value as (Positive=1, Negative=0).")
                
    return (adj_list, info_list) if not tensor else (torch.tensor(adj_list), torch.tensor(info_list))

def data_vis(data, undirected=False):
    """ Visualize graph data """
    import networkx as nx
    import torch_geometric
    
    g = torch_geometric.utils.to_networkx(data, to_undirected=undirected)
    return nx.draw(g)
