import os
import json
import logging

import numpy as np
import torch
import torch_geometric
from torch_geometric.data import DataLoader

from core.model import build_rstruc_model
from data.data_utils import adj_matrix_to_list, data_vis


"""
TODO:
- dataset도 node만이 있고, edge도가 있다 <- model에 따라 달라져야할듯
    - model이 edge okay면 empty edge 보내주고
    - model이 edge 필요없으면 있어도 어디에서 무시할까? 여기? 모델?

"""

def get_structure_loader(task='Reacher', batch_size=16):
    logging.info("Reacher")

    if task is 'Reacher':
        data_path = 'reacher_simulate/res'
    else:
        raise NotImplementedError
    data_file_names = [x for x in os.listdir(data_path) if _is_str_only(x)]
    data_file_names.sort()
    
    dataset = _to_dataset(data_path, data_file_names, node_zero_padding=True, node_max=8)
    # dataset = _to_dataset(data_path, data_file_names, node_zero_padding=False)

    return dataset
    # return DataLoader(dataset=dataset, 
    #                   batch_size=batch_size,
    #                   shuffle=True)


def _is_str_only(name):
    return name.endswith(".json") and "structure_only" in name


def _to_dataset(data_path, data_file_names, node_zero_padding=False, node_max=0):
    """
        node_zero_padding:
            Fixed size node features
        node_max:
            Max node feature vector size
    """
    logging.debug("--node feature: zero padding {}".format(node_zero_padding))
    if node_max > 0 and node_zero_padding is False:
        logging.warning("node_max ignored")

    flag = True
    dataset = []
    for data_file in data_file_names:
        logging.debug(data_file)
        
        with open(os.path.join(data_path, data_file)) as f:
            data_reacher = json.load(f)
        
        y = int(data_file.split('.')[0].split('_')[-1]) + 1 # set y as num of joints, indicated in file name

        for ith_urdf_data in data_reacher:
            data_raw = ith_urdf_data['structure']

            if node_zero_padding: # TODO: node_max = max(node_max, len())
                assert(node_max >= len(data_raw['node_feat']))
                data_raw['node_feat'] += [[0]] * (node_max - len(data_raw['node_feat']))
                # print(data_raw['node_feat'])
        
            """(v1)pure node feat"""
            # x = torch.tensor(data_raw['node_feat'], dtype=torch.float)
            # edge_index, _ = adj_matrix_to_list(data_raw['adj'], tensor=True)
            """(v2)edge info into node feat"""
            x = np.array(data_raw['node_feat'])
            edge_index, edge_info = adj_matrix_to_list(data_raw['adj'], 
                                                        info_matrix=data_raw['link_info'],
                                                        info_und=True,
                                                        tensor=False)
            edge_info = list(np.concatenate(([[0]], edge_info), axis=0))
            # print(edge_info)
            if node_zero_padding: # TODO: node_max = max(node_max, len())
                edge_info += [[0]] * (node_max - len(edge_info))
            x = np.concatenate((x, edge_info), axis=1)

            x = torch.tensor(x, dtype=torch.float)
            edge_index = torch.tensor(edge_index)
            """"""

            """for debugging"""
            # if logging.root.level == logging.DEBUG and flag and y == 5:
            #     print(x)
            #     print(x.shape)
            #     print(edge_index)
            #     print(edge_info)
            #     print(edge_index.shape)
            #     print(edge_info.shape)
            #     flag = False

            data = torch_geometric.data.Data(x=x, edge_index=edge_index.t().contiguous(), y=y)
            
            dataset.append(data)

    if logging.root.level == logging.DEBUG:
        import random

        print("-----------Sampled data-----------")
        random_data_idx = random.randrange(len(dataset))
        random_data = dataset[random_data_idx]
        print("Graph visualization of data #: ", random_data_idx)
        print("x: ", random_data.x)
        print("number of joint (y): ", random_data.y)
        # data_vis(random_data, undirected=True)
        print("-----------------=---------------")

    return dataset
        
