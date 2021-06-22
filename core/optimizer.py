import logging

import torch


def get_optimizer(config, net):
    # TODO: argument(select optimizer)

    opt = config['optimizer']
    learning_rate = config['learning_rate']
    eps = config['opt_epsilon']

    if opt == 'adam':
        return torch.optim.Adam(params=net.parameters(),
                            lr=learning_rate,
                            eps=eps,
                            weight_decay=1e-4)
    
    elif opt == 'sgd':
        return torch.optim.SGD(params=net.parameters(),
                            lr=learning_rate,
                            weight_decay=1e-4)

    else:
        logging.warning(f'Optimier {opt}: Not implemented ')
        raise NotImplementedError