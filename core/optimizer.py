import logging

import torch


def get_optimizer(args, config, net):
    # TODO: argument(select optimizer)

    opt = config['optimizer']
    learning_rate = config['learning_rate']
    eps = config['opt_epsilon']
    params = net[0].parameters()

    if opt == 'adam':
        optimizer = torch.optim.Adam(params=params,
                            lr=learning_rate,
                            eps=eps,
                            weight_decay=1e-4)
    
    elif opt == 'sgd':
        optimizer = torch.optim.SGD(params=params,
                            lr=learning_rate,
                            eps=eps,
                            weight_decay=1e-4)

    else:
        logging.warning(f'Optimier {opt}: Not implemented ')
        raise NotImplementedError


    if args.mode == 'train':
        if args.freeze:
            for param in net[1].parameters():
                param.requires_grad = False
            for param in net[2].parameters():
                param.requires_grad = False
        else:
            optimizer.add_param_group({'params': net[1].parameters()})
            optimizer.add_param_group({'params': net[2].parameters()})


    return optimizer