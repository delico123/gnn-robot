import torch


def get_optimizer(learning_rate, net):
    # TODO: argument(select optimizer)
    return torch.optim.Adam(params=net.parameters(),
                            lr=learning_rate,
                            weight_decay=1e-4)