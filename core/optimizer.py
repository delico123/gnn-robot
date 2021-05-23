import torch


def get_optimizer(args, net):
    # TODO: argument(select optimizer)
    return torch.optim.Adam(params=net.parameters(),
                            lr=args.rs_lr)