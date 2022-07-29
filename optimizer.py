import torch


def get_optimizer(model, lr=2.5e-4):
    return torch.optim.Adam(model.parameters(), lr=lr)
