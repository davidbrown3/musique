import torch


def continuous_to_discrete(A, B, dt):
    Ad = torch.eye(len(A)) + dt * A
    Bd = dt * B
    return Ad, Bd
