import numpy as np
import torch
from scipy import linalg


def continuous_to_discrete(A, B, dt):
    Ad = torch.eye(len(A)) + dt * A
    Bd = dt * B
    return Ad, Bd


class cost_function():

    def calculate_cost(self, x, u):
        pass

    def calculate_cost_hessian(self, x, u):
        return torch.autograd.functional.hessian(self.calculate_cost, (states, control))


class quadratic_cost_function(cost_function):

    def __init__(self, Cx_diag, Cu_diag, cx, cu):
        C_stacked = torch.cat((Cx_diag, Cu_diag))
        self.C = torch.eye(len(C_stacked)) * C_stacked
        self.c = torch.stack((cx, cu))

    def calculate_cost(self, x, u):
        stacked = torch.cat((x, u))
        return torch.matmul(stacked.T, torch.matmul(self.C, stacked)) / 2

    def calculate_cost_hessian(self, x, u):
        # For quadratic cost function, hessian is just the matrix
        return self.C


def convert_syntax(A, B, Cx, Cu, cx, cu):

    # Changing syntax of linear dynamics
    F = np.concatenate((A, B), axis=1)
    f = np.zeros([len(F), 1])
    C = linalg.block_diag(Cx, Cu)
    c = np.concatenate((cx, cu), axis=0)

    shape = B.shape
    N_x = shape[0]
    N_u = shape[1]

    return F, f, C, c, N_x, N_u
