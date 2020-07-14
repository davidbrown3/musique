import torch


def continuous_to_discrete(A, B, dt):
    Ad = torch.eye(len(A)) + dt * A
    Bd = dt * B
    return Ad, Bd


class cost_function():

    def calculate_cost(self, x, u):
        pass

    def calculate_cost_hessian(self, x, u):
        return torch.autograd.functional.hessian(self.calculate_cost, (x, u))

    def calculate_cost_jacobian(self, x, u):
        # For quadratic cost function, hessian is just the matrix
        return torch.autograd.functional.jacobian(self.calculate_cost, (x, u))


class quadratic_cost_function(cost_function):

    def __init__(self, Cx_diag, Cu_diag, cx, cu):
        C_stacked = torch.cat((Cx_diag, Cu_diag))
        self.C = torch.eye(len(C_stacked)) * C_stacked
        self.c = torch.cat((cx, cu))

    def calculate_cost(self, x, u):
        stacked = torch.cat((x, u))
        return torch.matmul(stacked.T, torch.matmul(self.C, stacked)) / 2

    def calculate_cost_hessian(self, x, u):
        # For quadratic cost function, hessian is just the matrix
        return self.C

    def calculate_cost_jacobian(self, x, u):
        # For quadratic cost function, hessian is just the matrix
        stacked = torch.cat((x, u))
        return torch.matmul(self.C, stacked) + self.c


def diagonalize(diagonal):

    return torch.eye(len(diagonal)) * diagonal
