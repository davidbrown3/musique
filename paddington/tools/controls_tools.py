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
        self.c = torch.stack((cx, cu))

    def calculate_cost(self, x, u):
        stacked = torch.cat((x, u))
        return torch.matmul(stacked.T, torch.matmul(self.C, stacked)) / 2

    def calculate_cost_hessian(self, x, u):
        # For quadratic cost function, hessian is just the matrix
        return self.C

    def calculate_cost_jacobian(self, x, u):
        # For quadratic cost function, hessian is just the matrix
        return self.c


def convert_syntax_transition(A, B):

    # Changing syntax of linear dynamics
    F = torch.cat((A, B), dim=1).float()  # TODO: Remove this
    f = torch.zeros([len(F), 1])

    return F, f


def convert_syntax_cost_diagonals(Cx_diag, Cu_diag, cx, cu):

    C_stacked = torch.cat((Cx_diag, Cu_diag), dim=0)
    C = torch.eye(len(C_stacked)) * C_stacked
    c = torch.cat((cx, cu), dim=0)

    return C, c
