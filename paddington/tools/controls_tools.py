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

    def __init__(self, g_xx, g_xu, g_ux, g_uu, g_x, g_u):

        if g_xx.shape[0] != g_xx.shape[1]:
            raise ValueError('g_xx must have shape (N_x, N_x)')
        if g_uu.shape[0] != g_uu.shape[1]:
            raise ValueError('g_uu must have  shape (N_u, N_u)')
        if g_xu.shape != (g_xx.shape[0], g_uu.shape[0]):
            raise ValueError('g_xu must have  shape (N_x, N_u)')
        if g_ux.shape != (g_uu.shape[0], g_xx.shape[0]):
            raise ValueError('g_ux must have  shape (N_u, N_x)')
        if g_x.shape != (1, g_xx.shape[0]):
            raise ValueError('g_x must have  shape (N_x, 1)')
        if g_u.shape != (1, g_uu.shape[0]):
            raise ValueError('g_u must have  shape (N_u, 1)')

        self.g_xx = g_xx
        self.g_xu = g_xu
        self.g_ux = g_ux
        self.g_uu = g_uu
        self.g_x = g_x
        self.g_u = g_u

    def calculate_quadratic_cost(self, x, u):

        return (torch.matmul(x.T, torch.matmul(self.g_xx, x)) +
                2 * torch.matmul(x.T, torch.matmul(self.g_xu, u)) +
                torch.matmul(u.T, torch.matmul(self.g_uu, u))) / 2

    def calculate_linear_cost(self, x, u):

        return torch.matmul(self.g_x, x) + torch.matmul(self.g_u, u)

    def calculate_cost(self, x, u):
        return self.calculate_quadratic_cost(x, u) + self.calculate_linear_cost(x, u)

    def calculate_g_xx(self, x, u):
        return self.g_xx

    def calculate_g_xu(self, x, u):
        return self.g_xu

    def calculate_g_ux(self, x, u):
        return self.g_ux

    def calculate_g_uu(self, x, u):
        return self.g_uu

    def calculate_g_x(self, x, u):

        g_xx = self.calculate_g_xx(x, u)
        g_ux = self.calculate_g_ux(x, u)

        return torch.matmul(g_xx, x).T + torch.matmul(u.T, g_ux) + self.g_x

    def calculate_g_u(self, x, u):

        g_ux = self.calculate_g_ux(x, u)
        g_uu = self.calculate_g_uu(x, u)

        return torch.matmul(g_ux, x) + torch.matmul(g_uu, u) + self.g_u


def diagonalize(diagonal):

    return torch.eye(len(diagonal)) * diagonal
