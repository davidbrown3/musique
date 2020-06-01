import torch

from paddington.tools.controls_tools import continuous_to_discrete


class NonLinearModel():

    @property
    def N_x(self):
        return 0

    @property
    def N_u(self):
        return 0

    def derivatives(self, x, u):
        return torch.tensor([])

    def step(self, x, u, dt):
        dx_dt = self.derivatives(x, u)
        return x + dx_dt * dt

    def calculate_statespace(self, x, u):
        return torch.autograd.functional.jacobian(self.derivatives, (x, u))

    def calculate_statespace_discrete(self, x, u, dt):
        A, B = self.calculate_statespace(x, u)
        return continuous_to_discrete(A, B, dt)

    def calculate_hessian(self, x, u):
        return [torch.autograd.functional.hessian(lambda x, u: self.derivatives(x, u)[i], (x, u)) for i in range(len(x))]

    def calculate_hessian_discrete(self, x, u, dt):
        return [torch.autograd.functional.hessian(lambda x, u: self.step(x, u, dt)[i], (x, u)) for i in range(len(x))]
