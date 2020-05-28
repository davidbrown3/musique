import torch

from paddington.tools.controls_tools import continuous_to_discrete


class NonLinearModel():

    def derivatives(self, x, u):
        return torch.tensor([])

    def step(self, x, u, dt):
        dx_dt = self.derivatives(x, u)
        return x + dx_dt * dt

    def calculate_statespace(self, x, u):
        return torch.autograd.functional.jacobian(self.derivatives, (x,u))

    def calculate_statespace_discrete(self, x, u, dt):
        A, B = self.calculate_statespace(x, u)
        return continuous_to_discrete(A, B, dt)

    def hessian(self, x, u):
        return [torch.autograd.functional.hessian(lambda x, u: self.derivatives(x, u)[i], (x, u)) for i in range(len(x))]
