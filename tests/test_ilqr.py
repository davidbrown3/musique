import math
import unittest

import torch

from paddington.examples.models.nonlinear.inverted_pendulum import \
    InvertedPendulum
from paddington.solvers.ilqr import iLQR
from paddington.tools.controls_tools import quadratic_cost_function


class TestQuadraticCost(unittest.TestCase):

    def setUp(self):

        Cx_diag = torch.tensor([0.1, 0.0, 2.0, 0.0])
        Cu_diag = torch.tensor([0.1])
        cx = torch.tensor([[0.0], [0.0], [0.0], [0.0]])
        cu = torch.tensor([[0.0]])
        self.cost_function = quadratic_cost_function(Cx_diag, Cu_diag, cx, cu)

    def test_jacobian(self):

        states_initial = torch.tensor([[-5.0], [0.0], [0.0], [0.0]])
        controls_initial = torch.tensor([[0.0]])
        jacobian_1 = self.cost_function.calculate_cost_jacobian(x=states_initial, u=controls_initial)
        jacobian_2 = torch.autograd.functional.jacobian(self.cost_function.calculate_cost, (states_initial[:, 0], controls_initial[:, 0]))

        [self.assertAlmostEqual(l, n, places=4) for l, n in zip(torch.squeeze(jacobian_1).tolist(), torch.squeeze(torch.cat(jacobian_2)).tolist())]

    def test_hessian(self):

        states_initial = torch.tensor([[-5.0], [0.0], [0.0], [0.0]])
        controls_initial = torch.tensor([[0.0]])
        hessian_1 = self.cost_function.calculate_cost_hessian(x=states_initial, u=controls_initial)
        hessian_2 = torch.autograd.functional.hessian(self.cost_function.calculate_cost, (states_initial[:, 0], controls_initial[:, 0]))


class TestILQR(unittest.TestCase):

    def setUp(self):

        self.plant = InvertedPendulum()

        Cx_diag = torch.tensor([0.1, 0.0, 0.2, 0.0])
        Cu_diag = torch.tensor([0.01])
        cx = torch.tensor([[0.0], [0.0], [0.0], [0.0]])
        cu = torch.tensor([[0.0]])
        self.cost_function = quadratic_cost_function(Cx_diag, Cu_diag, cx, cu)

        dt = 0.1

        self.solver = iLQR(plant=self.plant, cost_function=self.cost_function, dt=dt)

    def test_ilqr(self):

        states_initial = torch.tensor([[1.0], [0.0], [0.0], [0.0]], requires_grad=True)
        time_total = 40
        self.solver.solve(states_initial, time_total)


if __name__ == "__main__":
    unittest.main(verbosity=2)
