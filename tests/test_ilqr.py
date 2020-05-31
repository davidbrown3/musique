import unittest

import torch

from paddington.examples.models.nonlinear.inverted_pendulum import \
    InvertedPendulum
from paddington.solvers.ilqr import iLQR
from paddington.tools.controls_tools import quadratic_cost_function


class TestILQR(unittest.TestCase):

    def setUp(self):

        self.plant = InvertedPendulum()

        Cx_diag = torch.tensor([0.1, 0.0, 2.0, 0.0])
        Cu_diag = torch.tensor([1.0])
        cx = torch.tensor([0.0])
        cu = torch.tensor([0.0])
        self.cost_function = quadratic_cost_function(Cx_diag, Cu_diag, cx, cu)

        states_initial = torch.tensor([-5.0, 0.0, 0.0, 0.0], requires_grad=True)

        dt = 0.1

        self.solver = iLQR(plant=self.plant, cost_function=self.cost_function, dt=dt, states_initial=states_initial, K_guess=1, N_Steps=400)


if __name__ == "__main__":
    unittest.main(verbosity=2)
