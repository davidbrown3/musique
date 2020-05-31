import json
import unittest
from importlib.resources import open_text

import control
import numpy as np
import torch

from paddington.plants.linear_model import LinearModel
from paddington.solvers.lqr import LQR
from paddington.tools.controls_tools import (convert_syntax_cost_diagonals,
                                             convert_syntax_transition)


class TestAircraftPitch(unittest.TestCase):
    """
    Example documented at:
    http://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=ControlStateSpace
    """

    def setUp(self):

        with open_text("paddington.examples.models.linear", "aircraft_pitch.json") as f:
            data = json.load(f)
            self.plant = LinearModel.from_dict(data)

    def test_infinite_horizon(self):

        Cx_diag = torch.tensor([0.0, 0.0, 2.0])
        Cu_diag = torch.tensor([1.0])
        cx = torch.zeros([3, 1])
        cu = torch.zeros([1, 1])

        C, c = convert_syntax_cost_diagonals(Cx_diag, Cu_diag, cx, cu)

        F, f = convert_syntax_transition(self.plant.A_d, self.plant.B_d)

        # Discrete horizon
        solver = LQR(F=F, f=f, C=C, c=c, N_x=self.plant.N_x, N_u=self.plant.N_u, dt=0.1)

        K_discrete, _ = solver.K_horizon(N_Steps=100)
        K_discrete = torch.squeeze(K_discrete)

        # Infinite horizon
        K_infinite, _, _ = control.lqr(self.plant.A, self.plant.B, Cx_diag.numpy() * np.eye(len(Cx_diag)), Cu_diag.numpy() * np.eye(len(Cu_diag)))
        # Minus sign for convention
        K_infinite = -torch.squeeze(torch.tensor(K_infinite))

        # Assert close (7%)
        self.assertGreaterEqual(K_discrete[0] / K_infinite[0], 0.93)
        self.assertGreaterEqual(K_discrete[1] / K_infinite[1], 0.93)
        self.assertGreaterEqual(K_discrete[2] / K_infinite[2], 0.93)
        self.assertLessEqual(K_discrete[0] / K_infinite[0], 1.07)
        self.assertLessEqual(K_discrete[1] / K_infinite[1], 1.07)
        self.assertLessEqual(K_discrete[2] / K_infinite[2], 1.07)


if __name__ == "__main__":
    unittest.main(verbosity=2)
