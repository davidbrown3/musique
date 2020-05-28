import importlib
import json
import unittest
from importlib.resources import open_text

import control
import torch

from paddington.plants.linear_model import LinearModel
from paddington.solvers.lqr import LQR
from paddington.tools.controls_tools import convert_syntax


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

        Cx_diag = torch.tensor([0.1, 0.0, 2.0, 0.0])
        Cu_diag = torch.tensor([1.0])
        cx = torch.zeros([3, 1])
        cu = torch.zeros([1, 1])

        F, f, C, c, N_x, N_u = convert_syntax(self.plant.A_d, self.plant.B_d, Cx_diag, Cu_diag, cx, cu)

        # Discrete horizon
        solver = LQR(F=F, f=f, C=C, c=c, N_x=N_x, N_u=N_u)

        V_Tx, v_Tx, _, _ = solver.backward_pass(solver.V_Ty_i, solver.v_Ty_i)

        for _ in torch.arange(100):
            V_Tx, v_Tx, K_discrete, _ = solver.backward_pass(V_Tx, v_Tx)
        K_discrete = torch.squeeze(K_discrete)

        # Infinite horizon
        K_infinite, _, _ = control.lqr(self.plant.A, self.plant.B, Cx, Cu)
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
