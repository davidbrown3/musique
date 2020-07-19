import math
import unittest
from collections import namedtuple

import plotly.graph_objects as go
import torch
from genty import genty, genty_dataset

from paddington.examples.models.nonlinear.inverted_pendulum import \
    InvertedPendulum
from paddington.solvers.ilqr import iLQR
from paddington.tools.controls_tools import (diagonalize,
                                             quadratic_cost_function)


class TestILQR(unittest.TestCase):

    def setUp(self):

        self.plant = InvertedPendulum()

        g_xx = diagonalize(torch.tensor([0.1, 0.0, 0.2, 0.0], dtype=torch.float64))
        g_uu = torch.tensor([[0.01]], dtype=torch.float64)
        g_xu = torch.zeros([4, 1], dtype=torch.float64)
        g_x = torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
        g_u = torch.tensor([[0.0]], dtype=torch.float64)
        self.cost_function = quadratic_cost_function(g_xx=g_xx, g_xu=g_xu, g_uu=g_uu, g_x=g_x, g_u=g_u)

        dt = 0.1

        self.solver = iLQR(plant=self.plant, cost_function=self.cost_function, dt=dt)

    def test_ilqr(self):

        states_initial = torch.tensor([[5.0], [0.0], [math.pi/4], [0.0]], dtype=torch.float64)
        time_total = 40
        xs, us = self.solver.solve(states_initial, time_total)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                y=[x[2, 0] for x in xs],
                name='angular position'
            )
        )

        fig.add_trace(
            go.Scatter(
                y=[x[0, 0] for x in xs],
                name='linear position'
            )
        )

        fig.show()


if __name__ == "__main__":
    unittest.main(verbosity=2)
