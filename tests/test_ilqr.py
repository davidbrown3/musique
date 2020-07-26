import math
import unittest

import plotly.graph_objects as go
import torch

from paddington.examples.models.nonlinear.cart_pole import CartPole
from paddington.solvers.ilqr import iLQR
from paddington.tools.controls_tools import (diagonalize,
                                             quadratic_cost_function)


class TestILQR(unittest.TestCase):

    def setUp(self):

        self.plant = CartPole()

        g_xx = diagonalize(torch.tensor([0.1, 0.0, 0.2, 0.0]))
        g_uu = torch.tensor([[0.01]])
        g_xu = torch.zeros([4, 1])
        g_ux = torch.zeros([1, 4])
        g_x = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
        g_u = torch.tensor([[0.0]])
        self.cost_function = quadratic_cost_function(g_xx=g_xx, g_xu=g_xu, g_ux=g_ux, g_uu=g_uu, g_x=g_x, g_u=g_u)

        dt = 0.1

        self.solver = iLQR(plant=self.plant, cost_function=self.cost_function, dt=dt)

    def test_ilqr(self):

        states_initial = torch.tensor([[5.0], [0.0], [math.pi/4], [0.0]])
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

        fig.add_trace(
            go.Scatter(
                y=[u[0, 0] for u in us],
                name='control'
            )
        )

        fig.show()


if __name__ == "__main__":
    unittest.main(verbosity=2)
