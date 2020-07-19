import unittest
from collections import namedtuple

import torch
from genty import genty, genty_dataset

from paddington.tools.controls_tools import (diagonalize,
                                             quadratic_cost_function)


@genty
class TestQuadraticCost(unittest.TestCase):

    states = [
        torch.tensor([[-1.0], [0.0], [0.0], [0.0]], dtype=torch.float64),
        torch.tensor([[-1.0], [-1.0], [0.0], [0.0]], dtype=torch.float64),
        torch.tensor([[-1.0], [-1.0], [-1.0], [0.0]], dtype=torch.float64),
        torch.tensor([[-1.0], [-1.0], [-1.0], [-1.0]], dtype=torch.float64),
    ]

    controls = [
        torch.tensor([[0.0]], dtype=torch.float64),
        torch.tensor([[-1.0]], dtype=torch.float64),
    ]
    Case = namedtuple('Case', 'states controls')

    cases = []
    for state in states:
        for control in controls:
            cases.append(Case(state, control,))

    def setUp(self):
        g_xx = diagonalize(torch.tensor([0.1, 0.0, 2.0, 0.0], dtype=torch.float64))
        g_xu = torch.zeros([4, 1], dtype=torch.float64)
        g_uu = torch.tensor([[0.1]], dtype=torch.float64)
        g_x = torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
        g_u = torch.tensor([[0.0]], dtype=torch.float64)
        self.cost_function = quadratic_cost_function(g_xx=g_xx, g_xu=g_xu, g_uu=g_uu, g_x=g_x, g_u=g_u)

    @genty_dataset(
        *cases
    )
    def test_jacobian(self, states, controls):

        auto_jacobian = torch.autograd.functional.jacobian(self.cost_function.calculate_cost, (states[:, 0], controls[:, 0]))

        g_u = self.cost_function.calculate_g_u(x=states, u=controls)
        g_x = self.cost_function.calculate_g_x(x=states, u=controls)

        [self.assertAlmostEqual(l, n, places=4) for l, n in zip(auto_jacobian[1].tolist(), g_u.tolist())]
        [self.assertAlmostEqual(l, n, places=4) for l, n in zip(auto_jacobian[0].tolist(), g_x.tolist())]

    def test_hessian(self):

        states_initial = torch.tensor([[-5.0], [0.0], [0.0], [0.0]], dtype=torch.float64)
        controls_initial = torch.tensor([[0.0]], dtype=torch.float64)
        hessian_1 = self.cost_function.calculate_cost_hessian(x=states_initial, u=controls_initial)
        hessian_2 = torch.autograd.functional.hessian(self.cost_function.calculate_cost, (states_initial[:, 0], controls_initial[:, 0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
