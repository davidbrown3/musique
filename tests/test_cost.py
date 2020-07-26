import itertools
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
        torch.tensor([[-2.0], [-3.0], [-4.0], [-5.0]], dtype=torch.float64),
    ]

    controls = [
        torch.tensor([[0.0]], dtype=torch.float64),
        torch.tensor([[-1.0]], dtype=torch.float64),
        torch.tensor([[-2.0]], dtype=torch.float64),
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

    @genty_dataset(
        *cases
    )
    def test_hessian(self, states, controls):

        g_uu = self.cost_function.calculate_g_uu(x=states, u=controls)
        g_xx = self.cost_function.calculate_g_xx(x=states, u=controls)
        g_xu = self.cost_function.calculate_g_xu(x=states, u=controls)

        hessian = torch.autograd.functional.hessian(self.cost_function.calculate_cost, (states[:, 0], controls[:, 0]))

        hessian_xx_flat = list(itertools.chain.from_iterable(hessian[0][0].tolist()))
        g_xx_flat = list(itertools.chain.from_iterable(g_xx.tolist()))

        hessian_xu_flat = list(itertools.chain.from_iterable(hessian[0][1].tolist()))
        g_xu_flat = list(itertools.chain.from_iterable(g_xu.tolist()))

        hessian_uu_flat = list(itertools.chain.from_iterable(hessian[1][1].tolist()))
        g_uu_flat = list(itertools.chain.from_iterable(g_uu.tolist()))

        [self.assertAlmostEqual(l, n, places=4) for l, n in zip(hessian_xx_flat, g_xx_flat)]
        [self.assertAlmostEqual(l, n, places=4) for l, n in zip(hessian_xu_flat, g_xu_flat)]
        [self.assertAlmostEqual(l, n, places=4) for l, n in zip(hessian_uu_flat, g_uu_flat)]


if __name__ == "__main__":
    unittest.main(verbosity=2)
