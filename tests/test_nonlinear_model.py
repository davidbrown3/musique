import math
import unittest
from collections import namedtuple

import torch
from genty import genty, genty_dataset

from paddington.examples.models.nonlinear.inverted_pendulum import \
    InvertedPendulum


@genty
class TestPendulum(unittest.TestCase):

    states_0s = [
        torch.tensor([0.0, 0.0, 0.0, 0.0], requires_grad=True),
        torch.tensor([0.0, 0.0, math.pi/4, 0.0], requires_grad=True)
    ]

    controls_0s = [
        torch.tensor([0.0], requires_grad=True),
        torch.tensor([1.0], requires_grad=True)
    ]

    states_deltas = [
        torch.tensor([0.0, 0.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 1e-3, 0.0]),
        torch.tensor([0.0, 0.0, 0.0, 1e-3]),
        torch.tensor([0.0, 0.0, 1e-3, 1e-3]),
    ]

    controls_deltas = [
        torch.tensor([0.0]),
        torch.tensor([1e-3]),
        torch.tensor([1.0]),
    ]

    Case = namedtuple('Case', 'states_0 controls_0 states_delta controls_delta')

    cases = []
    for states_0 in states_0s:
        for controls_0 in controls_0s:
            for states_delta in states_deltas:
                for controls_delta in controls_deltas:
                    cases.append(Case(states_0, controls_0, states_delta, controls_delta))

    def setUp(self):
        self.problem = InvertedPendulum()

    @genty_dataset(
        *cases
    )
    def test_gradients(self, states_0, controls_0, states_delta, controls_delta):

        # Process
        derivatives_0 = self.problem.derivatives(states_0, controls_0)
        A, B = self.problem.calculate_statespace(states_0, controls_0)
        hessian = self.problem.calculate_hessian(states_0, controls_0)

        # Tests
        self.compare_jacobian(A, B, states_0, controls_0, states_delta, controls_delta, derivatives_0)
        self.compare_hessian(A, B, states_0, controls_0, states_delta, controls_delta, hessian)

    def compare_jacobian(self, A, B, states_0, controls_0, states_delta, controls_delta, derivatives_0):

        linear = torch.matmul(A, states_delta) + torch.matmul(B, controls_delta) + derivatives_0
        nonlinear = self.problem.derivatives(states_delta + states_0, controls_delta + controls_0)
        [self.assertAlmostEqual(l, n, places=4) for l, n in zip(linear.tolist(), nonlinear.tolist())]

    def compare_hessian(self, A, B, states_0, controls_0, states_delta, controls_delta, hessian):

        A_nonlinear, B_nonlinear = self.problem.calculate_statespace(states_delta + states_0, controls_delta + controls_0)
        dA_nonlinear = A_nonlinear - A
        dB_nonlinear = B_nonlinear - B

        A_rows = []
        B_rows = []

        for hess in hessian:

            # TODO: cat hess[0][0] & hess[0][1] along axis 1, cat states, controls along axis 0
            A_rows.append(
                torch.matmul(hess[0][0], states_delta) +  # Txx
                torch.matmul(hess[0][1], controls_delta)  # Txu
            )
            B_rows.append(
                torch.matmul(hess[1][0], states_delta) +  # Tux
                torch.matmul(hess[1][1], controls_delta)  # Tuu
            )

        dA_linear = torch.stack(A_rows)
        dB_linear = torch.stack(B_rows)

        self.assertTrue(torch.allclose(dA_linear, dA_nonlinear, atol=5e-5))
        self.assertTrue(torch.allclose(dB_linear, dB_nonlinear, atol=5e-5))


if __name__ == "__main__":
    unittest.main(verbosity=2)
