import math
import unittest

import torch

from paddington.plants.nonlinear_model import InvertedPendulum


class TestZeroLinearization(unittest.TestCase):

    def setUp(self):

        self.problem = InvertedPendulum()

        self.states_0 = torch.tensor([
            0.0,
            0.0,
            0.0,
            0.0
        ], requires_grad=True)

        self.control = torch.tensor([
            0.0
        ], requires_grad=True)

        # Linearisation
        self.derivatives = self.problem.derivatives(self.states_0, self.control)
        self.A, self.B = self.problem.calculate_statespace(self.states_0, self.control)

    def test_angular_position(self):

        linear = torch.matmul(
            self.A,
            torch.tensor([0.0, 0.0, 1e-3, 0.0]) + self.states_0
        ) + self.derivatives

        nonlinear = self.problem.derivatives(
            torch.tensor([0.0, 0.0, 1e-3, 0.0]),
            torch.tensor([0.0])
        )

        [self.assertAlmostEqual(l, n, places=5) for l, n in zip(linear.tolist(), nonlinear.tolist())]

    def test_angular_velocity(self):

        linear = torch.matmul(
            self.A,
            torch.tensor([0.0, 0.0, 0.0, 1e-3]) + self.states_0
        ) + self.derivatives

        nonlinear = self.problem.derivatives(
            torch.tensor([0.0, 0.0, 0.0, 1e-3]),
            torch.tensor([0.0])
        )

        [self.assertAlmostEqual(l, n, places=5) for l, n in zip(linear.tolist(), nonlinear.tolist())]

    def test_angular_combined(self):

        linear = torch.matmul(
            self.A,
            torch.tensor([0.0, 0.0, 1e-3, 1e-3]) + self.states_0
        )

        nonlinear = self.problem.derivatives(
            torch.tensor([0.0, 0.0, 1e-3, 1e-3]),
            torch.tensor([0.0])
        )

        [self.assertAlmostEqual(l, n, places=5) for l, n in zip(linear.tolist(), nonlinear.tolist())]


class TestNonZeroLinearization(unittest.TestCase):

    def setUp(self):

        self.problem = InvertedPendulum()

        self.states_0 = torch.tensor([
            0.0,
            0.0,
            math.pi/4,
            0.0
        ], requires_grad=True)

        self.control = torch.tensor([
            0.0
        ], requires_grad=True)

        # Linearisation
        self.derivatives = self.problem.derivatives(self.states_0, self.control)
        self.A, self.B = self.problem.calculate_statespace(self.states_0, self.control)
        self.hessian = self.problem.hessian(self.states_0, self.control)

    def test_hessian(self):

        A_nonlinear, B_nonlinear = self.problem.calculate_statespace(torch.tensor([0.0, 0.0, 0.0, 1e-3]) + self.states_0, self.control)

        A_linear = torch.zeros_like(A_nonlinear)
        rows = []
        for hess in self.hessian:
            rows.append(
                torch.matmul(
                    hess[0][0],
                    torch.tensor([0.0, 0.0, 0.0, 1e-3])
                )
            )

        A_linear = torch.stack(rows)
        A_nonlinear - self.A

    def test_angular_position(self):

        linear = torch.matmul(
            self.A,
            torch.tensor([0.0, 0.0, 1e-3, 0.0])
        ) + self.derivatives

        nonlinear = self.problem.derivatives(
            torch.tensor([0.0, 0.0, 1e-3, 0.0]) + self.states_0,
            torch.tensor([0.0])
        )

        [self.assertAlmostEqual(l, n, places=5) for l, n in zip(linear.tolist(), nonlinear.tolist())]

    def test_angular_velocity(self):

        linear = torch.matmul(
            self.A,
            torch.tensor([0.0, 0.0, 0.0, 1e-3])
        ) + self.derivatives

        nonlinear = self.problem.derivatives(
            torch.tensor([0.0, 0.0, 0.0, 1e-3]) + self.states_0,
            torch.tensor([0.0])
        )

        [self.assertAlmostEqual(l, n, places=5) for l, n in zip(linear.tolist(), nonlinear.tolist())]

    def test_angular_combined(self):

        linear = torch.matmul(
            self.A,
            torch.tensor([0.0, 0.0, 1e-3, 1e-3])
        ) + self.derivatives

        nonlinear = self.problem.derivatives(
            torch.tensor([0.0, 0.0, 1e-3, 1e-3]) + self.states_0,
            torch.tensor([0.0])
        )

        [self.assertAlmostEqual(l, n, places=5) for l, n in zip(linear.tolist(), nonlinear.tolist())]

if __name__ == "__main__":
    unittest.main(verbosity=2)
