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

        delta = torch.tensor([0.0, 0.0, 1e-3, 0.0])

        linear = torch.matmul(
            self.A,
            delta + self.states_0
        ) + self.derivatives

        nonlinear = self.problem.derivatives(
            delta,
            torch.tensor([0.0])
        )

        [self.assertAlmostEqual(l, n, places=5) for l, n in zip(linear.tolist(), nonlinear.tolist())]

    def test_angular_velocity(self):

        delta = torch.tensor([0.0, 0.0, 0.0, 1e-3])

        linear = torch.matmul(
            self.A,
            delta + self.states_0
        ) + self.derivatives

        nonlinear = self.problem.derivatives(
            delta,
            torch.tensor([0.0])
        )

        [self.assertAlmostEqual(l, n, places=5) for l, n in zip(linear.tolist(), nonlinear.tolist())]

    def test_angular_combined(self):

        delta = torch.tensor([0.0, 0.0, 1e-3, 1e-3])

        linear = torch.matmul(
            self.A,
            delta + self.states_0
        )

        nonlinear = self.problem.derivatives(
            delta,
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

        delta = torch.tensor([0.0, 0.0, 0.0, 1e-3])

        A_nonlinear, B_nonlinear = self.problem.calculate_statespace(
            delta + self.states_0,
            self.control
        )

        rows = []
        for hess in self.hessian:
            rows.append(
                torch.matmul(
                    hess[0][0],
                    delta
                )
            )

        A_linear = torch.stack(rows)
        A_nonlinear - self.A

    def test_angular_position(self):

        delta = torch.tensor([0.0, 0.0, 1e-3, 0.0])

        linear = torch.matmul(
            self.A,
            delta
        ) + self.derivatives

        nonlinear = self.problem.derivatives(
            delta + self.states_0,
            torch.tensor([0.0])
        )

        [self.assertAlmostEqual(l, n, places=5) for l, n in zip(linear.tolist(), nonlinear.tolist())]

    def test_angular_velocity(self):

        delta = torch.tensor([0.0, 0.0, 0.0, 1e-3])

        linear = torch.matmul(
            self.A,
            delta
        ) + self.derivatives

        nonlinear = self.problem.derivatives(
            delta + self.states_0,
            torch.tensor([0.0])
        )

        [self.assertAlmostEqual(l, n, places=5) for l, n in zip(linear.tolist(), nonlinear.tolist())]

    def test_angular_combined(self):

        delta = torch.tensor([0.0, 0.0, 1e-3, 1e-3])

        linear = torch.matmul(
            self.A,
            delta
        ) + self.derivatives

        nonlinear = self.problem.derivatives(
            delta + self.states_0,
            torch.tensor([0.0])
        )

        [self.assertAlmostEqual(l, n, places=5) for l, n in zip(linear.tolist(), nonlinear.tolist())]

if __name__ == "__main__":
    unittest.main(verbosity=2)
