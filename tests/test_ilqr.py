import math
import unittest

import torch

from paddington.plants.nonlinear_model import InvertedPendulum


class StandardTests():

    def compare_jacobian(self, delta):

        linear = torch.matmul(
            self.A,
            delta
        ) + self.derivatives

        nonlinear = self.problem.derivatives(
            delta + self.states_0,
            torch.tensor([0.0])
        )

        [self.assertAlmostEqual(l, n, places=5) for l, n in zip(linear.tolist(), nonlinear.tolist())]

    def compare_hessian(self, delta):

        A_nonlinear, B_nonlinear = self.problem.calculate_statespace(
            delta + self.states_0,
            self.control
        )

        dA_nonlinear = A_nonlinear - self.A

        rows = []
        for hess in self.hessian:
            rows.append(
                torch.matmul(
                    hess[0][0],
                    delta
                )
            )

        dA_linear = torch.stack(rows)

        self.assertTrue(torch.allclose(dA_linear, dA_nonlinear, atol=5e-5))

    def test_hessian_angular_position(self):

        delta = torch.tensor([0.0, 0.0, 0.0, 1e-3])
        self.compare_hessian(delta)

    def test_hessian_angular_velocity(self):

        delta = torch.tensor([0.0, 0.0, 1e-3, 0.0])
        self.compare_hessian(delta)

    def test_hessian_angular_combined(self):

        delta = torch.tensor([0.0, 0.0, 1e-3, 1e-3])
        self.compare_hessian(delta)

    def test_jacobian_angular_position(self):

        delta = torch.tensor([0.0, 0.0, 1e-3, 0.0])
        self.compare_jacobian(delta)

    def test_jacobian_angular_velocity(self):

        delta = torch.tensor([0.0, 0.0, 0.0, 1e-3])
        self.compare_jacobian(delta)

    def test_jacobian_angular_combined(self):

        delta = torch.tensor([0.0, 0.0, 1e-3, 1e-3])
        self.compare_jacobian(delta)

class TestZeroLinearization(unittest.TestCase, StandardTests):

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
        self.hessian = self.problem.hessian(self.states_0, self.control)

class TestNonZeroLinearization(unittest.TestCase, StandardTests):

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

if __name__ == "__main__":
    unittest.main(verbosity=2)
