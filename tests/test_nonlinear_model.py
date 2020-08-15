import unittest
from collections import namedtuple

import jax.numpy as np
from genty import genty, genty_dataset

from paddington.examples.models.nonlinear.cart_pole import \
    CartPole


@genty
class TestPendulum(unittest.TestCase):

    states_0s = [
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, np.pi/4, 0.0])
    ]

    controls_0s = [
        np.array([0.0]),
        np.array([1.0])
    ]

    states_deltas = [
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1e-3, 0.0]),
        np.array([0.0, 0.0, 0.0, 1e-3]),
        np.array([0.0, 0.0, 1e-3, 1e-3]),
    ]

    controls_deltas = [
        np.array([0.0]),
        np.array([1e-3]),
        np.array([1.0]),
    ]

    Case = namedtuple('Case', 'states_0 controls_0 states_delta controls_delta')

    cases = []
    for states_0 in states_0s:
        for controls_0 in controls_0s:
            for states_delta in states_deltas:
                for controls_delta in controls_deltas:
                    cases.append(Case(states_0, controls_0, states_delta, controls_delta))

    def setUp(self):
        dt = 1e-2
        self.problem = CartPole(dt)

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

        linear = np.matmul(A, states_delta) + np.matmul(B, controls_delta) + derivatives_0
        nonlinear = self.problem.derivatives(states_delta + states_0, controls_delta + controls_0)
        [self.assertAlmostEqual(l, n, places=4) for l, n in zip(linear.tolist(), nonlinear.tolist())]

    def compare_hessian(self, A, B, states_0, controls_0, states_delta, controls_delta, hessian):

        A_nonlinear, B_nonlinear = self.problem.calculate_statespace(states_delta + states_0, controls_delta + controls_0)
        dA_nonlinear = A_nonlinear - A
        dB_nonlinear = B_nonlinear - B

        dA_linear = np.tensordot(hessian[0][0], np.expand_dims(states_delta, 0).T, 1).squeeze() + \
            np.tensordot(hessian[0][1], np.expand_dims(controls_delta, 0).T, 1).squeeze()

        dB_linear = np.tensordot(hessian[1][0], np.expand_dims(states_delta, 0).T, 1).squeeze() + \
            np.tensordot(hessian[1][1], np.expand_dims(controls_delta, 0).T, 1).squeeze()

        self.assertTrue(np.allclose(dA_linear, dA_nonlinear, atol=5e-5))
        self.assertTrue(np.allclose(dB_linear, dB_nonlinear, atol=5e-5))


if __name__ == "__main__":
    unittest.main(verbosity=2)
