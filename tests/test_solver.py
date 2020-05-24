from lqr import LQR
import numpy as np
import control
import plotly.graph_objects as go
import unittest


class TestAircraftPitch(unittest.TestCase):
    """
    Example documented at:
    http://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=ControlStateSpace
    """

    def setUp(self):

        self.dt = 0.1

        self.A = np.array(
            [[-0.313, 56.7, 0], [-0.0139, -0.426, 0], [0, 56.7, 0], ])

        self.B = np.array([[0.232], [0.0203], [0]])

        C = np.eye(3)

        D = np.zeros([3, 1])

        sys = control.ss(self.A, self.B, C, D)

        sys_discrete = control.sample_system(sys, self.dt)

        self.A_d = np.array(sys_discrete.A)

        self.B_d = np.array(sys_discrete.B)

    def test_infinite_horizon(self):

        Cx = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 2], ])

        Cu = np.array([[1]])

        cx = np.zeros([3, 1])
        cu = np.zeros([1, 1])

        # Discrete horizon
        solver = LQR(A=self.A_d, B=self.B_d, Cx=Cx, Cu=Cu, cx=cx, cu=cu)

        V_Tx, v_Tx, _, _ = solver.backward_pass(solver.V_Ty_i, solver.v_Ty_i)

        for _ in np.arange(100):
            V_Tx, v_Tx, K_discrete, _ = solver.backward_pass(V_Tx, v_Tx)
        K_discrete = np.squeeze(K_discrete)

        # Infinite horizon
        K_infinite, _, _ = control.lqr(self.A, self.B, Cx, Cu)
        # Minus sign for convention
        K_infinite = -np.squeeze(np.array(K_infinite))

        # Assert close (7%)
        self.assertGreaterEqual(K_discrete[0] / K_infinite[0], 0.93)
        self.assertGreaterEqual(K_discrete[1] / K_infinite[1], 0.93)
        self.assertGreaterEqual(K_discrete[2] / K_infinite[2], 0.93)
        self.assertLessEqual(K_discrete[0] / K_infinite[0], 1.07)
        self.assertLessEqual(K_discrete[1] / K_infinite[1], 1.07)
        self.assertLessEqual(K_discrete[2] / K_infinite[2], 1.07)


# # %%
# Ks = []
# ks = []
# for t in np.arange(40, 0, -dt):
#     V_Tx, v_Tx, K_Tx, k_Tx = solver.backward_pass(V_Tx, v_Tx)
#     print(K_Tx)
#     Ks.append(K_Tx)
#     ks.append(k_Tx)

# xs = []
# ts = np.arange(0, 40, dt)
# x = np.array([
#     [0],
#     [0],
#     [-1]
# ])
# xs.append(x)
# for t, K, k in zip(ts, Ks[::-1], ks[::-1]):
#     x = solver.step(x, K, k)
#     xs.append(x)

# # %%
# fig = go.Figure()

# fig.add_trace(
#     go.Scatter(
#         x=ts,
#         y=[x[2, 0] for x in xs]
#     )
# )

# fig.show()

if __name__ == "__main__":
    unittest.main(verbosity=2)
