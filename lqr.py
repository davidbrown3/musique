import numpy as np
from scipy import linalg


class LQR:

    def __init__(self, A, B, Cx, Cu, cx, cu):

        shape = np.shape(B)
        self.N_x = shape[0]
        self.N_u = shape[1]

        F = np.concatenate((A, B), axis=1)
        f = np.zeros([self.N_x, 1])
        C = linalg.block_diag(Cx, Cu)
        c = linalg.block_diag(cx, cu)

        # LTI
        self.F_Tx = F
        self.f_Tx = f
        self.C_Tx = C
        self.c_Tx = c

        # Initialisation
        self.V_Ty_i = np.zeros([self.N_x, self.N_x])
        self.v_Ty_i = np.zeros([self.N_x, 1])

    def backward_pass(self, V_Ty, v_Ty):

        Q_Tx = self.C_Tx + np.dot(self.F_Tx.T, np.dot(V_Ty, self.F_Tx))
        q_Tx = self.c_Tx + np.dot(self.F_Tx.T, np.dot(V_Ty, self.f_Tx)) + np.dot(self.F_Tx.T, v_Ty)

        Q_xx_Tx = Q_Tx[0:self.N_x, 0:self.N_x]
        Q_uu_Tx = Q_Tx[self.N_x:(self.N_x+self.N_u), self.N_x:(self.N_x+self.N_u)]
        Q_ux_Tx = Q_Tx[self.N_x:(self.N_x+self.N_u), 0:self.N_x]
        Q_xu_Tx = Q_Tx[0:self.N_x, self.N_x:(self.N_x+self.N_u)]
        q_x_Tx = q_Tx[0:self.N_x]
        q_u_Tx = q_Tx[self.N_x:(self.N_x+self.N_u)]

        K_Tx = np.linalg.solve(Q_uu_Tx, Q_ux_Tx)
        k_Tx = np.linalg.solve(Q_uu_Tx, q_u_Tx)

        V_Tx = Q_xx_Tx + np.dot(Q_xu_Tx, K_Tx) + np.dot(K_Tx.T, Q_ux_Tx) + np.dot(K_Tx.T, np.dot(Q_uu_Tx, K_Tx))
        v_Tx = np.dot(Q_xu_Tx, k_Tx) + np.dot(K_Tx.T, np.dot(Q_uu_Tx, k_Tx)) + np.dot(K_Tx.T, q_u_Tx) + q_x_Tx

        return V_Tx, v_Tx
