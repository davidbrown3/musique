import torch


class LQR:

    def __init__(self, F, f, C, c, dt, N_x, N_u):

        # LTI
        self.N_x = N_x
        self.N_u = N_u
        self.F = F
        self.f = f
        self.C = C
        self.c = c
        self.dt = dt

        # Initialisation
        self.V_i, self.v_i, _, _ = self._backward_pass(
            V_Ty=torch.zeros([self.N_x, self.N_x]),
            v_Ty=torch.zeros([self.N_x, 1])
        )

    @staticmethod
    def step(x, K, k, F, f):

        u = torch.matmul(K, x) + k
        X = torch.cat((x, u))
        x_new = torch.matmul(F, X) + f

        return x_new, u

    @staticmethod
    def backward_pass(V_Ty, v_Ty, F_Tx, f_Tx, C_Tx, c_Tx, N_x, N_u):

        # Q Value constants
        Q_Tx = C_Tx + torch.matmul(F_Tx.T, torch.matmul(V_Ty, F_Tx))
        q_Tx = c_Tx + torch.matmul(F_Tx.T, torch.matmul(V_Ty, f_Tx)) + torch.matmul(F_Tx.T, v_Ty)

        # Q Value constant slices
        Q_xx_Tx = Q_Tx[0:N_x, 0:N_x]
        Q_uu_Tx = Q_Tx[N_x:(N_x+N_u), N_x:(N_x+N_u)]
        Q_ux_Tx = Q_Tx[N_x:(N_x+N_u), 0:N_x]
        Q_xu_Tx = Q_Tx[0:N_x, N_x:(N_x+N_u)]
        q_x_Tx = q_Tx[0:N_x]
        q_u_Tx = q_Tx[N_x:(N_x+N_u)]

        # Control constants
        K_Tx, _ = torch.solve(Q_ux_Tx, -Q_uu_Tx)
        k_Tx, _ = torch.solve(q_u_Tx, -Q_uu_Tx)

        # Value function constants
        V_Tx = Q_xx_Tx + torch.matmul(Q_xu_Tx, K_Tx) + torch.matmul(K_Tx.T, Q_ux_Tx) + torch.matmul(K_Tx.T, torch.matmul(Q_uu_Tx, K_Tx))
        v_Tx = torch.matmul(Q_xu_Tx, k_Tx) + torch.matmul(K_Tx.T, torch.matmul(Q_uu_Tx, k_Tx)) + torch.matmul(K_Tx.T, q_u_Tx) + q_x_Tx

        assert(v_Tx.shape[1] == 1)  # TODO: Replace with unittest

        return V_Tx, v_Tx, K_Tx, k_Tx

    def _backward_pass(self, V_Ty, v_Ty):
        return self.backward_pass(V_Ty, v_Ty, F_Tx=self.F, f_Tx=self.f, C_Tx=self.C, c_Tx=self.c, N_x=self.N_x, N_u=self.N_u)

    def K_horizon(self, N_Steps):

        V_Ty, v_Ty = self.V_i, self.v_i

        for t in range(N_Steps):
            V_Ty, v_Ty, K, k = self._backward_pass(V_Ty, v_Ty)

        return K, k

    def solve(self, states_initial, time_total):

        Ks = []
        ks = []
        V_Ty, v_Ty = self.V_i, self.v_i

        for t in torch.arange(time_total, 0, -self.dt):
            V_Ty, v_Ty, K_Ty, k_Ty = self._backward_pass(V_Ty, v_Ty)
            Ks.append(K_Ty)
            ks.append(k_Ty)

        xs = []
        us = []
        ts = torch.arange(0, time_total, self.dt)

        x = states_initial

        xs.append(x)
        for t, K, k in zip(ts, Ks[::-1], ks[::-1]):
            x, u = self.step(x=x, K=K, k=k, F=self.F, f=self.f)
            xs.append(x)
            us.append(u)
        us.append(u * 0.0)
        return ts, xs, us
