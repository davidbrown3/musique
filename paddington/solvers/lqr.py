import torch


class LQR:

    def __init__(self, T_x, T_u, g_x, g_u, g_xx, g_uu, g_xu, dt):

        # LTI
        self.T_x = T_x
        self.T_u = T_u
        self.g_x = g_x
        self.g_u = g_u
        self.g_xx = g_xx
        self.g_uu = g_uu
        self.g_xu = g_xu
        self.dt = dt

        # Initialisation
        self.R_x, self.R_xx, _, _ = self._backward_pass(
            R_xx=torch.zeros_like(g_xx, dtype=torch.float64),
            R_x=torch.zeros_like(g_x, dtype=torch.float64)
        )

    @staticmethod
    def step(x, beta, alpha, T_x, T_u):

        u = torch.matmul(beta, x) + alpha
        x_new = torch.matmul(T_x, x) + torch.matmul(T_u, u)

        return x_new, u

    @staticmethod
    def backward_pass(R_x, R_xx, T_x, T_u, g_x, g_u, g_xx, g_uu, g_xu):

        # Derivation
        g_ux = g_xu.T

        # Q Value
        Q_xx = g_xx + torch.matmul(T_x.T, torch.matmul(R_xx, T_x))
        Q_uu = g_uu + torch.matmul(T_u.T, torch.matmul(R_xx, T_u))
        Q_xu = g_xu + torch.matmul(T_x.T, torch.matmul(R_xx, T_u))
        Q_ux = Q_xu.T
        Q_x = g_x + torch.matmul(R_x, T_x)
        Q_u = torch.matmul(R_x, T_u)

        # Control constants
        beta, _ = torch.solve(Q_ux, -Q_uu)
        alpha, _ = torch.solve(Q_u, -Q_uu)

        # Return function constants
        R_xx = Q_xx + torch.matmul(Q_xu, beta) + torch.matmul(beta.T, Q_ux) + torch.matmul(beta.T, torch.matmul(Q_uu, beta))
        R_x = Q_x + torch.matmul(Q_u, beta) + torch.matmul(Q_xu, alpha).T + torch.matmul(alpha.T, torch.matmul(Q_uu, beta))

        assert(R_x.shape[0] == 1)  # TODO: Replace with unittest

        return R_x, R_xx, beta, alpha

    def _backward_pass(self, R_x, R_xx):
        return self.backward_pass(R_x=R_x, R_xx=R_xx, T_x=self.T_x, T_u=self.T_u, g_x=self.g_x, g_u=self.g_u, g_xx=self.g_xx, g_uu=self.g_uu, g_xu=self.g_xu)

    def K_horizon(self, N_Steps):

        R_x, R_xx = self.R_x, self.R_xx

        for t in range(N_Steps):
            R_x, R_xx, beta, alpha = self._backward_pass(R_x=R_x, R_xx=R_xx)

        return beta, alpha

    def solve(self, states_initial, time_total):

        betas = []
        alphas = []
        R_xx, R_x = self.R_xx, self.R_x

        for t in torch.arange(time_total, 0, -self.dt):
            R_x, R_xx, beta, alpha = self._backward_pass(R_x=R_x, R_xx=R_xx)
            betas.append(beta)
            alphas.append(alpha)

        xs = []
        us = []
        ts = torch.arange(0, time_total, self.dt)

        x = states_initial

        xs.append(x)
        for t, beta, alpha in zip(ts, betas[::-1], alphas[::-1]):
            x, u = self.step(x=x, beta=beta, alpha=alpha, T_x=self.T_x, T_u=self.T_u)
            xs.append(x)
            us.append(u)
        us.append(u * 0.0)
        return ts, xs, us
