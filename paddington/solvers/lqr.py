import torch


class LQR:

    def __init__(self, T_x, T_u, g_x, g_u, g_xx, g_uu, g_xu, g_ux, dt):

        # LTI
        self.T_x = T_x
        self.T_u = T_u
        self.g_x = g_x
        self.g_u = g_u
        self.g_xx = g_xx
        self.g_uu = g_uu
        self.g_xu = g_xu
        self.g_ux = g_ux
        self.dt = dt

        N_x = T_x.shape[0]
        N_u = T_u.shape[1]

        if T_x.shape != (N_x, N_x):
            raise ValueError('T_x must have be a square matrix')
        if T_u.shape != (N_x, N_u):
            raise ValueError('T_u must have shape (N_x, N_u)')
        if g_xx.shape != (N_x, N_x):
            raise ValueError('g_xx must have shape (N_x, N_x)')
        if g_xu.shape != (N_x, N_u):
            raise ValueError('g_xu must have  shape (N_x, N_u)')
        if g_ux.shape != (N_u, N_x):
            raise ValueError('g_ux must have  shape (N_u, N_x)')
        if g_uu.shape != (N_u, N_u):
            raise ValueError('g_uu must have  shape (N_u, N_u)')
        if g_x.shape != (1, N_x):
            raise ValueError('g_x must have  shape (N_x, 1)')
        if g_u.shape != (1, N_u):
            raise ValueError('g_u must have  shape (N_u, 1)')

        # Initialisation
        self.R_x, self.R_xx, _, _ = self._backward_pass(
            R_xx=torch.zeros_like(g_xx),
            R_x=torch.zeros_like(g_x)
        )

    @staticmethod
    def step(x, beta, alpha, T_x, T_u):

        u = torch.matmul(beta, x) + alpha
        x_new = torch.matmul(T_x, x) + torch.matmul(T_u, u)

        return x_new, u

    @staticmethod
    def calculate_Q_partials(R_x, R_xx, T_x, T_u, g_x, g_u, g_xx, g_uu, g_xu, g_ux):

        # Q Value
        Q_xx = g_xx + torch.matmul(T_x.T, torch.matmul(R_xx, T_x))
        Q_uu = g_uu + torch.matmul(T_u.T, torch.matmul(R_xx, T_u))
        Q_xu = g_xu + torch.matmul(T_x.T, torch.matmul(R_xx, T_u))
        Q_ux = g_ux + torch.matmul(T_u.T, torch.matmul(R_xx, T_x))
        Q_x = g_x + torch.matmul(R_x, T_x)
        Q_u = torch.matmul(R_x, T_u)

        return Q_xx, Q_uu, Q_xu, Q_ux, Q_x, Q_u

    def calculate_control_gains(Q_ux, Q_uu, Q_u):

        beta, _ = torch.solve(Q_ux, -Q_uu)
        alpha, _ = torch.solve(Q_u, -Q_uu)

        return beta, alpha

    @staticmethod
    def backward_pass(R_x, R_xx, T_x, T_u, g_x, g_u, g_xx, g_uu, g_xu, g_ux):

        Q_xx, Q_uu, Q_xu, Q_ux, Q_x, Q_u = LQR.calculate_Q_partials(R_x, R_xx, T_x, T_u, g_x, g_u, g_xx, g_uu, g_xu, g_ux)
        beta, alpha = LQR.calculate_control_gains(Q_ux, Q_uu, Q_u)

        # Return function constants
        R_xx = Q_xx + torch.matmul(Q_xu, beta) + torch.matmul(beta.T, Q_ux) + torch.matmul(beta.T, torch.matmul(Q_uu, beta))
        R_x = Q_x + torch.matmul(beta.T, Q_u).T + torch.matmul(Q_xu, alpha).T + torch.matmul(beta.T, torch.matmul(Q_uu, alpha)).T

        assert(R_x.shape[0] == 1)  # TODO: Replace with unittest

        return R_x, R_xx, beta, alpha

    def _backward_pass(self, R_x, R_xx):
        return LQR.backward_pass(R_x=R_x, R_xx=R_xx, T_x=self.T_x, T_u=self.T_u, g_x=self.g_x, g_u=self.g_u, g_xx=self.g_xx, g_uu=self.g_uu, g_xu=self.g_xu, g_ux=self.g_ux)

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
