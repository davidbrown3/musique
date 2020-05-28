import torch


class LQR:

    def __init__(self, F, f, C, c, N_x, N_u):

        # LTI
        self.N_x = N_x
        self.N_u = N_u
        self.F = F
        self.f = f
        self.C = C
        self.c = c

        # Initialisation
        self.V_Ty_i = torch.zeros([self.N_x, self.N_x])
        self.v_Ty_i = torch.zeros([self.N_x, 1])

    def step(self, x, K, k, F_Tx=None):

        if F_Tx is None:
            F_Tx = self.F

        u = torch.matmul(K, x) + k
        X = torch.cat((x, u))
        x_new = torch.matmul(F_Tx, X)

        return x_new

    def backward_pass(self, V_Ty, v_Ty, F_Tx=None, f_Tx=None, C_Tx=None, c_Tx=None):

        if F_Tx is None:
            F_Tx = self.F

        if f_Tx is None:
            f_Tx = self.f

        if C_Tx is None:
            C_Tx = self.C

        if c_Tx is None:
            c_Tx = self.c

        # Q Value constants
        Q_Tx = C_Tx + torch.matmul(F_Tx.T, torch.matmul(V_Ty, F_Tx))
        q_Tx = c_Tx + torch.matmul(F_Tx.T, torch.matmul(V_Ty, f_Tx)) + torch.matmul(F_Tx.T, v_Ty)

        # Q Value constant slices
        Q_xx_Tx = Q_Tx[0:self.N_x, 0:self.N_x]
        Q_uu_Tx = Q_Tx[self.N_x:(self.N_x+self.N_u), self.N_x:(self.N_x+self.N_u)]
        Q_ux_Tx = Q_Tx[self.N_x:(self.N_x+self.N_u), 0:self.N_x]
        Q_xu_Tx = Q_Tx[0:self.N_x, self.N_x:(self.N_x+self.N_u)]
        q_x_Tx = q_Tx[0:self.N_x, 0]
        q_u_Tx = q_Tx[self.N_x:(self.N_x+self.N_u), 0]

        if len(q_u_Tx.shape) == 1:
            q_u_Tx = torch.unsqueeze(q_u_Tx, dim=0)

        # Control constants
        K_Tx, _ = torch.solve(Q_ux_Tx, -Q_uu_Tx)
        k_Tx, _ = torch.solve(q_u_Tx, -Q_uu_Tx)

        # Value function constants
        V_Tx = Q_xx_Tx + torch.matmul(Q_xu_Tx, K_Tx) + torch.matmul(K_Tx.T, Q_ux_Tx) + torch.matmul(K_Tx.T, torch.matmul(Q_uu_Tx, K_Tx))
        v_Tx = torch.matmul(Q_xu_Tx, k_Tx) + torch.matmul(K_Tx.T, torch.matmul(Q_uu_Tx, k_Tx)) + torch.matmul(K_Tx.T, q_u_Tx) + q_x_Tx

        return V_Tx, v_Tx, K_Tx, k_Tx
