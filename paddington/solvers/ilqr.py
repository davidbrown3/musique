import torch

from paddington.solvers.lqr import LQR
from paddington.tools.controls_tools import convert_syntax_transition


class iLQR:

    def __init__(self, plant, cost_function, dt):

        self.plant = plant
        self.cost_function = cost_function
        self.dt = dt

    def initial_guess_lqr(self, states_initial, time_total):

        controls_initial = torch.zeros([self.plant.N_u, 1])
        A_d, B_d = self.plant.calculate_statespace_discrete(x=states_initial[:, 0], u=controls_initial[:, 0], dt=self.dt)
        F, f = convert_syntax_transition(A_d, B_d)
        C = self.cost_function.calculate_cost_hessian(x=states_initial, u=controls_initial)
        c = self.cost_function.calculate_cost_jacobian(x=states_initial, u=controls_initial)

        lqr = LQR(F=F, f=f, C=C, c=c, dt=self.dt, N_x=self.plant.N_x, N_u=self.plant.N_u)

        return lqr.solve(states_initial, time_total)

    def solve(self, states_initial, time_total):

        ts, x_bars, u_bars = self.initial_guess_lqr(states_initial=states_initial, time_total=time_total)

        print(
            torch.sum(
                torch.cat(
                    [self.cost_function.calculate_cost(x, u) for x, u in zip(x_bars, u_bars)]
                )
            )
        )

        for _ in range(5):
            Ks, ks = self.backward_pass(xs=x_bars, us=u_bars)
            x_bars, u_bars = self.forward_pass(x_bars=x_bars, u_bars=u_bars, Ks=Ks, ks=ks)

    def backward_pass(self, xs, us):

        # Initialisation
        A_d, B_d = self.plant.calculate_statespace_discrete(x=xs[-1][:, 0], u=us[-1][:, 0], dt=self.dt)
        F, f = convert_syntax_transition(A_d, B_d)

        V_Ty = torch.zeros([self.plant.N_x, self.plant.N_x])
        v_Ty = torch.zeros([self.plant.N_x, 1])

        Ks = []
        ks = []
        for x, u in zip(xs[::-1], us[::-1]):

            # TODO: Want jacobian, not state space
            A_d, B_d = self.plant.calculate_statespace_discrete(x=x[:, 0], u=u[:, 0], dt=self.dt)
            F_Tx, f_Tx = convert_syntax_transition(A_d, B_d)
            C_Tx = self.cost_function.calculate_cost_hessian(x=x, u=u)
            c_Tx = self.cost_function.calculate_cost_jacobian(x=x, u=u)

            V_Ty, v_Ty, K_Tx, k_Tx = LQR.backward_pass(V_Ty=V_Ty, v_Ty=v_Ty, F_Tx=F_Tx, f_Tx=f_Tx, C_Tx=C_Tx, c_Tx=c_Tx,
                                                       N_x=self.plant.N_x, N_u=self.plant.N_u)

            Ks.append(K_Tx)
            ks.append(k_Tx)

        return Ks[::-1], ks[::-1]

    def forward_pass(self, x_bars, u_bars, Ks, ks, alpha=1.0):

        x = x_bars[0]
        xs = []
        us = []
        cost = 0
        for x_bar, u_bar, K, k in zip(x_bars, u_bars, Ks, ks):
            xs.append(x)
            dx = x - x_bar
            du = torch.matmul(K, dx) + alpha * k
            u = u_bar + du
            cost += self.cost_function.calculate_cost(x, u)
            x = self.plant.step(x, u, dt=self.dt)
            us.append(u)

        us.append(u * 0.0)
        print(cost)

        return xs, us
