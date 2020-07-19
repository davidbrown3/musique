import torch

from paddington.solvers.lqr import LQR


class iLQR:

    def __init__(self, plant, cost_function, dt):

        self.plant = plant
        self.cost_function = cost_function
        self.dt = dt

    def initial_guess_lqr(self, states_initial, time_total):

        controls_initial = torch.zeros([self.plant.N_u, 1], dtype=torch.float64)
        A_d, B_d = self.plant.calculate_statespace_discrete(x=states_initial[:, 0], u=controls_initial[:, 0], dt=self.dt)

        lqr = LQR(T_x=A_d,
                  T_u=B_d,
                  g_x=self.cost_function.calculate_g_x(x=states_initial, u=controls_initial),
                  g_u=self.cost_function.calculate_g_u(x=states_initial, u=controls_initial),
                  g_xx=self.cost_function.calculate_g_xx(x=states_initial, u=controls_initial),
                  g_uu=self.cost_function.calculate_g_uu(x=states_initial, u=controls_initial),
                  g_xu=self.cost_function.calculate_g_xu(x=states_initial, u=controls_initial),
                  dt=self.dt)

        return lqr.solve(states_initial, time_total)

    def solve(self, states_initial, time_total, convergence=1e-4):

        ts, x_bars, u_bars = self.initial_guess_lqr(states_initial=states_initial, time_total=time_total)

        cost_prev = torch.sum(torch.cat(
            [self.cost_function.calculate_cost(x, u) for x, u in zip(x_bars, u_bars)]
        ))

        print(cost_prev)

        # Setting up while loop
        cost = cost_prev
        cost_prev = cost * 2.0
        while (torch.abs((cost - cost_prev)) / cost) > convergence:
            cost_prev = cost
            betas, alphas = self.backward_pass(xs=x_bars, us=u_bars)
            x_bars, u_bars, cost = self.forward_pass(x_bars=x_bars, u_bars=u_bars, betas=betas, alphas=alphas)
            print(cost)

        return x_bars, u_bars

    def backward_pass(self, xs, us):

        R_xx = torch.zeros([self.plant.N_x, self.plant.N_x], dtype=torch.float64)
        R_x = torch.zeros([1, self.plant.N_x], dtype=torch.float64)

        betas = []
        alphas = []
        for x, u in zip(xs[::-1], us[::-1]):

            T_x, T_u = self.plant.calculate_statespace_discrete(x=x[:, 0], u=u[:, 0], dt=self.dt)

            R_x, R_xx, beta, alpha = LQR.backward_pass(R_x=R_x,
                                                       R_xx=R_xx,
                                                       T_x=T_x,
                                                       T_u=T_u,
                                                       g_x=self.cost_function.calculate_g_x(x=x, u=u),
                                                       g_u=self.cost_function.calculate_g_u(x=x, u=u),
                                                       g_xx=self.cost_function.calculate_g_xx(x=x, u=u),
                                                       g_uu=self.cost_function.calculate_g_uu(x=x, u=u),
                                                       g_xu=self.cost_function.calculate_g_xu(x=x, u=u))

            betas.append(beta)
            alphas.append(alpha)

        return betas[::-1], alphas[::-1]

    def forward_pass(self, x_bars, u_bars, betas, alphas, line_alpha=0.1):

        x = x_bars[0]
        xs = []
        us = []
        cost = 0
        for x_bar, u_bar, beta, alpha in zip(x_bars, u_bars, betas, alphas):
            xs.append(x)
            dx = x - x_bar
            du = torch.matmul(beta, dx) + line_alpha * alpha
            u = u_bar + du
            cost += self.cost_function.calculate_cost(x, u)
            x = self.plant.step(x, u, dt=self.dt)
            us.append(u)

        return xs, us, cost
