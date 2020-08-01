import jax.numpy as np

from paddington.solvers.lqr import LQR


class iLQR:

    def __init__(self, plant, cost_function):

        self.plant = plant
        self.cost_function = cost_function

    def initial_guess_lqr(self, states_initial, time_total):

        controls_initial = np.zeros([self.plant.N_u, 1])
        A_d, B_d = self.plant.calculate_statespace_discrete(states_initial[:, 0], controls_initial[:, 0])

        lqr = LQR(T_x=A_d,
                  T_u=B_d,
                  g_x=self.cost_function.calculate_g_x(x=states_initial, u=controls_initial),
                  g_u=self.cost_function.calculate_g_u(x=states_initial, u=controls_initial),
                  g_xx=self.cost_function.g_xx,
                  g_uu=self.cost_function.g_uu,
                  g_xu=self.cost_function.g_xu,
                  g_ux=self.cost_function.g_ux,
                  dt=self.plant.dt)

        return lqr.solve(states_initial, time_total)

    def solve(self, states_initial, time_total, convergence=1e-4):

        ts, x_bars, u_bars = self.initial_guess_lqr(states_initial=states_initial, time_total=time_total)

        cost_prev = np.sum(np.concatenate(
            [self.cost_function.calculate_cost(x, u) for x, u in zip(x_bars, u_bars)]
        ))

        print(cost_prev)

        # Setting up while loops
        cost = cost_prev
        cost_prev = cost * 2.0
        # while (np.abs((cost - cost_prev)) / cost) > convergence:
        for _ in range(10):

            cost_prev = cost
            betas, alphas = self.backward_pass(xs=x_bars, us=u_bars)
            x_bars, u_bars = self.forward_pass(x_bars=x_bars, u_bars=u_bars, betas=betas, alphas=alphas)

            # cost = np.sum(np.concatenate(
            #     [self.cost_function.calculate_cost(x, u) for x, u in zip(x_bars, u_bars)]
            # ))
            # print(cost)

        return x_bars, u_bars

    def backward_pass(self, xs, us):

        _xs = np.concatenate(xs[::-1], 1)
        _us = np.concatenate(us[::-1], 0)
        # Might have to handle special case where len(u)==1
        T_xs, T_us = self.plant.calculate_statespace_discrete_batch(_xs, _us)
        g_xs = self.cost_function.calculate_g_u_batch(_xs, _us)
        g_us = self.cost_function.calculate_g_u_batch(_xs, _us)

        R_xx = np.zeros([self.plant.N_x, self.plant.N_x])
        R_x = np.zeros([1, self.plant.N_x])

        betas = []
        alphas = []

        g_xx = self.cost_function.g_xx
        g_uu = self.cost_function.g_uu
        g_xu = self.cost_function.g_xu
        g_ux = self.cost_function.g_ux

        for x, u, T_x, T_u, g_x, g_u in zip(xs[::-1], us[::-1], T_xs, T_us, g_xs, g_us):

            # Faster to do a 2nd diff on statespace than calculate hessian fresh

            R_x, R_xx, beta, alpha = LQR.backward_pass(R_x=R_x,
                                                       R_xx=R_xx,
                                                       T_x=T_x,
                                                       T_u=T_u,
                                                       g_x=g_x,
                                                       g_u=g_u,
                                                       g_xx=g_xx,
                                                       g_uu=g_uu,
                                                       g_xu=g_xu,
                                                       g_ux=g_ux)

            betas.append(beta)
            alphas.append(alpha)

        return betas[::-1], alphas[::-1]

    def forward_pass(self, x_bars, u_bars, betas, alphas, line_alpha=0.1):

        x = x_bars[0]
        xs = []
        us = []
        for x_bar, u_bar, beta, alpha in zip(x_bars, u_bars, betas, alphas):
            xs.append(x)
            dx = x - x_bar
            du = np.dot(beta, dx) + line_alpha * alpha
            u = u_bar + du
            x = self.plant.step(x, u)
            us.append(u)

        return xs, us
