import jax
import jax.numpy as np

from paddington.solvers.lqr import LQR


class iLQR:

    def __init__(self, plant, cost_function):

        self.plant = plant
        self.cost_function = cost_function
        self._forward_pass_inner = jax.jit(self.forward_pass_inner)
        self._forward_pass = jax.jit(self.forward_pass)
        self._backward_pass = jax.jit(self.backward_pass)

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

        xs = np.concatenate(x_bars, 1).T
        us = np.concatenate(u_bars, 1).T

        # cost_prev = np.sum(np.concatenate(
        #     [self.cost_function.calculate_cost(x, u) for x, u in zip(x_bars, u_bars)]
        # ))

        # print(cost_prev)

        # # Setting up while loops
        # cost = cost_prev
        # cost_prev = cost * 2.0
        # while (np.abs((cost - cost_prev)) / cost) > convergence:
        for _ in range(20):

            # cost_prev = cost
            betas, alphas = self._backward_pass(xs=xs, us=us)
            xs, us = self._forward_pass(xs=xs, us=us, betas=betas, alphas=alphas)

            # cost = np.sum(np.concatenate(
            #     [self.cost_function.calculate_cost(x, u) for x, u in zip(x_bars, u_bars)]
            # ))
            # print(cost)

        return xs, us

    def backward_pass(self, xs, us):

        xs = np.flip(xs, axis=0)
        us = np.flip(us, axis=0)

        T_xs, T_us = self.plant.calculate_statespace_discrete_batch(xs, us)
        g_xs = self.cost_function.calculate_g_x_batch(xs, us)
        g_us = self.cost_function.calculate_g_u_batch(xs, us)

        g_xx = self.cost_function.g_xx
        g_uu = self.cost_function.g_uu
        g_xu = self.cost_function.g_xu
        g_ux = self.cost_function.g_ux

        R_xx = g_xx * 0.0
        R_x = g_xs[0] * 0.0

        betas = np.empty([xs.shape[0], us.shape[1], xs.shape[1]])
        alphas = np.empty([us.shape[0], us.shape[1]])

        data = (R_x, R_xx, T_xs, T_us, g_xs, g_us, g_xx, g_uu, g_xu, g_ux, betas, alphas)
        data = jax.lax.fori_loop(0, xs.shape[0], iLQR.backward_pass_inner, data)

        return np.flip(data[-2], axis=0), np.flip(data[-1], axis=0)

    @staticmethod
    @jax.jit
    def backward_pass_inner(i, data):

        R_x, R_xx, T_xs, T_us, g_xs, g_us, g_xx, g_uu, g_xu, g_ux, betas, alphas = data

        R_x, R_xx, beta, alpha = LQR.backward_pass(R_x=R_x,
                                                   R_xx=R_xx,
                                                   T_x=T_xs[jax.ops.index[i, :, :]],
                                                   T_u=T_us[jax.ops.index[i, :, :]],
                                                   g_x=g_xs[jax.ops.index[i, :]],
                                                   g_u=g_us[jax.ops.index[i, :]],
                                                   g_xx=g_xx,
                                                   g_uu=g_uu,
                                                   g_xu=g_xu,
                                                   g_ux=g_ux)

        betas = jax.ops.index_update(betas, jax.ops.index[i, :], beta)
        alphas = jax.ops.index_update(alphas, jax.ops.index[i, :], alpha.squeeze())

        return (R_x, R_xx, T_xs, T_us, g_xs, g_us, g_xx, g_uu, g_xu, g_ux, betas, alphas)

    def forward_pass(self, xs, us, betas, alphas, line_alpha=0.1):

        x_ = xs[0, :]

        data = (xs, us, betas, alphas, x_, line_alpha)
        data = jax.lax.fori_loop(0, xs.shape[0], self._forward_pass_inner, data)

        return data[0], data[1]

    def forward_pass_inner(self, i, data):

        xs, us, betas, alphas, x_, line_alpha = data

        dx = x_ - xs[i]
        xs = jax.ops.index_update(xs, jax.ops.index[i, :], x_)
        du = np.dot(betas[i], dx) + line_alpha * alphas[i]
        u_ = us[i] + du
        us = jax.ops.index_update(us, jax.ops.index[i, :], u_.squeeze())
        x_ = self.plant.step(x_, u_)

        return (xs, us, betas, alphas, x_, line_alpha)
