import jax
import jax.numpy as np


class iLQR:

    def __init__(self, plant, running_cost_function, terminal_cost_function):

        self.plant = plant
        self.running_cost_function = running_cost_function
        self.terminal_cost_function = terminal_cost_function
        self._forward_pass_inner = jax.jit(self.forward_pass_inner)
        self._forward_pass = jax.jit(self.forward_pass)
        self._backward_pass = jax.jit(self.backward_pass)

    def initial_guess_lqr(self, states_initial, time_total):

        controls_initial = np.zeros([self.plant.N_u, 1])
        A_d, B_d = self.plant.calculate_statespace_discrete(states_initial[:, 0], controls_initial[:, 0])

        lqr = LQR(T_x=A_d,
                  T_u=B_d,
                  g_x=self.running_cost_function.calculate_g_x(x=states_initial, u=controls_initial),
                  g_u=self.running_cost_function.calculate_g_u(x=states_initial, u=controls_initial),
                  g_xx=self.running_cost_function.g_xx,
                  g_uu=self.running_cost_function.g_uu,
                  g_xu=self.running_cost_function.g_xu,
                  g_ux=self.running_cost_function.g_ux,
                  dt=self.plant.dt)

        return lqr.K_horizon(500)

    def solve(self, states_initial, time_total, convergence=1e-4):

        beta, alpha = self.initial_guess_lqr(states_initial=states_initial, time_total=time_total)

        N_steps = int(time_total / self.plant.dt)
        xs = np.zeros([N_steps, self.plant.N_x])
        us = np.zeros([N_steps, self.plant.N_u])
        betas = np.tile(beta, (N_steps, 1, 1))
        alphas = np.zeros([N_steps, self.plant.N_u])

        xs, us = self.forward_pass(np.squeeze(states_initial), xs, us, betas, alphas)

        costs = []
        for _ in range(300):

            # cost_prev = cost
            betas, alphas = self._backward_pass(xs=xs, us=us)
            xs, us = self._forward_pass(xi=np.squeeze(states_initial), xs=xs, us=us, betas=betas, alphas=alphas)
            cost = np.sum(self.running_cost_function.calculate_cost_batch(xs, us)) + \
                np.sum(self.terminal_cost_function.calculate_cost(xs[-1], us[-1]))
            costs.append(cost)

        return xs, us, costs

    def backward_pass(self, xs, us):

        xs = np.flip(xs, axis=0)
        us = np.flip(us, axis=0)

        T_xs, T_us = self.plant.calculate_statespace_discrete_batch(xs, us)
        T_hessians = self.plant.calculate_hessian_discrete_batch(xs, us)
        T_xxs = T_hessians[0][0]
        T_xus = T_hessians[0][1]
        T_uxs = T_hessians[1][0]
        T_uus = T_hessians[1][1]

        g_xs = self.running_cost_function.calculate_g_x_batch(xs, us)
        g_us = self.running_cost_function.calculate_g_u_batch(xs, us)

        g_xx = self.running_cost_function.g_xx
        g_uu = self.running_cost_function.g_uu
        g_xu = self.running_cost_function.g_xu
        g_ux = self.running_cost_function.g_ux

        R_xx = self.terminal_cost_function.g_xx
        R_x = self.terminal_cost_function.calculate_g_x(xs[0], us[0])

        betas = np.empty([xs.shape[0], us.shape[1], xs.shape[1]])
        alphas = np.empty([us.shape[0], us.shape[1]])

        data = (R_x, R_xx, T_xs, T_us, T_xxs, T_xus, T_uxs, T_uus, g_xs, g_us, g_xx, g_uu, g_xu, g_ux, betas, alphas)
        data = jax.lax.fori_loop(0, xs.shape[0], iLQR.backward_pass_inner, data)

        return np.flip(data[-2], axis=0), np.flip(data[-1], axis=0)

    @ staticmethod
    @ jax.jit
    def backward_pass_inner(i, data):

        R_x, R_xx, T_xs, T_us, T_xxs, T_xus, T_uxs, T_uus, g_xs, g_us, g_xx, g_uu, g_xu, g_ux, betas, alphas = data

        Q_xx, Q_uu, Q_xu, Q_ux, Q_x, Q_u = iLQR.calculate_Q_partials(
            R_x=R_x,
            R_xx=R_xx,
            T_x=T_xs[jax.ops.index[i, :, :]],
            T_u=T_us[jax.ops.index[i, :, :]],
            T_xx=T_xxs[jax.ops.index[i, :, :, :]],
            T_xu=T_xus[jax.ops.index[i, :, :, :]],
            T_ux=T_uxs[jax.ops.index[i, :, :, :]],
            T_uu=T_uus[jax.ops.index[i, :, :, :]],
            g_x=g_xs[jax.ops.index[i, :]],
            g_u=g_us[jax.ops.index[i, :]],
            g_xx=g_xx,
            g_uu=g_uu,
            g_xu=g_xu,
            g_ux=g_ux
        )

        beta, alpha = iLQR.calculate_control_gains(Q_ux, Q_uu, Q_u)

        # Return function constants
        R_xx = Q_xx + np.matmul(Q_xu, beta) + np.matmul(beta.T, Q_ux) + np.matmul(beta.T, np.matmul(Q_uu, beta))
        R_x = Q_x + np.matmul(Q_u, beta) + np.matmul(Q_xu, alpha).T + np.matmul(alpha.T, np.matmul(Q_uu, beta))

        assert(R_x.shape[0] == 1)  # TODO: Replace with unittest

        betas = jax.ops.index_update(betas, jax.ops.index[i, :], beta)
        alphas = jax.ops.index_update(alphas, jax.ops.index[i, :], alpha.squeeze())

        return (R_x, R_xx, T_xs, T_us, T_xxs, T_xus, T_uxs, T_uus, g_xs, g_us, g_xx, g_uu, g_xu, g_ux, betas, alphas)

    @staticmethod
    @jax.jit
    def calculate_Q_partials(R_x, R_xx, T_x, T_u, T_xx, T_xu, T_ux, T_uu, g_x, g_u, g_xx, g_uu, g_xu, g_ux):

        # Q Value
        Q_xx = g_xx + np.matmul(T_x.T, np.matmul(R_xx, T_x)) + np.tensordot(R_x, T_xx, 1)[0, :, :]
        Q_uu = g_uu + np.matmul(T_u.T, np.matmul(R_xx, T_u)) + np.tensordot(R_x, T_uu, 1)[0, :, :]
        Q_xu = g_xu + np.matmul(T_x.T, np.matmul(R_xx, T_u)) + np.tensordot(R_x, T_xu, 1)[0, :, :]
        Q_ux = g_ux + np.matmul(T_u.T, np.matmul(R_xx, T_x)) + np.tensordot(R_x, T_ux, 1)[0, :, :]
        Q_x = g_x + np.matmul(R_x, T_x)
        Q_u = np.matmul(R_x, T_u)

        return Q_xx, Q_uu, Q_xu, Q_ux, Q_x, Q_u

    @staticmethod
    @jax.jit
    def calculate_control_gains(Q_ux, Q_uu, Q_u):

        beta = np.linalg.solve(-Q_uu, Q_ux)
        alpha = np.linalg.solve(-Q_uu, Q_u)

        return beta, alpha

    def forward_pass(self, xi, xs, us, betas, alphas, line_alpha=0.01):

        x_ = xi
        data = (xs, us, betas, alphas, x_, line_alpha)
        data = jax.lax.fori_loop(0, xs.shape[0], self._forward_pass_inner, data)

        return data[0], data[1]

    def forward_pass_inner(self, i, data):

        xs, us, betas, alphas, x_, line_alpha = data
        dx = x_ - xs[i]
        xs = jax.ops.index_update(xs, jax.ops.index[i, :], x_)
        du = np.matmul(betas[i], dx) + line_alpha * alphas[i]
        u_ = us[i] + du
        us = jax.ops.index_update(us, jax.ops.index[i, :], u_)
        x_ = self.plant.step(x_, u_)

        return (xs, us, betas, alphas, x_, line_alpha)
