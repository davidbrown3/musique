import jax
import jax.numpy as np
import scipy
import tqdm


class DifferentialDynamicProgramming:

    def __init__(self, plant, running_cost_function, terminal_cost_function, order=1, debug=False):

        self.debug = debug
        self.plant = plant
        self.order = order
        self.running_cost_function = running_cost_function
        self.terminal_cost_function = terminal_cost_function

        # Dispatching
        if self.debug:
            self._forward_pass_inner = self.forward_pass_inner
            self._forward_pass = self.forward_pass
            self._backward_pass = self.backward_pass
            self._calculate_R_partials = self.calculate_R_partials
            self._calculate_Q_partials_1st = self.calculate_Q_partials_1st
            self._calculate_Q_partials_2nd = self.calculate_Q_partials_2nd
            self._calculate_control_gains = self.calculate_control_gains
            self._calculate_cost = self.calculate_cost
            if self.order == 1:
                self._backward_pass_loop = self.backward_pass_loop
                self._backward_pass_loop_jit_inner = None
            else:
                self._backward_pass_loop = self.backward_pass_loop
                self._backward_pass_loop_jit_inner = None
        else:
            self._forward_pass_inner = jax.jit(self.forward_pass_inner)
            self._forward_pass = jax.jit(self.forward_pass)
            self._backward_pass = jax.jit(self.backward_pass)
            self._calculate_R_partials = jax.jit(self.calculate_R_partials)
            self._calculate_Q_partials_1st = jax.jit(self.calculate_Q_partials_1st)
            self._calculate_Q_partials_2nd = jax.jit(self.calculate_Q_partials_2nd)
            self._calculate_control_gains = jax.jit(self.calculate_control_gains)
            self._calculate_cost = jax.jit(self.calculate_cost)
            if self.order == 1:
                self._backward_pass_loop = jax.jit(self.backward_pass_loop_1st_jit)
                self._backward_pass_loop_jit_inner = jax.jit(self.backward_pass_loop_1st_jit_inner)
            else:
                self._backward_pass_loop = jax.jit(self.backward_pass_loop_2nd_jit)
                self._backward_pass_loop_jit_inner = jax.jit(self.backward_pass_loop_2nd_jit_inner)

    def infinite_horizon_lqr(self, x, u):

        A_d, B_d = self.plant.calculate_statespace_discrete(x[:, 0], u[:, 0])

        P = scipy.linalg.solve_discrete_are(A_d,
                                            B_d,
                                            self.terminal_cost_function.g_xx*1e-5,
                                            self.running_cost_function.g_uu).squeeze()

        beta = -np.linalg.solve(
            a=self.running_cost_function.g_uu + np.matmul(B_d.T, np.matmul(P, B_d)),
            b=np.matmul(B_d.T, np.matmul(P, A_d)) + 2 * self.running_cost_function.g_ux
        )

        return beta

    def solve(self, states_initial, time_total, convergence=1e-4, iterations=300):

        beta = self.infinite_horizon_lqr(x=states_initial, u=np.zeros([self.plant.N_u, 1]))

        N_steps = int(time_total / self.plant.dt)
        xs = np.zeros([N_steps, self.plant.N_x])
        us = np.zeros([N_steps, self.plant.N_u])
        betas = np.tile(beta, (N_steps, 1, 1))
        alphas = np.zeros([N_steps, self.plant.N_u])

        xs, us = self.forward_pass(np.squeeze(states_initial), xs, us, betas, alphas)

        costs = []
        for _ in tqdm.trange(iterations):
            betas, alphas = self._backward_pass(xs=xs, us=us)
            xs, us = self._forward_pass(xi=np.squeeze(states_initial), xs=xs, us=us, betas=betas, alphas=alphas)
            cost = self._calculate_cost(xs, us)
            costs.append(cost)

        return xs, us, costs

    def calculate_cost(self, xs, us):
        return np.sum(self.running_cost_function.calculate_cost_batch(xs, us)) + \
            np.sum(self.terminal_cost_function.calculate_cost(xs[-1], us[-1]))

    def backward_pass(self, xs, us):

        xs = np.flip(xs, axis=0)
        us = np.flip(us, axis=0)

        T_xs, T_us = self.plant.calculate_statespace_discrete_batch(xs, us)

        g_xs = self.running_cost_function.calculate_g_x_batch(xs, us)
        g_us = self.running_cost_function.calculate_g_u_batch(xs, us)
        R_xx = self.terminal_cost_function.g_xx
        R_x = self.terminal_cost_function.calculate_g_x(xs[0], us[0])

        betas, alphas = self._backward_pass_loop(xs, us, R_x, R_xx, T_xs, T_us, g_xs, g_us)

        return betas, alphas

    def backward_pass_loop(self, xs, us, R_x, R_xx, T_xs, T_us, g_xs, g_us):

        if self.order == 2:
            T_hessians = self.plant.calculate_hessian_discrete_batch(xs, us)
            T_xxs = T_hessians[0][0]
            T_xus = T_hessians[0][1]
            T_uxs = T_hessians[1][0]
            T_uus = T_hessians[1][1]

        betas = np.empty([xs.shape[0], us.shape[1], xs.shape[1]])
        alphas = np.empty([us.shape[0], us.shape[1]])

        for i in range(xs.shape[0]):

            Q_xx, Q_uu, Q_xu, Q_ux, Q_x, Q_u = self._calculate_Q_partials_1st(
                R_x=R_x,
                R_xx=R_xx,
                T_x=T_xs[jax.ops.index[i, :, :]],
                T_u=T_us[jax.ops.index[i, :, :]],
                g_x=g_xs[jax.ops.index[i, :]],
                g_u=g_us[jax.ops.index[i, :]],
                g_xx=self.running_cost_function.g_xx,
                g_uu=self.running_cost_function.g_uu,
                g_xu=self.running_cost_function.g_xu,
                g_ux=self.running_cost_function.g_ux
            )

            if self.order == 2:

                dQ_xx, dQ_uu, dQ_xu, dQ_ux = self._calculate_Q_partials_2nd(
                    R_x=R_x,
                    T_xx=T_xxs[jax.ops.index[i, :, :, :]],
                    T_xu=T_xus[jax.ops.index[i, :, :, :]],
                    T_ux=T_uxs[jax.ops.index[i, :, :, :]],
                    T_uu=T_uus[jax.ops.index[i, :, :, :]]
                )

                Q_xx += dQ_xx
                Q_uu += dQ_uu
                Q_xu += dQ_xu
                Q_ux += dQ_ux

            beta, alpha = self._calculate_control_gains(Q_ux, Q_uu, Q_u)
            R_x, R_xx = self._calculate_R_partials(beta, alpha, Q_xx, Q_uu, Q_xu, Q_ux, Q_x, Q_u)

            betas = jax.ops.index_update(betas, jax.ops.index[i, :], beta)
            alphas = jax.ops.index_update(alphas, jax.ops.index[i, :], alpha.squeeze())

        return np.flip(betas, axis=0), np.flip(alphas, axis=0)

    def backward_pass_loop_1st_jit(self, xs, us, R_x, R_xx, T_xs, T_us, g_xs, g_us):

        betas = np.empty([xs.shape[0], us.shape[1], xs.shape[1]])
        alphas = np.empty([us.shape[0], us.shape[1]])

        data = (R_x, R_xx, T_xs, T_us, g_xs, g_us, betas, alphas)
        data = jax.lax.fori_loop(0, xs.shape[0], self._backward_pass_loop_jit_inner, data)

        return np.flip(data[-2], axis=0), np.flip(data[-1], axis=0)

    def backward_pass_loop_2nd_jit(self, xs, us, R_x, R_xx, T_xs, T_us, g_xs, g_us):

        T_hessians = self.plant.calculate_hessian_discrete_batch(xs, us)
        T_xxs = T_hessians[0][0]
        T_xus = T_hessians[0][1]
        T_uxs = T_hessians[1][0]
        T_uus = T_hessians[1][1]

        betas = np.empty([xs.shape[0], us.shape[1], xs.shape[1]])
        alphas = np.empty([us.shape[0], us.shape[1]])

        data = (R_x, R_xx, T_xs, T_us, T_xxs, T_xus, T_uxs, T_uus, g_xs, g_us, betas, alphas)
        data = jax.lax.fori_loop(0, xs.shape[0], self._backward_pass_loop_jit_inner, data)

        return np.flip(data[-2], axis=0), np.flip(data[-1], axis=0)

    def backward_pass_loop_1st_jit_inner(self, i, data):

        R_x, R_xx, T_xs, T_us, g_xs, g_us, betas, alphas = data

        Q_xx, Q_uu, Q_xu, Q_ux, Q_x, Q_u = self._calculate_Q_partials_1st(
            R_x=R_x,
            R_xx=R_xx,
            T_x=T_xs[jax.ops.index[i, :, :]],
            T_u=T_us[jax.ops.index[i, :, :]],
            g_x=g_xs[jax.ops.index[i, :]],
            g_u=g_us[jax.ops.index[i, :]],
            g_xx=self.running_cost_function.g_xx,
            g_uu=self.running_cost_function.g_uu,
            g_xu=self.running_cost_function.g_xu,
            g_ux=self.running_cost_function.g_ux
        )

        beta, alpha = self._calculate_control_gains(Q_ux, Q_uu, Q_u)
        R_x, R_xx = self._calculate_R_partials(beta, alpha, Q_xx, Q_uu, Q_xu, Q_ux, Q_x, Q_u)

        betas = jax.ops.index_update(betas, jax.ops.index[i, :], beta)
        alphas = jax.ops.index_update(alphas, jax.ops.index[i, :], alpha.squeeze())

        return (R_x, R_xx, T_xs, T_us, g_xs, g_us, betas, alphas)

    def backward_pass_loop_2nd_jit_inner(self, i, data):

        R_x, R_xx, T_xs, T_us, T_xxs, T_xus, T_uxs, T_uus, g_xs, g_us, betas, alphas = data

        Q_xx_1st, Q_uu_1st, Q_xu_1st, Q_ux_1st, Q_x, Q_u = self._calculate_Q_partials_1st(
            R_x=R_x,
            R_xx=R_xx,
            T_x=T_xs[jax.ops.index[i, :, :]],
            T_u=T_us[jax.ops.index[i, :, :]],
            g_x=g_xs[jax.ops.index[i, :]],
            g_u=g_us[jax.ops.index[i, :]],
            g_xx=self.running_cost_function.g_xx,
            g_uu=self.running_cost_function.g_uu,
            g_xu=self.running_cost_function.g_xu,
            g_ux=self.running_cost_function.g_ux
        )

        Q_xx_2nd, Q_uu_2nd, Q_xu_2nd, Q_ux_2nd = self._calculate_Q_partials_2nd(
            R_x=R_x,
            T_xx=T_xxs[jax.ops.index[i, :, :, :]],
            T_xu=T_xus[jax.ops.index[i, :, :, :]],
            T_ux=T_uxs[jax.ops.index[i, :, :, :]],
            T_uu=T_uus[jax.ops.index[i, :, :, :]]
        )

        Q_xx = Q_xx_1st + Q_xx_2nd
        Q_xu = Q_xu_1st + Q_xu_2nd
        Q_ux = Q_ux_1st + Q_ux_2nd
        Q_uu = Q_uu_1st + Q_uu_2nd

        beta, alpha = self._calculate_control_gains(Q_ux, Q_uu, Q_u)
        R_x, R_xx = self._calculate_R_partials(beta, alpha, Q_xx, Q_uu, Q_xu, Q_ux, Q_x, Q_u)

        betas = jax.ops.index_update(betas, jax.ops.index[i, :], beta)
        alphas = jax.ops.index_update(alphas, jax.ops.index[i, :], alpha.squeeze())

        return (R_x, R_xx, T_xs, T_us, T_xxs, T_xus, T_uxs, T_uus, g_xs, g_us, betas, alphas)

    @ staticmethod
    def calculate_R_partials(beta, alpha, Q_xx, Q_uu, Q_xu, Q_ux, Q_x, Q_u):

        R_xx = Q_xx + np.matmul(Q_xu, beta) + np.matmul(beta.T, Q_ux) + np.matmul(beta.T, np.matmul(Q_uu, beta))
        R_x = Q_x + np.matmul(Q_u, beta) + np.matmul(Q_xu, alpha).T + np.matmul(alpha.T, np.matmul(Q_uu, beta))

        return R_x, R_xx

    @ staticmethod
    def calculate_Q_partials_1st(R_x, R_xx, T_x, T_u, g_x, g_u, g_xx, g_uu, g_xu, g_ux):

        Q_xx = g_xx + np.matmul(T_x.T, np.matmul(R_xx, T_x))
        Q_uu = g_uu + np.matmul(T_u.T, np.matmul(R_xx, T_u))
        Q_xu = g_xu + np.matmul(T_x.T, np.matmul(R_xx, T_u))
        Q_ux = g_ux + np.matmul(T_u.T, np.matmul(R_xx, T_x))
        Q_x = g_x + np.matmul(R_x, T_x)
        Q_u = np.matmul(R_x, T_u)

        return Q_xx, Q_uu, Q_xu, Q_ux, Q_x, Q_u

    @ staticmethod
    def calculate_Q_partials_2nd(R_x, T_xx, T_xu, T_ux, T_uu):

        dQ_xx = np.tensordot(R_x, T_xx, 1)[0, :, :]
        dQ_uu = np.tensordot(R_x, T_uu, 1)[0, :, :]
        dQ_xu = np.tensordot(R_x, T_xu, 1)[0, :, :]
        dQ_ux = np.tensordot(R_x, T_ux, 1)[0, :, :]

        return dQ_xx, dQ_uu, dQ_xu, dQ_ux

    @ staticmethod
    def calculate_control_gains(Q_ux, Q_uu, Q_u):

        beta = np.linalg.solve(-Q_uu, Q_ux)
        alpha = np.linalg.solve(-Q_uu, Q_u)

        return beta, alpha

    def forward_pass(self, xi, xs, us, betas, alphas, line_alpha=1e-2):
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
