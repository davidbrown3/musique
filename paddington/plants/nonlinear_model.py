import jax
import jax.numpy as np

from paddington.tools.controls_tools import continuous_to_discrete


class NonLinearModel():

    def __init__(self, dt):
        self.dt = dt
        self._derivatives = jax.jit(self.derivatives)
        self.step = jax.jit(self._step)
        self.calculate_statespace_discrete = jax.jacfwd(self.step, [0, 1])
        self.calculate_statespace_discrete_batch = jax.vmap(self.calculate_statespace_discrete, in_axes=(0, 0))

    # @ property
    # def N_x(self):
    #     return 0

    # @ property
    # def N_u(self):
    #     return 0

    # def derivatives(self, x, u):
    #     return np.array([])

    def _step(self, x, u):
        dx_dt = self._derivatives(x, u)
        return x + dx_dt * self.dt

    def calculate_statespace_discrete(self, x, u):
        A, B = self.calculate_statespace(x, u)
        return continuous_to_discrete(A, B)

    def calculate_hessian(self, x, u):
        return [jax.hessian(lambda x, u: self.derivatives(x, u)[i], (x, u)) for i in range(len(x))]

    def calculate_hessian_discrete(self, x, u):
        return [jax.hessian(lambda x, u: self.step(x, u)[i], (x, u)) for i in range(len(x))]
