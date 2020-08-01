import jax.numpy as np

from paddington.plants.nonlinear_model import NonLinearModel


class CartPole(NonLinearModel):
    '''
    Example documented at:
    https://coneural.org/florian/papers/05_cart_pole.pdf
    '''

    # u = force
    # States = position, velocity, angular_position, angular_velocity

    def __init__(self, dt):

        self.mass_pendulum = 5
        self.mass_cart = 20
        self.length = 1
        self.gravity = 9.81
        self.angular_friction = 0.1

        super().__init__(dt)

    @property
    def N_x(self):
        return 4

    @property
    def N_u(self):
        return 1

    @property
    def mass_total(self):
        return self.mass_pendulum + self.mass_cart

    def derivatives(self, x, u):

        angular_position = x[2]
        angular_velocity = x[3]
        velocity = x[1]
        force = u[0]

        angular_position_sin = np.sin(angular_position)
        angular_position_cos = np.cos(angular_position)

        angular_acceleration = (
            self.gravity * angular_position_sin + angular_position_cos * (
                (-force - self.mass_pendulum * self.length * angular_velocity**2 * angular_position_sin) / self.mass_total
            )
        ) / (
            self.length * (
                4/3 - self.mass_pendulum * angular_position_cos**2 / self.mass_total
            )
        ) - angular_velocity * self.angular_friction

        acceleration = (
            force + self.mass_pendulum * self.length * (
                angular_velocity**2 * angular_position_sin - angular_acceleration * angular_position_cos
            )
        ) / self.mass_total

        derivatives = (
            velocity,
            acceleration,
            angular_velocity,
            angular_acceleration
        )

        return np.stack(derivatives)
