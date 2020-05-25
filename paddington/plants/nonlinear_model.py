import numpy as np


class InvertedPendulum():
    '''
    Example documented at:
    https://coneural.org/florian/papers/05_cart_pole.pdf
    '''

    # u = force
    # States = angular_position, angular_velocity, position, velocity

    def __init__(self):
        self.mass_pendulum = 5
        self.mass_cart = 20
        self.length = 1
        self.gravity = 9.81
        self.angular_friction = 0.1
    
    @property
    def mass_total(self):
        return self.mass_pendulum + self.mass_cart

    @staticmethod
    def decode_states(x):
        x = np.squeeze(x)
        return x[0], x[1], x[2], x[3]

    @staticmethod
    def decode_control(u):
        u = np.squeeze(u)
        return u

    def derivatives(self, x, u):

        angular_position, angular_velocity, _, velocity = self.decode_states(x)
        force = self.decode_control(u)

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
        
        return np.array([
            [angular_velocity],
            [angular_acceleration],
            [velocity],
            [acceleration]
        ])    
        
    def step(self, x, u, dt):
        dx_dt = self.derivatives(x, u)
        return x + dx_dt * dt
