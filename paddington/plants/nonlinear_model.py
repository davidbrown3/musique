import torch


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

    def derivatives(self, x, u):

        angular_position = x[0]
        angular_velocity = x[1]
        velocity = x[3] 
        force = u[0]

        angular_position_sin = torch.sin(angular_position)
        angular_position_cos = torch.cos(angular_position)

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
            angular_velocity,
            angular_acceleration,
            velocity,
            acceleration
        )

        return derivatives  
        
    def step(self, x, u, dt):
        dx_dt = self.derivatives(x, u)
        return x + dx_dt * dt

    def jacobian(self, x, u):
        return torch.autograd.functional.jacobian(self.derivatives, (x, u))
