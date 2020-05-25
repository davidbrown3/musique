import numpy as np


def calculate_angular_acceleration(angular_position, angular_velocity, force, mass_pendulum, mass_cart, length, gravity, angular_friction):

    mass_total = mass_pendulum + mass_cart

    angular_position_sin = np.sin(angular_position)
    angular_position_cos = np.cos(angular_position)

    angular_acceleration = (
        gravity * angular_position_sin + angular_position_cos * (
            (-force - mass_pendulum * length * angular_velocity**2 * angular_position_sin) / mass_total
        )
    ) / (
        length * (
            4/3 - mass_pendulum * angular_position_cos**2 / mass_total
        )
    ) - angular_velocity * angular_friction

    return angular_acceleration

def calculate_acceleration(angular_position, angular_velocity, angular_acceleration, force, mass_pendulum, mass_cart, length):

    mass_total = mass_pendulum + mass_cart

    angular_position_sin = np.sin(angular_position)
    angular_position_cos = np.cos(angular_position)

    acceleration = (
        force + mass_pendulum * length * (
            angular_velocity**2 * angular_position_sin - angular_acceleration * angular_position_cos
        ) 
    ) / mass_total

    return acceleration

def derivatives(angular_position, angular_velocity, position, velocity, force, mass_pendulum, mass_cart, length, gravity, angular_friction):

    angular_acceleration = calculate_angular_acceleration(
        angular_position, angular_velocity, force, mass_pendulum, mass_cart, length, gravity, angular_friction
    )
    acceleration = calculate_acceleration(
        angular_position, angular_velocity, angular_acceleration, force, mass_pendulum, mass_cart, length
    )

    return angular_velocity, angular_acceleration, velocity, acceleration

def step(X, u, mass_pendulum, mass_cart, length, gravity, angular_friction, dt):

    angular_position = X[0] 
    angular_velocity = X[1] 
    position = X[2]
    velocity = X[3]
    force = u

    d_angular_position, d_angular_velocity, d_position, d_velocity = derivatives(
        angular_position, angular_velocity, position, velocity, force, mass_pendulum, mass_cart, length, gravity, angular_friction
    )

    dX = np.array([
        d_angular_position,
        *d_angular_velocity,
        d_position,
        *d_velocity
    ])

    return X + dX * dt

def simulate(mass_pendulum, mass_cart, length, gravity, angular_friction, dt):

    angular_position_0 = np.deg2rad(150)
    angular_velocity_0 = 0.0
    position_0 = 0.0
    velocity_0 = 0.0

    X = np.array([
        angular_position_0,
        angular_velocity_0,
        position_0,
        velocity_0
    ])

    dt = 0.01
    states= [X]
    ts = np.arange(0, 1e4) * dt
    u = np.array([0])
    for i, _ in enumerate(ts):
        if i%10==0:
            print(i)
        X = step(X, u, mass_pendulum, mass_cart, length, gravity, angular_friction, dt)
        states.append(X)
    
    return states
