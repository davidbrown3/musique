import unittest
from collections import namedtuple

import jax.numpy as np
from genty import genty, genty_dataset

from simdynamics.examples.models import CartPole


@genty
class TestNaNs(unittest.TestCase):

    angular_positions = np.logspace(0, 7, num=5)
    angular_velocitys = np.logspace(0, 7, num=5)
    velocitys = np.logspace(0, 7, num=5)
    forces = np.logspace(0, 7, num=5)

    Case = namedtuple('Case', 'angular_position angular_velocity velocity force')

    cases = []
    for angular_position in angular_positions:
        for angular_velocity in angular_velocitys:
            for velocity in velocitys:
                for force in forces:
                    cases.append(Case(angular_position, angular_velocity, velocity, force))

    def setUp(self):
        self.plant = CartPole(dt=0.1)

    @genty_dataset(
        *cases
    )
    def test_plant(self, angular_position, angular_velocity, velocity, force):

        position = 0.0
        x = np.array([[position], [velocity], [angular_position], [angular_velocity]])
        u = np.array([[force]])

        xd = self.plant._derivatives(x, u)
        self.assertFalse(np.any(np.isnan(xd)))

        xnew = self.plant.step(x, u)
        self.assertFalse(np.any(np.isnan(xnew)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
