import json
import unittest
from importlib.resources import open_text

from paddington.plants.linear_model import LinearModel


class TestModelLoad(unittest.TestCase):
    def setUp(self):
        with open_text("paddington.example_models.linear", "aircraft_pitch.json") as f:
            self.data = json.load(f)

    def test_map_object(self):
        model = LinearModel.from_dict(self.data)
        self.assertEqual(model.dt, self.data["dt"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
