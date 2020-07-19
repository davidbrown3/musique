
import json
from importlib.resources import open_text

import plotly.graph_objects as go
import torch

from paddington.plants.linear_model import LinearModel
from paddington.solvers.lqr import LQR
from paddington.tools.controls_tools import diagonalize

with open_text("paddington.examples.models.linear", "aircraft_pitch.json") as f:
    data = json.load(f)
    plant = LinearModel.from_dict(data)

g_xx = diagonalize(torch.tensor([0.0, 0.0, 2.0], dtype=torch.float64))
g_uu = diagonalize(torch.tensor([1.0], dtype=torch.float64))
g_xu = torch.zeros([3, 1], dtype=torch.float64)
g_x = torch.zeros([1, 3], dtype=torch.float64)
g_u = torch.zeros([1, 1], dtype=torch.float64)

# Discrete horizon
solver = LQR(T_x=plant.A_d, T_u=plant.B_d, g_x=g_x, g_u=g_u, g_xx=g_xx, g_uu=g_uu, g_xu=g_xu, dt=plant.dt)

states_initial = torch.tensor([
    [0.0],
    [0.0],
    [-1.0]
], dtype=torch.float64)

ts, xs, us = solver.solve(states_initial=states_initial, time_total=40)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=ts,
        y=[x[2, 0] for x in xs]
    )
)

fig.show()
