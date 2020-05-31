
import json
from importlib.resources import open_text

import plotly.graph_objects as go
import torch

from paddington.plants.linear_model import LinearModel
from paddington.solvers.lqr import LQR
from paddington.tools.controls_tools import (convert_syntax_cost_diagonals,
                                             convert_syntax_transition)

with open_text("paddington.examples.models.linear", "aircraft_pitch.json") as f:
    data = json.load(f)
    plant = LinearModel.from_dict(data)

Cx_diag = torch.tensor([0.0, 0.0, 2.0])
Cu_diag = torch.tensor([1.0])
cx = torch.zeros([3, 1])
cu = torch.zeros([1, 1])

C, c = convert_syntax_cost_diagonals(Cx_diag, Cu_diag, cx, cu)
F, f = convert_syntax_transition(plant.A_d, plant.B_d)

# Discrete horizon
solver = LQR(F=F, f=f, C=C, c=c, dt=plant.dt, N_x=plant.N_x, N_u=plant.N_u)

states_initial = torch.tensor([
    [0.0],
    [0.0],
    [-1.0]
])

ts, xs, us = solver.solve(states_initial=states_initial, time_total=40)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=ts,
        y=[x[2, 0] for x in xs]
    )
)

fig.show()
