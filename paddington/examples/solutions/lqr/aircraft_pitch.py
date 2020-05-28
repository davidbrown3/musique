
import json
from importlib.resources import open_text

import plotly.graph_objects as go
import torch

from paddington.plants.linear_model import LinearModel
from paddington.solvers.lqr import LQR
from paddington.tools.controls_tools import convert_syntax

with open_text("paddington.examples.models.linear", "aircraft_pitch.json") as f:
    data = json.load(f)
    plant = LinearModel.from_dict(data)

Cx_diag = torch.tensor([0.0, 0.0, 2.0])
Cu_diag = torch.tensor([1.0])
cx = torch.zeros([3, 1])
cu = torch.zeros([1, 1])

F, f, C, c, N_x, N_u = convert_syntax(plant.A_d, plant.B_d, Cx_diag, Cu_diag, cx, cu)

# Discrete horizon
solver = LQR(F=F, f=f, C=C, c=c, N_x=N_x, N_u=N_u)

Ks = []
ks = []

V_Ty, v_Ty = solver.V_Ty_i, solver.v_Ty_i

for t in torch.arange(40, 0, -plant.dt):
    V_Ty, v_Ty, K_Ty, k_Ty = solver.backward_pass(V_Ty, v_Ty)
    Ks.append(K_Ty)
    ks.append(k_Ty)

xs = []
ts = torch.arange(0, 40, plant.dt)

x = torch.tensor([
    [0.0],
    [0.0],
    [-1.0]
])

xs.append(x)
for t, K, k in zip(ts, Ks[::-1], ks[::-1]):
    x = solver.step(x, K, k)
    xs.append(x)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=ts,
        y=[x[2, 0] for x in xs]
    )
)

fig.show()
