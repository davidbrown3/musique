
import importlib
import json
import os
import unittest
from importlib.resources import open_text

import control
import numpy as np
import plotly.graph_objects as go

from paddington.plants.linear_model import LinearModel
from paddington.solvers.lqr import LQR

with open_text("paddington.examples.models.linear", "aircraft_pitch.json") as f:
    data = json.load(f)
    plant = LinearModel.from_dict(data)

Cx = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 2], ])
Cu = np.array([[1]])
cx = np.zeros([3, 1])
cu = np.zeros([1, 1])

# Discrete horizon
solver = LQR(A=plant.A_d, B=plant.B_d, Cx=Cx, Cu=Cu, cx=cx, cu=cu)

Ks = []
ks = []

V_Ty, v_Ty = solver.V_Ty_i, solver.v_Ty_i

for t in np.arange(40, 0, -plant.dt):
    V_Ty, v_Ty, K_Ty, k_Ty = solver.backward_pass(V_Ty, v_Ty)
    Ks.append(K_Ty)
    ks.append(k_Ty)

xs = []
ts = np.arange(0, 40, plant.dt)

x = np.array([
    [0],
    [0],
    [-1]
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
