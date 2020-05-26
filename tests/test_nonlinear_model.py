import numpy as np
import plotly.graph_objects as go
import torch

from paddington.plants.nonlinear_model import InvertedPendulum

problem = InvertedPendulum()

angular_position_0 = np.deg2rad(0)
angular_velocity_0 = 0.0
position_0 = 0.0
velocity_0 = 0.0

states = torch.tensor([
    position_0,
    velocity_0,
    angular_position_0,
    angular_velocity_0
], requires_grad=True)

control = torch.tensor([
    0.0
], requires_grad=True)

dt = 0.01
d = problem.derivatives(states, control)

# Linearisation
jac = problem.jacobian(states, control, dt)

hess = problem.hessian(states, control)



states_log = [states]
ts = np.arange(0, 1e4) * dt

for i, t in enumerate(ts):
    if i%10==0:
        print(i)
    states = problem.step(x=states, u=control, dt=dt)
    states_log.append(states)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=ts,
        y=[np.rad2deg(x[0, 0]) for x in states_log]
    )
)

fig.show()
