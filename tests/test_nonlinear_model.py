import numpy as np
import plotly.graph_objects as go

from paddington.plants.nonlinear_model import InvertedPendulum

problem = InvertedPendulum()

angular_position_0 = np.deg2rad(150)
angular_velocity_0 = 0.0
position_0 = 0.0
velocity_0 = 0.0

states = np.array([
    [angular_position_0],
    [angular_velocity_0],
    [position_0],
    [velocity_0]
])

dt = 0.01
states_log = [states]
ts = np.arange(0, 1e4) * dt

for i, t in enumerate(ts):
    states = problem.step(x=states, u=np.array([[0]]), dt=dt)
    states_log.append(states)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=ts,
        y=[np.rad2deg(x[0, 0]) for x in states_log]
    )
)

fig.show()
