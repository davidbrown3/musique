import jax.numpy as np
import plotly.graph_objects as go
from jax import jit

from paddington.plants.nonlinear_model_static import simulate

states = simulate(mass_pendulum=5, mass_cart=20, length=1, gravity=9.81, angular_friction=0.1, dt=0.01)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        y=[np.rad2deg(x[0, 0]) for x in states]
    )
)

fig.show()
