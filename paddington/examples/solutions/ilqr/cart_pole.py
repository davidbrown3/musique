import jax.numpy as np
import plotly.graph_objects as go
from jax.config import config

from paddington.examples.models.nonlinear.cart_pole import CartPole
from paddington.solvers.ilqr import iLQR
from paddington.tools.controls_tools import (diagonalize,
                                             quadratic_cost_function)

dt = 1e-2
plant = CartPole(dt)

g_xx = diagonalize(np.array([0.0, 0.0, 0.0, 0.0]))
g_uu = np.array([[1e-7]])
g_xu = np.zeros([4, 1])
g_ux = np.zeros([1, 4])
g_x = np.array([[0.0, 0.0, 0.0, 0.0]])
g_u = np.array([[0.0]])
running_cost_function = quadratic_cost_function(g_xx=g_xx, g_xu=g_xu, g_ux=g_ux, g_uu=g_uu, g_x=g_x, g_u=g_u)

g_xx = diagonalize(np.array([1.0, 0.0, 1.0, 0.0]))
g_uu = np.array([[0.0]])
g_xu = np.zeros([4, 1])
g_ux = np.zeros([1, 4])
g_x = np.array([[0.0, 0.0, 0.0, 0.0]])
g_u = np.array([[0.0]])
terminal_cost_function = quadratic_cost_function(g_xx=g_xx, g_xu=g_xu, g_ux=g_ux, g_uu=g_uu, g_x=g_x, g_u=g_u)

'''
Likely one of the causes for failure is the quality of the initial guess
Could do multiple seeds of random inputs
'''

solver = iLQR(plant=plant, running_cost_function=running_cost_function, terminal_cost_function=terminal_cost_function)

states_initial = np.array([[5.0], [0.0], [np.deg2rad(180)], [0.0]])
time_total = 10

xs, us, costs = solver.solve(states_initial, time_total)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        y=costs,
        name='cost'
    )
)

fig.show()

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        y=xs[:, 2],
        name='angular position'
    )
)

fig.add_trace(
    go.Scatter(
        y=xs[:, 0],
        name='linear position'
    )
)

fig.add_trace(
    go.Scatter(
        y=us[:, 0],
        name='control'
    )
)

fig.show()
