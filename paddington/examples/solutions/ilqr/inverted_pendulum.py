import torch

from paddington.examples.models.nonlinear.inverted_pendulum import \
    InvertedPendulum
from paddington.solvers.ilqr import iLQR
from paddington.tools.controls_tools import quadratic_cost_function

plant = InvertedPendulum()

# TODO: Validate floats
Cx_diag = torch.tensor([0.1, 0.0, 2.0, 0.0])
Cu_diag = torch.tensor([1.0])
cx = torch.tensor([0.0])
cu = torch.tensor([0.0])

cost_function = quadratic_cost_function(Cx_diag=Cx_diag, Cu_diag=Cu_diag, cx=cx, cu=cu)

states = torch.tensor([-5.0, 0.0, 0.0, 0.0], requires_grad=True)
control = torch.tensor([0.0], requires_grad=True)

# Discrete horizon
solver = iLQR(plant=plant, cost_function=cost_function)



dt = 0.01

states_log = [states]
ts = torch.arange(0, 1e4) * dt

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
