from lqr import LQR
import numpy as np
import control

A = np.array([
    [-0.313, 56.7, 0],
    [-0.0139, -0.426, 0],
    [0, 56.7, 0],
])

B = np.array([
    [0.232],
    [0.0203],
    [0]
])

C = np.eye(3)

D = np.zeros([3, 1])

sys = control.ss(A, B, C, D)

sys_discrete = control.sample_system(sys, 0.1)

Cx = np.array([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 2],
])

Cu = np.array([
    [1]
])

cx = np.zeros([3, 1])
cu = np.zeros([1, 1])

solver = LQR(A=sys_discrete.A, B=sys_discrete.B, Cx=Cx, Cu=Cu, cx=cx, cu=cu)
V_Tx, v_Tx = solver.backward_pass(solver.V_Ty_i, solver.v_Ty_i)
print(V_Tx)
print(v_Tx)
V_Tx, v_Tx = solver.backward_pass(V_Tx, v_Tx)
print(V_Tx)
print(v_Tx)
