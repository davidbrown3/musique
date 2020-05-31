import torch

from paddington.solvers.lqr import LQR
from paddington.tools.controls_tools import convert_syntax_transition


class iLQR:

    def __init__(self, plant, cost_function, dt, states_initial):

        self.plant = plant
        self.cost_function = cost_function
        self.dt = dt

    def initial_guess_lqr(self, states_initial, time_total):

        controls_initial = torch.zeros([self.plant.N_u])
        A_d, B_d = self.plant.derivatives(x=states_initial, u=controls_initial)
        F, f, N_x, N_u = convert_syntax_transition(A_d, B_d)
        C = self.self.cost_function.calculate_cost_hessian(x=states_initial, u=controls_initial)
        c = self.cost_function.calculate_cost_jacobian(x=states_initial, u=controls_initial)

        lqr = LQR(F=F, f=f, C=C, c=c, dt=self.dt, N_x=self.plant.N_x, N_u=self.plant.N_u)

        return lqr.solve(states_initial, time_total)

    def solve(self, states_initial, K_guess, k_guess, time_total):

        ts, xs, us = self.initial_guess_lqr(states_initial=states_initial, K_guess=K_guess, k_guess=k_guess, time_total=time_total)

    def backward_pass(self, xs, us):

        # Initialisation
        A_d, B_d = self.plant.calculate_statespace_discrete(x=xs[-1], u=us[-1], dt=self.dt)
        F, f = convert_syntax_transition(A_d, B_d)
        C = self.self.cost_function.calculate_cost_hessian(x=xs[-1], u=us[-1])
        c = self.cost_function.calculate_cost_jacobian(x=xs[-1], u=us[-1])
        V_Ty, v_Ty, _, _ = LQR._backward_pass(V_Ty=torch.zeros([self.plant.N_x, self.plant.N_x]), v_Ty=torch.zeros([self.plant.N_x, 1]),
                                              F_Tx=F, f_Tx=f, C_Tx=C, c_Tx=c, N_x=self.plant.N_x, N_u=self.plant.N_u)

        Ks = []
        ks = []
        for x, u in zip(xs[::-1], us[::-1]):

            A_d, B_d = self.plant.calculate_statespace_discrete(x=x, u=u, dt=self.dt)
            F_Tx, f_Tx = convert_syntax_transition(A_d, B_d)
            C_Tx = self.self.cost_function.calculate_cost_hessian(x=x, u=u)
            c_Tx = self.cost_function.calculate_cost_jacobian(x=x, u=u)

            V_Ty, v_Ty, K_Tx, k_Tx = LQR.backward_pass(V_Ty=V_Ty, v_Ty=v_Ty, F_Tx=F_Tx, f_Tx=f_Tx, C_Tx=C_Tx, c_Tx=c_Tx,
                                                       N_x=self.plant.N_x, N_u=self.plant.N_u)
            Ks.append(K_Tx)
            ks.append(k_Tx)

    def integrate(self, N_steps):
        pass
