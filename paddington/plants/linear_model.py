import typing
from dataclasses import dataclass

import control
import torch
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class LinearModel:

    A: typing.List
    B: typing.List
    C: typing.List
    D: typing.List
    dt: float

    def __post_init__(self):
        self.A = torch.tensor(self.A)
        self.B = torch.tensor(self.B)
        self.C = torch.tensor(self.C)
        self.D = torch.tensor(self.D)
        self.sys = control.ss(self.A, self.B, self.C, self.D)
        self.discretize(dt=self.dt)

    def discretize(self, dt):
        self.sys_discrete = control.sample_system(self.sys, dt)
        self.A_d = torch.tensor(self.sys_discrete.A)
        self.B_d = torch.tensor(self.sys_discrete.B)
        self.C_d = torch.tensor(self.sys_discrete.C)
        self.D_d = torch.tensor(self.sys_discrete.D)
