
import numpy as np
from dataclasses import dataclass
from dataclasses_json import dataclass_json

import json
import os
import control
import typing


@dataclass_json
@dataclass
class LinearModel:

    A: typing.List
    B: typing.List
    C: typing.List
    D: typing.List
    dt: float

    def __post_init__(self):
        self.A = np.array(self.A)
        self.B = np.array(self.B)
        self.C = np.array(self.C)
        self.D = np.array(self.D)
        self.sys = control.ss(self.A, self.B, self.C, self.D)
        self.discretize(dt=1)

    def discretize(self, dt):
        self.sys_discrete = control.sample_system(self.sys, dt)
        self.A_d = np.array(self.sys_discrete.A)
        self.B_d = np.array(self.sys_discrete.B)
        self.C_d = np.array(self.sys_discrete.C)
        self.D_d = np.array(self.sys_discrete.D)
