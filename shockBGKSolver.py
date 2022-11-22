"""
Solver for BGK equation
@KaanKirmanoglu (github)
(2022) University of Illinois Urbana-Champaign
"""

import numpy as np
from physicalFuncs import *


class solverBGK:

    def __init__(self):
        print("Initializing Parameters")

        self.L_c = 2
        # Upstream Properties
        self.P_u = 6.6667  # [Pa]
        self.T_u = 300  # [K]
        self.M_u = 9
        # Argon collision diameter and gamma
        self.d_ar = 4.17e-10  # [m]
        self.gamma = 1.66667

        # Parameters
        self.nx = 40
        self.nv = 40
        self.t_tot = 100

        print("Initialized Parameters: ")
        print("M_u: ", self.M_u)
        print("T_u: ", self.T_u)

        # initialize distribution function to solve
        self.F = np.ones([self.nx, self.nv, 3])
        # calculate downstream quantities by applying Rankine-Hugoniot Conditions

        self.M_d, self.T_d, self.p_d = rankine_hugoniot(self.M_u, self.T_u, self.P_u, self.gamma)

    def initial_conditions(self):
        pass

    def drift(self):
        pass
        #   Apply drift function

    def collide(self):
        pass

    def bgk_ode(self):
        pass

    def execute_solver(self):
        self.initial_conditions()
        pass
