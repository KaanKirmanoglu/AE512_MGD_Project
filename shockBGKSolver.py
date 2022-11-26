"""
Solver for BGK equation
@KaanKirmanoglu (github)
(2022) University of Illinois Urbana-Champaign
"""
# TODO: BCs, biased maxwell on left? no BC on right?
# Collision frequency: v= v(x)? Equilibrium MB formula?

import numpy as np
import scipy
from physicalFuncs import *


class solverBGK:

    def __init__(self):
        print("Initializing Parameters")

        # Constants
        self.d_ar = 4.17e-10  # [m] collision diameter
        self.gamma = 1.66667  # heat capacity ratio
        self.sigma_col = np.pi * self.d_ar * self.d_ar  # collision cross section
        self.kb = 1.38064852e-23  # boltzmann constant
        self.n_avo = 6.022e23  # avogadro number
        self.m_ar = 39.948e-3 / self.n_avo  # kg, mass of argon particle

        self.length = 2
        # Upstream Properties
        self.p_u = 6.6667  # [Pa]
        self.T_u = 300  # [K]
        self.M_u = 9
        self.n_u = self.p_u / (self.kb * self.T_u)
        self.cs_u = np.sqrt(self.gamma * self.kb * self.T_u / self.m_ar)
        self.v_bar_u = self.M_u * self.cs_u

        # calculate downstream quantities by applying Rankine-Hugoniot Conditions

        self.M_d, self.T_d, self.p_d = rankine_hugoniot(self.M_u, self.T_u, self.p_u, self.gamma)
        self.n_d = self.p_d / (self.kb * self.T_d)
        self.cs_d = np.sqrt(self.gamma * self.kb * self.T_d / self.m_ar)
        self.v_bar_d = self.M_d * self.cs_d

        # Parameters
        self.mx = 40
        self.nv = 40
        self.t_tot = 100
        self.v_arr = np.linspace(-5500, 5500, self.nv)
        # initialize distribution function to solve
        self.F_0 = np.ones([self.mx, self.nv, 3])
        self.F = np.ones([self.mx, self.nv, 3])
        self.F_eq = np.ones([self.mx, self.nv, 3])
        self.temps = np.zeros([self.mx, 3])
        self.temps_hat = np.zeros([self.mx, 3])
        self.n_xi = np.zeros(self.mx)
        self.v_bar = np.zeros([self.mx, 3])
        self.coll_fq = np.zeros(self.mx)

        self.x_arr = np.linspace(-1, 1, self.mx)
        self.dx = self.x_arr[1] - self.x_arr[0]

        # Reference Quantities

        self.L_r = self.dx
        self.x_hat = self.x_arr/self.L_r
        self.C_r = self.v_bar_u
        self.c_hat = self.v_arr/self.C_r
        self.nu_r = np.sqrt(16 * self.kb * self.T_u / (np.pi * self.m_ar)) * self.n_u * self.sigma_col
        self.n_r = self.n_u
        self.Kn = self.C_r / (self.L_r * self.nu_r)
        self.t_span = [0, 5]

    def initial_conditions(self):
        # set initial conditions as non-dimensionalized form
        for i in range(self.mx):
            if self.x_arr[i] < 0:
                self.temps[i] = np.array([self.T_u, self.T_u, self.T_u])
                self.n_xi[i] = self.n_u / self.n_r
                self.v_bar[i] = [self.v_bar_u / self.C_r, 0, 0]
                self.coll_fq[i] = (np.sqrt(
                    16 * self.kb * self.T_u / (np.pi * self.m_ar)) * self.n_u * self.sigma_col) / self.nu_r
                self.F_0[i] = self.n_xi[i] * self.C_r * \
                              f_equilibrium(self.m_ar, self.kb, self.temps[i], self.C_r * self.v_bar[i], self.v_arr)
            else:
                self.temps[i] = np.array([self.T_d, self.T_d, self.T_d])
                self.n_xi[i] = self.n_d / self.n_r
                self.v_bar[i] = [self.v_bar_d / self.C_r, 0, 0]
                self.coll_fq[i] = (np.sqrt(
                    16 * self.kb * self.T_d / (np.pi * self.m_ar)) * self.n_d * self.sigma_col) / self.nu_r
                self.F_0[i] = self.n_xi[i] * self.C_r * \
                              f_equilibrium(self.m_ar, self.kb, self.temps[i], self.C_r*self.v_bar[i], self.v_arr)

        self.temps_hat = self.temps.copy() / (self.m_ar * self.C_r * self.C_r/self.kb)
        self.F = self.F_0.copy()
        pass

    def update_macros(self):
        for i in range(self.mx):
            n_i, temp_i, v_bar_i = get_macros(self.F[i], self.c_hat)
            self.n_xi[i] = n_i
            self.temps_hat[i] = temp_i
            # print(self.temps_hat[i])
            if min(temp_i) <= 0:
                print(temp_i)
                print(i)
                raise Exception('not pos temp')
            # print(self.m_ar * self.C_r * self.C_r * temp_i/self.kb)
            self.temps[i] = self.m_ar * self.C_r * self.C_r * temp_i/self.kb
            self.v_bar[i] = v_bar_i
            self.coll_fq[i] = np.sqrt(16 * self.kb * np.mean(self.temps[i]) / (np.pi * self.m_ar)) * n_i * self.sigma_col/self.nu_r
        pass

    def drift(self):
        v_dfdx = np.zeros(self.F.shape)
        c_up = np.where(self.c_hat > 0, self.c_hat, 0)
        c_down = np.where(self.c_hat < 0, self.c_hat, 0)
        dfdx_up = self.F[1:-1] - self.F[:-2]
        dfdx_down = self.F[2:] - self.F[1:-1]
        v_dfdx[1:-1] = c_up[np.newaxis, :, np.newaxis] * dfdx_up + c_down[np.newaxis, :, np.newaxis] * dfdx_down
        return v_dfdx
        #   Apply drift function

    def collide(self):
        f_coll = np.zeros(self.F.shape)
        for i in range(self.mx):
            f_eq_i = self.n_xi[i]*self.C_r * f_equilibrium(self.m_ar, self.kb, self.temps[i], self.C_r * self.v_bar[i], self.v_arr)
            f_coll[i] = self.coll_fq[i] * (f_eq_i - self.F[i]) / self.Kn
        return f_coll

    def bgk_ode(self, t, f):
        self.F = np.reshape(f, self.F.shape)
        self.update_macros()
        dfdt = self.collide() - self.drift()
        dfdt[0] = np.zeros(self.F[0].shape)
        return np.reshape(dfdt, dfdt.size)

    def execute_solver(self):
        self.initial_conditions()

        sol = scipy.integrate.solve_ivp(self.bgk_ode, self.t_span, np.reshape(self.F_0, self.F_0.size))
        return sol
