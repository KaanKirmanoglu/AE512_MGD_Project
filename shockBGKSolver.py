"""
solver for BGK equation
@KaanKirmanoglu (github)
(2022) University of Illinois Urbana-Champaign
"""
# TODO: BCs, biased maxwell on left? no BC on right?
# Collision frequency: v= v(x)? Equilibrium MB formula?

import numpy as np
import scipy
from matplotlib import pyplot as plt
from numpy import ndarray

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
        self.mx = 50
        self.nv = 30
        self.t_tot = 100
        self.v_arr = np.linspace(-6000, 7000, self.nv)
        # initialize distribution function to solve
        self.F_0 = np.ones([self.mx, self.nv, self.nv, self.nv])
        self.F = np.ones([self.mx, self.nv, self.nv, self.nv])
        self.F_eq = np.ones([self.mx, self.nv, self.nv, self.nv])
        self.F_rBC = np.ones([self.nv, self.nv, self.nv])
        self.F_lBC = np.ones([self.nv, self.nv, self.nv])
        self.temps = np.zeros([self.mx, 3])
        self.temps_hat = np.zeros([self.mx, 3])
        self.n_xi = np.zeros(self.mx)
        self.v_bar = np.zeros([self.mx, 3])
        self.coll_fq = np.zeros(self.mx)

        self.x_arr = np.linspace(-1, 1, self.mx)

        # Reference Quantities

        self.L_r = self.x_arr[-1] - self.x_arr[0]
        self.x_hat = self.x_arr / self.L_r
        self.dx = self.x_hat[1] - self.x_hat[0]
        self.C_r = self.v_bar_u
        self.c_hat = self.v_arr / self.C_r
        self.nu_r = np.sqrt(16 * self.kb * self.T_u / (np.pi * self.m_ar)) * self.n_u * self.sigma_col
        self.n_r = self.n_u
        self.Kn = self.C_r / (self.L_r * self.nu_r)
        self.t_span = [0, 5]

        # Solution quantities
        self.temps_t = []
        self.n_xi_t = []
        self.v_bar_t = []
        self.F_t = []
        self.t_arr = []
        self.q_tau = []

    def initial_conditions(self):
        # set initial conditions as non-dimensionalized form
        for i in range(self.mx):
            if self.x_arr[i] < 0:
                self.temps[i] = np.array([self.T_u, self.T_u, self.T_u])
                self.n_xi[i] = self.n_u / self.n_r
                self.v_bar[i] = [self.v_bar_u / self.C_r, 0, 0]
                self.coll_fq[i] = (np.sqrt(
                    16 * self.kb * self.T_u / (np.pi * self.m_ar)) * self.n_u * self.sigma_col) / self.nu_r
                self.F_0[i] = \
                    self.n_xi[i] * (self.C_r ** 3) * \
                    f_mb(self.m_ar, self.kb, self.temps[i], self.C_r * self.v_bar[i], self.v_arr)
                self.F[i] = \
                    self.n_xi[i] * (self.C_r ** 3) * \
                    f_mb(self.m_ar, self.kb, self.temps[i], self.C_r * self.v_bar[i], self.v_arr)
            else:
                self.temps[i] = np.array([self.T_d, self.T_d, self.T_d])
                self.n_xi[i] = self.n_d / self.n_r
                self.v_bar[i] = [self.v_bar_d / self.C_r, 0, 0]
                self.coll_fq[i] = (np.sqrt(
                    16 * self.kb * self.T_d / (np.pi * self.m_ar)) * self.n_d * self.sigma_col) / self.nu_r
                self.F_0[i] = \
                    self.n_xi[i] * (self.C_r ** 3) * \
                    f_mb(self.m_ar, self.kb, self.temps[i], self.C_r * self.v_bar[i], self.v_arr)
                self.F[i] = \
                    self.n_xi[i] * (self.C_r ** 3) * \
                    f_mb(self.m_ar, self.kb, self.temps[i], self.C_r * self.v_bar[i], self.v_arr)

        self.temps_hat = self.temps.copy() / (self.m_ar * self.C_r * self.C_r / self.kb)
        self.F_lBC = (self.n_u / self.n_r) * (self.C_r ** 3) * \
            f_mb(self.m_ar, self.kb, self.temps[0], self.C_r * self.v_bar[0], self.v_arr)
        self.F_rBC = (self.n_d / self.n_r) * (self.C_r ** 3) * \
            f_mb(self.m_ar, self.kb, self.temps[-1], self.C_r * self.v_bar[-1], self.v_arr)
        pass

    def update_macros(self):
        # Update Macroscopit quantities like density, temperature and also collision frequencies from VDF
        for i in range(self.mx):
            n_i, temp_i, v_bar_i = get_macros(self.F[i], self.c_hat)
            self.n_xi[i] = n_i
            self.temps_hat[i] = temp_i
            # print(self.temps_hat[i])
            if min(temp_i) <= 0:
                print(temp_i)
                print(i)
                print("time step:")
                print(self.t_arr[-1])
                raise Exception('not pos temp')
            # print(self.m_ar * self.C_r * self.C_r * temp_i/self.kb)
            self.temps[i] = self.m_ar * self.C_r * self.C_r * temp_i / self.kb
            self.v_bar[i] = v_bar_i
            self.coll_fq[i] = np.sqrt(16 * self.kb * np.mean(self.temps[i]) / (np.pi * self.m_ar)) * (
                    n_i * self.n_r) * self.sigma_col / self.nu_r
        pass

    def drift(self):
        # Approximate drift/streaming term for BGK Equation using first order upwinding
        # c_i*del(f)/del(x)
        v_dfdx: ndarray = np.zeros(self.F.shape)
        c_up = np.where(self.c_hat > 0, self.c_hat, 0)
        c_down = np.where(self.c_hat < 0, self.c_hat, 0)
        dfdx_up = (self.F[1:-1] - self.F[:-2]) / self.dx
        dfdx_down = (self.F[2:] - self.F[1:-1]) / self.dx
        v_dfdx[1:-1] = c_up[np.newaxis, :, np.newaxis, np.newaxis] * dfdx_up + \
            c_down[np.newaxis, :, np.newaxis, np.newaxis] * dfdx_down

        v_dfdx[0] = \
            c_up[:, np.newaxis, np.newaxis] * (self.F[0] - self.F_lBC) / self.dx
        v_dfdx[-1] = \
            c_down[:, np.newaxis, np.newaxis] * (self.F_rBC - self.F[-1]) / self.dx # + \
            #  c_up[:, np.newaxis, np.newaxis] * (self.F[-1] - self.F[-2]) / self.dx
        return v_dfdx
        #   Apply drift function

    def collide(self):
        # Calculate collision term for BGK equation
        # (1/Kn*nu) * (f_eq - f)
        f_coll = np.zeros(self.F.shape)
        for i in range(self.mx):
            temps_coll = np.mean(self.temps[i]) * np.ones(self.temps[i].shape)
            f_eq_i = self.n_xi[i] * (self.C_r ** 3) * f_mb(self.m_ar, self.kb, temps_coll, self.C_r * self.v_bar[i],
                                                           self.v_arr)
            f_coll[i] = self.coll_fq[i] * (f_eq_i - self.F[i]) / self.Kn
        return f_coll

    def bgk_ode(self, t, f):
        # calculate time derivatives
        # del(f)\del(t) = - c_i  * del(f)/del(x) + (1/Kn*nu) * (f_eq - f)
        self.F = np.reshape(f, self.F.shape)
        self.update_macros()
        dfdt = self.collide() - self.drift()
        c_up = np.where(self.c_hat > 0, 1, 0)
        c_down = np.where(self.c_hat < 0, 1, 0)
        dfdt[0] = c_up[:, np.newaxis, np.newaxis] * dfdt[0]
        dfdt[-1] = c_down[:, np.newaxis, np.newaxis] * dfdt[-1]
        # c_down[:, np.newaxis, np.newaxis] * dfdt[-1]
        return np.reshape(dfdt, dfdt.size)

    def bgk_ode2(self):
        # calculate time derivatives
        # del(f)\del(t) = - c_i  * del(f)/del(x) + (1/Kn*nu) * (f_eq - f)
        self.update_macros()
        dfdt = self.collide() - self.drift()
        c_up = np.where(self.c_hat > 0, 1, 0)
        c_down = np.where(self.c_hat < 0, 1, 0)
        dfdt[0] = c_up[:, np.newaxis, np.newaxis] * dfdt[0]
        dfdt[-1] = c_down[:, np.newaxis, np.newaxis] * dfdt[-1]
        # c_down[:, np.newaxis, np.newaxis] * dfdt[-1]
        return dfdt

    def execute_solver(self, t_s, solver_flag):
        # Execute solver and return solutions as macro quantities, resulting VDF, heat flux and shear stresses
        global sol
        self.initial_conditions()
        if solver_flag == 0:
            sol = scipy.integrate.solve_ivp(self.bgk_ode, t_s, np.reshape(self.F_0, self.F_0.size))
            temps_t = []
            n_xi_t = []
            v_bar_t = []
            F_t = []
            t_arr = []
            plt.figure()
            for i in range(len(sol.t)):
                t_arr.append(sol.t[i])
                F_t.append(sol.y[:, i].reshape(self.F_0.shape))
                temp_hat_it = np.zeros([self.mx, 3])
                n_it_hat = np.zeros(self.mx)
                v_bar_hat_it = np.zeros([self.mx, 3])
                for x in range(len(self.x_arr)):
                    n_i, temp_i, v_bar_i = get_macros(np.reshape(sol.y[:, i], self.F_0.shape)[x], self.c_hat)
                    temp_hat_it[x] = temp_i
                    n_it_hat[x] = n_i
                    v_bar_hat_it[x] = v_bar_i
                plt.plot(self.x_arr, temp_hat_it[:, 0])
                temps_t.append(temp_hat_it)
                n_xi_t.append(n_it_hat)
                v_bar_t.append(v_bar_hat_it)
            plt.show()
        if solver_flag == 1:
            sol = scipy.integrate.odeint(self.bgk_ode, np.reshape(self.F_0, self.F_0.size), t_s, tfirst=True)
            temps_t = []
            n_xi_t = []
            v_bar_t = []
            F_t = []
            t_arr = []
            plt.figure()
            for i in range(sol.shape[0]):
                t_arr.append(t_s[i])
                F_t.append(sol[i, :].reshape(self.F_0.shape))
                temp_hat_it = np.zeros([self.mx, 3])
                n_it_hat = np.zeros(self.mx)
                v_bar_hat_it = np.zeros([self.mx, 3])
                for x in range(len(self.x_arr)):
                    n_i, temp_i, v_bar_i = get_macros(np.reshape(sol[i, :], self.F_0.shape)[x], self.c_hat)
                    temp_hat_it[x] = temp_i
                    n_it_hat[x] = n_i
                    v_bar_hat_it[x] = v_bar_i
                plt.plot(self.x_arr, temp_hat_it[:, 0])
                temps_t.append(temp_hat_it)
                n_xi_t.append(n_it_hat)
                v_bar_t.append(v_bar_hat_it)
            plt.show()
        if solver_flag == 2:
            self.t_arr.append(t_s[0])
            self.temps_t.append(self.temps_hat.copy())
            self.n_xi_t.append(self.n_xi.copy())
            self.v_bar_t.append(self.v_bar.copy())
            self.F_t.append(self.F.copy())
            for t in t_s[1:]:
                dt = t - self.t_arr[-1]
                self.t_arr.append(t)
                dfdt_t = self.bgk_ode2()
                self.F = self.F + dt*dfdt_t
                # print(self.temps_hat[24])
                # print(abs(self.temps_hat[24, 0] - self.temps_t[-1][24][0]))
                self.temps_t.append(self.temps_hat.copy())
                self.n_xi_t.append(self.n_xi.copy())
                self.v_bar_t.append(self.v_bar.copy())



                print("Progress %:")
                print(100 * t / t_s[-1])
                # plt.figure()
                # plt.plot(self.x_arr, self.temps_hat[:, 0], 'x')
                # plt.show()

            for i in range(self.mx):
                self.q_tau.append(get_qi_taux(self.m_ar, (self.C_r**3)*self.F[i], self.v_arr))
            sol = [self.t_arr, self.F, self.temps_t, self.n_xi_t, self.v_bar_t, self.q_tau]
            self.update_macros()
        return sol
