import matplotlib.pyplot as plt
import numpy as np
import scipy
import shockBGKSolver
from physicalFuncs import *


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    solver = shockBGKSolver.solverBGK()
    print(solver.cs_u)
    print(solver.v_bar_u)
    fig_id = 1
    solver.initial_conditions()

    plt.figure(0)
    plt.plot(solver.x_arr, solver.temps[:, 0])

    # plt.show()
    solver.update_macros()
    temps_hat_0 = solver.temps_hat[:, 0].copy()
    v_bar0 = solver.v_bar[:, 0].copy()
    plt.plot(solver.x_arr, (solver.m_ar * solver.C_r * solver.C_r / solver.kb) * solver.temps_hat[:, 0], 'x')
    # plt.show()
    t_span0 = [0, 0.1]
    t_span1 = np.linspace(0, 0.1, 101)
    dt = 0.0002
    t_span2 = dt*np.arange(10)
    sol = solver.execute_solver(t_span2, 2)
    plt.figure()
    plt.plot(solver.x_arr, temps_hat_0, 'o')
    plt.plot(solver.x_arr, solver.temps_hat[:, 0], 'x', label='Tx')
    plt.plot(solver.x_arr, solver.temps_hat[:, 1], 'd', label='Ty')
    plt.plot(solver.x_arr, solver.temps_hat[:, 2], '*', label='Tz')
    plt.legend()
    plt.show()



    plt.figure()
    plt.plot(solver.x_arr, v_bar0, 'o')
    plt.plot(solver.x_arr, solver.v_bar[:, 0], 'x')
    plt.show()

    plt.figure()

    temps_t = np.array(sol[2])
    plt.plot(t_span2, temps_t[:, 24, 0])

    plt.show()
    v_bars = sol[4]
    v_bars = v_bars[::50]
    for t in range(len(v_bars)):
        plt.plot(solver.x_arr, np.array(v_bars[t])[:, 0])
    plt.show()

    # Execute Solver
    """
    t_span = np.linspace(0, 0.11, 100)
    sol = solver.execute_solver(t_span)
    temps_t = []
    n_xi_t = []
    v_bar_t = []
    F_t = []
    t_arr = []
    plt.figure()
    for i in range(len(sol.t)):
        t_arr.append(sol.t[i])
        F_t.append(sol.y[:, i].reshape(solver.F_0.shape))
        temp_hat_it = np.zeros([solver.mx, 3])
        n_it_hat = np.zeros(solver.mx)
        v_bar_hat_it = np.zeros([solver.mx, 3])
        for x in range(len(solver.x_arr)):
            n_i, temp_i, v_bar_i = get_macros(np.reshape(sol.y[:, i], solver.F_0.shape)[x], solver.c_hat)
            temp_hat_it[x] = temp_i
            n_it_hat[x] = n_i
            v_bar_hat_it[x] = v_bar_i
        plt.plot(solver.x_arr, temp_hat_it[:, 0])
        temps_t.append(temp_hat_it)
        n_xi_t.append(n_it_hat)
        v_bar_t.append(v_bar_hat_it)

    plt.show()
    """


    # solver.F = sol.y[:, -1].reshape(solver.F.shape)
    # solver.update_macros()

    # Chekc Dists
    """
    temp = solver.T_u
    v_bar = np.array([solver.v_bar_u, 200, 0])
    v_arr = np.linspace(-5500, 5500, 50)
    vx, vy, vz = np.meshgrid(v_arr, v_arr, v_arr)
    f_mb = f_mb(solver.m_ar, solver.kb, solver.T_u * np.ones(3), v_bar, v_arr)
    f_0mb = np.zeros([50, 50, 50])
    f_mbx = np.sqrt(solver.m_ar / (2 * temp * np.pi * solver.kb)) * np.exp(
        -0.5 * solver.m_ar * ((v_arr - v_bar[0]) ** 2) / (solver.kb * temp))

    for i in range(len(v_arr)):
        for j in range(len(v_arr)):
            for k in range(len(v_arr)):
                norm_mb = (solver.m_ar / (2 * temp * np.pi * solver.kb)) ** 1.5
                f_exp = (v_arr[i] - v_bar[0]) ** 2 + (v_arr[j] - v_bar[1]) ** 2 + (v_arr[k] - v_bar[2]) ** 2
                f_0mb[i, j, k] = norm_mb * np.exp(-0.5 * solver.m_ar * f_exp / (solver.kb * temp))

    v_arr2 = np.linspace(-50, 50, 11)

    f_x = np.trapz(np.trapz(f_mb, v_arr), v_arr)
    f_y = np.trapz(np.trapz(f_mb, v_arr), v_arr, axis=0)
    f_z = np.trapz(np.trapz(f_mb, v_arr, axis=0), v_arr, axis=0)
    n_x = np.trapz(f_x, v_arr)
    n_y = np.trapz(f_y, v_arr)
    n_z = np.trapz(f_z, v_arr)
    v_bar_x = np.trapz(v_arr * f_x, v_arr) / n_x
    v_bar_y = np.trapz(v_arr * f_y, v_arr) / n_y
    v_bar_z = np.trapz(v_arr * f_z, v_arr) / n_z
    v2_bar = np.trapz(v_arr * v_arr * f_x, v_arr) / n_x
    v2_bar_y = np.trapz(v_arr * v_arr * f_y, v_arr) / n_y
    v2_bar_z = np.trapz(v_arr * v_arr * f_z, v_arr) / n_z
    mT_k = v2_bar - v_bar_x ** 2

    Tx = mT_k * solver.m_ar / solver.kb
    Ty = (solver.m_ar / solver.kb) * (v2_bar_y - v_bar_y ** 2)
    Tz = (solver.m_ar / solver.kb) * (v2_bar_z - v_bar_z ** 2)
    """

    """
    sol = solver.execute_solver()
    solver.F = sol.y[:, -1].reshape(solver.F.shape)
    solver.update_macros()
    plt.figure()
    plt.plot(solver.x_arr, solver.temps[:, 0], label='T_x')
    plt.plot(solver.x_arr, solver.temps[:, 1], label='T_y')
    plt.plot(solver.x_arr, solver.temps[:, 2], label='T_z')
    plt.plot(solver.x_arr, temps_0, label='IC')
    plt.xlabel('x')
    plt.ylabel('T (K)')
    plt.legend()
    plt.show()
    
    sigma2_inv_mb = solver.m_ar / (solver.T_d * solver.kb)
    v_arr = np.linspace(-5200, 5500, 400)
    v_arr2 = np.linspace(-5200, 5500, 400)

    f_mb = np.sqrt(sigma2_inv_mb)/(np.sqrt(2*np.pi)) * np.exp(-0.5*sigma2_inv_mb*(v_arr-solver.v_bar_d)*(v_arr-solver.v_bar_d))
    f_mb2 = np.sqrt(sigma2_inv_mb) / (np.sqrt(2 * np.pi)) * np.exp(-0.5 * sigma2_inv_mb * v_arr2 * v_arr2)
    plt.figure()
    plt.plot(v_arr, f_mb)
    plt.show()
    int_n = np.trapz(f_mb, v_arr)
    c_bar = np.trapz(v_arr * f_mb, v_arr) / int_n
    c2_bar = np.trapz(v_arr * v_arr * f_mb, v_arr) / int_n
    c2_res_bar = c2_bar - c_bar**2
    c_r = solver.v_bar_u
    c_bar_hat = 1
    v_harr = v_arr/c_r
    v_harry = v_arr2/c_r
    t_hat = solver.T_u*solver.kb/(solver.m_ar * c_r * c_r)

    f_hat_0 = c_r*f_mb
    f_haty = c_r*f_mb2
        #(1/(np.sqrt(2*np.pi * t_hat))) * np.exp(-0.5 * (v_harr - c_bar_hat) * (v_harr - c_bar_hat)
         #                                                       / t_hat)

    plt.figure()
    plt.plot(v_harr, f_hat_0, 'x')
    plt.plot(v_harry, f_haty, 'o')
    plt.show()

    int_h0 = np.trapz(f_hat_0, v_harr)
    c_hat_bar = np.trapz(v_harr*f_hat_0, v_harr)/int_h0
    c2_hat_bar = np.trapz(v_harr*v_harr*f_hat_0, v_harr)/int_h0
    c2_hat_res_bar = c2_hat_bar - c_hat_bar**2
    print(c2_hat_res_bar)
"""

    """
    f_eq = np.ones([nv, 3])
    v_arr_i = []
    for i in range(len(v_bar)):
        sigma = np.sqrt(temp_r[i])
        v_arr = np.linspace(v_bar[i] - 4.2 * sigma, v_bar[i] + 4.2 * sigma, nv)
        v_arr_i.append(v_arr)
        # print(np.exp(-((v_arr - v_bar[i]) ** 2) / (2 * temp_r[i])))
        f_eq[:, i] = (1 / np.sqrt(2 * np.pi * temp_r[i])) * np.exp(-((v_arr - v_bar[i]) ** 2) / (2 * temp_r[i]))
        plt.figure(fig_id)
        print(fig_id)
        fig_id += 1
        plt.plot(v_arr, f_eq[:, i], 'o')
        plt.show()

    int_f1 = np.trapz(f_eq[:, 0], v_arr_i[0])
    print(int_f1)
    int_F_tot = np.trapz(f_eq[:, 0], v_arr_i[0]) * np.trapz(f_eq[:, 1], v_arr_i[1]) * np.trapz(f_eq[:, 2], v_arr_i[2])
    print(int_F_tot)
    # X,Y = np.meshgrid(range(inputs.nx)
    print(f_eq[-1, 1])
    """
