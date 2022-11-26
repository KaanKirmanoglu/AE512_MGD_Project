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


    sol = solver.execute_solver()
    # solver.F = sol.y[:, -1].reshape(solver.F.shape)
    # solver.update_macros()







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
