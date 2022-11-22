import matplotlib.pyplot as plt
import numpy as np
import scipy
import shockBGKSolver


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # solver = shockBGKSolver.solverBGK()
    fig_id = 1
    temp_r = [1, 1, 1]
    v_bar = [20, 0, 0]
    nv = 40
    f_eq = np.ones([nv, 3])
    v_arr_i = []
    for i in range(len(v_bar)):
        sigma = np.sqrt(temp_r[i])
        v_arr = np.linspace(v_bar[i] - 4.2 * sigma, v_bar[i] + 4.2 * sigma, nv)
        v_arr_i.append(v_arr)
        #print(np.exp(-((v_arr - v_bar[i]) ** 2) / (2 * temp_r[i])))
        f_eq[:, i] = (1 / np.sqrt(2 * np.pi * temp_r[i])) * np.exp(-((v_arr - v_bar[i]) ** 2) / (2 * temp_r[i]))
        plt.figure(fig_id)
        print(fig_id)
        fig_id += 1
        plt.plot(v_arr, f_eq[:, i],'o')
        plt.show()

    int_f1 = np.trapz(f_eq[:, 0], v_arr_i[0])
    print(int_f1)
    int_F_tot = np.trapz(f_eq[:, 0], v_arr_i[0])*np.trapz(f_eq[:, 1], v_arr_i[1])*np.trapz(f_eq[:, 2], v_arr_i[2])
    print(int_F_tot)
    # X,Y = np.meshgrid(range(inputs.nx)
    print(f_eq[-1, 1])