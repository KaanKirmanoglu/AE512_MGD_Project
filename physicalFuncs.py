"""
Physical functions to call for BGK solver (shock conditions, MB distribution etc.)
@KaanKirmanoglu (github)
(2022) University of Illinois Urbana-Champaign
"""
import numpy as np


def rankine_hugoniot(mu, temp_u, p_u, gamma):
    md = np.sqrt(((gamma - 1) * mu * mu + 2) / (2 * gamma * mu * mu - gamma + 1))

    pres_d = p_u * ((2 * gamma * mu * mu - gamma + 1) / (gamma + 1))
    t_ratio = (2 * gamma * mu * mu - (gamma - 1)) * ((gamma - 1) * mu * mu + 2) / (((gamma + 1) * mu) ** 2)
    temp_d = t_ratio * temp_u
    return md, temp_d, pres_d


def f_equilibrium(m, k, temp_r, v_bar, v_arr):
    # return maxwell boltzmann distribution
    f_eq = np.ones([len(v_arr), 3])
    if min(temp_r) <= 0:
        print(temp_r)

        raise Exception('not pos temp')
    sigma_inv_mb = np.sqrt(m/(k*temp_r))
    for i in range(len(v_bar)):
        f_eq[:, i] = (sigma_inv_mb[i] / np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((v_arr-v_bar[i])*sigma_inv_mb[i])**2)
    return f_eq


def get_macros(f_vdf, v_arr):
    n_i = np.trapz(f_vdf[:, 0], v_arr)
    v_bar_x = np.trapz(f_vdf[:, 0] * v_arr, v_arr) / np.trapz(f_vdf[:, 0], v_arr)
    v_bar_y = np.trapz(f_vdf[:, 1] * v_arr, v_arr) / np.trapz(f_vdf[:, 1], v_arr)
    v_bar_z = np.trapz(f_vdf[:, 2] * v_arr, v_arr) / np.trapz(f_vdf[:, 2], v_arr)
    v2_bar_x = np.trapz(f_vdf[:, 0] * v_arr * v_arr, v_arr) / np.trapz(f_vdf[:, 0], v_arr)
    v2_bar_y = np.trapz(f_vdf[:, 1] * v_arr * v_arr, v_arr) / np.trapz(f_vdf[:, 1], v_arr)
    v2_bar_z = np.trapz(f_vdf[:, 2] * v_arr * v_arr, v_arr) / np.trapz(f_vdf[:, 2], v_arr)
    temp_x = v2_bar_x - v_bar_x**2
    temp_y = v2_bar_y - v_bar_y**2
    temp_z = v2_bar_z - v_bar_z**2
    temp_i = np.array([temp_x, temp_y, temp_z])
    v_bar_i = np.array([v_bar_x, v_bar_y, v_bar_z])
    return n_i, temp_i, v_bar_i
