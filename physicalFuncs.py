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


def f_equilibrium(temp_r, nv, v_bar):
    f_eq = np.ones([nv, 3])

    for i in range(len(v_bar)):
        sigma = np.sqrt(temp_r[i])
        v_arr = np.linspace(v_bar[i]-4*sigma, v_bar[i]+4*sigma, nv)
        f_eq[:, i] = (1/np.sqrt(2*np.pi*temp_r[i])) * np.exp(-((v_arr-v_bar[i])**2) / (2*temp_r[i]))
    return f_eq
