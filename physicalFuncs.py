"""
Physical functions to call for BGK solver (shock conditions, MB distribution etc.)
@KaanKirmanoglu (github)
(2022) University of Illinois Urbana-Champaign
"""
import numpy as np


def rankine_hugoniot(mu, temp_u, p_u, gamma):
    # Jump conditions for normal shock
    md = np.sqrt(((gamma - 1) * mu * mu + 2) / (2 * gamma * mu * mu - gamma + 1))

    pres_d = p_u * ((2 * gamma * mu * mu - gamma + 1) / (gamma + 1))
    t_ratio = (2 * gamma * mu * mu - (gamma - 1)) * ((gamma - 1) * mu * mu + 2) / (((gamma + 1) * mu) ** 2)
    temp_d = t_ratio * temp_u
    return md, temp_d, pres_d


def f_mb(m, k, temp_r, v_bar, v_arr):
    # get Maxwell-Boltzmann VDF distribution
    norm_mb = (m / (2 * np.pi * k)) ** 1.5
    norm_mb = norm_mb / np.sqrt(temp_r[0] * temp_r[1] * temp_r[2])
    vx, vy, vz = np.meshgrid(v_arr, v_arr, v_arr, indexing='ij')
    f_exp = np.exp(-0.5 * m * ((vx - v_bar[0]) ** 2) / (k * temp_r[0])) * np.exp(
        -0.5 * m * ((vy - v_bar[1]) ** 2) / (k * temp_r[1])) \
            * np.exp(-0.5 * m * ((vz - v_bar[2]) ** 2) / (k * temp_r[2]))
    return norm_mb * f_exp


def get_macros(f_vdf, v_arr):
    # Acquire macroscopic quantities from a given distribution
    # zeroth moment: density, first moments: average (bulk) velocities, second moments: temperatures
    f_xy = np.trapz(f_vdf, v_arr)
    f_x = np.trapz(f_xy, v_arr)
    f_y = np.trapz(f_xy, v_arr, axis=0)
    f_z = np.trapz(np.trapz(f_vdf, v_arr, axis=0), v_arr, axis=0)
    n_x = np.trapz(f_x, v_arr)
    n_y = np.trapz(f_y, v_arr)
    n_z = np.trapz(f_z, v_arr)
    v_bar_x = np.trapz(v_arr * f_x, v_arr) / n_x
    v_bar_y = np.trapz(v_arr * f_y, v_arr) / n_y
    v_bar_z = np.trapz(v_arr * f_z, v_arr) / n_z
    v2_bar_x = np.trapz(f_x * v_arr * v_arr, v_arr) / n_x
    v2_bar_y = np.trapz(f_y * v_arr * v_arr, v_arr) / n_y
    v2_bar_z = np.trapz(f_z * v_arr * v_arr, v_arr) / n_z
    temp_x = v2_bar_x - v_bar_x ** 2
    temp_y = v2_bar_y - v_bar_y ** 2
    temp_z = v2_bar_z - v_bar_z ** 2
    temp_i = np.array([temp_x, temp_y, temp_z])
    v_bar_i = np.array([v_bar_x, v_bar_y, v_bar_z])
    n_i = (n_x + n_y + n_z) / 3
    return n_i, temp_i, v_bar_i


def get_qi_taux(m, f_vdf, v_arr):
    # Get heat flux and shear stresses from distribution
    f_xy = np.trapz(f_vdf, v_arr)
    f_x = np.trapz(f_xy, v_arr)
    f_y = np.trapz(f_xy, v_arr, axis=0)
    f_z = np.trapz(np.trapz(f_vdf, v_arr, axis=0), v_arr, axis=0)
    n_x = np.trapz(f_x, v_arr)
    n_y = np.trapz(f_y, v_arr)
    n_z = np.trapz(f_z, v_arr)
    v_bar_x = np.trapz(v_arr * f_x, v_arr) / n_x
    v_bar_y = np.trapz(v_arr * f_y, v_arr) / n_y
    v_bar_z = np.trapz(v_arr * f_z, v_arr) / n_z
    v2_bar_x = np.trapz(f_x * v_arr * v_arr, v_arr) / n_x
    v2_bar_y = np.trapz(f_y * v_arr * v_arr, v_arr) / n_y
    v2_bar_z = np.trapz(f_z * v_arr * v_arr, v_arr) / n_z
    vx, vy, vz = np.meshgrid(v_arr, v_arr, v_arr, indexing='ij')
    vxv2 = (vx - v_bar_x) * ((vx - v_bar_x) ** 2 + (vy - v_bar_y) ** 2 + (vz - v_bar_z) ** 2)
    vxv2fx = vxv2 * f_vdf
    q_xi = np.trapz(np.trapz(np.trapz(v_arr * vxv2fx, v_arr), v_arr), v_arr)

    p_i = n_x * ((v2_bar_x - v_bar_x ** 2) + (v2_bar_y - v_bar_y ** 2) + (v2_bar_z - v_bar_z ** 2)) / 3
    vyx_res = (vy - v_bar_y) * (vx - v_bar_x)
    vzx_res = (vz - v_bar_z) * (vx - v_bar_x)
    tau_xx = - (n_x * (v2_bar_x - v_bar_x ** 2) - p_i)
    tau_yx = - np.trapz(np.trapz(np.trapz(vyx_res * f_vdf, v_arr), v_arr), v_arr)
    tau_zx = - np.trapz(np.trapz(np.trapz(vzx_res * f_vdf, v_arr), v_arr), v_arr)
    return [q_xi, tau_xx, tau_yx, tau_zx]


def chapman_enskog(m, k, temp_r, v_bar, v_arr, nu, dlnTdx, dv_bardx):
    # Get Chapman-Enskog approximation for a VDF
    f0 = f_mb(m, k, np.array([temp_r, temp_r, temp_r]), v_bar, v_arr)
    vx, vy, vz = np.meshgrid(v_arr, v_arr, v_arr, indexing='ij')
    c_prime2 = (vx - v_bar[0]) ** 2 + (vy - v_bar[1]) ** 2 + (vz - v_bar[2]) ** 2
    cx_prime = (vx - v_bar[0])
    phi_1 = cx_prime * ((m * c_prime2 / (2 * k * temp_r) - 5 / 2) * dlnTdx) \
        + (m * k / temp_r) * (cx_prime * cx_prime - c_prime2 / 3) * dv_bardx
    phi_1 = -(1 / nu) * phi_1
    f_ce = f0 * (1 + phi_1)
    return f_ce


def log_temp_grad(temp_lBC, temp_rBC, temps, dx):
    # derivatives of ln(T) in space , \partial (lnT) / \partial x
    log_temps = np.log(temps)
    log_temp_lBC = np.log(temp_lBC)
    log_temp_rBC = np.log(temp_rBC)
    dlogTdx = np.zeros(temps.shape)
    dlogTdx[1:-1] = (log_temps[2:] - log_temps[:-2]) / (2 * dx)
    dlogTdx[0] = (log_temps[1] - log_temp_lBC) / (2 * dx)
    dlogTdx[-1] = (log_temp_rBC - log_temps[-2]) / (2 * dx)
    return dlogTdx


def v_bar_grad(v_bar_lBC, v_bar_rBC, v_bar, dx):
    # derivatives of bulk velocity in space , \partial (<C_i>) / \partial x
    dvdx = np.zeros(v_bar.shape)
    dvdx[1:-1] = (v_bar[2:] - v_bar[:-2]) / (2 * dx)
    dvdx[0] = (v_bar[1] - v_bar_lBC) / (2 * dx)
    dvdx[-1] = (v_bar_rBC - v_bar[-2]) / (2 * dx)
    return dvdx
