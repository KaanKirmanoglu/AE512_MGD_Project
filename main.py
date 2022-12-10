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
    # initialize solver
    solver = shockBGKSolver.solverBGK()

    solver.initial_conditions()

    solver.update_macros()

    dt = 0.0001
    t_span2 = dt * np.arange(900)
    # run the solver with given timesteps, using first order upwind in time
    sol = solver.execute_solver(t_span2, 2)

    # Chapman - Enskog Approximation
    dlogTdx = log_temp_grad(solver.T_u, solver.T_d, np.mean(solver.temps, axis=1), solver.L_r * solver.dx)
    dv_bardx = v_bar_grad(solver.v_bar_u, solver.v_bar_d, solver.C_r * solver.v_bar[:, 0], solver.L_r * solver.dx)
    f_ce = np.empty(solver.F_0.shape)
    for i in range(solver.mx):
        f_ce[i] = chapman_enskog(solver.m_ar, solver.kb, np.mean(solver.temps[i]), solver.C_r * solver.v_bar[i],
                                 solver.v_arr, solver.nu_r * solver.coll_fq[i], dlogTdx[i], dv_bardx[i])

    # Plotting Solutions
    """
    ## Plotting Temperatures
    plt.figure()
    plt.plot(solver.x_hat, solver.temps_hat[:, 0], '-x', label=r'$\^T_{x}$')
    plt.plot(solver.x_hat, solver.temps_hat[:, 1], '-d', label=r'$\^T_{y}$')
    plt.plot(solver.x_hat, solver.temps_hat[:, 2], '-*', label=r'$\^T_{z}$')
    plt.xlabel(r'$\^x$')
    plt.ylabel(r'$\^T$')
    plt.title(r'$\^x$ vs $\^T$, (Non-dimensionalized)')
    plt.legend()
    plt.savefig('nonDTemps.png')
    plt.show()


    plt.figure()
    plt.plot(solver.x_arr, solver.temps[:, 0], '-x', label=r'$T_{x}$')
    plt.plot(solver.x_arr, solver.temps[:, 1], '-d', label=r'$T_{y}$')
    plt.plot(solver.x_arr, solver.temps[:, 2], '-*', label=r'$T_{z}$')
    plt.xlabel(r'$x$, (m)')
    plt.ylabel(r'$T$, (K)')
    plt.title(r'$x$ vs $T$')
    plt.legend()
    plt.savefig('dimTemps.png')
    plt.show()

    ## Plotting densities
    plt.figure()
    plt.plot(solver.x_hat, solver.n_xi, '-x')
    plt.xlabel(r'$\^x$')
    plt.ylabel(r'$\^n$')
    plt.title(r'$\^x$ vs $\^n$, (Non-dimensionalized)')
    plt.savefig('nonDDens.png')
    plt.show()



    plt.figure()
    plt.plot(solver.x_arr, solver.n_r*solver.n_xi, '-x')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$n$')
    plt.title(r'$x$ vs $n$')
    plt.savefig('dimDens.png')
    plt.show()
    
    ## Plotting VDFs
    plt.figure()
    for i in range(4):
        xi = i + 23
        fx = np.trapz(np.trapz(solver.F[xi], solver.c_hat))
        nxi = np.trapz(fx, solver.c_hat)
        fx = fx/nxi
        lab = r"$\^x$=" + str(solver.x_hat[xi])
        plt.plot(solver.c_hat, fx, label=lab)

    plt.ylabel(r'$\^f_{c_x}$')
    plt.xlabel(r'$\^c_{x}$')
    plt.legend(loc='upper left')
    tit = "Normalized VDFs across shock"
    plt.title(tit)
    filename = "nonDimVDfs"
    plt.savefig(filename)
    plt.show()
    plt.figure()
    for j in range(4):
        xi = j + 23
        fx = np.trapz(np.trapz((solver.C_r**3)*solver.F[xi], solver.v_arr))
        nxi = np.trapz(fx, solver.v_arr)
        fx = fx/nxi
        lab = "x=" + str(solver.x_arr[xi])
        plt.plot(solver.v_arr, fx, label=lab)

    plt.legend(loc='upper left')
    plt.ylabel(r'$f_{c_x}$')
    plt.xlabel(r'$c_{x}$ (m/s)')
    tit = "VDFs across shock"
    plt.title(tit)
    filename = "dimVDfs"
    plt.savefig(filename)
    plt.show()
    
    ## Plotting heat fluxes and shear stress
    
    fluxes = np.array(sol[5])
    plt.figure()
    plt.plot(solver.x_arr, fluxes[:, 0])
    plt.title("Heat Fluxes vs x")
    plt.xlabel('x (m)')
    plt.ylabel('Q ($W/m^2$)')
    plt.savefig("heatFlux")
    plt.show()
    t_ref = solver.L_r/solver.C_r
    plt.figure()
    plt.plot(solver.x_hat, ((t_ref**3)/solver.m_ar)*fluxes[:, 0])
    plt.title("Heat Fluxes vs x, non-dimensionalized")
    plt.xlabel(r'$\^x$')
    plt.ylabel(r'$\^Q$')
    plt.savefig("nonDimHeatFlux")
    plt.show()

    plt.figure()
    plt.plot(solver.x_arr, fluxes[:, 1], '-x', label=r"$\tau_{xx}$")
    plt.plot(solver.x_arr, fluxes[:, 2], '-o', label=r"$\tau_{yx}$")
    plt.plot(solver.x_arr, fluxes[:, 3], '-*', label=r"$\tau_{zx}$")
    plt.title("Shear Stress vs x")
    plt.xlabel(r'$x$')
    plt.ylabel(r'$\tau$')
    plt.legend(loc='lower right')
    plt.savefig("shearStress")
    plt.show()

    plt.figure()
    plt.plot(solver.x_arr, (1 / solver.p_u) * fluxes[:, 1], '-x', label=r"$\^\tau_{xx}$")
    plt.plot(solver.x_arr, (1 / solver.p_u) * fluxes[:, 2], '-o', label=r"$\^\tau_{yx}$")
    plt.plot(solver.x_arr, (1 / solver.p_u) * fluxes[:, 3], '-*', label=r"$\^\tau_{zx}$")
    plt.title("Shear Stress vs x, non-dimensionalized")
    plt.xlabel(r'$\^x$')
    plt.ylabel(r'$\^\tau$')
    plt.legend(loc='lower right')
    plt.savefig("nonDimShearStress")
    plt.show()
   

    ## Plotting Chapman-Enskog vs BGK solution

    for j in range(4):
        xi = j + 23
        fx = np.trapz(np.trapz((solver.C_r ** 3) * solver.F[xi], solver.v_arr))
        nxi = np.trapz(fx, solver.v_arr)
        fx = fx / nxi
        fx_ce = np.trapz(np.trapz(f_ce[xi], solver.v_arr))
        fx_ce = fx_ce / np.trapz(fx_ce, solver.v_arr)
        plt.figure()
        lab = "BGK"
        lab2 = "CE"
        plt.plot(solver.v_arr, fx, label=lab)
        plt.plot(solver.v_arr, fx_ce, '-x', label=lab2)
        plt.legend(loc='upper left')
        plt.ylabel(r'$f_{c_x}$')
        plt.xlabel(r'$c_{x}$ (m/s)')
        tit = "Chapman-Enskog vs BGK at x=" + str(solver.x_arr[xi])
        plt.title(tit)
        filename = "ceVsBGK" + str(xi) + ".png"
        plt.savefig(filename, bbox_inches='tight')
        plt.show()

    xi = 8
    plt.figure()
    fx = np.trapz(np.trapz((solver.C_r ** 3) * solver.F[xi], solver.v_arr))
    nxi = np.trapz(fx, solver.v_arr)
    fx = fx / nxi
    fx_ce = np.trapz(np.trapz(f_ce[xi], solver.v_arr))
    fx_ce = fx_ce / np.trapz(fx_ce, solver.v_arr)
    lab = "BGK"
    lab2 = "CE"
    plt.plot(solver.v_arr, fx, label=lab)
    plt.plot(solver.v_arr, fx_ce, '-x', label=lab2)
    plt.legend(loc='upper left')
    plt.ylabel(r'$f_{c_x}$')
    plt.xlabel(r'$c_{x}$ (m/s)')
    tit = "Chapman-Enskog vs BGK at x=" + str(solver.x_arr[xi])
    plt.title(tit)
    filename = "ceVsBGK" + str(xi) + ".png"
    plt.savefig(filename, bbox_inches='tight')
    plt.show()

    xi = 38

    fx = np.trapz(np.trapz((solver.C_r ** 3) * solver.F[xi], solver.v_arr))
    nxi = np.trapz(fx, solver.v_arr)
    fx = fx / nxi
    fx_ce = np.trapz(np.trapz(f_ce[xi], solver.v_arr))
    fx_ce = fx_ce / np.trapz(fx_ce, solver.v_arr)
    lab = "BGK"
    lab2 = "CE"
    plt.plot(solver.v_arr, fx, label=lab)
    plt.plot(solver.v_arr, fx_ce, '-x', label=lab2)
    plt.legend(loc='upper left')
    plt.ylabel(r'$f_{c_x}$')
    plt.xlabel(r'$c_{x}$ (m/s)')
    tit = "Chapman-Enskog vs BGK at x=" + str(solver.x_arr[xi])
    plt.title(tit)
    filename = "ceVsBGK" + str(xi) + ".png"
    plt.savefig(filename, bbox_inches='tight')
    plt.show()
"""