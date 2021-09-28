import numpy as np

from evb_surf import EvbSurface, const_evb_off_diag
from lan_solv import LangevinSolver, momentum_autocorrelation


def run_evb_langevin(evb_surface: EvbSurface, gamma, mass, temp, init_q,
                     num_steps=100_000, record_steps=100, ensemble_size=1000, dt=0.01,
                     q_constr=None, k_constr=0.0, outfile="evb_lan", L=None):
    """

    :param evb_surface: EVB Surface class
    :param gamma: 2D [g1, g2] friction coefficients
    :param mass: 2D [m1, m2] mass parameters
    :param temp: temperature (k_b T) in kcal/mol
    :param ensemble_size: number of particles to run
    :param init_q: initial coordinates for [Q1, Q2]
    :return:
    """
    ldout = outfile+"_ld.txt"
    trajq1file = outfile+"_q1_traj.txt"
    trajq2file = outfile + "_q2_traj.txt"
    # note: the constraint potential is defined as U_constr = K (q-q1)^2
    # in the relevant literature, without a factor of 1/2
    if q_constr is not None:
        q_constr = np.asarray(q_constr) * evb_surface.delta

    def eval_force(q):
        net_force = evb_surface.force(q)
        if q_constr is not None:
            constraint_force = -2.0 * k_constr * (q - q_constr)
            net_force += constraint_force
        return net_force

    def eval_constraint_energy(q):
        if q_constr is not None:
            return k_constr * np.sum((q - q_constr)**2, axis=1)
        else:
            return np.zeros(np.shape(q)[0:1])

    def count_quadrants(q):
        Q1m = q[:, 0] < 0.0
        Q1p = np.logical_not(Q1m)
        Q2m = q[:, 1] < 0.0
        Q2p = np.logical_not(Q2m)
        mm = np.asarray(np.logical_and(Q1m, Q2m), dtype=np.int)
        mp = np.asarray(np.logical_and(Q1m, Q2p), dtype=np.int)
        pm = np.asarray(np.logical_and(Q1p, Q2m), dtype=np.int)
        pp = np.asarray(np.logical_and(Q1p, Q2p), dtype=np.int)
        return np.asarray([np.sum(mm), np.sum(mp), np.sum(pm), np.sum(pp)])

    # Construct [N, 2] ensemble
    delta = evb_surface.delta
    x0 = np.zeros([ensemble_size, 2])
    p0 = np.zeros([ensemble_size, 2])
    x0[:, 0] = init_q[0] * delta[0]
    x0[:, 1] = init_q[1] * delta[1]
    mass = np.asarray(mass)
    g = np.asarray(gamma)
    solver = LangevinSolver(x0, p0, mass, g, temp, dt, eval_force, L=L)
    solver.reset_p()
    print("Wiener increment norm |dW| = sqrt(2*k_b*T*g*dt) = {}".format(solver.wnorm))
    if np.any(solver.wnorm > 0.5):
        print("\tWARNING: |dW| > 0.5 may be unstable")
    m = num_steps // record_steps
    t_arr = np.zeros([m])
    x_hist = np.zeros([m, ensemble_size, 2])
    p_hist = np.zeros([m, ensemble_size, 2])
    q1_avg = np.zeros([m])
    q2_avg = np.zeros([m])
    e_avg = np.zeros([m])
    econst_avg = np.zeros([m])
    quadrants = np.zeros([m, 4])
    ti = 0  # time step
    ri = 0  # record step
    for i in range(num_steps):
        if ti%record_steps == 0:
            t = i * dt
            t_arr[ri] = t
            x_hist[ri] = solver.x[:]
            p_hist[ri] = solver.p[:]
            q1_avg[ri] = np.mean(solver.x[:, 0])
            q2_avg[ri] = np.mean(solver.x[:, 1])
            e_avg[ri] = np.mean(evb_surface.energy(solver.x))
            econst_avg[ri] = np.mean(eval_constraint_energy(solver.x))
            qcounts = np.asarray(count_quadrants(solver.x), dtype=np.float)
            print("t={} / {}, qcounts={}".format(t, num_steps*dt, qcounts))
            quadrants[ri, :] = qcounts[:]
            ri += 1
        ti += 1
        solver.step()

    acp = momentum_autocorrelation(p_hist)
    acx = momentum_autocorrelation(x_hist)
    mean_acp = np.mean(np.sum(acp, axis=2), axis=1)
    mean_acq1 = np.mean(acx[:, :, 0], axis=1)
    acp0 = np.copy(mean_acp[0])
    acx0 = np.copy(mean_acq1[0])
    for i in range(m):
        mean_acp[i] /= acp0
        mean_acq1[i] /= acx0

    np.savetxt(ldout, np.transpose([t_arr, q1_avg, q2_avg, e_avg, econst_avg,
                                      quadrants[:, 0], quadrants[:, 1], quadrants[:, 2], quadrants[:, 3],
                                      mean_acp, mean_acq1]), fmt='%8.6f',
               header='t       <q1>      <q2>     E_evb0   E_constr   QI(mm)   QII(mp)   QIII(pm)   QIV(pp)'
                      '   P_ACF   Q1_ACF')
    np.savetxt(trajq1file, np.hstack((t_arr[:, np.newaxis], x_hist[:, :, 0])), fmt='%8.6f')
    np.savetxt(trajq2file, np.hstack((t_arr[:, np.newaxis], x_hist[:, :, 1])), fmt='%8.6f')

def plot_evb_surface(evb_surface: EvbSurface, flnm):
    # Plot surface in kcal/mol (= .4184 [E])
    n = 101
    xgrid1 = np.linspace(evb_surface.lo_q1, evb_surface.hi_q1, n)
    ygrid1 = np.linspace(evb_surface.lo_q2, evb_surface.hi_q2, n)
    es = evb_surface.energy_mesh([xgrid1, ygrid1]) / .4184
    fs = evb_surface.force_mesh([xgrid1, ygrid1])
    Xgrid1, Ygrid1 = np.meshgrid(xgrid1, ygrid1, indexing='ij')

    cols = []
    for i, x in enumerate(xgrid1):
        for j, y in enumerate(ygrid1):
            cols.append([x, y, es[i, j]])
    cols = np.asarray(cols)
    np.savetxt(flnm+"_esurface.txt", cols)

    # Plot the surface.
    fig, ax = plt.subplots()
    #ax = fig.gca(projection='3d')
    cplot = ax.contourf(Xgrid1, Ygrid1, es, 15, cmap=cm.coolwarm, vmin=-15.0, vmax=120.0)
    ax.contour(Xgrid1, Ygrid1, es, cplot.levels, colors='k')
    #surf = ax.plot_surface(Xgrid1, Ygrid1, es, cmap=cm.coolwarm,
    #                       linewidth=0, antialiased=False)
    ax.set_xlabel("Q1 (Conformational)")
    ax.set_ylabel("Q2 (Substrate)")
    fig.colorbar(cplot, ax=ax, label="E (kcal/mol)")
    #ax.set_zlabel("E0")

    plt.savefig(flnm+"_energy_contour.pdf")

    fig, ax = plt.subplots()
    q = ax.quiver(Xgrid1, Ygrid1, fs[0]/.4184, fs[1]/.4184)
    ax.set_xlabel("Q1")
    ax.set_ylabel("Q2")
    # ax.quiverkey(q, X=0.3, Y=1.1, U=10,
    #             label='Quiver key, length = 10', labelpos='E')

    plt.show()


def plot_profile(evb_surface: EvbSurface, flnm):
    delta = evb_surface.delta
    # I -> III
    xgrid1 = np.linspace(-delta[0], delta[0], 51)
    ygrid1 = np.full((51,), -delta[1])
    gridpnts = np.asarray([xgrid1, ygrid1]).transpose()
    e1 = evb_surface.energy(gridpnts) / 0.4184 #kcal / mol
    ed1 = evb_surface.diabatic(gridpnts, 0) / 0.4184
    ed3 = evb_surface.diabatic(gridpnts, 2) / 0.4184
    np.savetxt(flnm+"_profile_I_III.txt", np.asarray([xgrid1, e1, ed1, ed3]).transpose())

    fig, ax = plt.subplots()
    ax.plot(xgrid1, e1)
    ax.plot(xgrid1, ed1)
    ax.plot(xgrid1, ed3)
    ax.set_xlabel("Q1 (Conformational)")
    ax.set_ylabel("PMF (kcal/mol)")
    plt.show()

    # III -> IV
    xgrid2 = np.full((51,), delta[0])
    ygrid2 = np.linspace(-delta[1], delta[1], 51)
    gridpnts2 = np.asarray([xgrid2, ygrid2]).transpose()
    e2 = evb_surface.energy(gridpnts2) / 0.4184
    ed3 = evb_surface.diabatic(gridpnts2, 2) / 0.4184
    ed4 = evb_surface.diabatic(gridpnts2, 3) / 0.4184
    np.savetxt(flnm+"_profile_III_IV.txt", np.asarray([ygrid2, e2, ed3, ed4]).transpose())

    fig, ax = plt.subplots()
    ax.plot(ygrid2, e2)
    ax.plot(ygrid2, ed3)
    ax.plot(ygrid2, ed4)
    ax.set_xlabel("Q2 (Substrate)")
    ax.set_ylabel("PMF (kcal/mol)")
    plt.show()


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D

    parser = argparse.ArgumentParser(description=
                                     "Runs a 2D Langevin Dynamics simulation parametrized by a four-state EVB surface. "
                                     "The coordinates Q1 and Q2 correspond to two renormalized degrees of freedom in a "
                                     "molecular system and are scaled so that the local minima are located at "
                                     "(+-0.5, +-0.5).")
    parser.add_argument("--jq1", nargs=2, type=float, help="Q1 barrier off-diagonals for [Q2 < 0, Q2 > 0]",
                        default=[0.0, 0.0], metavar=('JQ1-', 'JQ1+'))
    parser.add_argument("--jq2", nargs=2, type=float, help="Q2 barrier off-diagonals for [Q1 < 0, Q1 > 0]",
                        default=[0.0, 0.0], metavar=('JQ2-', 'JQ2+'))
    parser.add_argument("--jqx", nargs=2, type=float,
                        help="Additional off-diagonal contribution in the increasing"
                             " and decreasing directions at (Q1, Q2) = (0, 0)",
                        default=[0.0, 0.0], metavar=('JQX+', 'JQX-'))
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Overall scale factor for the energy surface."
                        )
    parser.add_argument("--minima", nargs=4, type=float,
                        help="The four EVB local minima in the following quadrants: "
                             " (-1, -1), (-1, 1), (1, -1), (1, 1)",
                        default=[0.0, 0.0, 0.0, 0.0], metavar=('A--', 'A-+', 'A+-', 'A++'))
    parser.add_argument("--freqs", nargs=2, type=float, help="EVB frequencies for Q1 and Q2 (in cm^-1)",
                        default=[1.0, 1.0], metavar=('W1', 'W2'))
    parser.add_argument("--reorg-energies", nargs=2, type=float,
                        help="Reorganization energies for Q1 and Q2 (in kcal/mol)")
    #parser.add_argument("--delta", nargs=2, type=float,
    #                    help="Delta length scales for Q1 and Q2, approximately in Angstroms")
    parser.add_argument("--temp", type=float, help="Temperature in K", metavar='T')
    parser.add_argument("--gamma", nargs=2, type=float, help="Friction coefficients for the two particles",
                        metavar=('G1', 'G2'))
    parser.add_argument("--mass", nargs=2, type=float, help="Masses of the two particles", metavar=('M1', 'M2'))
    parser.add_argument("--dt", type=float, help="Time step size (default=0.01)", default=0.01)
    parser.add_argument("--init-q", nargs=2, type=float, help="Initial condition for langevin dynamics (in delta)",
                        metavar=('Q1_0', 'Q2_0'), default=(-1.0, -1.0))
    parser.add_argument("--k-constr", type=float, help="Force constant for constraint force", default=0.0, metavar='K')
    parser.add_argument("--q-constr", nargs=2, type=float,
                        help="Harmonic constraint positions in units of delta (not used if not specified)", metavar=('Q1_1', 'Q2_1'))
    parser.add_argument("--num-steps", type=int, help="Number of steeps to run", metavar='N', default=10_000)
    parser.add_argument("--ensemble-size", type=int, help="Number of LD walkers to run simultaneously", default=100)
    parser.add_argument("--output", help="Output text file for EVB surface", default="evb_lan.txt")
    parser.add_argument("--outfreq", type=int, help="Number of steps between output records", default=100)
    parser.add_argument("--plot-surface", action='store_true')
    args = parser.parse_args()

    mass_amu = np.asarray(args.mass)

    # Set up energy surface
    # Convert frequency from cm^-1 to kcal/mol to calculate delta

    freqs_cm = np.asarray(args.freqs) / (2.0 * np.pi)
    # Freqs in ps^-1, where 0.1884943 = 2 pi c   in   cm/ps
    freqs_ps = 0.1884943 * freqs_cm

    w_kcalmol = 2.8951e-3 * freqs_cm

    print("w = {} (cm^-1),  \t{} (kcal/mol)\t{} (ps^-1)"
          .format(np.asarray(args.freqs), w_kcalmol, freqs_ps))
    reorg_energy = np.asarray(args.reorg_energies)
    # reorg_energy = 2.0 * reorg_energy
    print("Reorganization energy:")
    print("\tlambda = {} (kcal/mol)".format(reorg_energy))

    delta = np.sqrt(2.0 * reorg_energy / w_kcalmol)

    # Length scale conversion factor (1/A)
    # 0.02968 = 2 pi c / hbar  in   cm / (g/mol A^2)
    x_fact = np.sqrt(0.02968 * mass_amu * freqs_cm)
    delta_prime = delta * (x_fact**-1)
    print("Length Scale: {} A".format(x_fact**-1))
    print("*** delta = {}\t -> {} (A)".format(delta, delta_prime))

    # Hamiltonian matrix in natural units of energy
    # [E] = [(kg/mol) A^2/ps^2] = 10 [kJ / mol] = 2.39 [kcal/mol]
    mass = mass_amu * 0.001
    freqs_e = w_kcalmol * 4.184 * 0.1
    # QHO K in scaled coordinate units
    # [kg/mol] / [ps^2]  == [E] / [A]^2
    qho_k = mass * freqs_ps ** 2  # / (x_fact**2)
    print("QHO K (m omega^2): {}".format(qho_k))
    # Convert off-diagonals, gas shift from kcal/mol to [E]
    jq1, jq2, jqx = args.jq1, args.jq2, args.jqx
    jvals = np.asarray([jq2[0], jq1[0], jqx[0], jqx[1], jq1[1], jq2[1]])
    jmat = const_evb_off_diag(jvals) * 4.184 * 0.1
    gas_shift = np.reshape(np.asarray(args.minima), [2, 2]) * 4.184 * 0.1
    # *** Construct EVB surface in natural energy units
    #     as a function of unitless coordinates
    surf = EvbSurface(qho_k, jmat, gas_shift, n=101, delta=delta_prime, scale=args.scale)

    if args.plot_surface:
        plot_evb_surface(surf, args.output)
        plot_profile(surf, args.output)
    # Set up langevin dynamics
    # Convert to natural units of mass [kg/mol]

    # Convert to natural units of energy  [(kg/mol) A^2/ps^2]
    temp_kcalmol = args.temp * 0.001985875
    temp_e = temp_kcalmol * 4.184 * 0.1
    print(" Temperature: {} (K),   {} (kcal/mol)".format(args.temp, temp_kcalmol))
    q_constr = args.q_constr * 4.184 * 0.1 if args.q_constr is not None else None
    k_constr = args.k_constr * 4.184 * 0.1 if args.k_constr is not None else None
    # lamb = freqs_e * (delta ** 2)
    # print("Reorganization Energies (\\lambda = \\hbar \\omega \\delta^2):\n \t{}\t{}"
    #      .format(lamb[0], lamb[1]))
    run_evb_langevin(surf, gamma=args.gamma, mass=mass, temp=temp_e, init_q=args.init_q, num_steps=args.num_steps,
                     record_steps=args.outfreq, ensemble_size=args.ensemble_size, dt=args.dt, q_constr=q_constr,
                     k_constr=k_constr, outfile=args.output, L=None)
