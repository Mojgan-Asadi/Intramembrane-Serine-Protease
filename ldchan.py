import numpy as np

import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]
})
## for Palatino and other serif fonts use:
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Palatino"],
# })

from lan_solv import LangevinSolver, momentum_autocorrelation


def gaussian_potential(x, x0, s):
    return np.exp(-(x-x0)**2 / (2.0 * s))


def gaussian_gradient(x, x0, s):
    return (-(x-x0)/s) * np.exp(-(x-x0)**2 / (2.0 * s))


class Channel:
    """
    Simulates an axially symmetric channel with a Gaussian-shaped potential
    """
    def __init__(self, dz, r, cylinder_k, wall_r, wall_k, dG_barrier, barrier_width, barrier_position):
        self.dz = dz
        self.zmin = -dz / 2.0
        self.zmax = dz / 2.0
        self.r = r
        self.wall_r = wall_r
        self.wall_k = wall_k
        self.cylinder_k = cylinder_k
        self.s = barrier_width
        self.z0 = barrier_position
        # If the barrier is very wide, a correction to the height of the gaussian is needed
        # to simulate the correct energy barrier height
        self.e_correction = (1.0 - gaussian_potential(self.zmin, self.z0, self.s))
        self.dGb = dG_barrier / self.e_correction
        # Shift the zero-point of the potential energy so that \Delta G(z_min) = 0
        self._glo = 0.0
        self._glo = self.gaussian_potential(self.zmin)
        self.ghi = self.gaussian_potential(self.zmax)

    def gaussian_potential(self, z):
        return self.dGb * gaussian_potential(z, self.z0, self.s) - self._glo

    def gaussian_force(self, z):
        # (self.dGb / self.s) * (z - self.z0) * np.exp(-(z - self.z0) ** 2 / (2.0 * self.s))
        return self.dGb * gaussian_gradient(z, self.z0, self.s)

    def potential_1d(self, z):
        z = np.asarray(z)
        e_channel = self.gaussian_potential(z)
        e_le = np.where(z < self.zmin, self.wall_k * (z - self.zmin) ** 2, e_channel)
        e_bounds = np.where(z > self.zmax, self.ghi + self.wall_k * (z - self.zmax) ** 2, e_le)

        return e_bounds

    def force_1d(self, z):
        z = np.asarray(z)
        f_channel = -self.dGb * gaussian_gradient(z, self.z0, self.s)
        f_le = np.where(z < self.zmin, -2.0 * self.wall_k * (z - self.zmin), f_channel)
        f_bounds = np.where(z > self.zmax, -2.0 * self.wall_k * (z - self.zmax), f_le)

        return f_bounds

    def channel_energy(self, R):
        """
        R [N, 3] array
        """
        r2 = R[:, 0]**2 + R[:, 1]**2
        z = R[:, 2]
        return self.potential_1d(z) + np.where(r2 > self.r**2, self.cylinder_k * (r2 - 2.0 * self.r*np.sqrt(r2) + self.r**2), 0.0)

    def channel_force(self, R):

        forces = np.zeros_like(R)
        for r, f in zip(R, forces):
            r2 = r[0]**2 + r[1]**2
            z = r[2]
            if r2 > 1.0e-10:  # Steric component
                r1 = np.sqrt(r2)
                RxyHat = r[0:2]/r1
                if z < self.zmin or z > self.zmax:
                    dr = np.maximum(r1 - self.wall_r, 0.0)
                else:
                    dr = np.maximum(r1 - self.r, 0.0)
                f[0:2] = -self.cylinder_k * RxyHat * dr
        # Axial component
        forces[:, 2] = self.force_1d(R[:, 2])
        return forces

        # forces[:, 2] = self.force_1d(R[:, 2])
        # r2 = R[:, 0]**2 + R[:, 1]**2
        # r2 = r2[:, np.newaxis]
        # RxyHat = np.where(r2 > 1.0e-8, R[:, 0:2] / np.sqrt(r2), np.zeros_like(R[:, 0:2]))
        # forces[:, 0:2] = np.where(r2 > self.r**2, -2.0 * self.k * np.sqrt(r2 - self.r**2) * RxyHat, 0.0)
        # return forces


def waiting_times(t_arr, z_counts_hist):
    from scipy.stats import rv_histogram
    delta_counts = np.diff(z_counts_hist)
    mid_times = (t_arr[:-1] + t_arr[1:])/2.0
    jump_steps = np.nonzero(delta_counts)
    jump_weights = delta_counts[jump_steps]
    jump_times = mid_times[jump_steps]

    hist, bins = np.histogram(jump_times, bins=t_arr, weights=jump_weights, density=True)
    dist = rv_histogram((hist, bins))
    print("AVERAGE JUMP TIME:\n\t{} ps".format(dist.mean()))
    print("  STDEV:\n    {} ps".format(dist.std()))
    print("MEDIAN JUMP TIME:\n\t{} ps".format(dist.median()))

def run_channel_langevin(mass, gamma, temp, channel_length, channel_radius,
                         channel_k, dG_barrier, barrier_width, z_constr=None,
                         num_steps=200_000, record_steps=10, acp_len=500, ensemble_size=100,
                         barrier_position=0.0,
                         wall_r=None, wall_k=None, sample_size=None,
                         plot_pmf=None, plot_traj=None, plot_ac=None,
                         k_constr=0.0, dt=0.001, outfile="channel_ld.txt", pmffile="channel_pmf.txt",
                         acfile="channel_ac.txt", trajfile="channel_z_trajs.txt"):

    if wall_r is None:
        wall_r = channel_radius
    if wall_k is None:
        wall_k = channel_k
    if sample_size is None:
        sample_size = max(1, ensemble_size//10)

    def eval_constraint_potential_1d(z):
        if z_constr is not None:
            return k_constr * (z - z_constr)**2
        else:
            return np.zeros_like(z)

    def eval_constraint_force(R):
        if z_constr is not None:
            return -2.0 * k_constr * (R - np.asarray([0.0, 0.0, z_constr]))
        else:
            return np.zeros_like(R)

    channel = Channel(channel_length, channel_radius, channel_k, channel_radius, wall_k, dG_barrier,
                      barrier_width, barrier_position)
    if plot_pmf is not None:
        zrange = np.linspace(-1.5 * channel_length / 2.0, 1.5 * channel_length / 2.0, 101)
        fz = channel.potential_1d(zrange)*10.0/4.184
        f_constr_z = fz + (eval_constraint_potential_1d(zrange)
                           - eval_constraint_potential_1d(channel.zmin))*10.0/4.184
        fig, ax = plt.subplots()
        ax.plot(zrange, fz, label="PMF")
        ax.plot(zrange, f_constr_z, label="PMF + Pull")
        ax.set_xlabel(r'$z\, \mathrm{(\AA)}$')
        ax.set_ylabel(r'$\Delta G\, \mathrm{(kcal/mol)}$')
        ax.legend()
        plt.savefig(plot_pmf)
        np.savetxt(pmffile, np.transpose([zrange, fz, f_constr_z]), fmt='%8.6f',
                   header='z    PMF    PMF_Pull')
        print("Saved PMF:\n\t{}\n\t{}".format(pmffile, plot_pmf))

    def eval_force(R):
        constr_force = eval_constraint_force(R)
        return channel.channel_force(R) + constr_force

    x0 = np.zeros([ensemble_size, 3])
    p0 = np.zeros([ensemble_size, 3])
    x0[:, 2] = channel.zmin
    mass = np.asarray(mass)
    g = np.asarray(gamma)
    solver = LangevinSolver(x0, p0, mass, g, temp, dt, eval_force)
    solver.reset_p()
    print("Wiener increment norm |dW| = sqrt(2*k_b*T*g*dt) = {}".format(solver.wnorm))
    if np.any(solver.wnorm > 0.5):
        print("\tWARNING: |dW| > 0.5 may be unstable")
    m = num_steps // record_steps
    t_arr = np.zeros([m])
    p_hist = np.zeros([m, ensemble_size, 3])
    x_hist = np.zeros([m, sample_size, 3])
    x_avg = np.zeros([m, 3])
    z_counts_hist = np.zeros([m, 2])
    ti = 0  # time step
    ri = 0  # record step
    for i in range(num_steps):
        if ti % record_steps == 0:
            t = i * dt
            t_arr[ri] = t
            x_avg[ri] = np.mean(solver.x, axis=0)
            p_hist[ri] = solver.p[:]
            x_hist[ri] = solver.x[:sample_size]
            z_counts = np.asarray([np.sum(solver.x[:, 2] < 0.0), np.sum(solver.x[:, 2] >= 0.0)])
            z_counts_hist[ri] = z_counts
            ri += 1
        ti += 1
        solver.step()

    # Evaluate momentum autocorrelation
    # (Same as velocity autocorrelation for a single particle)
    acpz = momentum_autocorrelation(p_hist[:, :, 2], acp_len=acp_len)
    mean_acp = np.mean(acpz, axis=1)
    acp0 = np.copy(mean_acp[0])
    for i in range(acp_len):
        mean_acp[i] /= acp0

    if plot_traj is not None:
        fig, ax = plt.subplots()
        ax.plot(t_arr, x_avg[:, 2], label=r'$\langle z \rangle$')
        ax.plot(t_arr, x_hist[:, 0, 2], label=r'$1$')
        ax.plot(t_arr, x_hist[:, 1, 2], label=r'$2$')
        ax.plot(t_arr, x_hist[:, 2, 2], label=r'$3$')
        ax.plot(t_arr, x_hist[:, 3, 2], label=r'$4$')
        ax.set_xlabel(r'Time $\mathrm{(ps)}$')
        ax.set_ylabel(r'$z\,\mathrm{(\AA)}$')
        ax.legend(loc='lower right')
        plt.savefig(plot_traj)
        print("Trajectory plot:\n\t{}".format(plot_traj))

    if plot_ac is not None:
        fig, ax = plt.subplots()
        ax.plot(t_arr[:acp_len], mean_acp)
        ax.set_xlim([0.0, 2.0])
        plt.savefig(plot_ac)

    waiting_times(t_arr, z_counts_hist[:, 1])
    np.savetxt(outfile, np.transpose([t_arr, x_avg[:, 0], x_avg[:, 1], x_avg[:, 2],
                                      z_counts_hist[:, 0], z_counts_hist[:, 1]]), fmt='%8.6f',
               header='t       <x>      <y>     <z>     nlo     nhi')
    np.savetxt(acfile, np.transpose([t_arr[:acp_len], mean_acp]), fmt='%8.6f', header='tau    C_vz(tau)')
    np.savetxt(trajfile, np.hstack((t_arr[:, np.newaxis], x_hist[:, :, 2])), fmt='%8.6f')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=
                                     "Runs a Channel Langevin Dynamics simulation")
    parser.add_argument("--name", default="channel")
    # parser.add_argument("--plot-pmf", help="Plot the PMF profile to pdf file", default=None)
    parser.add_argument("--plot", action='store_true', help="Generate plots to output PDF files")

    pmf = parser.add_argument_group("PMF arguments")
    sim = parser.add_argument_group("Simulation arguments")

    pmf.add_argument("--dg-barrier", type=float, help="PMF Energy barrier (kcal/mol)", default=20.0)
    pmf.add_argument("--barrier-width", help="PMF barrier width (Å)", type=float, default=2.0)
    pmf.add_argument("--barrier-shift", help="PMF barrier position", type=float, default=0.0)

    pmf.add_argument("--constr-k", nargs='?', type=float, help="Harmonic constraint of pulling force (kcal/mol)",
                        default=None, const=0.0001)
    pmf.add_argument("--constr-z", type=float, default=20000.0, help="z center of pulling force (Å)")

    sim.add_argument("--temp", type=float, help="Temperature in K", metavar='T', default=300.0)
    sim.add_argument("--gamma", type=float, default=8.0, help="Friction constant in ps^-1")
    sim.add_argument("--mass", type=float, default=23.0, help="Particle mass in AMU")
    sim.add_argument("--ensemble-size", type=int, default=100, help="Number of simultaneous particles")
    sim.add_argument("--sample-size", nargs='?', type=int, default=None, const=10,
                        help="Number of example trajectories (default is ensemble_size/2)")
    sim.add_argument("--dt", type=float, help="Time step size (default=0.001)", default=0.001)
    sim.add_argument("--num-steps", type=int, help="Number of steps to run", default=100_000)
    sim.add_argument("--record-steps", type=int, help="Steps per record", default=10)
    sim.add_argument("--vac-steps", type=int, help="Maximum record steps for velocity autocorrelation", default=500)

    geometry = parser.add_argument_group("Geometry arguments")
    geometry.add_argument("--channel-length", type=float, help="Length of the channel (Å)", default=15.0)
    geometry.add_argument("--channel-radius", type=float, help="Radius of the channel (Á)", default=5.0)
    geometry.add_argument("--wall-r", type=float, help="Radius of pore wall", default=5.0)
    geometry.add_argument("--channel-k", type=float, help="Harmonic constraint of channel wall", default=5.0)
    geometry.add_argument("--wall-k", type=float, help="Harmonic constraint of pore wall", default=5.0)

    args = parser.parse_args()

    # convert amu to kg/mol
    mass = args.mass * 0.001
    # convert from kcal/mol to natural units of energy [(kg/mol) A^2/ps^2]
    temp = args.temp * 0.001985875 * 4.184 * 0.1
    dg_barrier = args.dg_barrier * 4.184 * 0.1
    constr_k = args.constr_k * 4.184 * 0.1 if args.constr_k is not None else None
    channel_k = args.channel_k * 4.184 * 0.1
    wall_k = args.wall_k * 4.184 * 0.1
    run_channel_langevin(mass, args.gamma, temp, args.channel_length, args.channel_radius, channel_k,
                         dg_barrier, args.barrier_width, z_constr=args.constr_z, k_constr=constr_k, wall_r=args.wall_r,
                         ensemble_size=args.ensemble_size, wall_k=wall_k, sample_size=args.sample_size,
                         barrier_position=args.barrier_shift,
                         dt=args.dt, num_steps=args.num_steps, record_steps=args.record_steps, acp_len=args.vac_steps,
                         outfile=args.name+"_ld.txt", acfile=args.name+"_ac.txt", trajfile=args.name+"_z_trajs.txt",
                         pmffile=args.name+"_pmf.txt",
                         plot_pmf=args.name+"_pmf.pdf" if args.plot else None,
                         plot_traj=args.name+"_z_trajs.pdf" if args.plot else None,
                         plot_ac=args.name+"_ac.pdf" if args.plot else None)


def test_channel():
    channel = Channel(15.0, 5.0, 5.0, 20.0, 2.0)
    x = np.asarray([[0.0, 0.0, -16.0], [0.0, 1.0, -10.0],
                    [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 16.0 ] ]).transpose()
    energies = channel.channel_energy(x)
    forces = channel.channel_force(x)
