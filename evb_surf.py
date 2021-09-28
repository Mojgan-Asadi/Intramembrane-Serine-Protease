import numpy as np
import scipy
# Minima in the q1 direction
d1 = np.asarray([[-1.0, -1.0], [1.0, 1.0]])
# Minima in the q2 direction
d2 = np.asarray([[-1.0, 1.0], [-1.0, 1.0]])
dmat = np.stack([d1, d2], axis=-1)  # [2, 2, 2]

def evb_diag(q, k, a, delta=1.0):
    """

    :param q: 2D coordinate vector [Q1, Q2]
    :param k: 2D frequency vector [w1, w2]
    :param a: [2, 2] energy_mesh minima
    :return: The diagonal EVB Hamiltonian and its gradient
       H_ii [2, 2],  dH_ii,q [2, 2, 2]
    """
    if np.isscalar(delta):
        delta = np.asarray([delta, delta])
    # Dimensions
    # [2, 2, 2] ~ [Q1 Surface Min, Q2 Surface Min, Q]
    dq = q - dmat*delta[np.newaxis, np.newaxis, :]
    hterms = 0.5 * k * (dq) ** 2
    grad = k * dq
    return np.sum(hterms, axis=-1) + a, grad


def const_evb_off_diag(h):
    """

    :param h: 6 off-diagonal elements [h12, h13, h14, h23, h24, h34]
    :return:
    """
    j = np.zeros([4, 4])
    j[0, 1] = h[0]
    j[0, 2] = h[1]
    j[0, 3] = h[2]
    j[1, 2] = h[3]
    j[1, 3] = h[4]
    j[2, 3] = h[5]
    j += np.transpose(j).copy()

    return j


def evb_matrix_forces(q, k, j, a, delta=1.0):
    """

    :param q: [2,] array (generalized coordinates)
    :param k: [2,] array (EVB frequencies)
    :param j: [4, 4] (Off-diagonal element matrix)
    :param a: [2, 2] (Gas shifts)
    :param delta: Scalar, or [2,] (EVB separation)
    :return:
      The EVB Hamiltonian matrix [4, 4] and its diabatic forces [4, 2]
      H_ij,  F_i,q
    """
    diag, grad = evb_diag(q, k, a, delta=delta)
    diag = np.reshape(diag, [-1])
    f = -np.reshape(grad, [-1, 2])
    return np.diag(np.reshape(diag, [-1])) + j, f


def make_energy_surface(w, J, a, delta=1.0):
    n = 31
    lo, hi = -2.0, 2.0
    q1range = np.linspace(lo, hi, n)
    q2range = np.linspace(lo, hi, n)
    dq = (hi - lo) / (n - 1)
    egrid = np.zeros([n, n])

    for i, q1 in enumerate(q1range):
        for j, q2 in enumerate(q2range):
            h, f = evb_matrix_forces(np.asarray([q1, q2]), w, J, a, delta=delta)
            vals, vecs = np.linalg.eigh(h)
            egrid[i, j] = vals[0]

    grad = np.gradient(egrid, dq)
    dfx = -grad[1]
    dfy = -grad[0]
    X, Y = np.meshgrid(q1range, q2range)
    return X, Y, egrid, [dfx, dfy]


class EvbSurface:
    def __init__(self, k, J, a, n=31, delta=1.0, lo=-2.5, hi=2.5, scale=1.0):
        from scipy.interpolate import interp2d

        if np.isscalar(n):
            n = np.asarray([n, n])
        if np.isscalar(delta):
            delta = np.asarray([delta, delta])
        self.scale=scale
        self.k = k
        self.J = J
        self.a = a

        self.n = n
        self.delta = delta
        self.lo_q1 = lo*delta[0]
        self.hi_q1 = hi*delta[0]
        self.lo_q2 = lo*delta[1]
        self.hi_q2 = hi*delta[1]

        q1range = np.linspace(self.lo_q1, self.hi_q1, n[0])
        q2range = np.linspace(self.lo_q2, self.hi_q2, n[1])
        self.x = q1range
        self.y = q2range
        dq1 = (self.hi_q1 - self.lo_q1) / (n[0] - 1)
        dq2 = (self.hi_q2 - self.lo_q2) / (n[1] - 1)
        egrid = np.zeros(n)
        for i, q1 in enumerate(q1range):
            for j, q2 in enumerate(q2range):
                h, f = evb_matrix_forces(np.asarray([q1, q2]), k, J, a, delta=delta)
                vals, vecs = np.linalg.eigh(h)
                egrid[i, j] = scale*vals[0]

        self.egrid = egrid
        grad = np.gradient(egrid, dq1, dq2)
        self.dfx = -grad[0]
        self.dfy = -grad[1]
        #self.X, self.Y = np.meshgrid(q1range, q2range, indexing='ij')
        # interp2d takes x for column index and y for row index
        self.e_interp = interp2d(self.x, self.y, self.egrid.transpose())
        self.fx_interp = interp2d(self.x, self.y, self.dfx.transpose())
        self.fy_interp = interp2d(self.x, self.y, self.dfy.transpose())

    def eval_evb(self, q):
        """
        Return the ground state energy, EVB density vector, and the EVB force
        """
        h, f = evb_matrix_forces(np.asarray(q), self.k, self.J, self.a, delta=self.delta)
        h = self.scale * h
        f = self.scale * f
        vals, vecs = np.linalg.eigh(h)
        e0 = vals[0]
        dens = vecs[:, 0]**2
        f = np.sum(dens[:, np.newaxis] * f, axis=0)
        return e0, dens, f

    def eval_diabatic(self, q):
        h, f = evb_matrix_forces(np.asarray(q), self.k, self.J, self.a, delta=self.delta)
        h = self.scale * h
        hdiag = np.diag(h)
        return hdiag

    def delta_mesh(self, n=51, scale=2.5):
        xgrid1 = np.linspace(-scale*self.delta[0], scale*self.delta[0], n)
        ygrid1 = np.linspace(-scale*self.delta[1], scale*self.delta[1], n)

        return np.meshgrid(xgrid1, ygrid1)

    def energy_mesh(self, q):
        """
        Evaluates the energy over an NxN rectangular mesh, where q has shape [2, N]
        :param q:
        :return:
        """
        n = np.shape(q[0])[0]
        energies = np.zeros([n, n])
        for i, q1 in enumerate(q[0]):
            for j, q2 in enumerate(q[1]):
                e0, _, _ = self.eval_evb(np.asarray([q1, q2]))
                energies[i, j] = e0
        return energies

    def diabatic(self, q: np.ndarray, k):
        sh = np.shape(q)
        energies = np.zeros(sh[0])
        for i, qi in enumerate(q):
            hdiag = self.eval_diabatic(qi)
            energies[i] = hdiag[k]
        return energies

    def energy(self, q: np.ndarray):
        """
        Evaluate the energy of a [N, 2] ensemble
        :param q: [N, 2] array
        :return: [N,] array
        """
        sh = np.shape(q)
        energies = np.zeros(sh[0])
        for i, qi in enumerate(q):
            e0, _, _ = self.eval_evb(qi)
            energies[i] = e0
        return energies

    def force(self, q):
        sh = np.shape(q)
        if sh[1] != 2:
            raise ValueError("EvbSurface.force: Expected [N, 2] array")
        forces = np.zeros(sh)
        for i, qi in enumerate(q):
            _, _, f = self.eval_evb(qi)
            forces[i, :] = f
        return forces

    def force_mesh(self, q):
        """
        Evaluates the force over an NxN rectangular mesh, where q has shape [2, N]
        :param q:
        :return:
        """
        #return [self.fx_interp(q[0], q[1]), self.fy_interp(q[0], q[1])]
        n = np.shape(q[0])[0]
        fx = np.zeros([n, n])
        fy = np.zeros([n, n])
        for i, q1 in enumerate(q[0]):
            for j, q2 in enumerate(q[1]):
                f = self.force(np.asarray([[q1, q2]]))
                fx[i,j] = f[0, 0]
                fy[i,j] = f[0, 1]
        return [fx, fy]


