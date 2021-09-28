import numpy as np


class LangevinSolver:
    """
    Stepping class for generic Langevin dynamics
    """
    def __init__(self, x0, p0, m, g, temp, dt, f, L=None):
        self.m = m
        self.g = g
        self.temp = temp
        self.f = f
        self.sh = np.shape(x0)
        self.x = x0
        self.p = p0
        self.dt = dt
        self.sqrtdt = np.sqrt(dt)
        self.wnorm = np.sqrt(2.0 * self.g * self.m * self.temp * dt)
        self.L = 1.0 if L is None else L

    def step(self):
        dxdt = self.p / self.m
        dw = np.random.standard_normal(self.sh) * self.wnorm / self.L
        dpdt = self.f(self.x) - (self.p * self.g)

        dx = dxdt * self.dt
        dp = dpdt * self.dt + dw

        self.x += dx
        self.p += dp

    def reset_p(self):
        """
        randomly distribute all momenta according to the Maxwell-Boltzmann distribution
        at temperature T
        :return:
        """
        self.p = np.random.standard_normal(self.sh) * np.sqrt(self.m * self.temp) / self.L


def momentum_autocorrelation(p0: np.ndarray, acp_len=None):
    """

    :param p0: Momentum history  [N, ...]
    :return: acp: [T, ...]
    """
    p0sh = np.shape(p0)
    sh = p0sh if acp_len is None else (acp_len,) + p0sh[1:]
    T = sh[0]
    N = p0sh[0]
    acp = np.zeros(sh)
    for t0 in range(N):
        for tau in range(T-t0):
            p1p2 = p0[t0] * p0[t0 + tau]
            acp[tau] += p1p2

    for tau in range(T):
        acp[tau] /= (T-tau)
    return acp


def test_langevin_solver():
    print("\n Testing Langevin Solver (Harmonic Oscillator)...")

    ensemble = 1000
    x0 = np.zeros([ensemble, 2])
    x0[:, 0] = 1.0
    x0[:, 1] = 1.5
    p0 = np.zeros([ensemble, 2])
    g = 2.0
    temp = 1.2
    k = 1.0
    m = np.asarray([[1.0, 2.0]])
    dt = 0.03
    nsteps = 400_000

    def f_harmonic(x):
        return - k * x

    solver = LangevinSolver(x0, p0, m, g, temp, dt, f_harmonic)
    print("|dW| = {}".format(solver.wnorm))
    solver.reset_p()
    for _ in range(nsteps):
        solver.step()

    x = solver.x
    p = solver.p
    x2 = x**2
    p2 = p**2
    mean_x2 = np.mean(x2, axis=0)
    mean_p2 = np.mean(p2, axis=0)
    mean_t = mean_p2/(2.0*m)
    mean_u = 0.5 * k * mean_x2
    mean_e = mean_t + mean_u
    print("statistics:")
    print("x^2: {}".format(mean_x2))
    print("p^2: {}".format(mean_p2))
    print("T : {}".format(mean_t))
    print("U : {}".format(mean_u))
    print("E: {}".format(mean_e))
    print("Expected energy: {}".format(temp))