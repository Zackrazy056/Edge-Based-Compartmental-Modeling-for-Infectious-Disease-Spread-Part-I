from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .distributions import ActualDegreePGF


@dataclass(frozen=True)
class CMSolution:
    t: np.ndarray
    theta: np.ndarray
    S: np.ndarray
    I: np.ndarray
    R: np.ndarray

    @property
    def cumulative(self) -> np.ndarray:
        return 1.0 - self.S


@dataclass(frozen=True)
class MassActionSolution:
    t: np.ndarray
    S: np.ndarray
    I: np.ndarray
    R: np.ndarray

    @property
    def cumulative(self) -> np.ndarray:
        return 1.0 - self.S


def empirical_degree_pgf(degrees: np.ndarray) -> ActualDegreePGF:
    deg = np.asarray(degrees, dtype=int)
    if deg.ndim != 1:
        raise ValueError("degrees must be a 1-D array")
    if np.any(deg < 0):
        raise ValueError("degrees must be non-negative")
    if deg.size == 0:
        raise ValueError("degrees must not be empty")

    ks, counts = np.unique(deg, return_counts=True)
    probs = counts / counts.sum()
    mean_k = float(np.sum(ks * probs))

    def _as_arr(x: np.ndarray | float) -> np.ndarray:
        return np.asarray(x, dtype=float)

    def psi(x: np.ndarray | float) -> np.ndarray | float:
        a = _as_arr(x)
        out = np.sum(probs * np.power(a[..., None], ks), axis=-1)
        return out if isinstance(x, np.ndarray) else float(out)

    def dpsi(x: np.ndarray | float) -> np.ndarray | float:
        a = _as_arr(x)
        out = np.sum(probs * ks * np.power(a[..., None], np.maximum(ks - 1, 0)), axis=-1)
        return out if isinstance(x, np.ndarray) else float(out)

    def ddpsi(x: np.ndarray | float) -> np.ndarray | float:
        a = _as_arr(x)
        kfac = ks * np.maximum(ks - 1, 0)
        out = np.sum(probs * kfac * np.power(a[..., None], np.maximum(ks - 2, 0)), axis=-1)
        return out if isinstance(x, np.ndarray) else float(out)

    return ActualDegreePGF(psi=psi, dpsi=dpsi, ddpsi=ddpsi, mean_degree=mean_k)


def solve_cm(
    pgf: ActualDegreePGF,
    beta: float,
    gamma: float,
    t_max: float,
    n_steps: int = 1201,
    theta0: float = 1.0 - 1e-6,
    r0: float = 0.0,
) -> CMSolution:
    if n_steps < 2:
        raise ValueError("n_steps must be at least 2")

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        theta, r = y
        s = pgf.psi(theta)
        i = 1.0 - s - r
        dtheta = -beta * theta + beta * pgf.dpsi(theta) / pgf.mean_degree + gamma * (1.0 - theta)
        dr = gamma * i
        return np.array([dtheta, dr], dtype=float)

    t_eval = np.linspace(0.0, t_max, n_steps)
    sol = solve_ivp(
        rhs,
        t_span=(0.0, t_max),
        y0=np.array([theta0, r0], dtype=float),
        t_eval=t_eval,
        method="RK45",
        rtol=1e-7,
        atol=1e-9,
    )
    theta = sol.y[0]
    r = sol.y[1]
    s = pgf.psi(theta)
    i = 1.0 - s - r
    i = np.clip(i, 0.0, 1.0)
    s = np.clip(s, 0.0, 1.0)
    r = np.clip(r, 0.0, 1.0)
    return CMSolution(t=sol.t, theta=theta, S=s, I=i, R=r)


def solve_mass_action(
    beta_hat: float,
    gamma: float,
    t_max: float,
    i0: float,
    r0: float = 0.0,
    n_steps: int = 1201,
) -> MassActionSolution:
    if n_steps < 2:
        raise ValueError("n_steps must be at least 2")
    if i0 <= 0 or i0 >= 1:
        raise ValueError("i0 must be in (0, 1)")

    s0 = 1.0 - i0 - r0

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        s, i, r = y
        ds = -beta_hat * s * i
        di = beta_hat * s * i - gamma * i
        dr = gamma * i
        return np.array([ds, di, dr], dtype=float)

    t_eval = np.linspace(0.0, t_max, n_steps)
    sol = solve_ivp(
        rhs,
        t_span=(0.0, t_max),
        y0=np.array([s0, i0, r0], dtype=float),
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )
    s, i, r = sol.y
    s = np.clip(s, 0.0, 1.0)
    i = np.clip(i, 0.0, 1.0)
    r = np.clip(r, 0.0, 1.0)
    return MassActionSolution(t=sol.t, S=s, I=i, R=r)
