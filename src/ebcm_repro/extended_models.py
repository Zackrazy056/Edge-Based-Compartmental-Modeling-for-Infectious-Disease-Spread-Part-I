from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

from .distributions import ActualDegreePGF, ExpectedDegreePGF


@dataclass(frozen=True)
class ModelSolution:
    t: np.ndarray
    S: np.ndarray
    I: np.ndarray
    R: np.ndarray
    aux: dict[str, np.ndarray]

    @property
    def cumulative(self) -> np.ndarray:
        return 1.0 - self.S


def pmf_degree_pgf(degree_probs: dict[int, float]) -> ActualDegreePGF:
    ks = np.array(sorted(degree_probs.keys()), dtype=float)
    probs = np.array([degree_probs[int(k)] for k in ks], dtype=float)
    probs = probs / np.sum(probs)
    mean_k = float(np.sum(ks * probs))

    def _as_arr(x: np.ndarray | float) -> np.ndarray:
        return np.asarray(x, dtype=float)

    def psi(x: np.ndarray | float) -> np.ndarray | float:
        a = _as_arr(x)
        out = np.sum(probs * np.power(a[..., None], ks), axis=-1)
        return out if isinstance(x, np.ndarray) else float(out)

    def dpsi(x: np.ndarray | float) -> np.ndarray | float:
        a = _as_arr(x)
        out = np.sum(probs * ks * np.power(a[..., None], np.maximum(ks - 1.0, 0.0)), axis=-1)
        return out if isinstance(x, np.ndarray) else float(out)

    def ddpsi(x: np.ndarray | float) -> np.ndarray | float:
        a = _as_arr(x)
        out = np.sum(
            probs * ks * np.maximum(ks - 1.0, 0.0) * np.power(a[..., None], np.maximum(ks - 2.0, 0.0)),
            axis=-1,
        )
        return out if isinstance(x, np.ndarray) else float(out)

    return ActualDegreePGF(psi=psi, dpsi=dpsi, ddpsi=ddpsi, mean_degree=mean_k)


def _theta_from_i0(func: Callable[[float], float], i0: float) -> float:
    target_s = 1.0 - i0
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        val = func(mid)
        if val > target_s:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _solve(rhs: Callable[[float, np.ndarray], np.ndarray], y0: np.ndarray, t_max: float, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    t_eval = np.linspace(0.0, t_max, n_steps)
    sol = solve_ivp(rhs, (0.0, t_max), y0, t_eval=t_eval, method="RK45", rtol=1e-7, atol=1e-9)
    return sol.t, sol.y


def solve_ad_mfsh(
    pgf: ActualDegreePGF,
    beta: float,
    gamma: float,
    t_max: float,
    n_steps: int = 1601,
    i0: float = 1e-4,
) -> ModelSolution:
    theta0 = _theta_from_i0(pgf.psi, i0)

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        theta = float(np.clip(y[0], 1e-12, 1.0))
        r = float(np.clip(y[1], 0.0, 1.0))
        s = pgf.psi(theta)
        i = 1.0 - s - r
        dtheta = -beta * theta + beta * theta * theta * pgf.dpsi(theta) / pgf.mean_degree - theta * gamma * np.log(theta)
        dr = gamma * i
        return np.array([dtheta, dr], dtype=float)

    t, y = _solve(rhs, np.array([theta0, 0.0], dtype=float), t_max=t_max, n_steps=n_steps)
    theta, r = y
    s = pgf.psi(theta)
    i = np.clip(1.0 - s - r, 0.0, 1.0)
    return ModelSolution(t=t, S=np.clip(s, 0.0, 1.0), I=i, R=np.clip(r, 0.0, 1.0), aux={"theta": theta})


def solve_dfd(
    pgf: ActualDegreePGF,
    beta: float,
    gamma: float,
    eta: float,
    t_max: float,
    n_steps: int = 2001,
    i0: float = 1e-4,
) -> ModelSolution:
    theta0 = _theta_from_i0(pgf.psi, i0)
    pi_s0 = theta0 * pgf.dpsi(theta0) / pgf.mean_degree
    pi_r0 = 0.0
    pi_i0 = max(1.0 - pi_s0 - pi_r0, 1e-12)
    phi_s0 = min(theta0, pi_s0)
    phi_i0 = max(theta0 - phi_s0, 1e-12) + 0.5 * i0

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        theta, phi_s, phi_i, pi_r, r = y
        theta = float(np.clip(theta, 1e-12, 1.0))
        phi_s = float(max(phi_s, 0.0))
        phi_i = float(max(phi_i, 0.0))
        pi_r = float(np.clip(pi_r, 0.0, 1.0))
        psi_p = max(float(pgf.dpsi(theta)), 1e-12)
        pi_s = theta * pgf.dpsi(theta) / pgf.mean_degree
        pi_i = 1.0 - pi_s - pi_r
        infect_flux = beta * phi_i * phi_s * float(pgf.ddpsi(theta)) / psi_p
        dtheta = -beta * phi_i
        dphi_s = -infect_flux + eta * theta * pi_s - eta * phi_s
        dphi_i = infect_flux + eta * theta * pi_i - (beta + gamma + eta) * phi_i
        dpi_r = gamma * pi_i
        s = pgf.psi(theta)
        i = 1.0 - s - r
        dr = gamma * i
        return np.array([dtheta, dphi_s, dphi_i, dpi_r, dr], dtype=float)

    y0 = np.array([theta0, phi_s0, phi_i0, pi_r0, 0.0], dtype=float)
    t, y = _solve(rhs, y0, t_max=t_max, n_steps=n_steps)
    theta, _, _, _, r = y
    s = pgf.psi(theta)
    i = np.clip(1.0 - s - r, 0.0, 1.0)
    return ModelSolution(
        t=t,
        S=np.clip(s, 0.0, 1.0),
        I=i,
        R=np.clip(r, 0.0, 1.0),
        aux={"theta": theta, "phi_s": y[1], "phi_i": y[2], "pi_r": y[3]},
    )


def solve_dc(
    pgf: ActualDegreePGF,
    beta: float,
    gamma: float,
    eta1: float,
    eta2: float,
    t_max: float,
    n_steps: int = 2201,
    i0: float = 1e-4,
) -> ModelSolution:
    theta0 = _theta_from_i0(pgf.psi, i0)
    xi = eta1 / (eta1 + eta2)
    pi = eta2 / (eta1 + eta2)

    phi_d0 = pi * theta0
    active0 = max(theta0 - phi_d0, 0.0)
    phi_i0 = max(active0 * i0, 1e-12)
    phi_s0 = max(active0 - phi_i0, 1e-12)
    xi_r0 = 0.0
    pi_r0 = 0.0

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        theta, phi_s, phi_i, phi_d, xi_r, pi_r, r = y
        theta = float(np.clip(theta, 1e-12, 1.0))
        phi_s = float(max(phi_s, 0.0))
        phi_i = float(max(phi_i, 0.0))
        phi_d = float(np.clip(phi_d, 0.0, 1.0))
        xi_r = float(np.clip(xi_r, 0.0, 1.0))
        pi_r = float(np.clip(pi_r, 0.0, 1.0))
        psi_p = max(float(pgf.dpsi(theta)), 1e-12)

        xi_s = (theta - phi_d) * pgf.dpsi(theta) / pgf.mean_degree
        xi_i = xi - xi_s - xi_r
        pi_s = phi_d * pgf.dpsi(theta) / pgf.mean_degree
        pi_i = pi - pi_s - pi_r

        infect_flux = beta * phi_i * phi_s * float(pgf.ddpsi(theta)) / psi_p
        dtheta = -beta * phi_i
        dphi_s = -infect_flux + eta1 * (pi_s / pi) * phi_d - eta2 * phi_s
        dphi_i = infect_flux + eta1 * (pi_i / pi) * phi_d - (eta2 + beta + gamma) * phi_i
        dphi_d = eta2 * (theta - phi_d) - eta1 * phi_d
        dxi_r = -eta2 * xi_r + eta1 * pi_r + gamma * xi_i
        dpi_r = eta2 * xi_r - eta1 * pi_r + gamma * pi_i
        s = pgf.psi(theta)
        i = 1.0 - s - r
        dr = gamma * i
        return np.array([dtheta, dphi_s, dphi_i, dphi_d, dxi_r, dpi_r, dr], dtype=float)

    y0 = np.array([theta0, phi_s0, phi_i0, phi_d0, xi_r0, pi_r0, 0.0], dtype=float)
    t, y = _solve(rhs, y0, t_max=t_max, n_steps=n_steps)
    theta, _, _, _, _, _, r = y
    s = pgf.psi(theta)
    i = np.clip(1.0 - s - r, 0.0, 1.0)
    return ModelSolution(
        t=t,
        S=np.clip(s, 0.0, 1.0),
        I=i,
        R=np.clip(r, 0.0, 1.0),
        aux={
            "theta": theta,
            "phi_s": y[1],
            "phi_i": y[2],
            "phi_d": y[3],
            "xi_r": y[4],
            "pi_r": y[5],
        },
    )


def solve_mp(
    pgf: ExpectedDegreePGF,
    beta: float,
    gamma: float,
    t_max: float,
    n_steps: int = 1801,
    i0: float = 1e-4,
) -> ModelSolution:
    theta0 = _theta_from_i0(pgf.Psi, i0)

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        theta, r = y
        theta = float(np.clip(theta, 1e-12, 1.0))
        r = float(np.clip(r, 0.0, 1.0))
        s = pgf.Psi(theta)
        i = 1.0 - s - r
        dtheta = -beta * theta + beta * pgf.dPsi(theta) / pgf.mean_degree + gamma * (1.0 - theta)
        dr = gamma * i
        return np.array([dtheta, dr], dtype=float)

    t, y = _solve(rhs, np.array([theta0, 0.0], dtype=float), t_max=t_max, n_steps=n_steps)
    theta, r = y
    s = pgf.Psi(theta)
    i = np.clip(1.0 - s - r, 0.0, 1.0)
    return ModelSolution(t=t, S=np.clip(s, 0.0, 1.0), I=i, R=np.clip(r, 0.0, 1.0), aux={"Theta": theta})


def solve_ed_mfsh(
    pgf: ExpectedDegreePGF,
    beta: float,
    gamma: float,
    t_max: float,
    n_steps: int = 2201,
    i0: float = 1e-5,
) -> ModelSolution:
    theta0 = _theta_from_i0(pgf.Psi, i0)

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        theta, r = y
        theta = float(np.clip(theta, 1e-12, 1.0))
        r = float(np.clip(r, 0.0, 1.0))
        s = pgf.Psi(theta)
        i = 1.0 - s - r
        dtheta = -beta + beta * pgf.dPsi(theta) / pgf.mean_degree + gamma * (1.0 - theta)
        dr = gamma * i
        return np.array([dtheta, dr], dtype=float)

    t, y = _solve(rhs, np.array([theta0, 0.0], dtype=float), t_max=t_max, n_steps=n_steps)
    theta, r = y
    s = pgf.Psi(theta)
    i = np.clip(1.0 - s - r, 0.0, 1.0)
    return ModelSolution(t=t, S=np.clip(s, 0.0, 1.0), I=i, R=np.clip(r, 0.0, 1.0), aux={"Theta": theta})


def solve_dvd(
    pgf: ExpectedDegreePGF,
    beta: float,
    gamma: float,
    eta: float,
    t_max: float,
    n_steps: int = 1801,
    i0: float = 1e-4,
) -> ModelSolution:
    theta0 = _theta_from_i0(pgf.Psi, i0)
    pi_r0 = 0.0

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        theta, pi_r, r = y
        theta = float(np.clip(theta, 1e-12, 1.0))
        pi_r = float(np.clip(pi_r, 0.0, 1.0))
        r = float(np.clip(r, 0.0, 1.0))
        pi_s = pgf.dPsi(theta) / pgf.mean_degree
        pi_i = 1.0 - pi_s - pi_r
        dtheta = -beta * theta + beta * pi_s + gamma * (1.0 - theta) + eta * (1.0 - theta - (beta / gamma) * pi_r)
        dpi_r = gamma * pi_i
        s = pgf.Psi(theta)
        i = 1.0 - s - r
        dr = gamma * i
        return np.array([dtheta, dpi_r, dr], dtype=float)

    t, y = _solve(rhs, np.array([theta0, pi_r0, 0.0], dtype=float), t_max=t_max, n_steps=n_steps)
    theta, pi_r, r = y
    s = pgf.Psi(theta)
    i = np.clip(1.0 - s - r, 0.0, 1.0)
    return ModelSolution(
        t=t,
        S=np.clip(s, 0.0, 1.0),
        I=i,
        R=np.clip(r, 0.0, 1.0),
        aux={"Theta": theta, "Pi_R": pi_r},
    )
