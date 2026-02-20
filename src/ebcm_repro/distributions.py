from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


EPS = 1e-12


@dataclass(frozen=True)
class ActualDegreePGF:
    psi: Callable[[np.ndarray | float], np.ndarray | float]
    dpsi: Callable[[np.ndarray | float], np.ndarray | float]
    ddpsi: Callable[[np.ndarray | float], np.ndarray | float]
    mean_degree: float


@dataclass(frozen=True)
class ExpectedDegreePGF:
    Psi: Callable[[np.ndarray | float], np.ndarray | float]
    dPsi: Callable[[np.ndarray | float], np.ndarray | float]
    ddPsi: Callable[[np.ndarray | float], np.ndarray | float]
    mean_degree: float


def _as_arr(x: np.ndarray | float) -> np.ndarray:
    return np.asarray(x, dtype=float)


def regular_degree(k: int) -> ActualDegreePGF:
    def psi(x: np.ndarray | float) -> np.ndarray | float:
        a = _as_arr(x)
        out = np.power(a, k)
        return out if isinstance(x, np.ndarray) else float(out)

    def dpsi(x: np.ndarray | float) -> np.ndarray | float:
        a = _as_arr(x)
        out = k * np.power(a, k - 1)
        return out if isinstance(x, np.ndarray) else float(out)

    def ddpsi(x: np.ndarray | float) -> np.ndarray | float:
        a = _as_arr(x)
        out = k * (k - 1) * np.power(a, k - 2)
        return out if isinstance(x, np.ndarray) else float(out)

    return ActualDegreePGF(psi=psi, dpsi=dpsi, ddpsi=ddpsi, mean_degree=float(k))


def poisson_degree(mean_k: float) -> ActualDegreePGF:
    def psi(x: np.ndarray | float) -> np.ndarray | float:
        a = _as_arr(x)
        out = np.exp(mean_k * (a - 1.0))
        return out if isinstance(x, np.ndarray) else float(out)

    def dpsi(x: np.ndarray | float) -> np.ndarray | float:
        a = _as_arr(x)
        out = mean_k * np.exp(mean_k * (a - 1.0))
        return out if isinstance(x, np.ndarray) else float(out)

    def ddpsi(x: np.ndarray | float) -> np.ndarray | float:
        a = _as_arr(x)
        out = mean_k * mean_k * np.exp(mean_k * (a - 1.0))
        return out if isinstance(x, np.ndarray) else float(out)

    return ActualDegreePGF(psi=psi, dpsi=dpsi, ddpsi=ddpsi, mean_degree=float(mean_k))


def bimodal_degree(k1: int, k2: int, p1: float = 0.5) -> ActualDegreePGF:
    p2 = 1.0 - p1

    def psi(x: np.ndarray | float) -> np.ndarray | float:
        a = _as_arr(x)
        out = p1 * np.power(a, k1) + p2 * np.power(a, k2)
        return out if isinstance(x, np.ndarray) else float(out)

    def dpsi(x: np.ndarray | float) -> np.ndarray | float:
        a = _as_arr(x)
        out = p1 * k1 * np.power(a, k1 - 1) + p2 * k2 * np.power(a, k2 - 1)
        return out if isinstance(x, np.ndarray) else float(out)

    def ddpsi(x: np.ndarray | float) -> np.ndarray | float:
        a = _as_arr(x)
        out = p1 * k1 * (k1 - 1) * np.power(a, k1 - 2) + p2 * k2 * (k2 - 1) * np.power(a, k2 - 2)
        return out if isinstance(x, np.ndarray) else float(out)

    mean_k = p1 * k1 + p2 * k2
    return ActualDegreePGF(psi=psi, dpsi=dpsi, ddpsi=ddpsi, mean_degree=float(mean_k))


def truncated_powerlaw_degree(nu: float, cutoff: float, kmax: int = 300) -> ActualDegreePGF:
    ks = np.arange(1, kmax + 1, dtype=float)
    weights = np.power(ks, -nu) * np.exp(-ks / cutoff)
    probs = weights / np.sum(weights)
    mean_k = float(np.sum(ks * probs))

    def psi(x: np.ndarray | float) -> np.ndarray | float:
        a = _as_arr(x)
        terms = np.power(a[..., None], ks)
        out = np.sum(terms * probs, axis=-1)
        return out if isinstance(x, np.ndarray) else float(out)

    def dpsi(x: np.ndarray | float) -> np.ndarray | float:
        a = _as_arr(x)
        terms = ks * np.power(a[..., None], ks - 1.0)
        out = np.sum(terms * probs, axis=-1)
        return out if isinstance(x, np.ndarray) else float(out)

    def ddpsi(x: np.ndarray | float) -> np.ndarray | float:
        a = _as_arr(x)
        terms = ks * (ks - 1.0) * np.power(a[..., None], ks - 2.0)
        out = np.sum(terms * probs, axis=-1)
        return out if isinstance(x, np.ndarray) else float(out)

    return ActualDegreePGF(psi=psi, dpsi=dpsi, ddpsi=ddpsi, mean_degree=mean_k)


def nb_degree_pgf(r: float, p: float) -> ActualDegreePGF:
    # P(k)=C(k+r-1,k) (1-p)^r p^k
    c = 1.0 - p

    def psi(x: np.ndarray | float) -> np.ndarray | float:
        a = _as_arr(x)
        out = np.power(c / (1.0 - p * a), r)
        return out if isinstance(x, np.ndarray) else float(out)

    def dpsi(x: np.ndarray | float) -> np.ndarray | float:
        a = _as_arr(x)
        out = r * p * np.power(c / (1.0 - p * a), r) / (1.0 - p * a)
        return out if isinstance(x, np.ndarray) else float(out)

    def ddpsi(x: np.ndarray | float) -> np.ndarray | float:
        a = _as_arr(x)
        out = r * (r + 1.0) * (p ** 2) * np.power(c / (1.0 - p * a), r) / np.power(1.0 - p * a, 2)
        return out if isinstance(x, np.ndarray) else float(out)

    mean_k = r * p / c
    return ActualDegreePGF(psi=psi, dpsi=dpsi, ddpsi=ddpsi, mean_degree=float(mean_k))


def mp_piecewise_uniform_expected() -> ExpectedDegreePGF:
    # rho(k)=1/4 on [0,2], 1/20 on [10,20], 0 otherwise
    intervals = [
        (0.0, 2.0, 0.25),
        (10.0, 20.0, 0.05),
    ]
    mean_k = sum(d * 0.5 * (b * b - a * a) for a, b, d in intervals)

    def _expint0(u: np.ndarray, a: float, b: float) -> np.ndarray:
        out = np.empty_like(u)
        small = np.abs(u) < 1e-8
        out[small] = b - a
        uu = u[~small]
        out[~small] = (np.exp(b * uu) - np.exp(a * uu)) / uu
        return out

    def _expint1(u: np.ndarray, a: float, b: float) -> np.ndarray:
        out = np.empty_like(u)
        small = np.abs(u) < 1e-8
        out[small] = 0.5 * (b * b - a * a)
        uu = u[~small]
        out[~small] = (
            np.exp(b * uu) * (b / uu - 1.0 / (uu * uu))
            - np.exp(a * uu) * (a / uu - 1.0 / (uu * uu))
        )
        return out

    def _expint2(u: np.ndarray, a: float, b: float) -> np.ndarray:
        out = np.empty_like(u)
        small = np.abs(u) < 1e-8
        out[small] = (b ** 3 - a ** 3) / 3.0
        uu = u[~small]
        out[~small] = (
            np.exp(b * uu) * (b * b / uu - 2.0 * b / (uu * uu) + 2.0 / (uu ** 3))
            - np.exp(a * uu) * (a * a / uu - 2.0 * a / (uu * uu) + 2.0 / (uu ** 3))
        )
        return out

    def Psi(x: np.ndarray | float) -> np.ndarray | float:
        a = _as_arr(x)
        u = a - 1.0
        total = np.zeros_like(a)
        for lo, hi, dens in intervals:
            total += dens * _expint0(u, lo, hi)
        return total if isinstance(x, np.ndarray) else float(total)

    def dPsi(x: np.ndarray | float) -> np.ndarray | float:
        a = _as_arr(x)
        u = a - 1.0
        total = np.zeros_like(a)
        for lo, hi, dens in intervals:
            total += dens * _expint1(u, lo, hi)
        return total if isinstance(x, np.ndarray) else float(total)

    def ddPsi(x: np.ndarray | float) -> np.ndarray | float:
        a = _as_arr(x)
        u = a - 1.0
        total = np.zeros_like(a)
        for lo, hi, dens in intervals:
            total += dens * _expint2(u, lo, hi)
        return total if isinstance(x, np.ndarray) else float(total)

    return ExpectedDegreePGF(Psi=Psi, dPsi=dPsi, ddPsi=ddPsi, mean_degree=float(mean_k))


def edmfs_h_expected() -> ExpectedDegreePGF:
    # rho(k)=exp(k)/(exp(3)-1) for k in (0,3)
    norm = np.exp(3.0) - 1.0

    def Psi(x: np.ndarray | float) -> np.ndarray | float:
        a = _as_arr(x)
        out = (np.exp(3.0 * a) - 1.0) / (np.maximum(a, EPS) * norm)
        # Stable limit as x->0 is 3/norm.
        close = np.abs(a) < 1e-8
        if np.any(close):
            out = np.array(out, copy=True)
            out[close] = 3.0 / norm
        return out if isinstance(x, np.ndarray) else float(out)

    def dPsi(x: np.ndarray | float) -> np.ndarray | float:
        a = _as_arr(x)
        out = (3.0 * np.exp(3.0 * a) * a - (np.exp(3.0 * a) - 1.0)) / (np.maximum(a, EPS) ** 2 * norm)
        close = np.abs(a) < 1e-6
        if np.any(close):
            out = np.array(out, copy=True)
            # Series around 0.
            out[close] = 9.0 / (2.0 * norm)
        return out if isinstance(x, np.ndarray) else float(out)

    def ddPsi(x: np.ndarray | float) -> np.ndarray | float:
        a = _as_arr(x)
        denom = np.maximum(a, EPS) ** 3 * norm
        num = np.exp(3.0 * a) * (9.0 * a * a - 6.0 * a + 2.0) - 2.0
        out = num / denom
        close = np.abs(a) < 1e-4
        if np.any(close):
            out = np.array(out, copy=True)
            out[close] = 9.0 / norm
        return out if isinstance(x, np.ndarray) else float(out)

    mean_k = float((2.0 * np.exp(3.0) + 1.0) / (np.exp(3.0) - 1.0))
    return ExpectedDegreePGF(Psi=Psi, dPsi=dPsi, ddPsi=ddPsi, mean_degree=mean_k)


def dvd_expected() -> ExpectedDegreePGF:
    # Psi(x) = (2 / (3 - exp(x-1)))^4
    def Psi(x: np.ndarray | float) -> np.ndarray | float:
        a = _as_arr(x)
        y = np.exp(a - 1.0)
        out = np.power(2.0 / (3.0 - y), 4)
        return out if isinstance(x, np.ndarray) else float(out)

    def dPsi(x: np.ndarray | float) -> np.ndarray | float:
        a = _as_arr(x)
        y = np.exp(a - 1.0)
        out = 64.0 * y / np.power(3.0 - y, 5)
        return out if isinstance(x, np.ndarray) else float(out)

    def ddPsi(x: np.ndarray | float) -> np.ndarray | float:
        a = _as_arr(x)
        y = np.exp(a - 1.0)
        out = 64.0 * y * (3.0 + 4.0 * y) / np.power(3.0 - y, 6)
        return out if isinstance(x, np.ndarray) else float(out)

    return ExpectedDegreePGF(Psi=Psi, dPsi=dPsi, ddPsi=ddPsi, mean_degree=2.0)

