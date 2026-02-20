from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SimulationResult:
    t: np.ndarray
    S: np.ndarray
    I: np.ndarray
    R: np.ndarray

    @property
    def cumulative(self) -> np.ndarray:
        return 1.0 - self.S


def sample_degree_sequence(
    name: str,
    n_nodes: int,
    rng: np.random.Generator,
    truncated_powerlaw_nu: float = 1.418,
    truncated_powerlaw_cutoff: float = 40.0,
    truncated_powerlaw_kmax: int = 400,
) -> np.ndarray:
    if n_nodes <= 0:
        raise ValueError("n_nodes must be positive")

    key = name.lower()
    if key == "homogeneous":
        deg = np.full(n_nodes, 5, dtype=np.int32)
    elif key == "poisson":
        deg = rng.poisson(5.0, size=n_nodes).astype(np.int32)
    elif key == "bimodal":
        deg = rng.choice(np.array([2, 8], dtype=np.int32), size=n_nodes, p=np.array([0.5, 0.5]))
    elif key == "truncated_powerlaw":
        ks = np.arange(1, truncated_powerlaw_kmax + 1, dtype=np.int32)
        w = np.power(ks.astype(float), -truncated_powerlaw_nu) * np.exp(-ks / truncated_powerlaw_cutoff)
        p = w / np.sum(w)
        deg = rng.choice(ks, size=n_nodes, p=p).astype(np.int32)
    else:
        raise ValueError(f"unknown degree distribution: {name}")

    if np.sum(deg, dtype=np.int64) % 2 == 1:
        idx = int(rng.integers(0, n_nodes))
        deg[idx] += 1
    return deg


def build_configuration_model_edges(degrees: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    deg = np.asarray(degrees, dtype=np.int32)
    if deg.ndim != 1:
        raise ValueError("degrees must be 1-D")
    if np.any(deg < 0):
        raise ValueError("degrees must be non-negative")

    stubs = np.repeat(np.arange(deg.size, dtype=np.int32), deg)
    if stubs.size % 2 == 1:
        stubs = stubs[:-1]
    rng.shuffle(stubs)
    u = stubs[0::2].copy()
    v = stubs[1::2].copy()
    return u, v


def run_sir_tauleap(
    edges_u: np.ndarray,
    edges_v: np.ndarray,
    n_nodes: int,
    beta: float,
    gamma: float,
    t_max: float,
    dt: float,
    initial_infected: int,
    rng: np.random.Generator,
) -> SimulationResult:
    if n_nodes <= 0:
        raise ValueError("n_nodes must be positive")
    if initial_infected <= 0 or initial_infected >= n_nodes:
        raise ValueError("initial_infected must satisfy 0 < initial_infected < n_nodes")
    if dt <= 0:
        raise ValueError("dt must be positive")

    u = np.asarray(edges_u, dtype=np.int32)
    v = np.asarray(edges_v, dtype=np.int32)
    if u.shape != v.shape:
        raise ValueError("edges_u and edges_v must have same shape")

    status = np.zeros(n_nodes, dtype=np.int8)
    seed_nodes = rng.choice(n_nodes, size=initial_infected, replace=False)
    status[seed_nodes] = 1

    s_count = n_nodes - initial_infected
    i_count = initial_infected
    r_count = 0

    p_recover = 1.0 - np.exp(-gamma * dt)

    t_vals = [0.0]
    s_vals = [s_count / n_nodes]
    i_vals = [i_count / n_nodes]
    r_vals = [0.0]

    t = 0.0
    while t < t_max and i_count > 0:
        infected = status == 1
        susceptible = status == 0

        uv = infected[u] & susceptible[v]
        vu = infected[v] & susceptible[u]
        pressure = np.bincount(v[uv], minlength=n_nodes) + np.bincount(u[vu], minlength=n_nodes)

        sus_idx = np.flatnonzero(susceptible)
        if sus_idx.size > 0:
            lambda_i = beta * dt * pressure[sus_idx]
            p_inf = 1.0 - np.exp(-lambda_i)
            inf_draw = rng.random(sus_idx.size) < p_inf
            new_inf = sus_idx[inf_draw]
        else:
            new_inf = np.empty(0, dtype=np.int32)

        inf_idx = np.flatnonzero(infected)
        rec_draw = rng.random(inf_idx.size) < p_recover
        new_rec = inf_idx[rec_draw]

        if new_inf.size > 0:
            status[new_inf] = 1
        if new_rec.size > 0:
            status[new_rec] = 2

        s_count -= int(new_inf.size)
        i_count += int(new_inf.size) - int(new_rec.size)
        r_count += int(new_rec.size)

        t += dt
        t_vals.append(t)
        s_vals.append(s_count / n_nodes)
        i_vals.append(i_count / n_nodes)
        r_vals.append(r_count / n_nodes)

    return SimulationResult(
        t=np.asarray(t_vals, dtype=float),
        S=np.asarray(s_vals, dtype=float),
        I=np.asarray(i_vals, dtype=float),
        R=np.asarray(r_vals, dtype=float),
    )
