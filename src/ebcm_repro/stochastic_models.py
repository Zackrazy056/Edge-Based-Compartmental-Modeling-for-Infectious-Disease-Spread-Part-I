from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .simulation import SimulationResult, build_configuration_model_edges, run_sir_tauleap


@dataclass(frozen=True)
class AveragedRuns:
    mean: SimulationResult
    n_total: int
    n_epidemic: int


def sample_nb_degrees(n_nodes: int, rng: np.random.Generator, r: int = 4, p: float = 1.0 / 3.0) -> np.ndarray:
    # Paper parameterization: P(k)=C(k+r-1,k)(1-p)^r p^k, numpy uses success-prob parameter.
    return rng.negative_binomial(r, 1.0 - p, size=n_nodes).astype(np.int32)


def sample_ad_mfsh_degrees(n_nodes: int, rng: np.random.Generator) -> np.ndarray:
    vals = np.array([1, 5, 25], dtype=np.int32)
    probs = np.array([25 / 31, 5 / 31, 1 / 31], dtype=float)
    return rng.choice(vals, size=n_nodes, p=probs).astype(np.int32)


def sample_kappa_mp_piecewise(n_nodes: int, rng: np.random.Generator) -> np.ndarray:
    half = n_nodes // 2
    k1 = rng.uniform(0.0, 2.0, size=half)
    k2 = rng.uniform(10.0, 20.0, size=n_nodes - half)
    kappa = np.concatenate([k1, k2])
    rng.shuffle(kappa)
    return kappa


def sample_kappa_ed_mfsh(n_nodes: int, rng: np.random.Generator) -> np.ndarray:
    # rho(k)=exp(k)/(exp(3)-1) on (0,3) => inverse CDF: k=ln(1+u*(e^3-1))
    u = rng.random(n_nodes)
    return np.log1p(u * (np.exp(3.0) - 1.0))


def _run_meanfield_heterogeneous(
    weights: np.ndarray,
    beta: float,
    gamma: float,
    t_max: float,
    dt: float,
    initial_infected: int,
    rng: np.random.Generator,
) -> SimulationResult:
    n_nodes = int(weights.size)
    status = np.zeros(n_nodes, dtype=np.int8)
    seeds = rng.choice(n_nodes, size=initial_infected, replace=False)
    status[seeds] = 1

    total_weight = float(np.sum(weights))
    p_rec = 1.0 - np.exp(-gamma * dt)

    s_count = n_nodes - initial_infected
    i_count = initial_infected
    r_count = 0

    t_vals = [0.0]
    s_vals = [s_count / n_nodes]
    i_vals = [i_count / n_nodes]
    r_vals = [0.0]

    t = 0.0
    while t < t_max and i_count > 0:
        infected = status == 1
        susceptible = status == 0
        pressure = float(np.sum(weights[infected])) / max(total_weight, 1e-12)

        sus_idx = np.flatnonzero(susceptible)
        if sus_idx.size > 0:
            lam = beta * dt * weights[sus_idx] * pressure
            p_inf = 1.0 - np.exp(-lam)
            new_inf = sus_idx[rng.random(sus_idx.size) < p_inf]
        else:
            new_inf = np.empty(0, dtype=np.int32)

        inf_idx = np.flatnonzero(infected)
        new_rec = inf_idx[rng.random(inf_idx.size) < p_rec]

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


def run_ad_mfsh_sim(
    n_nodes: int,
    beta: float,
    gamma: float,
    t_max: float,
    dt: float,
    initial_infected: int,
    rng: np.random.Generator,
) -> SimulationResult:
    degrees = sample_ad_mfsh_degrees(n_nodes, rng)
    return _run_meanfield_heterogeneous(
        weights=degrees.astype(float),
        beta=beta,
        gamma=gamma,
        t_max=t_max,
        dt=dt,
        initial_infected=initial_infected,
        rng=rng,
    )


def run_ed_mfsh_sim(
    n_nodes: int,
    beta: float,
    gamma: float,
    t_max: float,
    dt: float,
    initial_infected: int,
    rng: np.random.Generator,
) -> SimulationResult:
    kappa = sample_kappa_ed_mfsh(n_nodes, rng)
    return _run_meanfield_heterogeneous(
        weights=kappa,
        beta=beta,
        gamma=gamma,
        t_max=t_max,
        dt=dt,
        initial_infected=initial_infected,
        rng=rng,
    )


def run_ed_mfsh_binned_sim(
    n_nodes: int,
    beta: float,
    gamma: float,
    t_max: float,
    dt: float,
    initial_infected: int,
    rng: np.random.Generator,
    n_bins: int = 240,
) -> SimulationResult:
    # rho(k)=exp(k)/(exp(3)-1), k in (0,3). Use multinomial bin counts for fast large-N stochastic simulation.
    edges = np.linspace(0.0, 3.0, n_bins + 1)
    lo = edges[:-1]
    hi = edges[1:]
    mass = (np.exp(hi) - np.exp(lo)) / (np.exp(3.0) - 1.0)
    mass = mass / np.sum(mass)
    n_bin = rng.multinomial(n_nodes, mass).astype(np.int64)
    k_mid = 0.5 * (lo + hi)

    i_bin = rng.multinomial(initial_infected, n_bin / np.sum(n_bin)).astype(np.int64)
    s_bin = n_bin - i_bin
    r_bin = np.zeros_like(n_bin)

    total_weight = float(np.sum(k_mid * n_bin))
    p_rec = 1.0 - np.exp(-gamma * dt)

    s_count = int(np.sum(s_bin))
    i_count = int(np.sum(i_bin))
    r_count = 0

    t = 0.0
    t_vals = [0.0]
    s_vals = [s_count / n_nodes]
    i_vals = [i_count / n_nodes]
    r_vals = [0.0]

    while t < t_max and i_count > 0:
        pressure = float(np.sum(k_mid * i_bin)) / max(total_weight, 1e-12)
        p_inf = 1.0 - np.exp(-beta * k_mid * pressure * dt)
        p_inf = np.clip(p_inf, 0.0, 1.0)
        new_inf = rng.binomial(s_bin, p_inf).astype(np.int64)
        new_rec = rng.binomial(i_bin, p_rec).astype(np.int64)

        s_bin -= new_inf
        i_bin += new_inf - new_rec
        r_bin += new_rec

        s_count = int(np.sum(s_bin))
        i_count = int(np.sum(i_bin))
        r_count = int(np.sum(r_bin))

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


def run_mp_sim(
    n_nodes: int,
    beta: float,
    gamma: float,
    t_max: float,
    dt: float,
    initial_infected: int,
    rng: np.random.Generator,
) -> SimulationResult:
    kappa = sample_kappa_mp_piecewise(n_nodes, rng)
    degrees = rng.poisson(kappa).astype(np.int32)
    if np.sum(degrees, dtype=np.int64) % 2 == 1:
        idx = int(rng.integers(0, n_nodes))
        degrees[idx] += 1
    u, v = build_configuration_model_edges(degrees, rng)
    return run_sir_tauleap(
        edges_u=u,
        edges_v=v,
        n_nodes=n_nodes,
        beta=beta,
        gamma=gamma,
        t_max=t_max,
        dt=dt,
        initial_infected=initial_infected,
        rng=rng,
    )


def run_dfd_sim(
    n_nodes: int,
    beta: float,
    gamma: float,
    eta: float,
    t_max: float,
    dt: float,
    initial_infected: int,
    rng: np.random.Generator,
) -> SimulationResult:
    degrees = sample_nb_degrees(n_nodes=n_nodes, rng=rng, r=4, p=1.0 / 3.0)
    if np.sum(degrees, dtype=np.int64) % 2 == 1:
        idx = int(rng.integers(0, n_nodes))
        degrees[idx] += 1
    u, v = build_configuration_model_edges(degrees, rng)
    m = u.size

    status = np.zeros(n_nodes, dtype=np.int8)
    seeds = rng.choice(n_nodes, size=initial_infected, replace=False)
    status[seeds] = 1

    p_rec = 1.0 - np.exp(-gamma * dt)
    p_break = 1.0 - np.exp(-eta * dt)

    s_count = n_nodes - initial_infected
    i_count = initial_infected
    r_count = 0
    t = 0.0
    t_vals = [0.0]
    s_vals = [s_count / n_nodes]
    i_vals = [i_count / n_nodes]
    r_vals = [0.0]

    while t < t_max and i_count > 0:
        infected = status == 1
        susceptible = status == 0
        uv = infected[u] & susceptible[v]
        vu = infected[v] & susceptible[u]
        pressure = np.bincount(v[uv], minlength=n_nodes) + np.bincount(u[vu], minlength=n_nodes)

        sus_idx = np.flatnonzero(susceptible)
        if sus_idx.size > 0:
            lam = beta * dt * pressure[sus_idx]
            p_inf = 1.0 - np.exp(-lam)
            new_inf = sus_idx[rng.random(sus_idx.size) < p_inf]
        else:
            new_inf = np.empty(0, dtype=np.int32)

        inf_idx = np.flatnonzero(infected)
        new_rec = inf_idx[rng.random(inf_idx.size) < p_rec]

        if new_inf.size > 0:
            status[new_inf] = 1
        if new_rec.size > 0:
            status[new_rec] = 2

        # Edge swapping among edges broken during this step.
        break_mask = rng.random(m) < p_break
        broken_idx = np.flatnonzero(break_mask)
        if broken_idx.size > 0:
            stubs = np.empty(2 * broken_idx.size, dtype=np.int32)
            stubs[0::2] = u[broken_idx]
            stubs[1::2] = v[broken_idx]
            rng.shuffle(stubs)
            u[broken_idx] = stubs[0::2]
            v[broken_idx] = stubs[1::2]

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


def run_dc_sim(
    n_nodes: int,
    beta: float,
    gamma: float,
    eta1: float,
    eta2: float,
    t_max: float,
    dt: float,
    initial_infected: int,
    rng: np.random.Generator,
) -> SimulationResult:
    km = rng.poisson(3.0, size=n_nodes).astype(np.int32)
    if np.sum(km, dtype=np.int64) % 2 == 1:
        km[int(rng.integers(0, n_nodes))] += 1

    owner = np.repeat(np.arange(n_nodes, dtype=np.int32), km)
    l = owner.size
    partner = np.full(l, -1, dtype=np.int32)

    xi = eta1 / (eta1 + eta2)
    active = rng.random(l) < xi
    active_idx = np.flatnonzero(active)
    if active_idx.size % 2 == 1:
        active[active_idx[-1]] = False
        active_idx = active_idx[:-1]
    rng.shuffle(active_idx)
    a = active_idx[0::2]
    b = active_idx[1::2]
    partner[a] = b
    partner[b] = a

    status = np.zeros(n_nodes, dtype=np.int8)
    seeds = rng.choice(n_nodes, size=initial_infected, replace=False)
    status[seeds] = 1

    p_rec = 1.0 - np.exp(-gamma * dt)
    p_break = 1.0 - np.exp(-eta2 * dt)
    p_activate = 1.0 - np.exp(-eta1 * dt)

    s_count = n_nodes - initial_infected
    i_count = initial_infected
    r_count = 0
    t = 0.0
    t_vals = [0.0]
    s_vals = [s_count / n_nodes]
    i_vals = [i_count / n_nodes]
    r_vals = [0.0]

    stub_idx = np.arange(l, dtype=np.int32)
    while t < t_max and i_count > 0:
        active_idx = stub_idx[partner >= 0]
        edge_u_stub = active_idx[active_idx < partner[active_idx]]
        edge_v_stub = partner[edge_u_stub]
        nu = owner[edge_u_stub]
        nv = owner[edge_v_stub]

        infected = status == 1
        susceptible = status == 0
        uv = infected[nu] & susceptible[nv]
        vu = infected[nv] & susceptible[nu]
        pressure = np.bincount(nv[uv], minlength=n_nodes) + np.bincount(nu[vu], minlength=n_nodes)

        sus_idx = np.flatnonzero(susceptible)
        if sus_idx.size > 0:
            lam = beta * dt * pressure[sus_idx]
            p_inf = 1.0 - np.exp(-lam)
            new_inf = sus_idx[rng.random(sus_idx.size) < p_inf]
        else:
            new_inf = np.empty(0, dtype=np.int32)

        inf_idx = np.flatnonzero(infected)
        new_rec = inf_idx[rng.random(inf_idx.size) < p_rec]

        if new_inf.size > 0:
            status[new_inf] = 1
        if new_rec.size > 0:
            status[new_rec] = 2

        # Active edges break into dormant stubs.
        n_edges = edge_u_stub.size
        if n_edges > 0:
            break_e = rng.random(n_edges) < p_break
            if np.any(break_e):
                bu = edge_u_stub[break_e]
                bv = edge_v_stub[break_e]
                partner[bu] = -1
                partner[bv] = -1

        # Dormant stubs activate and pair among newly activated stubs.
        dormant = np.flatnonzero(partner < 0)
        if dormant.size > 1:
            act = dormant[rng.random(dormant.size) < p_activate]
            if act.size % 2 == 1:
                act = act[:-1]
            if act.size > 1:
                rng.shuffle(act)
                au = act[0::2]
                av = act[1::2]
                partner[au] = av
                partner[av] = au

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


def run_dvd_sim(
    n_nodes: int,
    beta: float,
    gamma: float,
    eta: float,
    t_max: float,
    dt: float,
    initial_infected: int,
    rng: np.random.Generator,
) -> SimulationResult:
    kappa = sample_nb_degrees(n_nodes=n_nodes, rng=rng, r=4, p=1.0 / 3.0).astype(float)
    # Equilibrium edge pool: Poisson degree around kappa.
    m0 = rng.poisson(kappa).astype(np.int32)
    stubs = np.repeat(np.arange(n_nodes, dtype=np.int32), m0)
    if stubs.size % 2 == 1:
        stubs = stubs[:-1]
    rng.shuffle(stubs)
    u = stubs[0::2].copy()
    v = stubs[1::2].copy()

    status = np.zeros(n_nodes, dtype=np.int8)
    seeds = rng.choice(n_nodes, size=initial_infected, replace=False)
    status[seeds] = 1

    p_rec = 1.0 - np.exp(-gamma * dt)
    p_break = 1.0 - np.exp(-eta * dt)

    s_count = n_nodes - initial_infected
    i_count = initial_infected
    r_count = 0
    t = 0.0
    t_vals = [0.0]
    s_vals = [s_count / n_nodes]
    i_vals = [i_count / n_nodes]
    r_vals = [0.0]

    while t < t_max and i_count > 0:
        m = u.size
        infected = status == 1
        susceptible = status == 0
        if m > 0:
            uv = infected[u] & susceptible[v]
            vu = infected[v] & susceptible[u]
            pressure = np.bincount(v[uv], minlength=n_nodes) + np.bincount(u[vu], minlength=n_nodes)
        else:
            pressure = np.zeros(n_nodes, dtype=np.int32)

        sus_idx = np.flatnonzero(susceptible)
        if sus_idx.size > 0:
            lam = beta * dt * pressure[sus_idx]
            p_inf = 1.0 - np.exp(-lam)
            new_inf = sus_idx[rng.random(sus_idx.size) < p_inf]
        else:
            new_inf = np.empty(0, dtype=np.int32)

        inf_idx = np.flatnonzero(infected)
        new_rec = inf_idx[rng.random(inf_idx.size) < p_rec]
        if new_inf.size > 0:
            status[new_inf] = 1
        if new_rec.size > 0:
            status[new_rec] = 2

        # Break existing edges.
        if m > 0:
            keep = rng.random(m) >= p_break
            u = u[keep]
            v = v[keep]

        # Create new edges from newly generated stubs.
        new_stubs_per_node = rng.poisson(kappa * eta * dt).astype(np.int32)
        new_stubs = np.repeat(np.arange(n_nodes, dtype=np.int32), new_stubs_per_node)
        if new_stubs.size % 2 == 1:
            new_stubs = new_stubs[:-1]
        if new_stubs.size > 1:
            rng.shuffle(new_stubs)
            new_u = new_stubs[0::2]
            new_v = new_stubs[1::2]
            u = np.concatenate([u, new_u]).astype(np.int32, copy=False)
            v = np.concatenate([v, new_v]).astype(np.int32, copy=False)

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


def align_and_average_runs(
    runs: list[SimulationResult],
    align_threshold: float,
    epidemic_threshold: float,
    t_min: float,
    t_max: float,
    n_grid: int = 801,
) -> AveragedRuns:
    epi = [r for r in runs if r.cumulative[-1] >= epidemic_threshold]
    if len(epi) == 0:
        epi = runs

    grid = np.linspace(t_min, t_max, n_grid)
    s_arr = np.empty((len(epi), n_grid), dtype=float)
    i_arr = np.empty((len(epi), n_grid), dtype=float)
    r_arr = np.empty((len(epi), n_grid), dtype=float)
    for j, run in enumerate(epi):
        shift = _first_crossing_time(run.t, run.cumulative, align_threshold)
        t = run.t - shift
        s_arr[j] = np.interp(grid, t, run.S, left=float(run.S[0]), right=float(run.S[-1]))
        i_arr[j] = np.interp(grid, t, run.I, left=float(run.I[0]), right=float(run.I[-1]))
        r_arr[j] = np.interp(grid, t, run.R, left=float(run.R[0]), right=float(run.R[-1]))

    s_mean = np.nanmean(s_arr, axis=0)
    i_mean = np.nanmean(i_arr, axis=0)
    r_mean = np.nanmean(r_arr, axis=0)
    mean = SimulationResult(t=grid, S=s_mean, I=i_mean, R=r_mean)
    return AveragedRuns(mean=mean, n_total=len(runs), n_epidemic=len(epi))


def _first_crossing_time(t: np.ndarray, y: np.ndarray, threshold: float) -> float:
    idx = np.flatnonzero(y >= threshold)
    if idx.size == 0:
        return float(t[0])
    j = int(idx[0])
    if j == 0:
        return float(t[0])
    t0, t1 = t[j - 1], t[j]
    y0, y1 = y[j - 1], y[j]
    if y1 == y0:
        return float(t1)
    frac = (threshold - y0) / (y1 - y0)
    return float(t0 + frac * (t1 - t0))
