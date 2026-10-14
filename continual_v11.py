"""One scenario, one timeline (paper, Sec. 4).

Toy factored MDP with continuous, non-Gaussian dynamics (v11_common).  Eight
phases of 2,000 transitions:
  phases 0-1   original behavior, original dynamics
  phase 2      BEHAVIOR SHIFT: states and policy change (mechanisms do not)
  phase 5      MECHANISM DRIFT: the effect of A on S2' is reversed (S1' unchanged)
World models: NBN with MDN mechanisms or kNN mechanisms (plus a linear-Gaussian
reference); queries answered by likelihood weighting.
Strategies (same model family, same data):
  frozen       fit on phase 0, never updated
  pooled       every mechanism refit on all data so far
  window       every mechanism refit on the newest phase only
  raw-restart  restart a mechanism when its out-of-sample log-likelihood drops
               (one-sided z-test), else append
  pipeline     Test (matched-pair permutation test) -> restart flagged
               mechanisms, append the others -> Plan
Both detectors run at level alpha/2 per mechanism (Bonferroni, alpha = 0.01).
Metrics per phase: W1 distance between model and true transition kernel
(also KL(true || model), which extrapolation makes heavy-tailed), success of the
planned action in the true MDP relative to optimal, flags, wall-clock time.
Output: results_v11/continual_v11.json

Run:  python continual_v11.py [n_seeds=5]
"""
import json
import os
import time

import numpy as np
import torch

import v11_common as C

import sys
SEEDS = range(int(sys.argv[1]) if len(sys.argv) > 1 else 5)   # first test: 5 seeds
ALPHA = 0.01
LEVEL = ALPHA / len(C.TRANSITION_NODES)
STRATS = ("frozen", "pooled", "window", "raw-restart", "pipeline")
FAMILIES = ("mdn", "knn", "lg")


def make_phases(seed):
    rng = np.random.default_rng(seed)
    return [C.simulate(rng, C.N_PER_PHASE, *C.phase_flags(t)) for t in range(C.N_PHASES)]


def record(model, t):
    drifted = C.phase_flags(t)[1]
    kl = C.kl_error(model, drifted)
    w1 = C.w1_error(model, drifted)
    return dict(joint=w1["joint"], S1n=w1["S1n"], S2n=w1["S2n"], kl=kl["joint"],
                plan=C.plan_score(model, drifted))


# Single-threaded torch for the MDN and linear-Gaussian runs: with the default
# thread pool, small MDN fits occasionally stall for minutes on macOS (an
# OpenMP issue; the same fit takes ~1 s single-threaded).  kNN keeps the pool
# for its neighbour searches.
DEFAULT_THREADS = torch.get_num_threads()


def run(family, phases, seed):
    torch.set_num_threads(DEFAULT_THREADS if family == "knn" else 1)
    torch.manual_seed(seed)
    m0 = C.new_model(family)
    for n in C.TRANSITION_NODES:
        C.fit_node(m0, family, n, phases[0])
    models = {s: C.clone(m0) for s in STRATS}
    buf = {s: {n: phases[0] for n in C.TRANSITION_NODES} for s in STRATS}
    ll_prev = {n: None for n in C.TRANSITION_NODES}
    out = {s: [record(models[s], 0)] for s in STRATS}
    flags = {"pipeline": [], "raw-restart": []}
    pvals = []
    secs = {"test": 0.0, "update": 0.0, "plan": 0.0}
    for t in range(1, C.N_PHASES):
        b = phases[t]
        # pooled: all data so far
        for n in C.TRANSITION_NODES:
            buf["pooled"][n] = C.concat([buf["pooled"][n], b])
            C.fit_node(models["pooled"], family, n, buf["pooled"][n], warm=True)
        # window: newest phase only
        models["window"] = C.new_model(family)
        for n in C.TRANSITION_NODES:
            C.fit_node(models["window"], family, n, b)
        # pipeline: Test -> Update (restart flagged, append others)
        f_pipe, p_t = {}, {}
        for n in C.TRANSITION_NODES:
            t0 = time.perf_counter()
            rng = np.random.default_rng([seed, t, C.TRANSITION_NODES.index(n)])
            p, _ = C.test_node(n, buf["pipeline"][n], b, rng)
            secs["test"] += time.perf_counter() - t0
            f_pipe[n], p_t[n] = bool(p < LEVEL), p
            t0 = time.perf_counter()
            if f_pipe[n]:
                buf["pipeline"][n] = b
                models["pipeline"].mechanisms[n] = C.new_mechanism(family)
                C.fit_node(models["pipeline"], family, n, b)
            else:
                buf["pipeline"][n] = C.concat([buf["pipeline"][n], b])
                C.fit_node(models["pipeline"], family, n, buf["pipeline"][n], warm=True)
            secs["update"] += time.perf_counter() - t0
        flags["pipeline"].append(f_pipe)
        pvals.append(p_t)
        # raw-restart: same update rule, raw-score detector
        f_raw = {}
        for n in C.TRANSITION_NODES:
            ll_new = C.node_logprob(models["raw-restart"], n, b)
            f_raw[n] = (ll_prev[n] is not None) and C.raw_score_flag(ll_new, ll_prev[n], LEVEL)
            ll_prev[n] = ll_new
            if f_raw[n]:
                buf["raw-restart"][n] = b
                models["raw-restart"].mechanisms[n] = C.new_mechanism(family)
                C.fit_node(models["raw-restart"], family, n, b)
            else:
                buf["raw-restart"][n] = C.concat([buf["raw-restart"][n], b])
                C.fit_node(models["raw-restart"], family, n, buf["raw-restart"][n], warm=True)
        flags["raw-restart"].append(f_raw)
        for s in STRATS:
            if s == "pipeline":
                t0 = time.perf_counter()
                out[s].append(record(models[s], t))
                secs["plan"] += time.perf_counter() - t0
            else:
                out[s].append(record(models[s], t))
    secs = {k: v / (C.N_PHASES - 1) for k, v in secs.items()}
    return out, flags, pvals, secs


def summarise(curves):
    res = {}
    for key in ("joint", "S1n", "S2n", "kl", "plan"):
        a = np.array([[ph[key] for ph in c] for c in curves])     # (seeds, phases)
        res[key] = dict(mean=a.mean(0).tolist(), sd=a.std(0, ddof=1).tolist(),
                        before=float(a[:, C.DRIFT_AT - 1].mean()),
                        after=float(a[:, C.DRIFT_AT].mean()),
                        end=float(a[:, -1].mean()), end_sd=float(a[:, -1].std(ddof=1)),
                        shift=float(a[:, C.SHIFT_AT].mean()))
    return res


def flag_counts(flag_runs):
    """flag_runs: list over seeds of list over t=1..7 of {node: bool}."""
    out = {}
    for n in C.TRANSITION_NODES:
        a = np.array([[f[n] for f in run] for run in flag_runs], dtype=float)  # (seeds, 7)
        out[n] = a.sum(0).astype(int).tolist()          # runs flagged, per phase 1..7
    return out


def main():
    raw = {f: {s: [] for s in STRATS} for f in FAMILIES}
    flags = {f: {"pipeline": [], "raw-restart": []} for f in FAMILIES}
    pv = {f: [] for f in FAMILIES}
    timing = {f: [] for f in FAMILIES}
    for seed in SEEDS:
        phases = make_phases(seed)
        for fam in FAMILIES:
            t0 = time.time()
            o, fl, p, secs = run(fam, phases, seed)
            for s in STRATS:
                raw[fam][s].append(o[s])
            for k in fl:
                flags[fam][k].append(fl[k])
            pv[fam].append(p)
            timing[fam].append(secs)
            print(f"seed {seed} {fam:3s} {time.time() - t0:6.1f}s | pipeline plan "
                  f"{[round(x['plan'], 2) for x in o['pipeline']]} | flags S2' "
                  f"{[int(f['S2n']) for f in fl['pipeline']]} raw S1' "
                  f"{[int(f['S1n']) for f in fl['raw-restart']]} raw S2' "
                  f"{[int(f['S2n']) for f in fl['raw-restart']]}", flush=True)
    summary = {f: {s: summarise(raw[f][s]) for s in STRATS} for f in FAMILIES}
    counts = {f: {k: flag_counts(flags[f][k]) for k in flags[f]} for f in FAMILIES}
    secs = {f: {k: float(np.mean([d[k] for d in timing[f]])) for k in timing[f][0]}
            for f in FAMILIES}
    same = all(flags["mdn"]["pipeline"][i] == flags[f]["pipeline"][i]
               for f in FAMILIES for i in range(len(SEEDS)))
    for f in FAMILIES:
        print(f"\n{f}: W1 joint before/after/end | plan before/after/end")
        for s in STRATS:
            e = summary[f][s]
            print(f"  {s:12s} {e['joint']['before']:.3f} / {e['joint']['after']:.3f} / "
                  f"{e['joint']['end']:.3f}±{e['joint']['end_sd']:.3f} | "
                  f"{e['plan']['before']:.3f} / {e['plan']['after']:.3f} / {e['plan']['end']:.3f}"
                  f" | at shift W1 {e['joint']['shift']:.3f} plan {e['plan']['shift']:.3f}")
        print(f"  flags (runs of {len(SEEDS)}, phases 1..7):", counts[f])
        print("  pipeline seconds per phase:", {k: round(v, 3) for k, v in secs[f].items()})
    print("test flags identical across families:", same)
    os.makedirs("results_v11", exist_ok=True)
    meta = dict(seeds=list(SEEDS), alpha=ALPHA, level=LEVEL, n_per_phase=C.N_PER_PHASE,
                shift_at=C.SHIFT_AT, drift_at=C.DRIFT_AT, n_phases=C.N_PHASES,
                lw_particles=C.LW_PARTICLES, eval_states=C.EVAL_N,
                test_flags_identical_across_families=same)
    with open("results_v11/continual_v11.json", "w") as fh:
        json.dump(dict(meta=meta, summary=summary, flag_counts=counts, seconds=secs,
                       pvalues=pv), fh, indent=1)
    print("written results_v11/continual_v11.json")


if __name__ == "__main__":
    main()
