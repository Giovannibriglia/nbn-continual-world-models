"""Planning-query benchmark: NBN against Pyro and NumPyro (pilot).

The query is the planner of Sec. 3 (4) and Sec. 4: for each state s,
a*(s) = mode of p(a | s, R = 1), by likelihood weighting (uniform action prior,
particles weighted by P(R = 1 | S2')), with the mode read from a weighted
Gaussian kernel (bandwidth 0.05) on the 201-point action grid.  One query =
one state.

Every engine runs the SAME world model (the MDN NBN fitted on phase-0 data,
as in planning_cost.py: its trained weights are copied into the Pyro and
NumPyro models) and the SAME estimator; they differ only in how the
computation is scheduled.  Engines:

  nbn            nbn's batched LikelihoodWeightingEngine (= planning_cost.plan)
  pyro-vec       Pyro model with plates over states and particles, one
                 poutine.trace; log-weights = log-probs of the observed sites
  numpyro-vec    the same model in NumPyro, jax.jit-compiled (compile time
                 excluded from the timings, reported separately)
  pyro-importance  pyro.infer.Importance(model, guide=None), Pyro's standard
                 importance sampler: one trace per particle, one state per call

Measurements (seed = MDN fit + particles):
  (a) time per state at S = 2,048 particles, for B states answered by one
      batched call (B in BATCHES) and by B calls of one state each;
  (b) plan score and time per state on the 200 evaluation states, for
      S in PARTICLES (batched calls only).
Devices: CPU (always) and the GPU for the torch engines when available (CUDA,
else Apple MPS); NumPyro runs on the platform given by --jax-platform (cpu,
cuda, or METAL); run it on a GPU in a separate process with --only-numpyro.
Timing: each measurement is preceded by one untimed call (the GPU may have sat
idle while CPU engines ran: the first call after that is slow), median of 5;
the engine order is rotated across seeds.

Run:    python engine_benchmark.py [--seeds 1] [--threads 0]
        --threads 0 keeps each library's default thread pool; --threads 1
        pins torch and XLA to one thread (must be set before jax is imported).
Output: results_v11/engine_benchmark_threads{N|default}.json
"""
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=1)
ap.add_argument("--seed-start", type=int, default=0, help="> 0: append seeds to the existing results file")
ap.add_argument("--threads", type=int, default=0)
ap.add_argument("--no-gpu", action="store_true")
ap.add_argument("--jax-platform", default="cpu", help="cpu, cuda, or METAL (Apple GPU; needs jax-metal)")
ap.add_argument("--only-numpyro", action="store_true", help="time the NumPyro engine only")
ap.add_argument("--workload", type=int, default=0,
                help="> 0: fixed-workload mode: answer this many queries for every B (in workload/B calls), "
                     "one timed pass; B in WORKLOAD_BATCHES; no one-per-call, importance or particle sweep")
ARGS = ap.parse_args()
if ARGS.threads == 1:
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
os.environ["JAX_PLATFORMS"] = ARGS.jax_platform.lower() if ARGS.jax_platform != "METAL" else "METAL"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import json
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as ndist
import pyro
import pyro.distributions as pdist
import torch
from numpyro import handlers as nh
from pyro import poutine

import nbn
import planning_cost as PC
import v11_common as C
from nbn.utils.batching import pack_parents

S_TIMING = C.LW_PARTICLES                      # 2,048
BATCHES = (1, 8, 64, 200, 1024)
WORKLOAD_BATCHES = (1, 8, 64, 256, 1024)      # all divide the default workload, 10,240 = 10 x 1,024
LOOP_B = 8                                     # one-at-a-time rate: constant in B, measured once
PARTICLES = (64, 256, 1024, 2048, 4096, 8192)
REPS = 5
CHUNK = PC.CHUNK
BW = 0.05
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_v11",
                   f"engine_benchmark_threads{ARGS.threads or 'default'}"
                   f"{'' if ARGS.jax_platform == 'cpu' else '_jax' + ARGS.jax_platform.lower()}"
                   f"{'_numpyro' if ARGS.only_numpyro else ''}"
                   f"{f'_workload{ARGS.workload}' if ARGS.workload else ''}.json")
JAX_DEV = "cpu" if ARGS.jax_platform == "cpu" else ARGS.jax_platform.lower()

pyro.enable_validation(False)                  # no argument checks in any engine


# ----------------------------------------------------------------------------
# The fitted MDNs, as plain parameters
# ----------------------------------------------------------------------------
def mdn_params(model, node):
    m = model.mechanisms[node]
    lin = [l for l in m.net if isinstance(l, torch.nn.Linear)]
    f = lambda t: t.detach().cpu().numpy().astype(np.float32)
    return {"parents": list(model.dag.parents(node)), "K": m.num_components,
            "W": [f(l.weight.T) for l in lin], "b": [f(l.bias) for l in lin],
            "pa_mean": f(m._pa_mean), "pa_std": f(m._pa_std),
            # child standardisation exists only in later nbn versions (identity in 0.20.2)
            "x_mean": float(m._x_mean.reshape(-1)[0]) if getattr(m, "_x_mean", None) is not None else 0.0,
            "x_std": float(m._x_std.reshape(-1)[0]) if getattr(m, "_x_std", None) is not None else 1.0,
            "pa_clamp": m._pa_clamp, "ls_clamp": m._log_scale_clamp, "min_scale": m.min_scale}


# ----------------------------------------------------------------------------
# Pyro
# ----------------------------------------------------------------------------
class PyroWM:
    """The NBN world model as a Pyro program (plates: particles x states)."""

    def __init__(self, params, device):
        self.dev = torch.device(device)
        t = lambda x: torch.as_tensor(x, device=self.dev)
        self.p = {n: {**p, "W": [t(w) for w in p["W"]], "b": [t(b) for b in p["b"]],
                      "pa_mean": t(p["pa_mean"]), "pa_std": t(p["pa_std"])}
                  for n, p in params.items()}
        self.zero, self.one = t(0.0), t(1.0)

    def mdn(self, node, cols):
        p = self.p[node]
        z = torch.stack([cols[q] for q in p["parents"]], -1)
        h = ((z - p["pa_mean"]) / p["pa_std"]).clamp(-p["pa_clamp"], p["pa_clamp"])
        for W, b in zip(p["W"][:-1], p["b"][:-1]):
            h = torch.relu(h @ W + b)
        out = h @ p["W"][-1] + p["b"][-1]
        K = p["K"]
        rest = out[..., K:].reshape(*out.shape[:-1], K, 2)
        loc = p["x_mean"] + p["x_std"] * rest[..., 0]
        scale = (p["x_std"] * rest[..., 1].clamp(max=p["ls_clamp"]).exp()).clamp_min(p["min_scale"])
        return pdist.MixtureSameFamily(pdist.Categorical(logits=out[..., :K]), pdist.Normal(loc, scale))

    def model(self, s1, s2, S):
        B = s1.shape[0]
        with pyro.plate("states", B, dim=-1), pyro.plate("particles", S, dim=-2):
            pyro.sample("S1", pdist.Normal(self.zero, self.one), obs=s1)
            pyro.sample("S2", pdist.Normal(self.zero, self.one), obs=s2)
            a = pyro.sample("A", pdist.Uniform(-self.one, self.one))
            cols = {"S1": s1.expand(S, B), "S2": s2.expand(S, B), "A": a}
            cols["S1n"] = pyro.sample("S1n", self.mdn("S1n", cols))
            cols["S2n"] = s2n = pyro.sample("S2n", self.mdn("S2n", cols))
            hit = ((s2n - C.GOAL).abs() < C.GOAL_HALF).float()
            pyro.sample("R", pdist.Bernoulli(probs=hit * (1 - 1e-6) + (1 - hit) * 1e-6), obs=self.one)

    def lw(self, s1, s2, S):
        """Vectorized LW: one trace, all states and particles. Returns a, log w [B, S]."""
        tr = poutine.trace(self.model).get_trace(s1, s2, S)
        tr.compute_log_prob(site_filter=lambda name, site: site["is_observed"])
        lw = sum(site["log_prob"] for site in tr.nodes.values()
                 if site["type"] == "sample" and site["is_observed"])
        return tr.nodes["A"]["value"].T, lw.T

    def plan(self, s1, s2, S, seed):
        torch.manual_seed(seed)
        with torch.no_grad():
            s1 = torch.as_tensor(s1, dtype=torch.float32, device=self.dev)
            s2 = torch.as_tensor(s2, dtype=torch.float32, device=self.dev)
            a, lw = self.lw(s1, s2, S)
            return mode_torch(a, torch.softmax(lw, -1), self.dev)

    def plan_importance(self, s1, s2, S, seed):
        """pyro.infer.Importance, one state per call, one trace per particle."""
        torch.manual_seed(seed)
        out = []
        with torch.no_grad():
            for j in range(len(s1)):
                x1 = torch.as_tensor(s1[j:j + 1], dtype=torch.float32, device=self.dev)
                x2 = torch.as_tensor(s2[j:j + 1], dtype=torch.float32, device=self.dev)
                imp = pyro.infer.Importance(self.model, guide=None, num_samples=S).run(x1, x2, 1)
                a = torch.stack([t.nodes["A"]["value"].reshape(()) for t in imp.exec_traces])
                lw = torch.stack([torch.as_tensor(v).reshape(()) for v in imp.log_weights])
                out.append(mode_torch(a[None], torch.softmax(lw, -1)[None], self.dev))
        return np.concatenate(out)


def mode_torch(a, w, dev):
    """The mode step of planning_cost.plan, for a, w [B, S]."""
    dt = torch.float64 if dev.type == "cpu" else torch.float32
    grid = torch.as_tensor(C.A_GRID, dtype=dt, device=dev)
    out = []
    for i in range(0, a.shape[0], CHUNK):
        ai, wi = a[i:i + CHUNK].to(dt), w[i:i + CHUNK].to(dt)
        kern = torch.exp(-0.5 * ((grid[None, None, :] - ai[..., None]) / BW) ** 2)
        out.append((wi[..., None] * kern).sum(1).argmax(1))
    return C.A_GRID[torch.cat(out).cpu().numpy()]


# ----------------------------------------------------------------------------
# NumPyro
# ----------------------------------------------------------------------------
J_GRID = jnp.asarray(C.A_GRID, dtype=jnp.float32)


def _mdn_np(p, cols):
    z = jnp.stack([cols[q] for q in p["parents"]], -1)
    h = jnp.clip((z - p["pa_mean"]) / p["pa_std"], -p["pa_clamp"], p["pa_clamp"])
    for W, b in zip(p["W"][:-1], p["b"][:-1]):
        h = jax.nn.relu(h @ W + b)
    out = h @ p["W"][-1] + p["b"][-1]
    K = p["K"]
    rest = out[..., K:].reshape(*out.shape[:-1], K, 2)
    loc = p["x_mean"] + p["x_std"] * rest[..., 0]
    scale = jnp.maximum(p["x_std"] * jnp.exp(jnp.minimum(rest[..., 1], p["ls_clamp"])), p["min_scale"])
    return ndist.MixtureSameFamily(ndist.Categorical(logits=out[..., :K]), ndist.Normal(loc, scale))


def np_model(params, s1, s2, S):
    B = s1.shape[0]
    with numpyro.plate("states", B, dim=-1), numpyro.plate("particles", S, dim=-2):
        numpyro.sample("S1", ndist.Normal(0.0, 1.0), obs=s1)
        numpyro.sample("S2", ndist.Normal(0.0, 1.0), obs=s2)
        a = numpyro.sample("A", ndist.Uniform(-1.0, 1.0))
        cols = {"S1": jnp.broadcast_to(s1, (S, B)), "S2": jnp.broadcast_to(s2, (S, B)), "A": a}
        cols["S1n"] = numpyro.sample("S1n", _mdn_np(params["S1n"], cols))
        cols["S2n"] = s2n = numpyro.sample("S2n", _mdn_np(params["S2n"], cols))
        hit = (jnp.abs(s2n - C.GOAL) < C.GOAL_HALF).astype(jnp.float32)
        numpyro.sample("R", ndist.Bernoulli(probs=hit * (1 - 1e-6) + (1 - hit) * 1e-6), obs=1.0)


def _np_plan_idx(params, key, s1, s2, S):
    tr = nh.trace(nh.seed(np_model, key)).get_trace(params, s1, s2, S)
    lw = sum(site["fn"].log_prob(site["value"]) for site in tr.values()
             if site["type"] == "sample" and site["is_observed"])
    a, w = tr["A"]["value"].T, jax.nn.softmax(lw.T, -1)          # [B, S]

    def one(aw):
        ai, wi = aw
        return jnp.argmax((wi[:, None] * jnp.exp(-0.5 * ((J_GRID[None, :] - ai[:, None]) / BW) ** 2)).sum(0))
    return jax.lax.map(one, (a, w), batch_size=CHUNK)


class NumPyroWM:
    def __init__(self, params):
        self.params = {n: {**p, "W": [jnp.asarray(w) for w in p["W"]], "b": [jnp.asarray(b) for b in p["b"]],
                           "pa_mean": jnp.asarray(p["pa_mean"]), "pa_std": jnp.asarray(p["pa_std"])}
                       for n, p in params.items()}
        array_keys = ("W", "b", "pa_mean", "pa_std")
        static = {n: {k: v for k, v in p.items() if k not in array_keys} for n, p in self.params.items()}
        self.arrays = {n: {k: p[k] for k in array_keys} for n, p in self.params.items()}
        self._fn = jax.jit(lambda arr, key, s1, s2, S: _np_plan_idx(
            {n: {**static[n], **arr[n]} for n in arr}, key, s1, s2, S), static_argnums=(4,))

    def plan_idx(self, s1, s2, S, seed):
        return self._fn(self.arrays, jax.random.PRNGKey(seed),
                        jnp.asarray(s1, dtype=jnp.float32), jnp.asarray(s2, dtype=jnp.float32), S)

    def plan(self, s1, s2, S, seed):
        return C.A_GRID[np.asarray(self.plan_idx(s1, s2, S, seed).block_until_ready())]


# ----------------------------------------------------------------------------
# Checks and timing
# ----------------------------------------------------------------------------
def check_ports(model, pyro_wm, np_wm):
    """The Pyro and NumPyro MDNs give nbn's log-densities on random inputs."""
    rng = np.random.default_rng(0)
    n = 512
    d = {k: torch.as_tensor(rng.standard_normal(n) if k != "A" else rng.uniform(-1, 1, n),
                            dtype=torch.float32).unsqueeze(-1) for k in ("S1", "S2", "A")}
    err = {}
    for node in C.TRANSITION_NODES:
        x = torch.as_tensor(rng.standard_normal(n), dtype=torch.float32)
        with torch.no_grad():
            ref = model.mechanisms[node].log_prob(x.unsqueeze(-1), pack_parents(d, model.dag.parents(node)))
            cols = {k: v[:, 0].to(pyro_wm.dev) for k, v in d.items()}
            lp_pyro = pyro_wm.mdn(node, cols).log_prob(x.to(pyro_wm.dev)).cpu()
        lp_np = _mdn_np(np_wm.params[node], {k: jnp.asarray(v[:, 0].numpy()) for k, v in d.items()}).log_prob(
            jnp.asarray(x.numpy()))
        ref = ref.reshape(-1)
        rel = lambda lp: float((np.abs(np.asarray(lp) - ref.numpy()) / (1 + np.abs(ref.numpy()))).max())
        err[node] = (rel(lp_pyro), rel(lp_np))      # float32: |log p| reaches ~1e4 far from the data
    print(f"  port check, max |log p - log p_nbn| / (1 + |log p_nbn|) (pyro, numpyro): {err}", flush=True)
    assert all(max(v) < 1e-4 for v in err.values()), err      # float32 through a 64-64 MLP
    return err


def sync_for(dev):
    if dev == "cuda":
        return torch.cuda.synchronize
    return (lambda: torch.mps.synchronize()) if dev == "mps" else (lambda: None)


def timed(fn, sync):
    sync()
    fn()                    # untimed: wakes the device and warms the caches for this shape
    ts = []
    for _ in range(REPS):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    torch.set_num_threads(ARGS.threads or torch.get_num_threads())
    gpu = None if ARGS.no_gpu else ("cuda" if torch.cuda.is_available()
                                    else "mps" if torch.backends.mps.is_available() else None)
    s1e, s2e = C.eval_states()
    res = {"meta": {"versions": {"nbn": nbn.__version__, "torch": torch.__version__, "pyro": pyro.__version__,
                                 "numpyro": numpyro.__version__, "jax": jax.__version__},
                    "torch_threads": torch.get_num_threads(), "threads_arg": ARGS.threads,
                    "jax_devices": [str(d) for d in jax.devices()], "gpu": gpu,
                    "gpu_name": torch.cuda.get_device_name(0) if gpu == "cuda" else gpu,
                    "s_timing": S_TIMING, "batches": BATCHES, "loop_b": LOOP_B, "particles": PARTICLES,
                    "reps": REPS, "seeds": list(range(ARGS.seeds))},
           "port_check": {}, "compile_s": {}, "batch": {}, "loop": {}, "sweep": {}}
    seeds = range(ARGS.seed_start, ARGS.seed_start + ARGS.seeds)
    if ARGS.seed_start > 0:
        with open(OUT) as f:
            res = json.load(f)
        assert len(res["meta"]["seeds"]) == ARGS.seed_start, res["meta"]["seeds"]
    res["meta"]["seeds"] = list(range(ARGS.seed_start + ARGS.seeds))

    for seed in seeds:
        mdn = PC.fit_phase0(seed)
        params = {n: mdn_params(mdn, n) for n in C.TRANSITION_NODES}
        engines = {"nbn/cpu": (lambda m: lambda s1, s2, S, sd: PC.plan(m, s1, s2, S, sd))(mdn),
                   "pyro-vec/cpu": PyroWM(params, "cpu").plan}
        syncs = {"nbn/cpu": sync_for("cpu"), "pyro-vec/cpu": sync_for("cpu")}
        if gpu:
            m_gpu = PC.to_device(mdn, gpu)
            engines[f"nbn/{gpu}"] = lambda s1, s2, S, sd: PC.plan(m_gpu, s1, s2, S, sd)
            engines[f"pyro-vec/{gpu}"] = PyroWM(params, gpu).plan
            syncs[f"nbn/{gpu}"] = syncs[f"pyro-vec/{gpu}"] = sync_for(gpu)
        if ARGS.only_numpyro:
            engines, syncs = {}, {}
        np_wm = NumPyroWM(params)
        engines[f"numpyro-vec/{JAX_DEV}"] = np_wm.plan
        syncs[f"numpyro-vec/{JAX_DEV}"] = sync_for("cpu")      # plan() blocks until ready
        pimp = PyroWM(params, "cpu")
        res["port_check"][seed] = check_ports(mdn, PyroWM(params, gpu or "cpu"), np_wm)
        rot = seed % len(engines)                                    # rotate the timing order
        engines = dict(list(engines.items())[rot:] + list(engines.items())[:rot])
        res["meta"]["numpyro_version"] = numpyro.__version__

        # JIT compile cost of NumPyro (first call per new shape), excluded below
        for B in (WORKLOAD_BATCHES if ARGS.workload else sorted(set(BATCHES) | {1, LOOP_B, C.EVAL_N})):
            t0 = time.perf_counter()
            np_wm.plan(np.zeros(B), np.zeros(B), S_TIMING, 0)
            res["compile_s"].setdefault(str(B), []).append(time.perf_counter() - t0)
        for S in (() if ARGS.workload else PARTICLES):
            np_wm.plan(s1e, s2e, S, 0)
        print(f"seed {seed} numpyro compile (s) per B: "
              f"{ {k: round(v[-1], 2) for k, v in res['compile_s'].items()} }", flush=True)
        for name, fn in engines.items():
            fn(s1e[:8], s2e[:8], S_TIMING, seed)                    # warm-up

        if ARGS.workload:
            # (a') fixed workload: the same N queries for every B, in N / B calls, one timed pass
            N = ARGS.workload
            rng = np.random.default_rng([seed, 99])
            w1, w2 = rng.standard_normal(N), rng.standard_normal(N)       # the same states for every engine
            res["meta"].update(batches=list(WORKLOAD_BATCHES), workload=N)
            for B in WORKLOAD_BATCHES:
                assert N % B == 0, (N, B)
                for name, fn in engines.items():
                    fn(w1[:B], w2[:B], S_TIMING, seed)                     # warm-up for this shape
                    syncs[name]()
                    t0 = time.perf_counter()
                    for i in range(0, N, B):
                        fn(w1[i:i + B], w2[i:i + B], S_TIMING, seed)
                    syncs[name]()
                    total = time.perf_counter() - t0
                    res["batch"].setdefault(name, {}).setdefault(str(B), []).append(total / N)
                    res.setdefault("workload_total_s", {}).setdefault(name, {}).setdefault(str(B), []).append(total)
                    print(f"seed {seed} B={B:5d} {name:16s} {N} queries in {N // B:5d} calls: {total:8.2f} s "
                          f"({1e3 * total / N:7.3f} ms/query)", flush=True)
            os.makedirs(os.path.dirname(OUT), exist_ok=True)
            with open(OUT, "w") as f:
                json.dump(res, f, indent=1)
            continue

        # (a) time per state vs B at S = 2,048
        rng = np.random.default_rng([seed, 99])
        for B in BATCHES:
            b1, b2 = rng.standard_normal(B), rng.standard_normal(B)
            for name, fn in engines.items():
                t = timed(lambda: fn(b1, b2, S_TIMING, seed), syncs[name]) / B
                res["batch"].setdefault(name, {}).setdefault(str(B), []).append(t)
                print(f"seed {seed} B={B:5d} {name:16s} batched      {1e3 * t:9.3f} ms/state", flush=True)
        b1, b2 = rng.standard_normal(LOOP_B), rng.standard_normal(LOOP_B)
        for name, fn in engines.items():
            t = timed(lambda: [fn(b1[j:j + 1], b2[j:j + 1], S_TIMING, seed) for j in range(LOOP_B)],
                      syncs[name]) / LOOP_B
            res["loop"].setdefault(name, []).append(t)
            print(f"seed {seed}          {name:16s} one per call {1e3 * t:9.3f} ms/state", flush=True)
        if not ARGS.only_numpyro:
            t = timed(lambda: pimp.plan_importance(b1[:1], b2[:1], S_TIMING, seed), lambda: None)
            res["loop"].setdefault("pyro-importance/cpu", []).append(t)
            print(f"seed {seed}          {'pyro-importance/cpu':16s} one per call {1e3 * t:9.3f} ms/state",
                  flush=True)
            sc = PC.score(pimp.plan_importance(s1e[:20], s2e[:20], S_TIMING, seed), s1e[:20], s2e[:20])
            res.setdefault("importance_score_20states", []).append(sc)

        # (b) plan score and time per state vs particles, 200 evaluation states
        for S in PARTICLES:
            for name, fn in engines.items():
                acts = fn(s1e, s2e, S, seed)
                t = timed(lambda: fn(s1e, s2e, S, seed), syncs[name]) / C.EVAL_N
                d = res["sweep"].setdefault(name, {}).setdefault(str(S), {"score": [], "time": []})
                d["score"].append(PC.score(acts, s1e, s2e))
                d["time"].append(t)
                print(f"seed {seed} S={S:5d} {name:16s} score {d['score'][-1]:.3f}  "
                      f"{1e3 * t:8.3f} ms/state", flush=True)

        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "w") as f:
            json.dump(res, f, indent=1)
    print(f"wrote {os.path.relpath(OUT)}")


if __name__ == "__main__":
    main()
