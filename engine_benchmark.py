"""The planner of the paper in NBN, Pyro and NumPyro (Appendix C, Figure 4).

The query is the planner of Sec. 3 (4) and Sec. 4: for each state s,
a*(s) = mode of p(a | s, R = 1), by likelihood weighting (uniform action
prior, particles weighted by P(R = 1 | S2')), with the mode read from a
weighted Gaussian kernel (bandwidth 0.05) on the 201-point action grid.
One query = one state.

Every engine runs the SAME world model (the MDN NBN fitted on phase-0 data,
as in planning_cost.py; its trained weights are copied into the Pyro and
NumPyro models, and check_ports verifies the copy) and the SAME estimator;
the engines differ only in how the computation is carried out:

  nbn          nbn's batched LikelihoodWeightingEngine (= planning_cost.plan)
  pyro-vec     Pyro model with plates over states and particles, one
               poutine.trace; log-weights = log-probs of the observed sites
  numpyro-vec  the same model in NumPyro, compiled with jax.jit (JAX compiles
               once per input shape; the first call of each shape is timed)

Two modes:

  accuracy   Figure 4b: plan score on the 200 evaluation states for
             S in PARTICLES particles per state, CPU engines, --seeds seeds
             (the seed fixes the phase-0 data, the MDN fit and the particles).
             Output: results_v11/engine_accuracy.json
  speed      Figure 4a: each engine answers the same --workload random
             states (default 10,240) at 2,048 particles, in workload / B
             calls of B states, B in BATCHES; one untimed call per engine and
             B, then the whole workload is timed once.  Torch engines run on
             the CPU and, when available, on the CUDA GPU; NumPyro runs on
             --jax-platform.  JAX and torch share the GPU badly in one
             process, so NumPyro on the GPU runs in a second process:
               python engine_benchmark.py speed
               python engine_benchmark.py speed --jax-platform cuda --only-numpyro
             Output: results_v11/engine_speed.json and
                     results_v11/engine_speed_numpyro_cuda.json

Run:  python engine_benchmark.py accuracy [--seeds 5]
      python engine_benchmark.py speed [--workload 10240] [--jax-platform cpu|cuda] [--only-numpyro]
"""
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=("accuracy", "speed"))
ap.add_argument("--seeds", type=int, default=5, help="accuracy: number of seeds")
ap.add_argument("--workload", type=int, default=10240, help="speed: queries answered for every B")
ap.add_argument("--jax-platform", default="cpu", choices=("cpu", "cuda"), help="device of the NumPyro engine")
ap.add_argument("--only-numpyro", action="store_true", help="speed: time the NumPyro engine only")
ARGS = ap.parse_args()
os.environ["JAX_PLATFORMS"] = ARGS.jax_platform
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import json
import time

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

S_TIMING = C.LW_PARTICLES                      # 2,048, as in Sec. 4
BATCHES = (1, 8, 64, 256, 1024)                # all divide the default workload, 10,240 = 10 x 1,024
PARTICLES = (64, 256, 1024, 2048, 4096, 8192)
CHUNK = PC.CHUNK
BW = 0.05
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_v11")

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


# ----------------------------------------------------------------------------
# Modes
# ----------------------------------------------------------------------------
def sync_for(dev):
    return torch.cuda.synchronize if dev == "cuda" else (lambda: None)


def meta():
    return {"versions": {"nbn": nbn.__version__, "torch": torch.__version__, "pyro": pyro.__version__,
                         "numpyro": numpyro.__version__, "jax": jax.__version__},
            "torch_threads": torch.get_num_threads(), "jax_devices": [str(d) for d in jax.devices()],
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def save(res, name):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {os.path.relpath(path)}")


def accuracy():
    s1e, s2e = C.eval_states()
    res = {"meta": {**meta(), "particles": PARTICLES, "seeds": list(range(ARGS.seeds))},
           "port_check": {}, "score": {}}
    for seed in range(ARGS.seeds):
        mdn = PC.fit_phase0(seed)
        params = {n: mdn_params(mdn, n) for n in C.TRANSITION_NODES}
        np_wm = NumPyroWM(params)
        res["port_check"][seed] = check_ports(mdn, PyroWM(params, "cpu"), np_wm)
        engines = {"nbn/cpu": lambda s1, s2, S, sd: PC.plan(mdn, s1, s2, S, sd),
                   "pyro-vec/cpu": PyroWM(params, "cpu").plan,
                   "numpyro-vec/cpu": np_wm.plan}
        for S in PARTICLES:
            for name, fn in engines.items():
                sc = PC.score(fn(s1e, s2e, S, seed), s1e, s2e)
                res["score"].setdefault(name, {}).setdefault(str(S), []).append(sc)
                print(f"seed {seed} S={S:5d} {name:16s} score {sc:.3f}", flush=True)
        save(res, "engine_accuracy.json")


def speed():
    n = ARGS.workload
    gpu = "cuda" if torch.cuda.is_available() else None
    assert all(n % b == 0 for b in BATCHES), (n, BATCHES)
    mdn = PC.fit_phase0(0)
    params = {k: mdn_params(mdn, k) for k in C.TRANSITION_NODES}
    np_wm = NumPyroWM(params)
    res = {"meta": {**meta(), "workload": n, "batches": BATCHES, "s_timing": S_TIMING, "seed": 0},
           "port_check": check_ports(mdn, PyroWM(params, gpu or "cpu"), np_wm),
           "compile_s": {}, "total_s": {}}

    engines, syncs = {}, {}
    if not ARGS.only_numpyro:
        engines["nbn/cpu"] = lambda s1, s2, S, sd: PC.plan(mdn, s1, s2, S, sd)
        engines["pyro-vec/cpu"] = PyroWM(params, "cpu").plan
        if gpu:
            m_gpu = PC.to_device(mdn, gpu)
            engines[f"nbn/{gpu}"] = lambda s1, s2, S, sd: PC.plan(m_gpu, s1, s2, S, sd)
            engines[f"pyro-vec/{gpu}"] = PyroWM(params, gpu).plan
    engines[f"numpyro-vec/{ARGS.jax_platform}"] = np_wm.plan     # plan() blocks until the result is ready
    syncs = {k: sync_for(k.split("/")[1]) for k in engines}

    # JAX compiles once per input shape: time the first call of each B (compile + one call)
    for b in BATCHES:
        t0 = time.perf_counter()
        np_wm.plan(np.zeros(b), np.zeros(b), S_TIMING, 0)
        res["compile_s"][str(b)] = time.perf_counter() - t0
    print(f"numpyro first call (s) per B: { {k: round(v, 2) for k, v in res['compile_s'].items()} }", flush=True)

    rng = np.random.default_rng([0, 99])
    w1, w2 = rng.standard_normal(n), rng.standard_normal(n)        # the same states for every engine
    for b in BATCHES:
        for name, fn in engines.items():
            fn(w1[:b], w2[:b], S_TIMING, 0)                          # untimed: wakes the device for this shape
            syncs[name]()
            t0 = time.perf_counter()
            for i in range(0, n, b):
                fn(w1[i:i + b], w2[i:i + b], S_TIMING, 0)
            syncs[name]()
            total = time.perf_counter() - t0
            res["total_s"].setdefault(name, {})[str(b)] = total
            print(f"B={b:5d} {name:16s} {n} queries in {n // b:5d} calls: {total:8.2f} s", flush=True)
    save(res, f"engine_speed{'_numpyro_' + ARGS.jax_platform if ARGS.only_numpyro else ''}.json")


if __name__ == "__main__":
    accuracy() if ARGS.mode == "accuracy" else speed()
