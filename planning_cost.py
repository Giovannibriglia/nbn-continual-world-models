"""Cost and accuracy of planning by inference (paper, Appendix C).

The planner of Sec. 3 (4) and Sec. 4: a*(s) = mode of p(a | s, R = 1), computed
by nbn's batched likelihood weighting (uniform action prior; particles weighted
by P(R = 1 | S2')).  Two questions, on the toy MDP of Sec. 4 (original dynamics):

  (a) accuracy vs particles: plan score (true success of the chosen action /
      best grid action, in closed form) on the 200 evaluation states, for
      S in {64, ..., 8192} particles per state; world model = the true dynamics
      or the MDN NBN fitted on phase-0 data (2,000 transitions);
  (b) cost of batching: wall-clock time per planned state when B states are
      answered by ONE batched call versus B calls of one state each, at
      S = 2,048, on a laptop CPU (1 thread and PyTorch's default thread count) and on the Apple
      GPU (MPS) when available.

Seeds 0..n_seeds-1: the seed fixes the phase-0 data, the MDN fit and the
particles.  The mode-finding step is the same as v11_common.plan_actions
(weighted Gaussian kernel, bandwidth 0.05, 201-point action grid), computed in
chunks of states so that S = 8192 fits in memory; test_v11.py checks that it
returns the same actions as plan_actions.

Output: results_v11/planning_cost.json
Run:    python planning_cost.py [n_seeds=5]
"""
import json
import os
import sys
import time

import numpy as np
import torch
from torch.distributions import Normal, Uniform

import nbn
import v11_common as C
from nbn.inference.likelihood_weighting import LikelihoodWeightingEngine
from test_v11 import oracle_model

PARTICLES = (64, 128, 256, 512, 1024, 2048, 4096, 8192)
BATCHES = (1, 8, 64, 200, 1024)
S_TIMING = C.LW_PARTICLES          # 2,048, as in Sec. 4
REPS = 3                           # timing repetitions (median is kept)
CHUNK = 32                         # states per chunk in the mode-finding step
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_v11", "planning_cost.json")


class DeviceRoot(C.FixedRoot):
    """C.FixedRoot with its distribution built on the model's device (for MPS)."""

    def __init__(self, kind, device):
        super().__init__(kind)
        self.dev = torch.device(device)

    def forward(self, parents):
        one = torch.ones(1, device=self.dev)
        if self.kind == "uniform":
            return Uniform(-one, one, validate_args=False)
        return Normal(0.0 * one, one)


def to_device(model, device):
    m = C.clone(model).to(device)
    for n, kind in (("S1", "normal"), ("S2", "normal"), ("A", "uniform")):
        m.set_mechanism(n, DeviceRoot(kind, device))
    return m


def plan(model, s1, s2, n_particles, seed, bw=0.05):
    """Same planner as C.plan_actions, with S particles and a chunked mode step."""
    dev = model.device
    torch.manual_seed(seed)
    eng = LikelihoodWeightingEngine(n_samples=n_particles)
    ev = {"S1": torch.as_tensor(s1, dtype=torch.float32, device=dev).unsqueeze(-1),
          "S2": torch.as_tensor(s2, dtype=torch.float32, device=dev).unsqueeze(-1),
          "R": torch.ones(len(s1), 1, device=dev)}
    w, samp = eng.query(model, ["A"], ev)
    a = samp[..., 0]
    dt = torch.float64 if dev.type == "cpu" else torch.float32   # MPS has no float64
    grid = torch.as_tensor(C.A_GRID, dtype=dt, device=dev)
    out = []
    for i in range(0, len(s1), CHUNK):
        ai, wi = a[i:i + CHUNK].to(dt), w[i:i + CHUNK].to(dt)
        kern = torch.exp(-0.5 * ((grid[None, None, :] - ai[..., None]) / bw) ** 2)
        out.append((wi[..., None] * kern).sum(1).argmax(1))
    return C.A_GRID[torch.cat(out).cpu().numpy()]


def score(actions, s1, s2, drifted=False):
    got = C.true_success(s1, s2, actions, drifted)
    best = C.true_success(s1[:, None], s2[:, None], C.A_GRID[None, :], drifted).max(1)
    return float(got.mean() / best.mean())


def fit_phase0(seed):
    rng = np.random.default_rng(seed)
    data0 = C.simulate(rng, C.N_PER_PHASE, *C.phase_flags(0))   # = phase 0 of continual_v11
    torch.manual_seed(seed)
    m = C.new_model("mdn")
    for n in C.TRANSITION_NODES:
        C.fit_node(m, "mdn", n, data0)
    return m


def sync(dev):
    if dev.type == "mps":
        torch.mps.synchronize()


def timed(fn, dev):
    ts = []
    for _ in range(REPS):
        sync(dev)
        t0 = time.perf_counter()
        fn()
        sync(dev)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    seeds = range(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
    s1, s2 = C.eval_states()
    n_threads = torch.get_num_threads()
    devices = [("cpu_1thread", "cpu", 1), (f"cpu_{n_threads}threads", "cpu", n_threads)]
    if torch.backends.mps.is_available():
        devices.append(("mps", "mps", n_threads))
    res = {"meta": {"particles": PARTICLES, "batches": BATCHES, "s_timing": S_TIMING,
                    "reps": REPS, "seeds": list(seeds), "nbn": nbn.__version__,
                    "torch": torch.__version__, "devices": [d[0] for d in devices]},
           "score": {"true": {}, "mdn": {}}, "time": {}}

    oracle = oracle_model(False)
    for seed in seeds:
        mdn = fit_phase0(seed)
        torch.set_num_threads(1)
        # (a) accuracy vs particles
        for S in PARTICLES:
            for name, m in (("true", oracle), ("mdn", mdn)):
                sc = score(plan(m, s1, s2, S, seed), s1, s2)
                res["score"][name].setdefault(str(S), []).append(sc)
            print(f"seed {seed} S={S:5d}: true {res['score']['true'][str(S)][-1]:.3f}  "
                  f"mdn {res['score']['mdn'][str(S)][-1]:.3f}", flush=True)
        # (b) cost of batching
        rng = np.random.default_rng([seed, 99])
        for label, dev_name, threads in devices:
            torch.set_num_threads(threads)
            m = mdn if dev_name == "cpu" else to_device(mdn, dev_name)
            dev = m.device
            plan(m, s1[:8], s2[:8], S_TIMING, seed)                       # warm-up
            for B in BATCHES:
                b1, b2 = rng.standard_normal(B), rng.standard_normal(B)
                t_batch = timed(lambda: plan(m, b1, b2, S_TIMING, seed), dev)
                t_loop = timed(lambda: [plan(m, b1[j:j + 1], b2[j:j + 1], S_TIMING, seed)
                                        for j in range(B)], dev)
                d = res["time"].setdefault(label, {}).setdefault(str(B), {"batched": [], "one_at_a_time": []})
                d["batched"].append(t_batch / B)
                d["one_at_a_time"].append(t_loop / B)
                print(f"seed {seed} {label:14s} B={B:4d}: per state batched {1e3 * t_batch / B:8.3f} ms, "
                      f"one at a time {1e3 * t_loop / B:8.3f} ms", flush=True)
        torch.set_num_threads(n_threads)
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "w") as f:
            json.dump(res, f, indent=1)
    print(f"wrote {os.path.relpath(OUT)}")


if __name__ == "__main__":
    main()
