"""Shared code for the experiments of the paper (Sec. 3: Test / Update / Plan;
Sec. 4: the single continuous, non-Gaussian toy).  Requires `pip install nbn`.

Contents
  * the toy factored MDP (continuous, non-Gaussian, heteroscedastic) with a
    behavior shift and a mechanism drift on one timeline;
  * NBN world models whose transition mechanisms are MDN or kNN (nbn), with
    fixed root mechanisms (states are observed, actions are chosen by the
    planner, so roots never need learning);
  * Test: matched-pair permutation test of "mechanism unchanged", per
    mechanism, model-free (uses data only);
  * raw-score detector: drop of the out-of-sample mean log-likelihood;
  * Update: per mechanism, append (no flag) or restart (flag);
  * Plan: planning as inference, p(a | s, R = 1) with R = 1{goal reached},
    answered by nbn's likelihood weighting; action = posterior mode;
  * metrics: KL(true || model) of the transition kernel and success
    probability of the planned action in the true MDP, relative to optimal.
"""
from __future__ import annotations

import copy
import math
from typing import Dict

import numpy as np
import torch
from scipy import stats
from scipy.spatial import cKDTree
from torch.distributions import Normal, Uniform

from nbn import NeuralBayesianNetwork
from nbn.inference.likelihood_weighting import LikelihoodWeightingEngine
from nbn.mechanisms.base import Mechanism
from nbn.mechanisms.non_parametric.knn_conditional import KNNConditionalMechanism
from nbn.mechanisms.parametric.linear_gaussian import LinearGaussianMechanism
from nbn.mechanisms.parametric.mdn import MDNMechanism
from nbn.utils.batching import pack_parents

# ----------------------------------------------------------------------------
# Timeline
# ----------------------------------------------------------------------------
N_PHASES = 8
SHIFT_AT = 2          # behavior shift: from this phase on, states and policy change
DRIFT_AT = 5          # mechanism drift: from this phase on, S2' reacts differently to A
N_PER_PHASE = 2000
EDGES = [("S1", "S1n"), ("A", "S1n"), ("S2", "S2n"), ("A", "S2n"), ("S1", "S2n")]
REWARD_EDGE = ("S2n", "R")      # R = 1 when the goal is reached (known to the agent)
TRANSITION_NODES = ("S1n", "S2n")
ROOTS = ("S1", "S2", "A")

# ----------------------------------------------------------------------------
# True dynamics
#   S1' = 0.8 S1 + 0.6 A + 0.3 e1,              e1 = (G - 2)/sqrt(2), G ~ Gamma(2, 1)
#   S2' = 0.6 S2 - 0.4 S1 + c tanh(2A) + sig(S1) e2,
#         e2 = 0.9 B + sqrt(1 - 0.81) Z,  B = +-1 (prob 1/2), Z ~ N(0, 1)
#         sig(S1) = 0.15 + 0.45 * logistic(2 S1)
#   c = +1.2 before the drift, -1.2 after (the action's effect is reversed)
# e1 is skewed, e2 is bimodal, and the noise of S2' grows with S1.
# ----------------------------------------------------------------------------
C_PRE, C_POST = 1.2, -1.2
M_B = 0.9
SD_Z = math.sqrt(1.0 - M_B ** 2)
GOAL, GOAL_HALF = 0.0, 0.2       # success: |S2'| < 0.2


def sig(s1):
    return 0.15 + 0.45 / (1.0 + np.exp(-2.0 * s1))


def mean2(s1, s2, a, drifted):
    c = C_POST if drifted else C_PRE
    return 0.6 * s2 - 0.4 * s1 + c * np.tanh(2.0 * a)


def sample_parents(rng, n, shifted):
    if not shifted:              # original behavior: wide states, uniform actions
        s1 = rng.standard_normal(n)
        s2 = rng.standard_normal(n)
        a = rng.uniform(-1.0, 1.0, n)
    else:                        # new behavior: states move, the policy depends on S1
        s1 = 0.8 + 0.7 * rng.standard_normal(n)
        s2 = -0.4 + 1.0 * rng.standard_normal(n)
        a = np.clip(-0.4 * s1 + 0.5 * rng.standard_normal(n), -1.0, 1.0)
    return s1, s2, a


def sample_children(rng, s1, s2, a, drifted):
    n = len(s1)
    e1 = (rng.gamma(2.0, 1.0, n) - 2.0) / math.sqrt(2.0)
    s1n = 0.8 * s1 + 0.6 * a + 0.3 * e1
    b = np.where(rng.random(n) < 0.5, -1.0, 1.0)
    e2 = M_B * b + SD_Z * rng.standard_normal(n)
    s2n = mean2(s1, s2, a, drifted) + sig(s1) * e2
    return s1n, s2n


def simulate(rng, n, shifted, drifted):
    s1, s2, a = sample_parents(rng, n, shifted)
    s1n, s2n = sample_children(rng, s1, s2, a, drifted)
    return {k: torch.as_tensor(v, dtype=torch.float32).unsqueeze(-1)
            for k, v in dict(S1=s1, S2=s2, A=a, S1n=s1n, S2n=s2n).items()}


def phase_flags(t):
    return t >= SHIFT_AT, t >= DRIFT_AT


def true_logpdf(node, x, s1, s2, a, drifted):
    if node == "S1n":
        g = (x - 0.8 * s1 - 0.6 * a) * math.sqrt(2.0) / 0.3 + 2.0
        return stats.gamma.logpdf(g, 2.0) + math.log(math.sqrt(2.0) / 0.3)
    m, s = mean2(s1, s2, a, drifted), sig(s1)
    l1 = stats.norm.logpdf(x, m + s * M_B, s * SD_Z)
    l2 = stats.norm.logpdf(x, m - s * M_B, s * SD_Z)
    return np.logaddexp(l1, l2) + math.log(0.5)


def true_success(s1, s2, a, drifted):
    """P(|S2'| < GOAL_HALF | s1, s2, a) in the true MDP (closed form)."""
    m, s = mean2(s1, s2, a, drifted), sig(s1)
    out = 0.0
    for sign in (1.0, -1.0):
        mu = m + sign * s * M_B
        sd = s * SD_Z
        out = out + 0.5 * (stats.norm.cdf((GOAL + GOAL_HALF - mu) / sd)
                           - stats.norm.cdf((GOAL - GOAL_HALF - mu) / sd))
    return out


def concat(datasets):
    return {k: torch.cat([d[k] for d in datasets]) for k in datasets[0]}


# ----------------------------------------------------------------------------
# NBN world model: fixed roots, learned transition mechanisms
# ----------------------------------------------------------------------------
class FixedRoot(Mechanism):
    """A root with a fixed distribution (not learned).

    States are observed in every query (their log-density is a constant that
    cancels in the self-normalised weights); the action prior is the
    planner's proposal, uniform on [-1, 1].
    """

    def __init__(self, kind: str):
        super().__init__()
        self.kind = kind

    def forward(self, parents):
        one = torch.ones(1)
        if self.kind == "uniform":
            return Uniform(-one, one, validate_args=False)
        return Normal(0.0 * one, one)

    def fit_local(self, x, parents, **kw):
        return {}

    def is_fitted(self):
        return True


FAMILIES = ("mdn", "knn", "lg")
MDN_EPOCHS, MDN_EPOCHS_WARM = 200, 60


def new_mechanism(family):
    if family == "mdn":
        return MDNMechanism(num_components=5, hidden=(64, 64))
    if family == "knn":
        return KNNConditionalMechanism()
    return LinearGaussianMechanism()


class GoalReward(Mechanism):
    """P(R = 1 | s2') = 1{|s2' - goal| < GOAL_HALF}, floored at 1e-6.

    The agent's own goal, not learned.  Conditioning on R = 1 turns planning
    into inference: p(a | s, R = 1) is proportional to P(success | s, a) under
    a uniform action prior.
    """
    is_discrete = True

    def forward(self, parents):
        raise NotImplementedError

    def log_prob(self, x, parents):
        hit = ((parents.reshape(-1) - GOAL).abs() < GOAL_HALF).float()
        p1 = hit * (1.0 - 1e-6) + (1.0 - hit) * 1e-6
        r = x.reshape(-1).float()
        return torch.log(r * p1 + (1.0 - r) * (1.0 - p1))

    def fit_local(self, x, parents, **kw):
        return {}

    def is_fitted(self):
        return True


def new_model(family) -> NeuralBayesianNetwork:
    nodes = ("S1", "S2", "A", "S1n", "S2n")
    variables = dict.fromkeys(nodes, ("continuous", 1))
    variables["R"] = ("discrete", 2)
    m = NeuralBayesianNetwork(EDGES + [REWARD_EDGE], variables=variables, device="cpu")
    m.set_mechanism("R", GoalReward())
    m.set_mechanism("S1", FixedRoot("normal"))
    m.set_mechanism("S2", FixedRoot("normal"))
    m.set_mechanism("A", FixedRoot("uniform"))
    for n in TRANSITION_NODES:
        m.set_mechanism(n, new_mechanism(family))
    return m


def fit_node(model, family, node, data, warm=False):
    """(Re)fit one transition mechanism on `data` (all rows since its last restart)."""
    pa = pack_parents(data, model.dag.parents(node))
    mech = model.mechanisms[node]
    if family == "mdn":
        mech.fit_local(data[node], pa, epochs=MDN_EPOCHS_WARM if warm else MDN_EPOCHS,
                       consolidate=False, warm_start=warm)
    else:   # kNN stores the sample; LG solves the normal equations (both exact)
        model.mechanisms[node] = new_mechanism(family)
        model.mechanisms[node].fit_local(data[node], pa)
    model._cache_version += 1


def node_logprob(model, node, data):
    pa = pack_parents(data, model.dag.parents(node))
    with torch.no_grad():
        return model.mechanisms[node].log_prob(data[node], pa).reshape(-1).double().numpy()


# ----------------------------------------------------------------------------
# Test: matched-pair permutation test, per mechanism (model-free)
# ----------------------------------------------------------------------------
def _poly3(z):
    cols = [np.ones(len(z))]
    d = z.shape[1]
    for i in range(d):
        cols += [z[:, i], z[:, i] ** 2, z[:, i] ** 3]
        for j in range(i + 1, d):
            cols.append(z[:, i] * z[:, j])
    return np.stack(cols, 1)


def matched_pair_test(pa_ref, x_ref, pa_new, x_new, rng, radius=0.25, n_perm=999):
    """Test H0: p(x | pa) is the same in the reference data and in the new batch.

    1. Pair each new transition with the closest unused reference transition in
       (standardised) parent space; drop it if no partner lies within `radius`
       (no overlap = no evidence, as a table test skips unseen configurations).
    2. Residuals r = x - g(pa), with g a cubic regression fitted on the
       reference data; this corrects the small parent mismatch of each pair
       (bias-corrected matching).
    3. Under H0 the two members of a pair are exchangeable, so swapping them
       (= flipping the sign of d = r_new - r_ref and of
       d2 = r_new^2 - r_ref^2) leaves the law of the data unchanged.  The
       statistic is how well the parents predict d (location change) plus how
       well they predict d2 (spread or shape change): the sum of the two R^2 of
       a cubic regression on the parents.  The p-value compares it with random
       sign flips.
    Returns (p-value, number of pairs).
    """
    mu = pa_ref.mean(0)
    sd = pa_ref.std(0) + 1e-9
    zr, zn = (pa_ref - mu) / sd, (pa_new - mu) / sd
    tree = cKDTree(zr)
    k = min(32, len(zr))
    dist, idx = tree.query(zn, k=k)
    dist, idx = dist.reshape(len(zn), -1), idx.reshape(len(zn), -1)
    used = np.zeros(len(zr), bool)
    pairs = []
    for i in rng.permutation(len(zn)):
        for d, j in zip(dist[i], idx[i]):
            if d > radius:
                break
            if not used[j]:
                used[j] = True
                pairs.append((i, j))
                break
    m = len(pairs)
    if m < 50:
        return 1.0, m
    pi = np.array(pairs)
    beta = np.linalg.lstsq(_poly3(zr), x_ref, rcond=None)[0]
    rn = x_new[pi[:, 0]] - _poly3(zn[pi[:, 0]]) @ beta
    rr = x_ref[pi[:, 1]] - _poly3(zr[pi[:, 1]]) @ beta
    d, d2 = rn - rr, rn ** 2 - rr ** 2
    zmid = 0.5 * (zn[pi[:, 0]] + zr[pi[:, 1]])
    Q, _ = np.linalg.qr(_poly3(zmid))
    signs = np.vstack([np.ones((1, m)), np.where(rng.random((n_perm, m)) < 0.5, -1.0, 1.0)])
    T = (((signs * d) @ Q) ** 2).sum(1) / (d @ d) + (((signs * d2) @ Q) ** 2).sum(1) / (d2 @ d2)
    p = (1.0 + (T[1:] >= T[0]).sum()) / (1.0 + n_perm)
    return float(p), m


def test_node(node, ref, new, rng, **kw):
    pa_names = [p for p, c in EDGES if c == node]
    pr = torch.cat([ref[p] for p in pa_names], 1).numpy().astype(float)
    pn = torch.cat([new[p] for p in pa_names], 1).numpy().astype(float)
    return matched_pair_test(pr, ref[node].numpy().ravel().astype(float),
                             pn, new[node].numpy().ravel().astype(float), rng, **kw)


def raw_score_flag(ll_new, ll_prev, level):
    """One-sided two-sample z-test: has the out-of-sample mean log-lik dropped?"""
    z = (ll_prev.mean() - ll_new.mean()) / math.sqrt(
        ll_new.var(ddof=1) / len(ll_new) + ll_prev.var(ddof=1) / len(ll_prev))
    return bool(stats.norm.sf(z) < level)


# ----------------------------------------------------------------------------
# Plan: planning as inference, answered by likelihood weighting
# ----------------------------------------------------------------------------
EVAL_N = 200
A_GRID = np.linspace(-1.0, 1.0, 201)
LW_PARTICLES = 2048
_LW = LikelihoodWeightingEngine(n_samples=LW_PARTICLES)


def eval_states():
    rng = np.random.default_rng(12345)
    return rng.standard_normal(EVAL_N), rng.standard_normal(EVAL_N)


def plan_actions(model, s1, s2, bw=0.05, seed=0):
    """a*(s) = mode of p(a | S1=s1, S2=s2, R=1) under a uniform action prior.

    One batched likelihood-weighting call for all states: each particle draws
    an action from the prior and S1', S2' from the learned mechanisms, and is
    weighted by P(R = 1 | S2'), i.e. by whether the goal is reached.  The
    posterior is proportional to the success probability of each action; its
    mode is read from a weighted kernel density on a fine action grid (the
    posterior can be bimodal, so its mean would be a poor action).
    """
    torch.manual_seed(seed)
    ev = {"S1": torch.as_tensor(s1, dtype=torch.float32).unsqueeze(-1),
          "S2": torch.as_tensor(s2, dtype=torch.float32).unsqueeze(-1),
          "R": torch.ones(len(s1), 1)}
    w, samp = _LW.query(model, ["A"], ev)
    w = w.double().numpy()                                  # [B, S]
    a = samp[..., 0].double().numpy()                       # [B, S]
    kern = np.exp(-0.5 * ((A_GRID[None, None, :] - a[..., None]) / bw) ** 2)
    dens = (w[..., None] * kern).sum(1)                     # [B, grid]
    return A_GRID[dens.argmax(1)]


def plan_score(model, drifted):
    s1, s2 = eval_states()
    a = plan_actions(model, s1, s2)
    got = true_success(s1, s2, a, drifted)
    best = true_success(s1[:, None], s2[:, None], A_GRID[None, :], drifted).max(1)
    return float(got.mean() / best.mean())


# ----------------------------------------------------------------------------
# Model error: KL(true || model) of the transition kernel
# ----------------------------------------------------------------------------
def kl_error(model, drifted, n_pa=400, n_x=16):
    rng = np.random.default_rng(777)
    s1 = np.repeat(rng.standard_normal(n_pa), n_x)
    s2 = np.repeat(rng.standard_normal(n_pa), n_x)
    a = np.repeat(rng.uniform(-1.0, 1.0, n_pa), n_x)
    s1n, s2n = sample_children(rng, s1, s2, a, drifted)
    data = {k: torch.as_tensor(v, dtype=torch.float32).unsqueeze(-1)
            for k, v in dict(S1=s1, S2=s2, A=a, S1n=s1n, S2n=s2n).items()}
    out = {}
    for node, x in (("S1n", s1n), ("S2n", s2n)):
        lp_t = true_logpdf(node, x, s1, s2, a, drifted)
        lp_m = node_logprob(model, node, data)
        out[node] = float(np.mean(lp_t - lp_m))
    out["joint"] = out["S1n"] + out["S2n"]
    return out


def w1_error(model, drifted, n_pa=400, n_x=256):
    """Model error as Wasserstein-1 between model and true next-state laws.

    Mean over parent points from the original behavior; per point, W1 of the
    two 1-D conditionals from n_x sorted samples each.  With children that are
    conditionally independent given the parents (true of the model and of the
    MDP) and an L1 ground cost, the W1 of the joint kernel is exactly the sum
    of the per-mechanism W1, reported as 'joint'.
    """
    rng = np.random.default_rng(778)
    s1 = rng.standard_normal(n_pa)
    s2 = rng.standard_normal(n_pa)
    a = rng.uniform(-1.0, 1.0, n_pa)
    rep = lambda v: np.repeat(v, n_x)
    t1, t2 = sample_children(rng, rep(s1), rep(s2), rep(a), drifted)
    truth = {"S1n": t1.reshape(n_pa, n_x), "S2n": t2.reshape(n_pa, n_x)}
    data = {k: torch.as_tensor(rep(v), dtype=torch.float32).unsqueeze(-1)
            for k, v in dict(S1=s1, S2=s2, A=a).items()}
    out = {}
    torch.manual_seed(0)
    for node in TRANSITION_NODES:
        pa = pack_parents(data, model.dag.parents(node))      # one row per sample
        with torch.no_grad():
            xm = model.mechanisms[node].sample(pa, n=1).reshape(n_pa, n_x).double().numpy()
        out[node] = float(np.abs(np.sort(xm, 1) - np.sort(truth[node], 1)).mean())
    out["joint"] = out["S1n"] + out["S2n"]
    return out


def clone(model):
    return copy.deepcopy(model)
