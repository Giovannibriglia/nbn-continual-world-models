# Neural Bayesian Networks as Continual World Models

Code for the NeurIPS 2026 workshop paper *Neural Bayesian Networks as Continual World Models for Reinforcement Learning* (Workshop on Continual World Models).

## The idea in brief

A Neural Bayesian Network (NBN) is a Bayesian network whose mechanisms, one per variable, are learned conditional densities, such as mixture density networks. The paper turns an NBN into a world model that keeps learning while the agent plans with it. For every new batch of experience, the pipeline does three things:

1. **Test** each mechanism for change. The test compares conditionals given the parents, so a change in the agent's behavior does not raise false alarms. The per-mechanism likelihood, by contrast, does raise them.
2. **Update** only the flagged mechanisms, by a restart. All other mechanisms keep their history.
3. **Plan** by inference in the network. The query is p(a | s, goal reached), computed with batched likelihood weighting.

The experiment is a single toy MDP with continuous, non-Gaussian dynamics (skewed and bimodal noise). Its timeline has two events: a change of the agent's behavior at phase 2 and a drift of one mechanism at phase 5. The pipeline is compared with a frozen model, pooled refitting, a sliding window, and a restart triggered by the raw likelihood.

The NBN library is [`nbn`](https://github.com/Giovannibriglia/NeuralBayesianNetworks) ([PyPI](https://pypi.org/project/nbn/)).

## Install

Python 3.10 or later.

```bash
git clone https://github.com/Giovannibriglia/nbn-continual-world-models.git
cd nbn-continual-world-models
pip install -r requirements.txt      # nbn (with PyTorch), scipy, matplotlib
```

## Run the experiments

All scripts run on a laptop CPU; no GPU is needed.

| Command | What it does | Time |
|---|---|---|
| `python test_v11.py` | Sanity checks: calibration and power of the drift test, planner with the true dynamics (score 0.964), raw likelihood moving under a behavior shift, the W1 floor of the true model, and the exact kNN append. | ~30 s |
| `python continual_v11.py [n_seeds]` | The full timeline (8 phases × 2,000 transitions) for MDN, kNN and linear-Gaussian NBNs and the five maintenance strategies. Writes `results_v11/continual_v11.json`. The paper uses 5 seeds (the default); use at least 2 seeds, since the summary reports standard deviations. | ~5 min per seed |
| `python make_figures_v11_mdn.py` | Figure 2 of the paper (MDN mechanisms) from the JSON; prints the whole-timeline summary quoted in Section 5. Writes `Images/continual_v11_mdn.pdf/.png`. | a few s |

The results used in the paper are included in `results_v11/`, so the figure can be rebuilt without rerunning the experiment. Rerunning `continual_v11.py` overwrites them. The paper reports the MDN results; the script also runs kNN and linear-Gaussian mechanisms.

## Files

- `v11_common.py`: the toy MDP, the NBN world models, the drift test, the raw-likelihood detector, the planner (likelihood weighting), and the metrics.
- `continual_v11.py`: the experiment.
- `test_v11.py`: the checks.
- `make_figures_v11_mdn.py`: the figure.

## Citation

```bibtex
@inproceedings{briglia2026nbn,
  title     = {Neural Bayesian Networks as Continual World Models for Reinforcement Learning},
  author    = {Briglia, Giovanni and Lippi, Marco and Mariani, Stefano and Zambonelli, Franco},
  booktitle = {NeurIPS 2026 Workshop on Continual World Models},
  year      = {2026}
}
```

## License

Apache License 2.0 (see `LICENSE`).
