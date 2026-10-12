# Separating Intelligence from Authority — reference implementation

Companion code for the IEEE Software manuscript *"Separating Intelligence from Authority:
Patterns for Trustworthy AI Integration."* It implements the hybrid payment platform from the
paper (Figure 2) and runs four experiments on **synthetic data only**. No real customer, bank,
or production data is used; all parameters are illustrative.

## What is implemented
| Module | Role in the architecture |
|---|---|
| `aiarch/data.py` | Synthetic card-payment generator (1% fraud) with two fraud regimes for drift |
| `aiarch/model.py` | Fraud model (gradient boosting). Returns a score, never a decision |
| `aiarch/policy.py` | Policy engine (decision authority), model-direct anti-pattern, deterministic fallback rules |
| `aiarch/resilience.py` | Discrete-event simulation of the synchronous path: worker pool, time budget, circuit breaker |
| `aiarch/events.py` | Event-driven anomaly detection with and without idempotent (event-ID) consumers |
| `aiarch/experiments.py` | Experiments E1–E4 |

## Experiments
- **E1 Infrastructure failure**: model 4x slower for 120 s, then unresponsive for 60 s, at 200 authorizations/s.
  Compares no time budget (2 s client timeout), time budget + fallback, and time budget + fallback + circuit breaker.
- **E2 Model failure**: over 60 days fraud shifts to small card-testing payments. Compares input-drift
  monitoring (PSI on all inputs and the score) with outcome monitoring on delayed fraud labels.
- **E3 Decision failure**: a retrained model ships with a feature-pipeline units bug. Compares direct
  deployment, deployment behind the policy engine, and a shadow gate on 10% of traffic.
- **E4 At-least-once delivery**: 2% duplicate events plus a one-hour replay. Compares naive and idempotent consumers.

## Reproduce
Tested with Python 3.13 on Windows.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python run_all.py        # about 3 minutes; writes results/*.csv, summary.json, series_seed1.npz
python make_figure.py    # writes results/Figure3_evaluation.png and .pdf
```

All runs are seeded (seeds 1–10). The committed files in `results/` are the ones reported in the paper.

**Expected variation.** Results that come only from the simulation (failed payments, recovery
times, detection delays, alert counts) are identical across machines. Results that depend on the
trained fraud model (fraud intercepted, fraud value approved, PSI values) can differ slightly with
a different operating system, Python version, or library version, because gradient-boosting
training is not bit-for-bit identical across platforms. The conclusions do not change. Use the
pinned versions in `requirements.txt` to match the paper most closely.

## Key results (mean ± SD over 10 seeds)
See `results/summary.json`, the per-seed CSV files, and Table 2 of the paper.

## Limitations
This is a simulation, not a production measurement. It shows the direction and relative size of
each effect. Workload, latency, fraud, and label-delay parameters are assumptions and can be changed
in `aiarch/`.


## License
MIT. See [LICENSE](LICENSE).
