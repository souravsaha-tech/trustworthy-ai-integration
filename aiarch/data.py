"""Synthetic card-transaction generator with two fraud regimes.

Regime A: classic fraud (large amounts, new devices, high velocity, risky merchants).
Regime B: shifted fraud (small card-testing amounts, known devices, long distance).
Nothing here is derived from real bank data; all parameters are illustrative.
"""
import numpy as np
import pandas as pd

FEATURES = ["log_amount", "new_device", "velocity_1h", "merchant_risk",
            "log_distance", "log_card_age", "night"]


def _legit(n, rng):
    return dict(
        amount=rng.lognormal(3.6, 1.0, n),           # median about $37
        new_device=rng.random(n) < 0.05,
        velocity_1h=rng.poisson(1.0, n),
        merchant_risk=rng.beta(2, 8, n),
        distance=rng.lognormal(2.0, 1.0, n),         # km from usual location
        card_age=rng.lognormal(6.0, 0.8, n),         # days
        hour=rng.integers(0, 24, n),
    )


def _fraud_a(n, rng):
    return dict(
        amount=rng.lognormal(5.2, 0.9, n),           # median about $180
        new_device=rng.random(n) < 0.60,
        velocity_1h=rng.poisson(4.0, n),
        merchant_risk=rng.beta(5, 3, n),
        distance=rng.lognormal(3.5, 1.2, n),
        card_age=rng.lognormal(5.0, 1.0, n),
        hour=np.where(rng.random(n) < 0.5, rng.integers(0, 6, n), rng.integers(0, 24, n)),
    )


def _fraud_b(n, rng):
    return dict(
        amount=rng.lognormal(2.3, 0.6, n),           # median about $10
        new_device=rng.random(n) < 0.08,
        velocity_1h=rng.poisson(1.6, n),
        merchant_risk=rng.beta(3, 6, n),
        distance=rng.lognormal(4.5, 0.8, n),
        card_age=rng.lognormal(6.0, 0.8, n),
        hour=rng.integers(0, 24, n),
    )


def generate(n, rng, fraud_rate=0.01, drift_share=0.0):
    """Return a DataFrame of n transactions.

    drift_share: fraction of fraudulent transactions drawn from regime B.
    """
    is_fraud = rng.random(n) < fraud_rate
    k = int(is_fraud.sum())
    kb = int(rng.binomial(k, drift_share)) if k else 0
    cols = {c: np.empty(n, dtype=v.dtype) for c, v in _legit(1, rng).items()}
    parts = [(~is_fraud, _legit((~is_fraud).sum(), rng))]
    idx = np.flatnonzero(is_fraud)
    rng.shuffle(idx)
    ia, ib = idx[kb:], idx[:kb]
    for sel, gen in ((ia, _fraud_a), (ib, _fraud_b)):
        m = np.zeros(n, bool); m[sel] = True
        parts.append((m, gen(len(sel), rng)))
    for m, d in parts:
        for c, v in d.items():
            cols[c][m] = v
    df = pd.DataFrame(cols)
    df["is_fraud"] = is_fraud
    df["regime_b"] = False
    df.loc[ib, "regime_b"] = True
    return add_features(df)


def add_features(df, amount_scale=1.0):
    """Feature pipeline. amount_scale != 1 simulates a units bug (e.g., cents vs dollars)."""
    out = df.copy()
    out["log_amount"] = np.log1p(out["amount"] * amount_scale)
    out["log_distance"] = np.log1p(out["distance"])
    out["log_card_age"] = np.log1p(out["card_age"])
    out["night"] = (out["hour"] < 6).astype(int)
    out["new_device"] = out["new_device"].astype(int)
    return out
