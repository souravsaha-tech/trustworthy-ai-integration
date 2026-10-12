"""Event-driven anomaly detection with and without idempotent consumers."""
import numpy as np
from scipy.stats import poisson


def make_events(rng, merchants=200, minutes=24 * 60, tx_per_min_mean=20.0, fail_p=0.01,
                incidents=((37, 600, 20), (121, 1080, 20)), incident_fail_p=0.30):
    """Payment-failure events: (event_id, merchant, minute). Returns events and ground truth."""
    rate = rng.gamma(2.0, tx_per_min_mean / 2.0, merchants)  # heterogeneous merchant volume
    rows = []
    truth = set()
    for m in range(merchants):
        tx = rng.poisson(rate[m], minutes)
        p = np.full(minutes, fail_p)
        for (im, start, dur) in incidents:
            if im == m:
                p[start:start + dur] = incident_fail_p
                truth.update((m, w) for w in range(start // 5, (start + dur) // 5 + 1))
        fails = rng.binomial(tx, p)
        mins = np.repeat(np.arange(minutes), fails)
        rows.append(np.column_stack([np.full(len(mins), m), mins]))
    ev = np.vstack(rows)
    ids = np.arange(len(ev))
    return np.column_stack([ids, ev]), rate, truth


def deliver(events, rng, dup_p=0.02, replay=(720, 780)):
    """At-least-once delivery: random duplicates plus a replay of one hour after a restart."""
    dups = events[rng.random(len(events)) < dup_p]
    rep = events[(events[:, 2] >= replay[0]) & (events[:, 2] < replay[1])]
    return np.vstack([events, dups, rep])


def detect(delivered, rate, merchants, minutes, idempotent, fail_p=0.01, window=5, alpha=1e-6):
    """Alert when a merchant's failures in a 5-minute window exceed a Poisson threshold."""
    ev = delivered
    if idempotent:
        _, first = np.unique(ev[:, 0], return_index=True)
        ev = ev[np.sort(first)]
    nwin = minutes // window
    counts = np.zeros((merchants, nwin), int)
    np.add.at(counts, (ev[:, 1], ev[:, 2] // window), 1)
    thresh = poisson.isf(alpha, rate * window * fail_p)  # per merchant
    alerts = counts > thresh[:, None]
    actions = int(alerts.sum())
    return alerts, actions, len(delivered) - len(ev)
