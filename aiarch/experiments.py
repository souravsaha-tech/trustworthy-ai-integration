"""Experiments E1-E4. Each function returns a dict of metrics for one random seed."""
import numpy as np
from .data import generate, add_features, FEATURES
from .model import train, score
from .policy import policy_decide, model_direct, fallback_rules, outcomes, DECLINE
from .resilience import simulate, fault_schedule
from .events import make_events, deliver, detect

SLO_MS = 150.0


# ---------------------------------------------------------------- E1: infrastructure failure
def e1(seed, rate_per_s=200, duration_s=600, return_series=False):
    rng = np.random.default_rng(seed)
    model = train(rng, seed=seed)
    n = rng.poisson(rate_per_s * duration_s)
    arrivals = np.sort(rng.uniform(0, duration_s * 1000, n))
    tx = generate(n, rng)
    s = score(model, tx)
    svc = fault_schedule(arrivals, rng)
    fault = arrivals >= 200_000                     # fault (200-380 s) and its aftermath
    normal = arrivals < 200_000
    sec = (arrivals // 1000).astype(int)
    base_o = outcomes(policy_decide(s[normal], tx[normal]), tx[normal])
    res, series = {"normal_period": dict(fraud_intercept_pct=100 * base_o["fraud_intercept_rate"],
                                          legit_friction_pct=100 * base_o["legit_friction_rate"])}, {}
    for mode in ("naive", "budget", "budget_cb"):
        r = simulate(arrivals, svc, mode)
        lat = r["latency_ms"]
        d = np.where(r["used_model"], policy_decide(s, tx), fallback_rules(tx))
        d = np.where(r["error"], DECLINE, d)  # failed authorization = payment not completed
        p99 = np.array([np.percentile(lat[sec == k], 99) for k in range(duration_s)])
        model_share = np.array([r["used_model"][sec == k].mean() for k in range(duration_s)])
        # model scoring resumes: first second after 380 s from which the next 10 s are >=95% model-scored
        ok = model_share >= 0.95
        resume = next((k for k in range(0, duration_s - 390) if ok[380 + k:390 + k].all()), np.nan)
        legit = ~tx["is_fraud"].to_numpy()
        o = outcomes(d[fault], tx[fault])
        res[mode] = dict(
            p99_ms=float(np.percentile(lat[fault], 99)),
            slo_violation_pct=float(100 * (lat[fault] > SLO_MS).mean()),
            legit_failed_pct=float(100 * r["error"][fault & legit].mean()),
            model_scored_pct=float(100 * r["used_model"][fault].mean()),
            fallback_pct=float(100 * (~r["used_model"][fault] & ~r["error"][fault]).mean()),
            model_calls=int(r["sent"][fault].sum()),
            model_resume_s_after_fault=float(resume),
            fraud_intercept_pct=100 * o["fraud_intercept_rate"],
            legit_friction_pct=100 * o["legit_friction_rate"],
        )
        err_share = np.array([r["error"][sec == k].mean() for k in range(duration_s)])
        series[mode] = dict(p99=p99, model_share=model_share, failed_share=err_share)
    return (res, series) if return_series else res


# ---------------------------------------------------------------- E2: model failure (drift)
def psi(expected, actual, bins):
    e = np.histogram(expected, bins)[0] / len(expected) + 1e-6
    a = np.histogram(actual, bins)[0] / len(actual) + 1e-6
    return float(np.sum((a - e) * np.log(a / e)))


def e2(seed, days=60, per_day=20_000, onset=25, ramp=10, max_share=0.7,
       label_delay_median_d=7.0, return_series=False):
    rng = np.random.default_rng(seed)
    model = train(rng, seed=seed)
    dfs = []
    for day in range(days):
        share = 0.0 if day < onset else min(max_share, max_share * (day - onset + 1) / ramp)
        df = generate(per_day, rng, drift_share=share)
        df["day"] = day
        dfs.append(df)
    import pandas as pd
    df = pd.concat(dfs, ignore_index=True)
    df["score"] = score(model, df)
    df["intercepted"] = policy_decide(df["score"].to_numpy(), df) != 0
    fraud = df[df.is_fraud].copy()
    fraud["label_day"] = fraud["day"] + rng.lognormal(np.log(label_delay_median_d), 0.5, len(fraud))

    # input drift monitor: daily PSI on every model input and on the score, alarm at 0.2
    base = df[df.day < 14]
    cols = FEATURES + ["score"]
    bins = {c: np.unique(np.quantile(base[c], np.linspace(0, 1, 11))) for c in cols}
    for c in ("new_device", "night"):
        bins[c] = np.array([-0.5, 0.5, 1.5])
    bins["velocity_1h"] = np.array([-0.5, 0.5, 1.5, 2.5, 3.5, 5.5, 100])
    max_psi = np.array([max(psi(base[c], df[df.day == d][c], bins[c]) for c in cols) for d in range(days)])
    psi_alarm = np.flatnonzero((max_psi > 0.2) & (np.arange(days) >= 14))

    # outcome monitor: recall on fraud whose labels have arrived, 7-day window by txn date
    base_recall = fraud[fraud.day < 14]["intercepted"].mean()
    recall_seen = np.full(days, np.nan)
    out_alarm = []
    for d in range(14, days):
        w = fraud[(fraud.day > d - 7) & (fraud.day <= d) & (fraud.label_day <= d)]
        if len(w) >= 30:
            recall_seen[d] = w["intercepted"].mean()
            if recall_seen[d] < base_recall - 0.10:
                out_alarm.append(d)
    true_recall = fraud.groupby("day")["intercepted"].mean().reindex(range(days)).to_numpy()
    first_psi = next((d for d in psi_alarm if d >= onset), np.nan)
    first_out = next((d for d in out_alarm if d >= onset), np.nan)
    false_psi = int(sum(d < onset for d in psi_alarm))
    false_out = int(sum(d < onset for d in out_alarm))
    # fraud dollars that got through between onset and outcome-based detection
    if not np.isnan(first_out):
        w = fraud[(fraud.day >= onset) & (fraud.day <= first_out)]
        loss_undetected = float(w.loc[~w.intercepted, "amount"].sum())
    else:
        loss_undetected = float("nan")
    res = dict(
        baseline_recall_pct=100 * float(base_recall),
        post_drift_recall_pct=100 * float(fraud[fraud.day >= onset + ramp]["intercepted"].mean()),
        max_psi_pre=float(max_psi[14:onset].max()), max_psi_post=float(max_psi[onset:].max()),
        psi_detect_delay_d=float(first_psi - onset) if not np.isnan(first_psi) else np.nan,
        psi_false_alarms=false_psi,
        outcome_detect_delay_d=float(first_out - onset) if not np.isnan(first_out) else np.nan,
        outcome_false_alarms=false_out,
        fraud_loss_before_detection_usd=loss_undetected,
    )
    series = dict(true_recall=true_recall, recall_seen=recall_seen, max_psi=max_psi, base_recall=base_recall)
    return (res, series) if return_series else res


# ---------------------------------------------------------------- E3: decision failure (bad release)
def e3(seed, n=100_000, shadow_frac=0.10):
    rng = np.random.default_rng(seed)
    v1 = train(rng, seed=seed)
    tx = generate(n, rng)
    s1 = score(v1, tx)
    # v2 is retrained on more data, but its feature pipeline divides amounts by 100 (units bug)
    v2 = train(rng, n=300_000, seed=seed + 1000)
    s2_bug = v2.predict_proba(add_features(tx, amount_scale=0.01)[FEATURES])[:, 1]
    s2_ok = score(v2, tx)

    def gate(s_old, s_new, frac):
        """Shadow gate: run the candidate on a slice of live traffic without acting on it."""
        k = int(len(s_old) * frac)
        a, b, t = s_old[:k], s_new[:k], tx.iloc[:k]
        bins = np.array([-1e-9, 0.01, 0.05, 0.1, 0.2, 0.5, 0.9, 1 + 1e-9])
        p = psi(a, b, bins)
        da, db = policy_decide(a, t), policy_decide(b, t)
        flag_ratio = float(((db != 0).mean() + 1e-9) / ((da != 0).mean() + 1e-9))
        disagree = float((da != db).mean())
        passed = (p <= 0.2) and (0.8 <= flag_ratio <= 1.25) and (disagree <= 0.01)
        return passed, p, flag_ratio, disagree

    res = {}
    res["v1_policy"] = outcomes(policy_decide(s1, tx), tx)
    res["v2bug_model_direct"] = outcomes(model_direct(s2_bug, tx), tx)
    res["v2bug_policy"] = outcomes(policy_decide(s2_bug, tx), tx)
    ok_bug, psi_bug, fr_bug, dis_bug = gate(s1, s2_bug, shadow_frac)
    ok_good, psi_good, fr_good, dis_good = gate(s1, s2_ok, shadow_frac)
    # with the shadow gate, the buggy v2 only reaches production if the gate passes
    res["v2bug_shadow_gate"] = res["v2bug_policy"] if ok_bug else res["v1_policy"]
    res["v2good_policy"] = outcomes(policy_decide(s2_ok, tx), tx)
    res["gate"] = dict(bug_promoted=bool(ok_bug), bug_psi=psi_bug, bug_flag_ratio=fr_bug, bug_disagree=dis_bug,
                       good_promoted=bool(ok_good), good_psi=psi_good, good_flag_ratio=fr_good,
                       good_disagree=dis_good)
    return res


# ---------------------------------------------------------------- E4: at-least-once delivery
def e4(seed):
    rng = np.random.default_rng(seed)
    merchants, minutes = 200, 24 * 60
    ev, rate, truth = make_events(rng, merchants=merchants, minutes=minutes)
    delivered = deliver(ev, rng)
    res = {}
    for name, idem in (("naive", False), ("idempotent", True)):
        alerts, actions, dropped = detect(delivered, rate, merchants, minutes, idem)
        hits = {(int(m), int(w)) for m, w in zip(*np.nonzero(alerts))}
        incidents_found = sum(any((m, w) in hits for (mm, w) in truth if mm == m) for m in {m for m, _ in truth})
        res[name] = dict(alerts=len(hits), false_alerts=len(hits - truth),
                         incidents_detected=incidents_found, duplicates_dropped=dropped)
    res["events"] = int(len(ev)); res["delivered"] = int(len(delivered))
    return res
