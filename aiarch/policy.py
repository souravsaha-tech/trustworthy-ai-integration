"""Policy engine (the 'authority'): turns scores and business rules into decisions.

Decisions: APPROVE, STEP_UP (e.g., one-time passcode), REVIEW (manual), DECLINE.
Anything other than APPROVE is treated as intercepting a fraudulent transaction.
"""
import numpy as np

APPROVE, STEP_UP, REVIEW, DECLINE = 0, 1, 2, 3


def policy_decide(score, df):
    """Model score combined with business rules, thresholds and caps."""
    amt, nd, vel = df["amount"].to_numpy(), df["new_device"].to_numpy(), df["velocity_1h"].to_numpy()
    d = np.full(len(df), APPROVE)
    d[(score >= 0.50) | ((amt > 1000) & (score >= 0.20)) | ((nd == 1) & (amt > 1500))] = STEP_UP
    d[(amt > 5000) & ((nd == 1) | (score >= 0.10))] = REVIEW
    d[(score >= 0.90) | (vel >= 12)] = DECLINE
    return d


def model_direct(score, df, threshold=0.5):
    """Anti-pattern: the model's score is the decision."""
    return np.where(score >= threshold, DECLINE, APPROVE)


def fallback_rules(df):
    """Pre-approved deterministic fallback used when no model score is available."""
    amt, nd = df["amount"].to_numpy(), df["new_device"].to_numpy()
    vel, mr = df["velocity_1h"].to_numpy(), df["merchant_risk"].to_numpy()
    d = np.full(len(df), APPROVE)
    d[((nd == 1) & (amt > 150)) | (vel >= 5) | ((mr > 0.6) & (amt > 250)) | (amt > 3000)] = STEP_UP
    d[(mr > 0.8) & (nd == 1)] = DECLINE
    return d


def outcomes(decisions, df):
    """Business metrics for a set of decisions."""
    fraud = df["is_fraud"].to_numpy()
    amt = df["amount"].to_numpy()
    approved = decisions == APPROVE
    legit = ~fraud
    return dict(
        fraud_intercept_rate=float((~approved[fraud]).mean()) if fraud.any() else float("nan"),
        fraud_loss_usd=float(amt[fraud & approved].sum()),
        fraud_usd_total=float(amt[fraud].sum()),
        legit_friction_rate=float((~approved[legit]).mean()),
        legit_decline_rate=float((decisions[legit] == DECLINE).mean()),
    )
