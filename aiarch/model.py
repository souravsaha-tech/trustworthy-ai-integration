"""Fraud model (the 'intelligence'): returns a score, never a decision."""
from sklearn.ensemble import HistGradientBoostingClassifier
from .data import FEATURES, generate


def train(rng, n=200_000, drift_share=0.0, seed=0):
    df = generate(n, rng, drift_share=drift_share)
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=seed)
    m.fit(df[FEATURES], df["is_fraud"])
    return m


def score(model, df):
    return model.predict_proba(df[FEATURES])[:, 1]
