"""Baselines. Each returns a score for every (client, product)."""
import numpy as np
import pandas as pd


def fit_popularity(train_splits):
    return np.vstack([s["new"] for s in train_splits]).mean(axis=0)


def score_popularity(pop, split):
    return np.tile(pop, (len(split["held"]), 1))


def fit_segment(train_splits):
    seg = np.concatenate([s["seg"] for s in train_splits])
    new = np.vstack([s["new"] for s in train_splits])
    return pd.DataFrame(new).groupby(seg).mean()


def score_segment(rates, pop, split):
    r = rates.reindex(split["seg"]).to_numpy(dtype="float64")
    return np.where(np.isnan(r), pop, r)


def fit_cooc(train_splits):
    H = np.vstack([s["held"] for s in train_splits]).astype("float64")
    N = np.vstack([s["new"] for s in train_splits]).astype("float64")
    return (H.T @ N) / (H.sum(axis=0)[:, None] + 1.0)


def score_cooc(M, split):
    return split["held"].astype("float64") @ M


def top_k(scores, held, k):
    masked = np.where(held == 1, -np.inf, scores)
    order = np.argsort(-masked, axis=1, kind="stable")[:, :k]
    picked = np.take_along_axis(masked, order, axis=1)
    return np.where(np.isfinite(picked), order, -1)
