"""Ranking metrics. `ranked` is clients x K product indices (best first, -1 = empty)."""
import numpy as np


def evaluate(ranked, new, ks):
    n = len(ranked)
    valid = ranked >= 0
    rel = np.where(valid, new[np.arange(n)[:, None], np.where(valid, ranked, 0)], 0).astype(float)
    n_true = new.sum(axis=1)
    out = {}
    for k in ks:
        r = rel[:, :k]
        disc = 1.0 / np.log2(np.arange(2, k + 2))
        dcg = (r * disc).sum(axis=1)
        m = np.minimum(n_true, k)
        idcg = np.cumsum(disc)[np.maximum(m, 1) - 1]
        out[f"NDCG@{k}"] = float(np.mean(dcg / idcg))
        prec = np.cumsum(r, axis=1) / np.arange(1, k + 1)
        out[f"MAP@{k}"] = float(np.mean((prec * r).sum(axis=1) / np.maximum(m, 1)))
        out[f"HitRate@{k}"] = float(np.mean(r.sum(axis=1) > 0))
    return out
