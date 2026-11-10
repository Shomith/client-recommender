"""Stage two: re-rank candidates with a LightGBM LambdaRank model."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from . import config as C

CAT_FEATURES = C.CAT_COLS + ["product"]


def build_candidates(split, als_ids, als_scores, pop, cooc, seg_scores, mask):
    idx = np.where(mask)[0]
    k = als_ids.shape[1]
    rows = np.repeat(idx, k)
    prod = als_ids[idx].ravel()
    s = als_scores[idx].ravel()
    r = np.tile(np.arange(k), len(idx))
    ok = prod >= 0
    rows, prod, s, r = rows[ok], prod[ok], s[ok], r[ok]
    X = split["X"].iloc[rows].reset_index(drop=True)
    held = split["held"][rows]
    for j in range(held.shape[1]):
        X[f"held_{j}"] = held[:, j]
    X["product"] = prod
    X["als_score"] = s
    X["als_rank"] = r
    X["pop_rate"] = pop[prod]
    X["cooc"] = cooc[rows, prod]
    X["seg_rate"] = seg_scores[rows, prod]
    y = split["new"][rows, prod]
    group = np.bincount(np.searchsorted(idx, rows), minlength=len(idx))
    return X, y, group[group > 0], rows, prod


def fit_ranker(train, val):
    Xt, yt, gt = train
    Xv, yv, gv = val
    model = lgb.LGBMRanker(objective="lambdarank", n_estimators=400, learning_rate=0.05,
                           num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, random_state=C.SEED, verbose=-1)
    model.fit(Xt, yt, group=gt, eval_set=[(Xv, yv)], eval_group=[gv], eval_at=[5],
              categorical_feature=CAT_FEATURES,
              callbacks=[lgb.early_stopping(30, verbose=False)])
    return model


def ranked_from_scores(rows, prod, pred, n_clients, width):
    df = pd.DataFrame({"row": rows, "prod": prod, "pred": pred}).sort_values(["row", "pred"], ascending=[True, False])
    df["pos"] = df.groupby("row").cumcount()
    out = -np.ones((n_clients, width), dtype=int)
    df = df[df["pos"] < width]
    out[df["row"].to_numpy(), df["pos"].to_numpy()] = df["prod"].to_numpy()
    return out
