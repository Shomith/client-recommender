"""Loading, cleaning, and turning monthly snapshots into (features, what was added next month)."""
import numpy as np
import pandas as pd
from . import config as C


def load_raw(path=None, months=None, fraction=None):
    path = path or C.DATA_PATH
    months = set(months or C.MONTHS)
    fraction = C.CLIENT_FRACTION if fraction is None else fraction
    keep = int(round(fraction * 1000))
    usecols = [C.ID_COL, C.DATE_COL] + C.NUM_COLS + C.CAT_COLS + C.PRODUCTS
    parts = []
    for chunk in pd.read_csv(path, usecols=usecols, dtype=str, chunksize=500_000):
        chunk["month"] = chunk[C.DATE_COL].str[:7]
        chunk = chunk[chunk["month"].isin(months)]
        ids = pd.to_numeric(chunk[C.ID_COL], errors="coerce")
        chunk = chunk[(ids * 7919 % 1000) < keep]
        if len(chunk):
            parts.append(chunk)
    if not parts:
        raise ValueError("No rows matched. Check DATA_PATH and the months in config.py.")
    return pd.concat(parts, ignore_index=True)


def clean(df):
    df = df.copy()
    df[C.ID_COL] = pd.to_numeric(df[C.ID_COL], errors="coerce").astype("int64")
    for c in C.NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["age"] = df["age"].clip(18, 100)
    df.loc[df["antiguedad"] < 0, "antiguedad"] = np.nan
    for c in ["age", "antiguedad", "renta"]:
        df[c] = df[c].fillna(df[c].median())
    for c in ["ind_nuevo", "indrel", "ind_actividad_cliente"]:
        df[c] = df[c].fillna(0)
    for c in C.CAT_COLS:
        df[c] = pd.factorize(df[c].fillna("NA").str.strip())[0]
    for p in C.PRODUCTS:
        df[p] = pd.to_numeric(df[p], errors="coerce").fillna(0).clip(0, 1).astype("int8")
    return df.drop_duplicates([C.ID_COL, "month"])


def make_split(df, prev_m, feat_m, next_m):
    """held[i,j]=1 if client i holds product j in feat_m; new[i,j]=1 if added by next_m."""
    A = df[df["month"] == feat_m].set_index(C.ID_COL)
    N = df[df["month"] == next_m].set_index(C.ID_COL)[C.PRODUCTS]
    P = df[df["month"] == prev_m].set_index(C.ID_COL)[C.PRODUCTS]
    common = A.index.intersection(N.index)
    A, N = A.loc[common], N.loc[common]
    held = A[C.PRODUCTS].to_numpy()
    new = ((N.to_numpy() == 1) & (held == 0)).astype(np.int8)
    prev = P.reindex(common).to_numpy(dtype="float64")
    prev = np.where(np.isnan(prev), held, prev)
    recent_adds = ((held == 1) & (prev == 0)).sum(axis=1)
    X = A[C.NUM_COLS + C.CAT_COLS].copy()
    X["n_held"] = held.sum(axis=1)
    X["recent_adds"] = recent_adds
    age_band = pd.cut(X["age"], [0, 25, 35, 45, 55, 101], labels=False).astype(int)
    seg = (age_band.astype(str) + "_" + X["segmento"].astype(str)).to_numpy()
    return {"ids": common.to_numpy(), "X": X.reset_index(drop=True), "held": held.astype(np.int8),
            "new": new, "seg": seg, "name": f"{feat_m}->{next_m}"}


def eval_clients(split):
    return split["new"].sum(axis=1) > 0
