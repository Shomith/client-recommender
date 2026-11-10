"""ALS matrix factorization on 'which products does each client hold'."""
import numpy as np
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares
from . import config as C


def fit_als(train_splits):
    held = np.vstack([s["held"] for s in train_splits]).astype(np.float32)
    model = AlternatingLeastSquares(factors=C.ALS_FACTORS, regularization=C.ALS_REG,
                                    iterations=C.ALS_ITERS, random_state=C.SEED)
    model.fit(sp.csr_matrix(held), show_progress=False)
    return model


def als_top(model, held, n=None):
    n = n or C.N_CANDIDATES
    ui = sp.csr_matrix(held.astype(np.float32))
    ids, scores = model.recommend(np.arange(ui.shape[0]), ui, N=n,
                                  filter_already_liked_items=True, recalculate_user=True)
    ids = np.asarray(ids).copy()
    scores = np.asarray(scores, dtype="float64").copy()
    bad = (ids < 0) | ~np.isfinite(scores) | (np.take_along_axis(held, np.clip(ids, 0, held.shape[1] - 1), axis=1) == 1)
    ids[bad] = -1
    scores[bad] = -1e9
    return ids, scores
