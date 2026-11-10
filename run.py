"""Run the whole pipeline.  python run.py [--fraction 0.05] [--data path] [--out results]"""
import argparse, os, time
import numpy as np
import pandas as pd
from src import config as C
from src import data, baselines as B, als as A, rank as R, metrics as M, explain as E


def main(data_path, fraction, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    print("Loading and cleaning data...")
    df = data.clean(data.load_raw(data_path, fraction=fraction))
    print(f"  {len(df):,} client-month rows, {df[C.ID_COL].nunique():,} clients")
    train = [data.make_split(df, *s) for s in C.TRAIN_SPLITS]
    val = data.make_split(df, *C.VAL_SPLIT)
    test = data.make_split(df, *C.TEST_SPLIT)
    for s in train + [val, test]:
        m = data.eval_clients(s)
        print(f"  {s['name']}: {len(m):,} clients, {m.sum():,} added something ({m.mean():.1%})")

    print("Fitting baselines and ALS...")

    def fit_feats(history):
        return (B.fit_popularity(history), B.fit_segment(history),
                B.fit_cooc(history), A.fit_als(history))

    def prepare(split, feats):
        pop, seg_rates, cooc_m, als_model = feats
        mask = data.eval_clients(split)
        ids, sc = A.als_top(als_model, split["held"])
        seg = B.score_segment(seg_rates, pop, split)
        cooc = B.score_cooc(cooc_m, split)
        X, y, g, rows, prod = R.build_candidates(split, ids, sc, pop, cooc, seg, mask)
        return {"ids": ids, "seg": seg, "cooc": cooc, "mask": mask, "X": X, "y": y, "g": g, "rows": rows, "prod": prod}

    # Each training month only sees features built from the months before it
    p_train = [prepare(train[k], fit_feats(train[:k])) for k in range(1, len(train))]

    # Validation and test use features fitted on all 4 training months
    test_feats = fit_feats(train)
    pop = test_feats[0]
    p_val, p_test = prepare(val, test_feats), prepare(test, test_feats)

    print("Training the LightGBM ranker...")
    Xt = pd.concat([p["X"] for p in p_train], ignore_index=True)
    yt = np.concatenate([p["y"] for p in p_train])
    gt = np.concatenate([p["g"] for p in p_train])
    ranker = R.fit_ranker((Xt, yt, gt), (p_val["X"], p_val["y"], p_val["g"]))
    print(f"  best iteration: {ranker.best_iteration_}")

    print("Evaluating on the test month...")
    W, n = C.TOP_N, len(test["held"])
    rankings = {
        "Popularity": B.top_k(B.score_popularity(pop, test), test["held"], W),
        "Segment popularity": B.top_k(p_test["seg"], test["held"], W),
        "Co-occurrence": B.top_k(p_test["cooc"], test["held"], W),
        "ALS only": p_test["ids"],
        "ALS + LightGBM ranker": R.ranked_from_scores(p_test["rows"], p_test["prod"], ranker.predict(p_test["X"]), n, W),
    }
    mask = p_test["mask"]
    cold = mask & (test["X"]["antiguedad"].to_numpy() <= C.COLD_START_MAX_TENURE)
    slices = {"All clients who added a product": mask,
              f"Cold-start (tenure <= {C.COLD_START_MAX_TENURE} months)": cold}
    out = []
    for sname, sl in slices.items():
        if sl.sum() < 30:
            print(f"  skipping slice '{sname}' (only {sl.sum()} clients)")
            continue
        for mname, ranked in rankings.items():
            out.append({"slice": sname, "method": mname, "n_clients": int(sl.sum()),
                        **M.evaluate(ranked[sl], test["new"][sl], C.KS)})
    results = pd.DataFrame(out)
    results.to_csv(f"{out_dir}/results.csv", index=False)
    show = ["NDCG@5", "MAP@7", "HitRate@3"]
    with open(f"{out_dir}/results.md", "w") as f:
        for sname, g in results.groupby("slice", sort=False):
            f.write(f"### {sname} (n = {g['n_clients'].iloc[0]:,})\n\n| Method | " + " | ".join(show) + " |\n")
            f.write("|---|" + "---|" * len(show) + "\n")
            for _, r in g.iterrows():
                f.write(f"| {r['method']} | " + " | ".join(f"{r[c]:.4f}" for c in show) + " |\n")
            f.write("\n")
    print(open(f"{out_dir}/results.md").read())
    print("Computing SHAP explanations...")
    E.explain(ranker, p_test["X"], p_test["rows"], p_test["prod"], out_dir)
    print(open(f"{out_dir}/example_explanation.txt").read())
    print(f"Done in {time.time() - t0:.0f}s. Outputs in {out_dir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=C.DATA_PATH)
    ap.add_argument("--fraction", type=float, default=C.CLIENT_FRACTION)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    main(a.data, a.fraction, a.out)
