"""SHAP: which features pushed the ranker's scores up or down."""
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from . import config as C


def explain(ranker, X, rows, prod, out_dir, sample=3000, seed=None):
    rng = np.random.default_rng(C.SEED if seed is None else seed)
    pick = rng.choice(len(X), size=min(sample, len(X)), replace=False)
    Xs = X.iloc[pick]
    sv = shap.TreeExplainer(ranker.booster_).shap_values(Xs)
    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=Xs.columns).sort_values(ascending=False)
    mean_abs.to_csv(f"{out_dir}/shap_importance.csv", header=["mean_abs_shap"])
    top = mean_abs.head(15)[::-1]
    plt.figure(figsize=(7, 5))
    plt.barh(top.index, top.values, color="#1F4E5C")
    plt.xlabel("Mean |SHAP value|")
    plt.title("What drives the ranker's scores")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/shap_importance.png", dpi=150)
    plt.close()
    i = int(np.argmax(ranker.predict(Xs)))
    contrib = pd.Series(sv[i], index=Xs.columns).sort_values(key=np.abs, ascending=False).head(8)
    name = C.PRODUCTS[int(Xs.iloc[i]["product"])]
    lines = [f"Example: why the ranker scored product '{name}' highly for one client", ""]
    for feat, val in contrib.items():
        lines.append(f"  {feat} = {Xs.iloc[i][feat]:.3g}: {'pushed up' if val > 0 else 'pushed down'} ({val:+.3f})")
    open(f"{out_dir}/example_explanation.txt", "w").write("\n".join(lines))
    return mean_abs
