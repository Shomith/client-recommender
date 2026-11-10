"""Makes a SMALL FAKE file with the Santander columns. Only checks the code runs; numbers mean nothing."""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, ".")
from src import config as C

rng = np.random.default_rng(0)
n = 6000
age = rng.integers(18, 80, n); tenure0 = rng.integers(0, 200, n)
income = rng.lognormal(11.5, 0.6, n); activity = rng.integers(0, 2, n)
seg = rng.choice(["01 - TOP", "02 - PARTICULARES", "03 - UNIVERSITARIO"], n, p=[.05, .75, .2])
sex = rng.choice(["H", "V"], n); chan = rng.choice(["KHE", "KAT", "KFC", "KHK", "KFA"], n)
rel = rng.choice(["A", "I", "P"], n, p=[.9, .08, .02])
held = np.zeros((n, 24), dtype=int)
held[:, 2] = (rng.random(n) < .8); held[:, 22] = (rng.random(n) < .3)
base = rng.random(24) * 0.012 + 0.002
rows = []
for mi, m in enumerate(C.MONTHS):
    present = rng.random(n) > 0.03
    df = pd.DataFrame({"ncodpers": 100000 + np.arange(n), "fecha_dato": f"{m}-28", "age": age,
        "antiguedad": tenure0 + mi, "renta": income, "ind_nuevo": (tenure0 + mi < 6).astype(int),
        "indrel": 1, "ind_actividad_cliente": activity, "sexo": sex, "segmento": seg,
        "canal_entrada": chan, "tiprel_1mes": rel})
    for j, p in enumerate(C.PRODUCTS):
        df[p] = held[:, j]
    rows.append(df[present])
    p_add = np.tile(base, (n, 1))
    p_add[:, 18] *= 1 + 4 * held[:, 2] * ((age > 25) & (age < 55))
    p_add[:, 13] *= 1 + 3 * held[:, 18]; p_add[:, 12] *= 1 + 3 * held[:, 2]; p_add[:, 5] *= 1 + 3 * (age > 60)
    p_add *= (0.5 + activity[:, None])
    held = np.maximum(held, (rng.random((n, 24)) < p_add).astype(int))
pd.concat(rows).to_csv("data/synthetic_sample.csv", index=False)
print("wrote data/synthetic_sample.csv")
