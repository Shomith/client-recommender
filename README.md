# Next-Best-Product Recommender for Banking Clients

Predicts which financial products a bank client is likely to add next month, using the
Santander Product Recommendation data from Kaggle. It works in two stages: ALS (matrix
factorization) scores candidate products, then a LightGBM LambdaRank model re-ranks them
using client features, product co-occurrence and segment-level adoption rates. SHAP
explains each recommendation.

## Results

Test month: April → May 2016, scored only on clients who added at least one product.
Every model and baseline was fitted on earlier months only.

**All clients who added a product (n = 5,503)**

| Method | NDCG@5 | MAP@7 | HitRate@3 |
|---|---|---|---|
| Popularity | 0.6971 | 0.6443 | 0.7125 |
| Segment popularity | 0.7457 | 0.6901 | 0.7863 |
| Co-occurrence | 0.7400 | 0.6828 | 0.8286 |
| ALS only | 0.3495 | 0.3528 | 0.3921 |
| **ALS + LightGBM ranker** | **0.8747** | **0.8434** | **0.9402** |

**Cold start: clients with 6 months' tenure or less (n = 1,004)**

| Method | NDCG@5 | MAP@7 | HitRate@3 |
|---|---|---|---|
| Popularity | 0.5223 | 0.4859 | 0.4781 |
| Segment popularity | 0.6714 | 0.6250 | 0.6853 |
| Co-occurrence | 0.6791 | 0.6259 | 0.8386 |
| ALS only | 0.2109 | 0.2159 | 0.2689 |
| **ALS + LightGBM ranker** | **0.8863** | **0.8623** | **0.9402** |

The ranker beats the strongest baseline by about 0.13 NDCG@5, and it holds up for new
clients, where popularity drops off sharply.

Full numbers (K = 3, 5, 7) are in `results/results.csv`, and feature importance is in
`results/shap_importance.png`.

## What I learned along the way

- **ALS on its own is weak here.** With only 24 products, simple counts (segment
  popularity, co-occurrence) already capture most of what matrix factorization can learn.
  My first version used 32 factors, more than there are products, so ALS just memorised
  who holds what and did worse than random (0.05 NDCG@5). Dropping to 8 factors with
  stronger regularization brought it to 0.35. It's still useful as one feature for the
  ranker, but the lift comes from combining signals per client.
- **Fixed a target-leakage bug.** Originally, the ranker's training months got features
  (segment rates, co-occurrence, ALS) fitted on those same months, so the features partly
  contained the answer. SHAP showed `seg_rate` dominating every prediction. Now each
  training month only sees features built from the months before it. The test score
  went up slightly, and the explanations are spread across popularity, co-occurrence and
  segment rate instead of resting on one leaked feature.

## Setup

- **Task:** given a client's products in month *t*, rank the products they don't hold yet
  by how likely they are to be added in month *t + 1*.
- **Time-based split, nothing shuffled:**
  - Train: Nov 2015 → Mar 2016 (four monthly transitions)
  - Validation (early stopping): Mar → Apr 2016
  - Test: Apr → May 2016
- **Sample:** 20% of clients (selected by hashing the client id, so it's the same clients
  every month), giving about 1.47M client-month rows and 189K clients.
- **Metrics:** NDCG@K, MAP@K and HitRate@K for K = 3, 5, 7.

## Run it

1. Download `train_ver2.csv` from Kaggle (competition: `santander-product-recommendation`)
   into `data/`.
2. `pip install -r requirements.txt`
3. `python run.py` (add `--fraction 0.05` for a quick run)

On Windows, set `OPENBLAS_NUM_THREADS=1` first, or ALS runs much slower.

Outputs go to `results/`.

`python tests/make_synthetic.py` writes a small fake dataset with the same columns. It's
only for checking that the code runs, so don't read anything into its numbers.

## Project layout

```
run.py              end-to-end pipeline
src/data.py         loading, cleaning, monthly splits
src/baselines.py    popularity, segment popularity, co-occurrence
src/als.py          ALS candidate scoring (implicit)
src/rank.py         candidate features + LightGBM LambdaRank
src/metrics.py      NDCG / MAP / HitRate
src/explain.py      SHAP explanations
```
