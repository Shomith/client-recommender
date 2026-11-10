### All clients who added a product (n = 5,503)

| Method | NDCG@5 | MAP@7 | HitRate@3 |
|---|---|---|---|
| Popularity | 0.6971 | 0.6443 | 0.7125 |
| Segment popularity | 0.7457 | 0.6901 | 0.7863 |
| Co-occurrence | 0.7400 | 0.6828 | 0.8286 |
| ALS only | 0.3495 | 0.3528 | 0.3921 |
| ALS + LightGBM ranker | 0.8747 | 0.8434 | 0.9402 |

### Cold-start (tenure <= 6 months) (n = 1,004)

| Method | NDCG@5 | MAP@7 | HitRate@3 |
|---|---|---|---|
| Popularity | 0.5223 | 0.4859 | 0.4781 |
| Segment popularity | 0.6714 | 0.6250 | 0.6853 |
| Co-occurrence | 0.6791 | 0.6259 | 0.8386 |
| ALS only | 0.2109 | 0.2159 | 0.2689 |
| ALS + LightGBM ranker | 0.8863 | 0.8623 | 0.9402 |

