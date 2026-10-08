# Experiment log

A curated version of the development log: what was tried, what happened, and what it taught.
**Negative results are kept on purpose** - most of the learning is in them.

> **How to read the numbers**
>
> * Rows **#0-#34** come from the original development log. They were measured on *earlier code
>   states* (mostly the 247-cluster scheme, one LightGBM seed, interim feature sets), so they are
>   **not comparable across rows**, and this repository does not reproduce them row by row - only
>   the final configuration is reproducible (see the README). Read each row as *what was learned*,
>   not as a benchmark. IDs are kept so they can be cross-referenced with the original log.
> * Rows **#35-#37** were run on the final code during the refactor, with the 148-cluster scheme.
> * **Noise floor (#36):** re-seeding LightGBM moves WAPE by σ ≈ 0.002-0.006 on a regular month and
>   σ ≈ 0.02-0.03 on the seasonal-spike month. **Differences below ~0.01 (regular) or ~0.05 (spike)
>   from a single run are not distinguishable from noise** - which includes several small "wins"
>   below.
> * Month names are Jalali: Azar 1403 = Nov 21 - Dec 20 2024, Mordad 1404 = Jul 23 - Aug 22 2025,
>   Shahrivar 1404 = Aug 23 - Sep 22 2025 (the seasonal peak), Mehr 1404 = Sep 23 - Oct 22 2025,
>   Aban 1404 = Oct 23 - Nov 21 2025. Lower WAPE is better.

Legend: ✅ kept · ❌ rejected / reverted · ⚠️ mixed or not adopted · ℹ️ informational

## At a glance

| # | Experiment | Outcome | |
|---|---|---|---|
| 1 | Nowruz 1402 as one 13-day block (data fix) | was fragmented 4+2+2 | ✅ |
| 2 | `jalali_week` feature | 0.1569 → 0.1355 (Azar 1403) | ✅ |
| 4 | Per-horizon calibration, fit on validation, no floor | 0.1355 → 0.1504 | ❌ |
| 5, 7 | `holiday_magnitude` instead of binary `is_holiday` | 2 of 4 windows better, all \|Δ\| < 0.005 | ⚠️ inconclusive |
| 8 | 4-window robustness check | **found the Shahrivar failure** | ℹ️ |
| 9 | `month` + `dow` + `week_of_month` as a group | 0.1603 → 0.1335, 0.1799 → 0.1391 | ✅ |
| 10 | Jalali-only `month` (drop Gregorian) | 0.1391 → 0.1516 | ❌ kept both |
| 11 | Demand-weighted training | Shahrivar 0.4283 → 0.2137; Azar +0.022, Aban +0.011 | ❌ |
| 12 | Thursday+Friday weekend, Saturday-first `dow` | 0.1373 → 0.1365 | ✅ correctness |
| 13 | Per-cluster × weekday history | 0.1370 → 0.1361 | ✅ |
| 14 | Raw `cluster_code` as a categorical feature | 0.1361 → 0.1386 | ❌ overfit |
| 15 | Stage A/B guardrail | no change on Aban; large effect on Shahrivar (#37) | ⚠️ kept as an option |
| 16 | Deterministic LightGBM | same inputs → same model | ✅ |
| 17 | Shrunk per-cluster Stage-A bias feature | 0.1361 → 0.1386 | ❌ |
| 18 | Bayesian recency update of the reveal curve | Stage A worse in all 3 windows | ❌ |
| 19 | `days_to_yalda` | abandoned before any measurement | ℹ️ |
| 20 | Quantile / Huber objectives | Huber 0.6028 on Shahrivar; quantile mixed | ❌ |
| 21 | √-demand-weighted training | Shahrivar 0.4666 → 0.2220; others +0.005…+0.013 | ⚠️ available |
| 22 | Hierarchical-Bayes shrinkage of cluster history | Aban 0.1361 → 0.1401; Shahrivar 0.4666 → 0.5078 | ❌ |
| 23 | Leakage-free global calibration, floored at 1 | Aban 0.1361 → 0.1226; Azar unchanged | ✅ |
| 24 | CatBoost + categorical cluster | Aban 0.1361 → 0.1400 | ❌ |
| 25 | Clustering: volume **and** weekly-coverage criterion | Azar 1403 0.1361 → 0.1683 | ❌ |
| 26 | Clustering: regional aggregates of the sparse tail | Azar 1403 0.1361 → 0.1594 | ❌ |
| 27 | Per-horizon-bucket calibration | Aban 0.1218; Shahrivar 0.3853 | ✅ shipped (but see #37) |
| 28 | Clustering: 90 searches/month rate threshold | worse on all 3 windows | ❌ |
| 29 | Clustering: 200/month agglomerative floor | Azar 1403 0.1361 → 0.1684 | ❌ |
| 30 | Per-cluster × month baseline | worse on all 3 windows, incl. Shahrivar | ❌ |
| 31 | Clustering: strict 500/month floor, 148 clusters | Aban a wash; Shahrivar 0.3904 → 0.3296 | ✅ **final scheme** |
| 32 | √-weighting × 148 clusters | the two fixes do not compound | ℹ️ |
| 33 | Holiday shoulder-day flags (±1-3 days) | helps 2 months, hurts Shahrivar | ⚠️ not in final features |
| 34 | ML-based, trajectory-aware Stage A (+ ensemble) | prototyped separately | ⚠️ not in this repo |
| 35 | Zero-volume centroid fix in clustering | 42 cities move; no measurable effect over 4 seeds | ⚠️ optional flag |
| 36 | Seed sensitivity (noise floor) | σ 0.002-0.006 regular, 0.02-0.03 spike | ℹ️ |
| 37 | Post-hoc: guardrail + calibration, short-lead routing | spike month 0.365 → 0.260 | ⚠️ opt-in |

---

## A. Features and calendar

**#1 Nowruz 1402.** The first year's Nowruz arrived as three fragments (4+2+2 days) because it was
joined against a stale cached calendar. Fixed to one 13-day block (1-13 Farvardin) - a data fix,
not a modelling experiment. (The bundled calendar keeps this convention; the official public
holidays are 1-4 and 12-13 Farvardin.)

**#2 `jalali_week`** (week of the Jalali year). Expected a modest gain with a real overfitting risk
(only 2-3 repeats per week). Result: 0.1569 → 0.1355, a ~14 % relative reduction; it also changed
how much the model used `lead` (best iteration 198 → 601), i.e. opened useful lead × week splits.

**#5 / #7 `holiday_magnitude`.** The binary `is_holiday` cannot tell a 13-day Nowruz from a
one-day observance, and named holidays were under-forecast by 12 %. Replacing it with the block
length was a wash across four windows (Azar 1403 +0.0036, Aban −0.0019, Dey 1403 +0.0048,
Mehr −0.0028): kept the binary flag for simplicity. Note the first test window contained one
holiday, in a 2-day block, so it could not discriminate anything.

**#9 Calendar group.** Ablating `month`, `dow` and `week_of_month` together: Azar 1403 0.1603 →
0.1335 (−0.027), Aban 1404 0.1799 → 0.1391 (−0.041) - the largest gains of any feature work.
Importance ranking was consistent: `dow` > `month` > `week_of_month`.

**#10 One Jalali month instead of two encodings.** Dropping the Gregorian `month` and keeping only
`jalali_month` cost 0.012 (0.1391 → 0.1516), so both are kept. Not a clean test of Gregorian vs
Jalali - it shows the redundant pair beats the single feature (either Gregorian months carry
information a Jalali month does not, or the trees simply benefit from two encodings).

**#12 Weekend fix.** `is_weekend` flagged only Friday and `dow` used pandas' Monday-first order
- both wrong for Iran. Fixed to Thursday + Friday and Saturday = 0 … Friday = 6. Aban 0.1373 →
0.1365 (validation 0.139 → 0.125). Kept as a correctness fix; it was motivated by under-prediction
concentrated on Thursdays and the Wednesdays before them.

**#13 Per-cluster × weekday history.** Wednesday's error persisted after #12, but not because it is
more volatile: it simply has a higher *level* (second only to Thursday; +25-40 % over Mon/Tue for
big resort clusters). That is a level effect, not a timing effect, so #12's reveal-curve fix could
never address it. Adding per-cluster-per-weekday mean/std: 0.1370 → 0.1361 (Wednesday 0.1664 →
0.1597), immediately the #2 feature by gain.

**#14 `cluster_code` as a raw categorical.** Instantly the #2 feature by gain but the backtest
regressed (0.1361 → 0.1386) while validation was flat - the textbook signature of memorising a
high-cardinality ID.

**#17 Shrunk per-cluster Stage-A bias** (median of `target / stageA` per cluster, shrunk toward 1
with 50 pseudo-observations). The smoothed version of #14's idea: 0.1361 → 0.1386, and validation
got worse too, so it was not even a generalisation problem.

**#19 `days_to_yalda`.** Implemented, killed before measurement, fully removed.

**#22 Hierarchical-Bayes (James-Stein) shrinkage of cluster history.** Each cluster's mean
shrunk toward its province, itself shrunk toward the global mean, with weights driven by each
cluster's own variance. Aban 0.1361 → 0.1401, Shahrivar 0.4666 → 0.5078. *Root cause:* the resort
and coastal clusters are exactly the highest-variance ones, so the estimator pulled their means down
20-29 % toward calmer neighbours - but their variance is **real seasonality** (quiet most of the
year, huge in Shahrivar), not sampling noise. No pooling, shrinkage or target-encoding technique can
tell the two apart from the demand series alone.

**#30 Per-cluster × month baseline.** To give the model "this cluster's typical Shahrivar level":
worse on all three windows, including Shahrivar itself. A sharper seasonal anchor made the model
regress *harder* toward it, with only ~2 prior Shahrivars per cluster.

**#33 Holiday shoulder-day flags** (exactly 1, 2, 3 days before/after a holiday block). Azar 1403
0.1401 → 0.1330, Aban 0.1279 → 0.1242, Shahrivar 0.3904 → 0.4121. The log marks it "kept", **but the
shipped 21-feature model does not use them** (the trained model's feature list has none of the six
flags), so they are not part of this repository. Re-test them with a feature list that includes
them if desired.

## B. Stage A (reveal curve)

**#18 Bayesian recency update of the weekday reveal curve** (long-run prior + the most recent 60
days as evidence, prior strength growing with horizon). Stage A got *worse* in all three windows
(Shahrivar 1403 0.1801 → 0.1862, Shahrivar 1404 0.1664 → 0.2371, Aban 1404 0.1567 → 0.2047). The
Aban regression exposed the design flaw: its "recent" 60 days overlapped the tail of the
Shahrivar/Mehr spike, contaminating a normal month with evidence from the other anomaly.

**#34 ML-based, trajectory-aware Stage A** (a bounded, monotone LightGBM reveal model using recent
search velocity and Jalali calendar features). It improved Stage A in all four windows (mean
0.1897 → 0.1638) and a 50/50 ensemble with the baseline helped three windows (Azar 0.1330 →
0.1311, Aban 0.1242 → 0.1223, Shahrivar 0.4121 → 0.3537) with a small Mordad regression (0.1778 →
0.1791). Prototyped on a separate branch; **not part of this repository**. Given #36, gains of
0.002 are inside the noise.

## C. Training objective and model family

**#11 / #21 Demand-weighted training.** WAPE is demand-weighted, but the loss is not. Weighting by
raw demand (#11) nearly halved the Shahrivar failure (0.4283 → 0.2137) but hurt the healthy months
(Azar 0.1335 → 0.1554, Aban 0.1391 → 0.1498) and once failed to converge at all. √-weighting (#21)
is gentler: Shahrivar 0.4666 → 0.2220, but Aban +0.0045, Azar +0.0128, Mehr +0.0071. No free lunch;
available via `--sample-weighting sqrt`. A conditional scheme (switch weighting only for
spike-prone windows) is on the roadmap.

**#20 Objectives.** Quantile regression (α > 0.5 directly targets under-prediction): Shahrivar
0.4666 → 0.4214 (α = 0.6) / 0.4781 (α = 0.65); Aban 0.1361 → 0.1373 / 0.1347. Huber was the worst
option tried (Shahrivar 0.6028). A shifted quantile moves *where* the prediction sits but not
*which rows* the model must get right, so it cannot replace reweighting.

**#24 CatBoost** with `cluster_code` as a native categorical (its ordered target statistics are
designed for exactly this): Aban 1404 0.1361 → 0.1400, markedly slower. An earlier **XGBoost**
attempt hung and was killed. LightGBM stays.

**#16 Determinism.** `best_iteration` had varied (907 / 1052 / 907) across identical runs, which had
wrecked a per-cell diagnosis (#15). `deterministic=True, force_row_wise=True` fixed it (identical
iteration counts and WAPE to four decimals). **Caveat from #36:** determinism fixes *run-to-run*
variance for a given seed; it does not make results robust *to the seed*.

## D. Post-processing: calibration and the guardrail

**#4 Per-horizon calibration, first attempt.** Error analysis found Stage B over-forecasting by 15 %
at 22-30 days and under-forecasting a few percent below that. A correction fit on the validation
fold alone made things worse (0.1355 → 0.1504): the validation fold's bias direction (over-forecast
everywhere) did not match the scored window's. Reverted.

**#23 Global calibration, floored at 1.** The refined idea: the log-L1 objective predicts the median,
which is below the mean WAPE rewards, so the right correction is always ≥ 1 - estimate it on the
validation fold and floor it. Without the floor it backfired on Azar 1403 (factor 0.925, wrong
sign: 0.136 → 0.155). With it: Azar unchanged, Aban 0.1361 → 0.1226, Mehr 0.1156 → 0.1123,
Shahrivar 0.4666 → 0.3952.

**#27 Per-horizon-bucket calibration.** The residual bias flips sign with the horizon (mid-lead
under-predicts 4-9 %, 22-30 days over-predicts 16 %), so a single factor cannot fix it. Per bucket:
Aban 0.1218, Shahrivar 0.3853 - never worse than the global factor on those windows. **Shipped.**
*Later finding (#37):* on Mordad 1404, with the final model, it *hurts* (+0.027).

**#15 Guardrail.** For Aban's 10 worst cells Stage A was already close for 8, and Stage B made them
much worse. Clipping Stage B to [0.7, 1.5] × Stage A: on all cells 0.1409 (worse - it mostly clipped
tiny cells where ratios are noisy); restricted to Stage A ≥ 1,000: no change on Aban. Kept as an
option; its real value shows on the spike month (#37).

## E. Clustering

The first study varied one threshold of a "merge sparse cities into the nearest strong neighbour"
rule on Aban 1404: no clustering 0.1409 (321 clusters), threshold 200 → 247 clusters 0.1361 (best),
100 → 0.1369, 500 → 0.1389, 1000 → 0.1365. Even though sparse cities contribute ~0.008 % of the
error, **training on 74 extra near-empty series measurably degraded the shared model for everyone**.

**#25-#29 - four re-clustering variants, four regressions** (Azar 1403: +0.032 for #25, +0.023 for #26,
+0.005 for #28, +0.032 for #29). #26 was designed to rule out the obvious explanation (merging into
strong cities contaminates them): strong cities were left *completely untouched*, yet their forecasts
shifted by up to −14.8 % anyway. **The mechanism:** the reveal curve's `global_curve` and
`dow_adjustment` are medians pooled over *every* cluster-day, and the training panel is shared, so
any change to the cluster set - however small - re-calibrates the large clusters that dominate WAPE.
*Clustering is not a locally containable choice in this architecture.* A real fix is to fit the
pooled statistics from a fixed reference set of large clusters.

**#31 Strict 500/month floor, 148 clusters.** A cluster must clear 500 searches in **each** of the 8
months of the reference year (not a cumulative or averaged total): it fixes a diagnosed bug where a
city passed the old "strong" test with 54 stale searches from two years ago. 247 → 148 clusters.
Aban was a wash (0.1279 → 0.1286) and Shahrivar improved (0.3904 → 0.3296, like for like with #32),
so it became the **final scheme**. **#32** checked whether √-weighting compounds with it: it does not
(√ on the old clustering 0.2137 beat √ on the new 0.2399).

## F. Experiments run during the refactor (final code, 148 clusters)

### #35 Zero-volume centroid fix

**Finding.** While reproducing the clustering (it matches the shipped `clusters.csv` exactly) a
latent bug turned up: when two **zero-demand** clusters merge, their demand-weighted centroid
divides by zero demand and collapses to (0°, 0°), so the merged group later becomes "nearest" to
whichever cluster lies closest to the Gulf of Guinea. 63 of the 321 cities had no demand in the
reference year, so this happened 15 times and produced three merges spanning 6,400-7,100 km.

**Fix.** `--fix-zero-volume-centroid` uses the plain mean of the coordinates in that case. It
changes the cluster of 42 cities (still 148 clusters).

**Effect** - final model (Stage B + calibration), paired over the same four seeds (42, 1, 2, 3);
Δ = fixed − legacy, ± standard error of the paired difference:

| Month | legacy | fixed | Δ |
|---|---:|---:|---:|
| Azar 1403 | 0.1385 | 0.1376 | −0.0009 ± 0.0050 |
| Mordad 1404 | 0.1863 | 0.1888 | +0.0025 ± 0.0018 |
| Shahrivar 1404 | 0.3645 | 0.3879 | +0.0234 ± 0.0217 |
| Aban 1404 | 0.1244 | 0.1226 | −0.0018 ± 0.0023 |
| *mean, 3 regular months* | *0.1497* | *0.1496* | |

**Verdict: no measurable effect** - every Δ is within ~1.4 standard errors of zero, and the
regular-month mean is identical to four decimals. The *single-seed* comparison (seed 42 only) had
suggested a large gain (Azar 1403 −0.013, Aban −0.008, Shahrivar +0.044): a clean illustration of
why #36 matters. The default keeps the legacy behaviour so the shipped 148-cluster mapping is
reproduced exactly; the fix is recommended for new datasets because it yields geographically
sensible clusters at no accuracy cost.

### #36 Seed sensitivity - the noise floor

Four LightGBM seeds (42, 1, 2, 3) on the final configuration; mean ± standard deviation:

| Month | Stage A | Stage B | Stage B + calibration | range of Stage B + calibration |
|---|---:|---:|---:|---|
| Azar 1403 | 0.182 | 0.139 ± 0.005 | 0.139 ± 0.005 | 0.135 - 0.145 |
| Mordad 1404 | 0.241 | 0.161 ± 0.001 | 0.186 ± 0.006 | 0.182 - 0.195 |
| Shahrivar 1404 | 0.159 | 0.426 ± 0.034 | 0.365 ± 0.024 | 0.330 - 0.383 |
| Aban 1404 | 0.152 | 0.135 ± 0.002 | 0.124 ± 0.003 | 0.121 - 0.129 |

* **Robust to the seed:** Stage B beats Stage A on the three regular months; calibration helps Aban,
  hurts Mordad (0.182-0.195 vs 0.159-0.162 uncalibrated, *every* seed), is neutral on Azar;
  Shahrivar fails (0.33-0.45 vs Stage A 0.159) in every seed.
* **Not robust:** the shipped seed (42) is the *best* of the four on Shahrivar (0.330 vs 0.370-0.383)
  and the *worst* on Azar 1403 (0.1446 vs 0.1345-0.1397), so the reported single-seed table is
  neither systematically flattering nor pessimistic.
* **Implication:** determinism (#16) removes run-to-run noise for a fixed seed but not
  seed-to-seed noise. Any single-run delta under ~0.01 on a regular month (~0.05 on the spike
  month) cannot be told apart from re-seeding - including several small gains recorded above
  (#12, #13, #34), which were kept for principled reasons (correctness, mechanism) rather than on
  the measured delta alone.

### #37 Post-hoc diagnostics: guardrail and short-horizon routing

Post-processing the final predictions, using the saved per-cell backtest outputs. **These
thresholds were chosen after seeing the same four windows, so treat them as leads.**

*Guardrail on calibrated predictions* (clip to [0.7, 1.5] × Stage A where Stage A ≥ 1,000; four
seeds, mean ± sd):

| Month | Stage B + calibration | + guardrail |
|---|---:|---:|
| Azar 1403 | 0.139 ± 0.005 | 0.142 ± 0.006 |
| Mordad 1404 | 0.186 ± 0.006 | 0.185 ± 0.005 |
| Shahrivar 1404 | 0.365 ± 0.024 | **0.260 ± 0.010** |
| Aban 1404 | 0.124 ± 0.003 | 0.125 ± 0.003 |
| *mean, 3 regular months* | 0.150 | 0.150 |
| *mean, all 4 months* | 0.203 | **0.178** |

The effect on the spike month is large and consistent across seeds (also lower variance), and costs
essentially nothing elsewhere: cheap insurance. It is therefore exposed as
`forecast --guardrail`, but **off by default** so the shipped configuration matches the original
solution.

*Routing short horizons to Stage A* (seed 42 only): by horizon, Stage A beats Stage B + calibration
for 1-7 days ahead (WAPE 0.041 vs 0.059 and 0.088 vs 0.141, pooled over the regular months) while
Stage B wins from ~8 days. Using Stage A for horizons ≤ 7 days:

| | shipped | route ≤ 3 d | route ≤ 7 d | route ≤ 7 d + guardrail |
|---|---:|---:|---:|---:|
| mean, 3 regular months | 0.1532 | 0.1513 | 0.1469 | 0.1476 |
| mean, all 4 months | 0.1973 | 0.1914 | 0.1844 | 0.1666 |

Promising (it helps regular months *and* the spike month) but based on one seed and a threshold
picked from the by-horizon table - validate on more windows before adopting.

---

## Lessons

1. **Determinism ≠ robustness.** Fixing the seed made experiments repeatable, not meaningful.
   Quantify the noise floor early (#36), then ignore deltas inside it.
2. **Fixes that help typical months tend to hurt the spike month,** because the spike is driven by
   seasonal *level*, which calendar and timing features do not capture (#12, #13, #30, #33).
3. **In these clusters, variance is signal.** Pooling, shrinkage and target encoding all failed for
   the same reason: the noisiest clusters are the most seasonal ones (#14, #17, #22, #30).
4. **A better model family did not beat a well-featured GBM** (#24); the wins came from the problem
   structure - censoring, replayed training data, calendar blocks.
5. **Post-processing beat modelling for regime robustness:** the guardrail (#37) did more for the
   spike month than any model change, at almost no cost elsewhere.
6. **Pooled statistics make clustering non-local** (#25-#29): shared components couple every cluster
   to every other.

## Code removed during the refactor

Features and helpers that the shipped model did not use were dropped from the package to keep it
focused; their behaviour and measured effect are documented above. They remain in the original
hackathon archive: per-cluster × month baseline (#30), Bayesian shrinkage of history (#22),
shrunk Stage-A bias (#17), `holiday_magnitude` (#5), the shoulder-day flags (#33), `cluster_hist_n`
(a constant column LightGBM ignored - removing it changes no prediction), and the unused
categorical-feature plumbing. The √-weighting and objective options, the guardrail and the
calibration variants are kept.
