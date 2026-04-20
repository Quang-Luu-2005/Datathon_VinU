# Rescuing RMSE without surrendering MAE: a prioritized metric-squeezing playbook

**Your RMSE got worse while MAE improved by 3.3% — that is a textbook symptom, not a mystery.** Optimizing an L1/Tweedie (high variance_power) objective pushes predictions toward the conditional *median* of a right-skewed revenue distribution; since median < mean on skewed data, you systematically under-predict the biggest days, and their squared residuals dominate RMSE. The single highest-ROI move for your pipeline is therefore **not** more Optuna trials — it is to train a parallel RMSE-aware learner, blend it with your current base, and calibrate the blend on walk-forward OOF. Realistic target with the roadmap below: **Revenue MAE 568K → 510–540K, RMSE 787K → 640–700K, R² 0.788 → 0.82–0.86**. COGS should move proportionally.

The rest of this report develops a diagnostic-first framework, gives ready-to-use code for every technique, ranks them by ROI against your specific symptom, and closes with a four-week sequencing plan that front-loads the rescue moves and defers low-value work (20-bucket Optuna, pseudo-labeling, BMA) that would waste compute or overfit your small validation footprint.

---

## 1. Diagnose before you optimize: a 90-minute triage

Run this panel *before* any modeling changes. Every subsequent technique choice keys off what it reveals.

```python
import numpy as np, pandas as pd, matplotlib.pyplot as plt, scipy.stats as stats
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import linregress

def rmse_regression_diagnosis(y_true, y_pred):
    return pd.Series({
        'var_ratio': np.var(y_pred)/np.var(y_true),              # <1 → preds compressed
        'std_ratio': np.std(y_pred)/np.std(y_true),
        'mean_ratio': y_pred.mean()/y_true.mean(),               # <1 → under-predict bias
        'top5_sq_err_share': np.sort((y_true-y_pred)**2)[-5:].sum()/((y_true-y_pred)**2).sum(),
        'large_target_mae': np.mean(np.abs(y_true-y_pred)[y_true>np.quantile(y_true,0.9)]),
        'small_target_mae': np.mean(np.abs(y_true-y_pred)[y_true<np.quantile(y_true,0.1)]),
        'ols_slope': linregress(y_pred, y_true).slope,           # want ≈1.0
        'resid_skew': stats.skew(y_true-y_pred),
        'resid_kurt': stats.kurtosis(y_true-y_pred),
    })
```

**Read the output like this** (decision tree for *which* intervention comes first):

| Symptom | Root cause | First intervention |
|---|---|---|
| `var_ratio < 0.85` AND `ols_slope < 0.9` | Predictions compressed toward median (**your likely case**) | **Linear rescale + add L2 sibling model** (§5.1) |
| `top5_sq_err_share > 0.25` | 5 days drive RMSE | **Winsorization + event-day model** (§5.2) |
| `resid_kurt > 6`, fat QQ tails | Heavy-tailed residuals | **Huber loss with δ ≈ MAD(residuals)** (§3.2) |
| `mean_ratio < 0.95` and log1p used | Missing Duan/half-variance correction | **Duan smearing** (§6.3) — often a 5–15% RMSE freebie |
| Residuals grow with horizon bucket | Trend drift accumulates | **STL/Prophet detrend + XGB on residuals** (§6.4) |
| Residual ACF bars outside CI at lag 7/365 | Missed seasonality | **Tune SARIMA orders or add lag features** |
| Adversarial-validation AUC > 0.85 | Feature drift train↔holdout | **Drop top-drifting features** (§7.1) |

Complement the table with four error-breakdown tables (error by horizon bucket, by day-of-week, by month, by target-magnitude decile). These tell you *where* to deploy specialized treatment — event multipliers, horizon-specific stacking, weekend interactions — instead of spraying regularization across everything.

---

## 2. The decision tree: which lever to pull first

```
Is RMSE regressing while MAE improving?  ← YOUR CASE
│
├── YES → §5 RMSE Rescue Path (execute in order)
│         1. Add reg:squarederror sibling model + blend with L1 base
│         2. Drop Tweedie variance_power 1.5+ → 1.1–1.3
│         3. Duan smearing if log1p in pipeline
│         4. Linear OOF calibration (Platt-for-regression)
│         5. Winsorize predictions at 1st/99th percentile
│         6. Huber objective replacing L1 in one ensemble member
│
└── NO (both metrics plateaued)
    │
    ├── MAE >> RMSE/n (uniform errors, no big tails)
    │   → Feature engineering + target encoding + CQR q=0.5 point forecast
    │
    ├── Low R² but MAE fine → model under-reacts
    │   → Reduce regularization, add dynamic features, Chronos/TimesFM features
    │
    ├── Errors cluster by horizon → direct per-horizon models
    │
    └── Errors cluster on events → event-specific model + oracle features
```

---

## 3. Loss function engineering: the core MAE/RMSE tradeoff

**The foundational result** (Gneiting 2011): different metrics have different Bayes-optimal point predictors. L2 → conditional mean; L1 → conditional median; pinball(q) → q-quantile. You cannot simultaneously minimize MAE and RMSE with one model on non-Gaussian residuals. You must either (a) train multiple models and blend, or (b) post-hoc shift toward the mean. Both are cheap; doing them is what separates top-100 from top-10 finishers in M5.

### 3.1 Huber loss with δ tuned to residual scale

Use XGBoost's native `reg:pseudohubererror` with `huber_slope=δ`. Rule of thumb: **δ ≈ 1.0–1.5 × MAD(residuals)** from a baseline run. For your Revenue MAE ≈ 568K, sweep δ ∈ {300K, 500K, 800K, 1.2M} in Optuna. Huber is L2 near 0 (honest for RMSE) and L1 in the tail (honest for MAE) — **the best single-model choice when balancing both**.

```python
xgb.XGBRegressor(objective="reg:pseudohubererror", huber_slope=5e5,
                 tree_method="hist", learning_rate=0.03, max_depth=7,
                 n_estimators=5000, early_stopping_rounds=250)
```
Expected lift vs current: **RMSE −2 to −6%, MAE ±0 to −2%.**

### 3.2 Multi-loss ensemble (THE technique every M5/Favorita winner used)

Train four diverse base learners — `reg:squarederror`, `reg:absoluteerror`, `reg:pseudohubererror`, `reg:tweedie` with low variance_power — on identical walk-forward folds, then fit convex blend weights on OOF to minimize a composite loss. **Expected combined lift: MAE −2 to −6%, RMSE −3 to −8%, R² +0.01 to +0.03.**

```python
from scipy.optimize import minimize
def combined_loss(w, preds, y, alpha=0.5):
    w = np.clip(w, 0, None); w = w / w.sum()
    p = preds @ w
    mae, rmse = np.mean(np.abs(p-y)), np.sqrt(np.mean((p-y)**2))
    return alpha*(mae/568_000) + (1-alpha)*(rmse/787_000)   # scale-normalize

res = minimize(combined_loss, np.ones(4)/4, args=(oof_preds, y_oof, 0.5),
               method="Nelder-Mead")
w_opt = np.clip(res.x, 0, None); w_opt /= w_opt.sum()
```

Empirical weight patterns from winning solutions: **balanced target** ≈ (0.3 L2, 0.3 L1, 0.3 Huber, 0.1 Tweedie p=1.2); **RMSE-heavy** ≈ (0.6 L2, 0.1 L1, 0.3 Huber). Your current pipeline likely sits too far toward Tweedie+L1.

### 3.3 Tweedie variance_power: push it down, not up

For right-skewed but strictly positive revenue: `p ∈ [1.1, 1.3]` behaves convex (good for RMSE); `p ≥ 1.5` pulls toward the median (what you have now, and what hurt RMSE). **Drop your Tweedie to p=1.2 and accept ~0–1% MAE cost for 1–3% RMSE recovery**, cost-free in compute.

### 3.4 Quantile regression with multi-output + asymmetric losses

XGBoost 2.0+ natively fits multiple quantiles simultaneously with `reg:quantileerror, quantile_alpha=[0.1, 0.5, 0.9]`. Use the q=0.5 output as a CQR-regularized MAE-optimal point forecast (§4.3) and keep q=0.1/0.9 for prediction intervals. Use **LinEx asymmetric loss** only if you have an asymmetric business cost (e.g., stock-out penalty ≫ overstock cost for COGS).

### 3.5 Custom eval_metric for early stopping

Even while using `reg:squarederror` as the objective, bias *early stopping* toward your evaluation blend:

```python
def mae_rmse_combo(y_pred, dtrain, w=0.5):
    y = dtrain.get_label()
    return "combo", w*np.mean(np.abs(y_pred-y)) + (1-w)*np.sqrt(np.mean((y_pred-y)**2))
```

Zero overfitting risk, zero compute cost, typically 0.5–1.5% gain on held-out composite metric.

---

## 4. Post-hoc calibration: the 10-minute free wins

These all train on walk-forward OOF predictions and apply at inference. They are individually small but nearly always positive in expectation, and stack well.

### 4.1 Linear OOF calibration (Platt-for-regression)

```python
from sklearn.linear_model import LinearRegression
a, b = LinearRegression().fit(oof.reshape(-1,1), y_oof).coef_[0], ...intercept_
y_cal = a*test_preds + b
```

**The single most underrated trick in the stack.** GBDT predictors are structurally shrinkage estimators (they never extrapolate past training max), so on trending series you typically need `a ≈ 1.05–1.10, b > 0`. Expected: **MAE −1 to −3%, RMSE −1 to −3%, R² +0.005 to +0.015** with near-zero overfitting risk. Run it always.

### 4.2 Quantile mapping (delta method)

Multiplicatively re-inflates the tail compression that L1/Tweedie causes:

```python
qs = np.linspace(1e-4, 1-1e-4, 1000)
pq, tq = np.quantile(oof,qs), np.quantile(y_oof,qs)
delta = tq / np.maximum(pq, 1e-6)
d = np.interp(test_preds, pq, delta)
y_cal = test_preds * d
```

The climate-science delta-QM pattern, adapted for revenue. Expected: **MAE −1 to −2%, RMSE −2 to −4%.** Use *instead of* isotonic for your RMSE issue (isotonic is piecewise-constant and can hurt RMSE).

### 4.3 Conformalized Quantile Regression as a point forecast

CQR (Romano, Patterson, Candès 2019) trains multi-quantile models then uses an OOF calibration split to adjust intervals. The **q=0.5 output** is a multi-quantile-regularized median estimator that often **beats single-output L1 models by 1–3% MAE**. Implementation is straightforward with XGBoost's native multi-quantile support.

### 4.4 Residual bootstrap bias correction (block-bootstrap for time series)

Fit per-prediction-bucket trimmed-mean residuals on OOF with a **moving-block bootstrap** (block length ≈ 30 days) to preserve autocorrelation:

```python
edges = np.quantile(oof, np.linspace(0,1,20))
bias = [np.mean(np.clip(y_oof[m]-oof[m], *np.quantile(y_oof[m]-oof[m], [0.1,0.9])))
        for m in [(oof>=edges[i])&(oof<=edges[i+1]) for i in range(19)]]
centers = 0.5*(edges[:-1]+edges[1:])
y_cal = test_preds + np.interp(test_preds, centers, bias)
```

Expected: **MAE −0.5 to −2%, RMSE −1 to −3%.** Essential complement to the linear calibrator when bias is non-linear in prediction magnitude.

### 4.5 When NOT to use isotonic regression

Isotonic is rank-preserving but **piecewise-constant output destroys RMSE on smooth series** (introduces step artifacts). If you must use it, wrap with a PCHIP interpolator (`scipy.interpolate.PchipInterpolator`) over the fitted isotonic knots for smooth monotone calibration. Isotonic remains excellent for constraining the COGS/Revenue ratio into sensible bounds — that is its best use in your pipeline.

---

## 5. RMSE-specific rescue: treat this as the top-priority track

### 5.1 Stack an RMSE booster on top of your current MAE base

This is the technique that separated top-10 from top-100 in M5. Train a shallow L2 learner to predict residuals of your current blend:

```python
resid_oof = y_oof - base_oof
stack_feats = np.column_stack([base_oof, X_oof])      # base pred is the key feature
xgb_stack = xgb.XGBRegressor(
    objective="reg:squarederror", max_depth=4,        # shallow: correction, not competition
    n_estimators=1500, learning_rate=0.02, reg_lambda=5.0,
    early_stopping_rounds=200).fit(stack_feats, resid_oof, eval_set=[...])
y_final = base_test + xgb_stack.predict(stack_feats_test)
```

**Expected: RMSE −3 to −7%, MAE −0.5 to −1.5%.** Targets the squared-residual pattern without destroying MAE-good properties of the base.

### 5.2 Winsorization at empirical percentiles

```python
y_cal = np.clip(test_preds, np.quantile(y_train,0.01), np.quantile(y_train,0.99))
```

Cheapest insurance in the playbook. If Tweedie/L1 training occasionally fires 3σ-low predictions, clipping removes the squared-residual catastrophes. **RMSE −1 to −4%, MAE neutral.**

### 5.3 James-Stein shrinkage toward historical mean (tail-only)

For predictions in the top/bottom decile, shrink toward recent mean using the prediction-ensemble variance as the shrinkage signal. This is RMSE-specific theory: James-Stein dominates MLE under squared-error risk by trading bias for variance. **RMSE −1 to −3%** when applied only to tails.

### 5.4 Dynamic output clipping with growth headroom

```python
growth = (1.15)**(horizon_days/365)                     # 15% annual headroom
lo, hi = np.quantile(y_train,0.01)*0.5, np.quantile(y_train,0.99)*1.5*growth
y_cal = np.clip(test_preds, lo, hi)
```

Addresses long-horizon extrapolation blowups; tighter than winsorization but keeps room for trend growth. **RMSE −2 to −8%.**

---

## 6. Target transformations that move the needle

### 6.1 Duan smearing on log-transformed predictions (biggest quick win if missing)

If your pipeline uses `log1p` anywhere, **you almost certainly need this**. `exp(ŷ_log)` estimates the *median*, not the mean, under-predicting skewed revenue by 5–30%.

```python
resid_log = y_val_log - model.predict(X_val)
smear = np.mean(np.exp(resid_log))                     # Duan factor ≥ 1
y_pred_mean = np.exp(model.predict(X_test)) * smear - 1
# Better: horizon-bucket-specific smearing factor
horizon_smear = {h: np.mean(np.exp(resid_log[horizon==h])) for h in buckets}
```

**MAE −2 to −8%, RMSE −5 to −15%, R² +0.01 to +0.04.** Top-ROI technique if not already applied.

### 6.2 asinh target transform (handles zeros and heavy tails natively)

For Revenue with occasional zeros or small negatives (returns days), `arcsinh(y/SCALE)` with SCALE ≈ median(y)/10 beats log1p. Caveat: not scale-invariant (Bellemare & Wichman 2020) — tune SCALE on validation. Expected: **MAE −2 to −5%, RMSE −3 to −8%** over log1p when zeros present.

### 6.3 STL/Prophet detrending before XGBoost (biggest lever at long horizon)

For a 548-day horizon, trend drift dominates residual error past day 180. Decompose with STL, forecast trend+season separately (STLForecast + ARIMA), then train XGBoost on the residual:

```python
from statsmodels.tsa.forecasting.stl import STLForecast
stlf = STLForecast(y_train, ARIMA, model_kwargs=dict(order=(2,1,0), trend='t'),
                   period=7).fit()
y_final = stlf.forecast(548).values + xgb_on_residuals.predict(X_test)
```

**MAE −3 to −10%, RMSE −5 to −15%, R² +0.02 to +0.06.** Best single lever for your long horizon — do not skip.

### 6.4 Box-Cox and Yeo-Johnson

Box-Cox picks the optimal λ via MLE; typically gains 1–4% MAE over log1p when optimal λ is 0.2–0.4 (moderate rather than extreme skew). Yeo-Johnson is Box-Cox that tolerates negatives — use if Revenue adjustments can go negative. Both require the same Duan-style inverse-transform bias correction as log1p.

### 6.5 Rolling z-score per time period (leak-safe)

```python
mu = y.shift(1).rolling(365, min_periods=90).mean()
sd = y.shift(1).rolling(365, min_periods=90).std()
y_z = (y - mu) / sd
```

Removes 10-year level drift; forecast mu/sd separately (EWMA or linear trend) to invert. **MAE −3 to −10%** when non-stationary level dominates, but implementation-error-prone — always audit the shift.

---

## 7. Feature selection and adversarial validation

### 7.1 Adversarial validation — highest-ROI feature move for 8-table pipelines

Train a LightGBM classifier to distinguish old vs recent rows across your 8 tables' features:

```python
# Label recent 25% as 1, rest 0; stratified 5-fold LightGBM
auc, imp = adversarial_validation(X_old, X_recent, features)
# AUC ≈ 0.5: healthy.  0.7–0.85: moderate drift — inspect top 5–10 features.
# >0.9: severe drift — top features are leaking time info (levels, cumulative sums,
# naive target encodings).
drifting = imp[imp > imp.quantile(0.90)].index.tolist()
```

Two strategies: **drop** drifting features, OR use the classifier's OOF probability as **sample weights** to up-weight training rows that look like the holdout period. **MAE −2 to −6%, RMSE −3 to −10%** on long-horizon forecasts — a particularly large win given your 548-day horizon.

### 7.2 Null importance (Altmann et al.)

Shuffle target 80 times, record LightGBM feature importances for each run, build a null distribution per feature, keep features whose real importance exceeds the 75th null percentile. Catches features that look important but aren't — common in 8-table cross features. **MAE −2 to −6%** by pruning noise, ~80 model fits cost.

### 7.3 Time-aware target encoding for product_id / promotion_id

Expanding-window mean encoding with Bayesian smoothing against the global mean. Use K-fold time-aware splits if volumes per ID are small. Critical error to avoid: naive target encoding leaks future target into past — adversarial validation will immediately flag this (AUC > 0.9). **MAE −3 to −8%** on promotion-heavy periods when implemented correctly; same range *negative* if botched.

### 7.4 KS drift filter

Screen every feature with `ks_2samp(train, recent)`, flag those with `ks_stat > 0.1` regardless of p-value (large-n makes p-values meaningless). Use as a safety net, not a primary selector.

---

## 8. Ensemble advancements beyond your current blend

### 8.1 Caruana forward-stepwise selection with replacement (bagged)

**The single highest-ROI ensemble technique for competition forecasting.** Build a library of 30–60 candidate predictors (different seeds, objectives, DART vs gbtree, snapshot rounds, small HP variations), then greedily add models one at a time — with replacement, so the same model can be added multiple times to effectively learn weights in 1/N increments. Reference: Caruana ICML 2004.

```python
def ensemble_selector(loss_fn, y_hats, y_true, init_size=1, max_iter=80):
    losses = pd.Series({m: loss_fn(p, y_true) for m,p in y_hats.items()}).sort_values()
    init = losses.index[:init_size].tolist()
    y_avg = np.mean([y_hats[m] for m in init], axis=0); n=len(init)
    members = [init]
    for _ in range(max_iter):
        wc, wn = n/(n+1), 1/(n+1)
        best = min(y_hats, key=lambda m: loss_fn(wc*y_avg + wn*y_hats[m], y_true))
        y_avg = wc*y_avg + wn*y_hats[best]; n += 1
        members.append(members[-1]+[best])
    # weights from member-selection counts
    return members
```

**Critical refinement**: bag the selector — run 20 bagged selectors on random 50% subsets of the library, average resulting weights. This prevents overfitting the selection set, which is the dominant failure mode. **Expected: MAE −2 to −5%, RMSE −2 to −5%.**

### 8.2 Convex non-negative weighted blend with Dirichlet prior

For a smaller library (≤6 members), SLSQP-constrained simplex optimization with a `Dirichlet(α=2–5)` shrinkage prior toward uniform beats both OLS and unconstrained optimization. Add per-horizon-bucket weights *only if* you have ≥200 validation rows per bucket — with your 548-day × 6-fold = 3,288 validation rows, **stay at ≤6 horizon buckets** and use hierarchical smoothing across adjacent buckets (penalize `||W[b]-W[b-1]||²`). Going to 20 buckets with your data is a negative-expected-value move.

### 8.3 Snapshot ensembles + SWA analog for GBDT

Save XGBoost boosters at rounds {best−500, best−400, ..., best}; blend their predictions. Pair with cyclical learning rate for extra diversity. Zero extra training cost, **0.5–2% RMSE gain**. Particularly strong as library members for Caruana selection.

### 8.4 DART booster as ensemble member

DART (Dropouts meet MART) drops a fraction of already-built trees each round — internal ensembling. Add 3 DART variants to your Caruana library. Warning: DART breaks early-stopping prediction buffers, so use fixed `num_boost_round` from a reserved fold. **0.3–1.5% standalone gain, more when blended.**

### 8.5 Feature-Weighted Linear Stacking (FWLS)

When relative model accuracy depends on context (Prophet wins at H=30, XGB at H=548, seasonal naive near recurring events), FWLS learns row-level blend weights as linear functions of meta-features (horizon, month, rolling vol, holiday proximity). With 4 base models × 6 meta-features = 24 ridge coefficients, fit-able on your 3,288 validation rows. **Expected: 0.5–2%**, proven on Netflix Prize 2nd place.

### 8.6 Skip: Bayesian Model Averaging, pseudo-labeling, teacher-student

BMA with BIC-approximated weights empirically collapses toward the single best model and is dominated by Caruana selection (Caruana 2006 Table 2). Pseudo-labeling on a 548-day horizon risks confirmation bias — Kaggle retail post-mortems report null or negative results. Teacher-student distillation is an inference-speed technique with ≤1% accuracy gain on regression. **Deprioritize all three.**

---

## 9. Cross-validation improvements that matter

### 9.1 Purged walk-forward with embargo

Your rolling features with window W and lag features up to 365 days leak across fold boundaries. Purge `max(rolling_window, max_lag)` days between train and val — typically 30–90 days for your pipeline. This does not change current metric values directly but makes tuning decisions based on them *correct*, typically worth **0.5–1.5% honest RMSE improvement** because you stop picking overfit hyperparameters.

### 9.2 Nested walk-forward CV for blend-weight selection

The biggest upgrade you can make to your evaluation harness. Outer walk-forward for final metric reporting; inner walk-forward on each outer train segment for blend-weight fitting. Prevents the common failure of tuning blend weights on the same folds you report. **Expected reported metrics will worsen 1–3% initially** (optimism bias removed) — but the numbers will be trustworthy and transfer to the true holdout. Do this before you believe any of the Kaggle-level gains claimed in this report.

### 9.3 CPCV for hyperparameter selection only

Combinatorial Purged Cross-Validation (de Prado) generates ~9 reconstructable test paths from one history. **Do not use for production metric reporting** (you want the causal expanding-window last fold). Do use for hyperparameter and blend-weight selection — variance of the metric estimate drops by ~3×, letting you distinguish real 2% gains from noise. Library: `skfolio.model_selection.CombinatorialPurgedCV`.

### 9.4 Block-bootstrap confidence intervals on metrics

Before celebrating any improvement, report 95% CI on each metric via moving-block bootstrap (block length = 30 days). A 2% MAE improvement with ±4% CI is not an improvement. This operational discipline is the difference between iterating forward and iterating in circles.

### 9.5 Seed budget: stay at 3–5, not 10+

Seed bagging gives `Var(y_bar_S) = σ²((1-ρ)/S + ρ)` with ρ ≈ 0.9 for XGBoost — 1→3 seeds captures 67% of achievable reduction, 3→5 adds 13%, 5→10 adds 13%. **Going 3→5 is a small free win (~0.2–0.5%); going beyond 10 is diminishing returns.** Use the spared compute for more objective-diverse library members (§8.1).

---

## 10. Kaggle 2023–2025 tactics worth importing

**Recurring patterns across Store Sales Ecuador, M5, Optiver, Enefit, GoDaddy, LEAP winners** (appearing in 3+ top-3 solutions):

1. **Tweedie/Poisson for zero-inflated right-skew** with `p ∈ [1.05, 1.25]` (not 1.5+). You already have Tweedie — fix the variance_power.
2. **Direct horizon-specific models** for long horizons, or direct-recursive hybrid (direct for weeks 1–2, recursive thereafter). Your horizon-bucket blend is a coarse version; consider separate models per bucket.
3. **Seed-bag identical model 5× with different seeds + subsamples**. You have 3 — bump to 5.
4. **Multi-window CV + select on `mean(CV) + k·std(CV)`** — never on a single fold.
5. **Custom asymmetric or auxiliary loss** (smooth L1, confidence head). LEAP 1st-place used a confidence-bin auxiliary task that reweights losses per-sample — novel 2024 technique worth trying.
6. **Magic multiplier / isotonic calibration** on OOF to minimize the evaluation metric. M5 5th place won gold with naive public notebook × 0.95. For your composite MAE+RMSE+R², fit a single multiplier per horizon bucket that minimizes composite on OOF.
7. **Hierarchical reconciliation** (MinT-shrink) on Revenue + COGS + GrossProfit hierarchy, via `hierarchicalforecast` (Nixtla) or `darts`. Or model `margin_t` as a slow-moving series and derive COGS = Revenue × (1 − margin). **Expected 2–6% on the correlated target** plus business-logic coherence — this is the COGS-specific move.
8. **Event-aware feature set for Tet/11.11/12.12/Black Friday**: days-to/from event (±14 days), event-type categorical, same-event-last-year lag, same-event-2y-ago lag, pre-event ramp ratio, lunar-calendar conversion for Tet. Your event multipliers are Bayesian-shrunk — upgrade to full feature interactions plus a separate event-day model blended at a learned weight.
9. **Drop ancient history**: Favorita 1st place used only 1–5 months. With 10 years you're likely giving weight to 2016 dynamics that don't apply to 2026. Try training on only the last 2–3 years plus seasonal-year-ago features.
10. **Arithmetic mean in log-space (≈ geometric mean in real space)** for ensembling right-skewed targets — typically 1–3% better than arithmetic mean in real space.

**Foundation model features** (TimesFM 2.5, Chronos-2, Moirai): as of Nov 2025, zero-shot TSFMs underperform feature-engineered XGBoost on finance/retail benchmarks (arxiv 2511.18578), **but** their quantile outputs used as **auxiliary features** inside XGBoost give 1–4% lift in recent ensembles (arxiv 2508 bagging-boosting paper). Chronos-2 uniquely supports future-known covariates (promos, events) — this is the one blend member worth adding in 2026. LoRA fine-tuning TimesFM-2.5 on your own Revenue/COGS costs ~1 GPU-hour.

**Ethical metric-gaming for mixed MAE+RMSE+R²**: R² and RMSE share the same argmin (conditional mean), so predict the mean then apply **small shrinkage toward the median** (5–15%). This pays a tiny RMSE/R² cost for a meaningful MAE cut — usually strict Pareto improvement on the composite.

---

## 11. The four-week roadmap (front-loaded by ROI)

The single biggest mistake would be to implement this in order of technical novelty rather than diagnostic fit to your symptom. **Week 1 is mandatory before anything else.**

**Week 1 — Rescue RMSE, instrument honestly.** Run the §1 diagnostic panel. Apply combined MAE+RMSE eval_metric for early stopping. Apply linear OOF calibration. Apply winsorization at 1st/99th percentile. Drop Tweedie variance_power from whatever you have to 1.2. Add Duan smearing if `log1p` is in the pipeline. Install nested walk-forward CV harness and block-bootstrap CI reporting. *Expected: RMSE −4 to −8%, MAE −1 to −2%, R² +0.01 to +0.03, plus trustworthy metric reporting.*

**Week 2 — Add the missing L2 learner, blend properly.** Train a parallel `reg:squarederror` XGBoost on the same folds. Add a Huber learner with δ ≈ MAD(residuals). Fit convex blend weights on OOF with scipy SLSQP + Dirichlet(α=3) prior, minimizing the composite loss. Stack a shallow L2 residual-correction booster on the MAE base (§5.1). Drop ancient history — retrain everything on last 3 years plus year-ago lags. *Expected additional: RMSE −3 to −5%, MAE −1 to −3%.*

**Week 3 — Feature quality and long-horizon structure.** Run adversarial validation; drop top-10 drifting features. Null-importance prune the remaining set. Rebuild target encoding for product_id/promotion_id with expanding-window Bayesian smoothing. Implement STL/Prophet detrending + XGBoost on residuals as a parallel branch. Add full event-interaction features (days-to/from, same-event lags, lunar calendar). Add Chronos-2 quantile forecasts as auxiliary features. *Expected additional: MAE −3 to −6%, RMSE −3 to −6%.*

**Week 4 — Polish and consolidate.** Build a library of 30–60 candidate predictors (seeds × objectives × DART × snapshots × detrend-branch × Chronos-feature variants). Run bagged Caruana forward selection to produce final blend weights. Apply quantile-mapping (delta) post-calibration as the last step. Apply MinT-shrink reconciliation on the (Revenue, COGS, GrossProfit) hierarchy. Lock in hyperparameters via CPCV. *Expected additional: MAE −1 to −3%, RMSE −1 to −3%.*

Cumulative realistic target: **Revenue MAE 568K → 510–540K (−5 to −10%), RMSE 787K → 640–700K (−11 to −19%), R² 0.788 → 0.82–0.86.** Cumulative stretch target with every technique firing: MAE ~480K, RMSE ~590K, R² ~0.88. COGS moves proportionally with an extra 2–6% from ratio reconciliation.

---

## 12. What NOT to do (negative-expected-value for your pipeline)

**Skip these, regardless of how clever they sound.** Going to 20-bucket Optuna blend weights without hierarchical smoothing will overfit your ~3,288 validation rows catastrophically. Pseudo-labeling across a 548-day horizon amplifies any existing bias into a feedback loop — Kaggle retail post-mortems consistently report null or negative results. Bayesian Model Averaging via BIC collapses to the single best model and is dominated by Caruana selection. Teacher-student distillation is for inference speed, not accuracy. Log-cosh custom objective gains less than 0.5% over Huber at 2× implementation cost. Adaptive conformal inference is excellent for interval calibration across your long horizon but contributes ~0% to point-forecast accuracy. Rank-based blending destroys scale for continuous-target regression and should never be used for MAE/RMSE/R². And **do not** add more Optuna trials to the base model — you are already past the point where single-model HP tuning moves the metric; the next order of magnitude comes from ensemble diversity and post-hoc calibration, not from `max_depth=9` vs `max_depth=10`.

## Conclusion: the meta-lesson

Your RMSE regression is a diagnostic gift, not a setback. It tells you exactly what is wrong — your training pushed the model onto the conditional-median manifold, and you need a second model on the conditional-mean manifold plus a calibration bridge between them. The M5, Favorita, and Optiver winners all converged on the same pattern: **many diverse learners, Caruana-selected, post-hoc calibrated, validated honestly**. Your pipeline already has the diversity infrastructure (Prophet, seasonal naive, SARIMA, XGBoost, COGS ratio branch); what it lacks is loss-function diversity within the XGBoost family and a proper composite-metric-aware calibration layer. Fix those two gaps in Week 1–2 and the remaining work is incremental. The deepest shift in mindset for a mature pipeline is to stop thinking about "making the model better" and start thinking about "measuring honestly and shifting predictions toward the metric's optimum" — which is exactly what distinguishes a top-10 Kaggle finish from a top-100 one.