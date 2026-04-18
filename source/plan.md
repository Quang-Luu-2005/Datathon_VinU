# A production-grade LightGBM pipeline for daily revenue forecasting in Vietnamese e-commerce fashion

**Bottom line:** the winning recipe for this problem is an **expanding-window walk-forward LightGBM** trained on `log1p(revenue)` with an `regression_l1` objective, Optuna-tuned with **fold-level Hyperband pruning**, augmented by (i) a **residual SARIMA/LGBM corrector**, (ii) **event-specific multiplicative bias calibration** for Tết and mega-sales, and (iii) a **constrained-weight ensemble** with XGBoost/CatBoost/Prophet/NHITS on walk-forward OOF. Against a seasonal-naïve baseline this stack typically yields 20–30% RMSE reduction, with a further 3–8% from residual correction on Tết/11.11 spike days. The **single highest-leverage decisions** are: use lunar-relative features (`days_to_tet`, `tet_proximity`), apply `.shift(horizon)` before every rolling statistic, and never let the 2023-01-01 → 2024-07-01 test block influence any tuning decision. SHAP (`TreeExplainer` with `tree_path_dependent`) doubles as an explainability layer and a leak-detector; it should be integrated into the final reporting artifact, not treated as a post-hoc diagnostic.

The plan below is organised in the order you will execute it: data prep → walk-forward CV → Optuna tuning → residual correction → ensembling → SHAP → leakage audits. Every section contains copy-pastable Python.

---

## 1. Data preparation and feature engineering

### 1.1 Master pipeline and time spine

Build a single daily spine from 2012-01-01 to 2024-07-01, left-join all other tables on `date`, and **defer the train/test split until after features are built** (features are deterministic functions of lagged values; splitting first would cause NaN proliferation at fold edges).

```python
import numpy as np, pandas as pd, lightgbm as lgb, shap
SEED = 42
TRAIN_END  = pd.Timestamp('2022-12-31')
TEST_START = pd.Timestamp('2023-01-01')
TEST_END   = pd.Timestamp('2024-07-01')
DATE, TARGET = 'date', 'revenue'
```

### 1.2 Calendar features (basic, cyclical, Vietnamese holidays)

```python
def add_basic_calendar(df):
    d = df[DATE].dt
    df['day_of_week']=d.dayofweek; df['day_of_month']=d.day
    df['day_of_year']=d.dayofyear; df['week_of_year']=d.isocalendar().week.astype(int)
    df['month']=d.month; df['quarter']=d.quarter; df['year']=d.year
    df['is_weekend']=(d.dayofweek>=5).astype('int8')
    df['is_month_start']=d.is_month_start.astype('int8')
    df['is_month_end']  =d.is_month_end.astype('int8')
    df['is_quarter_start']=d.is_quarter_start.astype('int8')
    df['is_quarter_end']  =d.is_quarter_end.astype('int8')
    return df

def add_cyclical(df):
    for col, period in [('day_of_week',7),('month',12),
                         ('day_of_year',365.25),('day_of_month',31)]:
        df[f'{col}_sin'] = np.sin(2*np.pi*df[col]/period)
        df[f'{col}_cos'] = np.cos(2*np.pi*df[col]/period)
    return df
```

**Tết is the single most important feature** for this problem. Hard-code verified solar dates for the training+test span; compute **days_to_tet, days_since_tet, tet_proximity** (exponential bump) and flags for pre-Tết shopping windows (7/14/30 days). Fashion revenue begins ramping **≈ 21 days before** Tết and crashes for 5-7 days during the holiday itself.

```python
TET_DATES = pd.to_datetime([
    "2012-01-23","2013-02-10","2014-01-31","2015-02-19","2016-02-08",
    "2017-01-28","2018-02-16","2019-02-05","2020-01-25","2021-02-12",
    "2022-02-01","2023-01-22","2024-02-10","2025-01-29"])

def add_tet_features(df):
    d   = df[DATE].values.astype('datetime64[D]')
    tet = TET_DATES.values.astype('datetime64[D]')
    nxt = np.clip(np.searchsorted(tet, d, side='left'), 0, len(tet)-1)
    prv = np.clip(nxt-1, 0, len(tet)-1)
    days_to    = (tet[nxt] - d).astype(int)
    days_since = (d - tet[prv]).astype(int)
    df['days_to_tet']=days_to; df['days_since_tet']=days_since
    df['is_tet_day']   = (days_to==0).astype('int8')
    df['is_tet_week']  = ((days_to<=3)|(days_since<=3)).astype('int8')
    df['is_pre_tet_14d']= ((days_to>=1)&(days_to<=14)).astype('int8')
    df['is_pre_tet_30d']= ((days_to>=1)&(days_to<=30)).astype('int8')
    df['tet_proximity']= np.exp(-np.minimum(days_to, days_since)/7)
    return df
```

Add Vietnamese public holidays (`python-holidays` VN), Hung Kings (hard-coded lunar dates), mega-sale double-digit days (**3.3, 6.6, 9.9, 10.10, 11.11, 12.12**), Black Friday / Cyber Monday, Women's Day (8/3), Mid-Autumn, and **paydays** (15th + end-of-month, which matter in Vietnam because salaries cycle on these dates).

```python
import holidays
MEGA = {'dd_9_9':(9,9),'dd_10_10':(10,10),'dd_11_11':(11,11),'dd_12_12':(12,12),
        'dd_3_3':(3,3),'dd_6_6':(6,6),'womens_day':(3,8)}
def add_events(df):
    years = df[DATE].dt.year.unique()
    vn = holidays.country_holidays('VN', years=list(years))
    df['is_vn_holiday'] = df[DATE].isin(vn).astype('int8')
    for name,(m,day) in MEGA.items():
        df[f'is_{name}']      = ((df[DATE].dt.month==m)&(df[DATE].dt.day==day)).astype('int8')
        this = pd.to_datetime(dict(year=df[DATE].dt.year,month=m,day=day),errors='coerce')
        nxt  = pd.to_datetime(dict(year=df[DATE].dt.year+1,month=m,day=day),errors='coerce')
        dt   = (this-df[DATE]).dt.days
        dt   = dt.where(dt>=0, (nxt-df[DATE]).dt.days)
        df[f'days_to_{name}'] = dt.clip(0,120)
    # Black Friday = 4th Friday of November
    def bf(y):
        nov = pd.date_range(f'{y}-11-01', f'{y}-11-30')
        return nov[nov.dayofweek==4][3]
    bfd = pd.to_datetime([bf(y) for y in range(df[DATE].dt.year.min(), df[DATE].dt.year.max()+2)])
    df['is_black_friday']=df[DATE].isin(bfd).astype('int8')
    df['is_cyber_monday']=df[DATE].isin(bfd+pd.Timedelta(days=3)).astype('int8')
    # Paydays
    eom = df[DATE]+pd.offsets.MonthEnd(0)
    df['is_mid_month_pay']=(df[DATE].dt.day==15).astype('int8')
    df['is_eom_pay']=(df[DATE]==eom).astype('int8')
    df['is_payday_window']=(((df[DATE].dt.day.between(14,17)))|((eom-df[DATE]).dt.days<=2)).astype('int8')
    return df
```

### 1.3 Lag and rolling statistics — `.shift(horizon).rolling(w)` is mandatory

The universal anti-leakage idiom is **shift first, roll second**. For a 1-step-ahead recursive model `shift(1)`; for an H-step direct model `shift(H)`. A rolling mean written as `df['revenue'].rolling(7).mean()` silently includes the current day — the single most common fatal bug.

```python
LAGS  = [1,2,3,7,14,21,28,30,60,90,180,365]
WINDS = [7,14,28,56,90]
EWM_A = [0.05,0.1,0.2,0.4]

def add_lag_roll(df, horizon=1):
    s = df[TARGET]
    for L in LAGS:
        df[f'lag_{L}'] = s.shift(L)
    for L in [7,14,21,28,35,49]:
        df[f'dow_lag_{L}'] = s.shift(L)
    base = s.shift(horizon)
    for W in WINDS:
        r = base.rolling(W, min_periods=max(3,W//3))
        df[f'rmean_{W}']=r.mean();   df[f'rstd_{W}']=r.std()
        df[f'rmin_{W}']=r.min();     df[f'rmax_{W}']=r.max()
        df[f'rmed_{W}']=r.median();  df[f'rskew_{W}']=r.skew()
        df[f'rkurt_{W}']=r.kurt()
        df[f'rcv_{W}']=df[f'rstd_{W}']/(df[f'rmean_{W}'].abs()+1e-6)
    for a in EWM_A:
        df[f'ewm_a{a}']    = base.ewm(alpha=a,adjust=False).mean()
        df[f'ewm_std_a{a}']= base.ewm(alpha=a,adjust=False).std()
    # Momentum and YoY
    df['diff_1']=s.shift(1)-s.shift(2)
    df['diff_7']=s.shift(1)-s.shift(8)
    df['yoy_lag']=s.shift(365)
    df['yoy_diff']=s.shift(1)-s.shift(365)
    df['yoy_ratio']=s.shift(1)/(s.shift(365)+1)
    df['yoy_roll28']=s.shift(365).rolling(28, min_periods=10).mean()
    return df
```

### 1.4 Cross-table features (web_traffic, promotions, inventory)

Three golden rules for cross-table joins: **(1)** web_traffic and inventory are populated *after* sales occur, so every column must be `.shift(1)` or deeper before it enters the model; **(2)** promotions are known in advance — same-day features are legitimate but must be **auditable** (i.e., reproducible from a plan that existed before the forecast was made); **(3)** prefer `pd.merge_asof(..., direction='backward', allow_exact_matches=False)` to enforce strict-past semantics.

```python
def add_web_traffic(df, wt):
    df = df.merge(wt, on=DATE, how='left')
    for col in ['sessions','visitors','page_views']:
        for L in [1,2,3,7,14,28]: df[f'{col}_lag_{L}']=df[col].shift(L)
        for W in [7,14,28]:
            b=df[col].shift(1)
            df[f'{col}_rmean_{W}']=b.rolling(W,min_periods=3).mean()
            df[f'{col}_rstd_{W}'] =b.rolling(W,min_periods=3).std()
        df[f'{col}_yoy']      = df[col].shift(365)
        df[f'{col}_yoy_ratio']= df[col].shift(1)/(df[col].shift(365)+1)
    # Conversion-rate proxy using strictly lagged values
    df['conv_rate_lag1']  = df[TARGET].shift(1)/(df['sessions'].shift(1)+1)
    df['conv_rate_rmean7']= df['conv_rate_lag1'].rolling(7,min_periods=3).mean()
    df['corr_rev_sess_28']= (df[TARGET].shift(1).rolling(28,min_periods=10)
                               .corr(df['sessions'].shift(1)))
    df = df.drop(columns=['sessions','visitors','page_views'])   # prevent same-day leak
    return df

def add_promos(df, promos):
    daily=[]
    for _,r in promos.iterrows():
        for dt in pd.date_range(r.start_date, r.end_date):
            daily.append({'date':dt,'promo_id':r.promo_id,
                          'discount_pct':r.discount_pct,'promo_type':r.promo_type})
    agg = (pd.DataFrame(daily).groupby('date')
             .agg(count_active_promos=('promo_id','nunique'),
                  max_discount=('discount_pct','max'),
                  mean_discount=('discount_pct','mean'),
                  n_types=('promo_type','nunique'))
             .reset_index())
    df = df.merge(agg, on=DATE, how='left').fillna({'count_active_promos':0,
         'max_discount':0,'mean_discount':0,'n_types':0})
    df['is_active_promo'] = (df['count_active_promos']>0).astype('int8')
    for W in [7,14,30]:
        df[f'promo_days_last_{W}']  = df['is_active_promo'].shift(1).rolling(W,min_periods=1).sum()
        df[f'promo_depth_last_{W}'] = df['max_discount'].shift(1).rolling(W,min_periods=1).mean()
    df['promo_x_tet']     = df['is_active_promo']*df['is_tet_week']
    df['promo_x_1111']    = df['is_active_promo']*df['is_dd_11_11']
    df['depth_x_tetprox'] = df['max_discount']*df['tet_proximity']
    return df

def add_inventory(df, inv):
    daily = (inv.groupby('date')
               .agg(total_stock=('stock_level','sum'),
                    sku_count=('sku_id','nunique'),
                    n_oos=('stock_level',lambda s:(s==0).sum()))
               .reset_index())
    daily['oos_rate']=daily['n_oos']/daily['sku_count']
    df = df.merge(daily, on=DATE, how='left')
    for col in ['total_stock','sku_count','oos_rate']:
        df[f'{col}_lag1'] = df[col].shift(1)
        df[f'{col}_rmean_7']  = df[col].shift(1).rolling(7,min_periods=3).mean()
        df[f'{col}_rmean_28'] = df[col].shift(1).rolling(28,min_periods=3).mean()
    df['stock_delta_7d'] = df['total_stock'].shift(1).pct_change(7)
    df['lowstock_x_promo']= ((df['oos_rate_lag1']>0.3).astype(int)*df['is_active_promo'])
    df = df.drop(columns=['total_stock','sku_count','n_oos','oos_rate'])
    return df
```

---

## 2. Expanding-window walk-forward validation

**Why expanding, not sliding:** with 11 years spanning COVID (2020-21 shock) and post-COVID recovery, discarding early history throws away regime-rich signal. Anchor every fold at 2012-01-01 and advance only the training-end cursor. **Why 6-8 folds × 180-day validation:** each fold becomes a *mini version of the 548-day test*, so fold-level metrics are directly comparable to what you'll see in production. **Why a gap of 28 days:** the embargo must be ≥ the longest rolling window that touches the lag chain; with a 28-day rolling mean, 28 days is the correct purge.

### 2.1 The custom splitter

`sklearn.model_selection.TimeSeriesSplit` cannot enforce a minimum initial train size, cannot anchor splits to calendar dates, and cannot purge after the validation block — all three matter here.

```python
from typing import Iterator, Tuple, Optional

class ExpandingWindowWalkForward:
    def __init__(self, initial_train_days=365*5, val_days=180,
                 step_days=180, gap_days=28, max_splits=8):
        self.initial_train_days=initial_train_days; self.val_days=val_days
        self.step_days=step_days; self.gap_days=gap_days; self.max_splits=max_splits
    def get_n_splits(self, X=None, y=None, groups=None):
        n=len(X); k=max(0,(n-self.initial_train_days-self.gap_days-self.val_days)//self.step_days+1)
        return min(k,self.max_splits) if self.max_splits else k
    def split(self, X, y=None, groups=None)->Iterator[Tuple[np.ndarray,np.ndarray]]:
        n=len(X); idx=np.arange(n); train_end=self.initial_train_days; fold=0
        while True:
            vs=train_end+self.gap_days; ve=vs+self.val_days
            if ve>n or (self.max_splits and fold>=self.max_splits): break
            yield idx[:train_end], idx[vs:ve]
            train_end += self.step_days; fold += 1
```

With `initial_train_days=5*365`, `val_days=180`, `step_days=180`, `gap_days=28` you get ~8 folds whose validation windows roll across 2017-H1 → 2022-H1. Every training index stays strictly `< 2023-01-01`, so the test block is never seen during tuning.

### 2.2 Purged variant for aggressive protection

For multi-step forecasts where rolling features reach into the validation window, also **drop the tail of each training fold**:

```python
class PurgedWalkForward(ExpandingWindowWalkForward):
    def __init__(self, *a, purge_days=28, **kw):
        super().__init__(*a, **kw); self.purge_days=purge_days
    def split(self, X, y=None, groups=None):
        for tr, va in super().split(X, y, groups):
            yield (tr[:-self.purge_days] if self.purge_days>0 else tr), va
```

---

## 3. LightGBM hyperparameter tuning with Optuna

### 3.1 Objective choice for skewed, non-negative revenue

For this target I recommend **benchmarking two objectives**: `regression_l1` trained on `log1p(revenue)` as primary (directly optimises MAE, robust to Tết spikes), and `tweedie(variance_power=1.3)` on **raw** revenue as a second candidate (handles zero-days and heavy tails natively via its log-link). Do **not** log-transform the target when using Tweedie or Poisson — those losses already have an internal log-link. Report MAE/RMSE/R² on the **original** revenue scale regardless of training space.

### 3.2 Search space and Optuna study

Use **TPE with `multivariate=True`** (correlated hyperparameters like `num_leaves`/`max_depth`/`min_data_in_leaf` benefit from joint modelling), and **HyperbandPruner** so aggressive trials die early. Pruning is applied at the **fold level**, not the boosting-step level, because `LightGBMPruningCallback` only reports correctly for the first fold in a CV loop (Optuna issue [#3203](https://github.com/optuna/optuna/issues/3203)). Target **75–150 trials**; with 8 folds of ~1 second each, 100 trials cost roughly 15 min on a laptop.

```python
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATS  = [c for c in df_feat.columns if c not in (DATE, TARGET)]
train_df = df_feat[df_feat[DATE] < TEST_START].dropna(subset=['lag_365']).reset_index(drop=True)
test_df  = df_feat[(df_feat[DATE]>=TEST_START)&(df_feat[DATE]<=TEST_END)].reset_index(drop=True)
train_df['y_log'] = np.log1p(train_df[TARGET].clip(lower=0))
test_df['y_log']  = np.log1p(test_df[TARGET].clip(lower=0))
cv = ExpandingWindowWalkForward(365*5, 180, 180, 28, 8)

def objective(trial):
    params = {
      'objective':'regression_l1','metric':'mae','boosting_type':'gbdt',
      'learning_rate'   : trial.suggest_float('learning_rate',5e-3,1e-1,log=True),
      'num_leaves'      : trial.suggest_int('num_leaves',15,255,log=True),
      'max_depth'       : trial.suggest_int('max_depth',4,12),
      'min_data_in_leaf': trial.suggest_int('min_data_in_leaf',5,100),
      'feature_fraction': trial.suggest_float('feature_fraction',0.5,1.0),
      'bagging_fraction': trial.suggest_float('bagging_fraction',0.5,1.0),
      'bagging_freq'    : trial.suggest_int('bagging_freq',1,7),
      'lambda_l1'       : trial.suggest_float('lambda_l1',1e-8,10.0,log=True),
      'lambda_l2'       : trial.suggest_float('lambda_l2',1e-8,10.0,log=True),
      'min_gain_to_split': trial.suggest_float('min_gain_to_split',0.0,1.0),
      'max_bin'         : trial.suggest_int('max_bin',63,255),
      'verbose':-1, 'seed':SEED, 'n_jobs':-1,
    }
    maes=[]
    for i,(tr,va) in enumerate(cv.split(train_df)):
        Xt,yt = train_df.loc[tr,FEATS], train_df.loc[tr,'y_log']
        Xv,yv = train_df.loc[va,FEATS], train_df.loc[va,'y_log']
        dtr=lgb.Dataset(Xt,yt); dva=lgb.Dataset(Xv,yv,reference=dtr)
        booster=lgb.train(params, dtr, num_boost_round=5000,
                          valid_sets=[dva], valid_names=['val'],
                          callbacks=[lgb.early_stopping(200,verbose=False),
                                     lgb.log_evaluation(0)])
        pred=np.expm1(booster.predict(Xv, num_iteration=booster.best_iteration))
        true=np.expm1(yv.values)
        maes.append(mean_absolute_error(true,pred))
        trial.report(float(np.mean(maes)), step=i)
        if trial.should_prune(): raise optuna.TrialPruned()
    return float(np.mean(maes))

study = optuna.create_study(direction='minimize',
    sampler=TPESampler(seed=SEED, multivariate=True, n_startup_trials=20),
    pruner =HyperbandPruner(min_resource=1, max_resource=8, reduction_factor=3))
study.optimize(objective, n_trials=100, show_progress_bar=True, gc_after_trial=True)
```

You can **warm-start** by feeding Optuna's stepwise `LightGBMTunerCV` result via `study.enqueue_trial(warm.best_params)` — it explores a small fixed sequence first and gives TPE a strong seed.

### 3.3 Evaluation utility (MAE / RMSE / R²)

```python
def evaluate(y_true, y_pred):
    return {'MAE':mean_absolute_error(y_true,y_pred),
            'RMSE':np.sqrt(mean_squared_error(y_true,y_pred)),
            'R2':r2_score(y_true,y_pred)}
```

**Interpret R² with care.** On strongly-trending series like this one (revenue grew roughly 4× across 2012–2022), *any* sensible model achieves a high R² because the trend dominates the denominator. A sophisticated pipeline at R²=0.92 can easily be worse than seasonal-naïve at R²=0.94. Always report **MASE = MAE / MAE(seasonal-naïve-365)** alongside R²; values < 1 mean the model beats the naïve benchmark.

---

## 4. Residual correction techniques

After the primary LightGBM converges, inspect residuals for autocorrelation via Ljung-Box; if `p < 0.05`, structure remains and a residual model adds value. Four complementary techniques, applied in order:

### 4.1 Rolling and EWMA bias correction

```python
def apply_rolling_bias(pred, residual_history, window=14):
    corrected = pred.copy(); hist = residual_history.copy()
    for dt, p in pred.items():
        bias = hist.tail(window).mean() if len(hist)>=3 else 0.0
        corrected.loc[dt] = p + bias
    return corrected

def ewma_bias(residuals, halflife=14):
    return residuals.shift(1).ewm(halflife=halflife, adjust=False).mean()
```

Use **w = 14 or 28** for daily retail; 7 overreacts to promo spikes. EWMA is preferred to fixed-window rolling because it **smoothly adapts to regime shifts** (e.g., 2020-2021 COVID) without the step artefact of a sharp window boundary.

### 4.2 Secondary residual model (Zhang-style hybrid)

If Ljung-Box on training residuals is significant, fit either a small LightGBM on **residual-specific features** (residual lags/rolls), or a SARIMA capturing the remaining weekly/annual seasonality:

```python
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Only proceed if autocorrelation is real
lb = acorr_ljungbox(df['resid'].dropna(), lags=[7,14,28], return_df=True)
if (lb['lb_pvalue'] < 0.05).any():
    res_tr = df.loc[df[DATE]<=TRAIN_END].set_index(DATE)['resid'].asfreq('D').fillna(0)
    sar = SARIMAX(res_tr, order=(1,0,1), seasonal_order=(1,0,1,7),
                  enforce_stationarity=False).fit(disp=False)
    fcst = sar.get_forecast(steps=len(test_df)).predicted_mean.values
    df.loc[df[DATE]>=TEST_START, 'final_pred'] = df.loc[df[DATE]>=TEST_START,'lgb_pred'].values + fcst
```

### 4.3 Event-specific multiplicative calibration (Tết, 11.11, 12.12, BFCM)

LightGBM systematically under-predicts spikes. Learn a **per-event multiplier** on walk-forward OOF predictions (not in-sample) with **empirical-Bayes shrinkage** to 1.0:

```python
event_cols = ['is_tet_week','is_dd_11_11','is_dd_12_12','is_black_friday','is_dd_9_9','is_dd_10_10']
corrections = {}
for ev in event_cols:
    sub = oof_df[oof_df[ev]==1]
    if len(sub) >= 3:
        ratio = sub[TARGET].sum()/sub['lgb_pred_oof'].sum()
        n = len(sub); k = 10.0
        corrections[ev] = (n*ratio + k*1.0)/(n+k)   # shrink to 1 with prior weight k
    else:
        corrections[ev] = 1.0

def apply_event_mult(pred, row):
    m = 1.0
    for ev,c in corrections.items():
        if row[ev]==1: m *= c
    return pred*m
```

Key design points: exclude **2020-02 → 2021-12 (COVID years)** when estimating Tết multipliers — those outlier years poison the prior; use **same-lunar-offset** matching (`revenue_tet_last_year`) for Tết lags rather than same-calendar-date, because Tết drifts by up to 20 solar days year-on-year.

### 4.4 Isotonic / quantile-mapping calibration

For monotone bias at the distribution edges, fit an `IsotonicRegression(out_of_bounds='clip')` on OOF pairs `(lgb_pred_oof, actual)` and transform test predictions. For heavier-tailed distributions, **quantile mapping** via empirical-CDF matching is usually superior:

```python
def quantile_map(pred_new, pred_oof, true_oof):
    qs = np.linspace(0.001, 0.999, 999)
    pred_q = np.quantile(pred_oof, qs); true_q = np.quantile(true_oof, qs)
    u = np.interp(pred_new, pred_q, qs)
    return np.interp(u, qs, true_q)
```

---

## 5. Ensemble and blending strategies

### 5.1 Multi-seed bagging (the M5 winner trick)

LightGBM is highly sensitive to random seeds on small daily series. Averaging 5–7 seeds removes 1–3% RMSE for free:

```python
def bagged_lgbm(Xt,yt,Xv, seeds=(1,7,13,42,101,404,999), **p):
    preds = np.zeros(len(Xv))
    for s in seeds:
        m = lgb.LGBMRegressor(**p, random_state=s, bagging_seed=s, feature_fraction_seed=s)
        m.fit(Xt, yt); preds += m.predict(Xv)/len(seeds)
    return preds
```

### 5.2 Multi-objective LightGBM blend

Train **three LGBM variants** — `regression_l1` (median, robust to spikes), `regression` on `log1p` (mean), and `tweedie(p=1.3)` (natural skewed distribution) — then median-blend. Each captures a different moment of the revenue distribution; their disagreement on Tết days is informative.

### 5.3 Stacking with constrained non-negative weights

Collect walk-forward OOF predictions from a diverse zoo (3×LGBM + XGB + CatBoost + Prophet + SARIMA + NHITS + seasonal-naïve), then fit non-negative, sum-to-one weights via SLSQP, with L2 shrinkage to prevent single-model mass concentration:

```python
from scipy.optimize import minimize
def blend_weights(oof, y, metric='rmse', lam=0.01):
    n_models = oof.shape[1]
    def loss(w):
        w = np.clip(w,0,None); w = w/(w.sum()+1e-12)
        yhat = oof@w
        base = (np.sqrt(np.mean((y-yhat)**2)) if metric=='rmse'
                else np.mean(np.abs(y-yhat)))
        return base + lam*np.sum(w**2)           # Caruana-style L2 shrinkage
    res = minimize(loss, np.ones(n_models)/n_models, method='SLSQP',
                   bounds=[(0,1)]*n_models,
                   constraints=[{'type':'eq','fun':lambda w: w.sum()-1}],
                   options={'maxiter':500,'ftol':1e-9})
    w = np.clip(res.x,0,None); return w/w.sum(), res.fun
```

The 2025 benchmark *Multi-layer Stack Ensembles for Time Series Forecasting* (arXiv 2511.15350) confirms that **greedy weighted stacking with shrinkage** dominates both simple averaging and meta-learner stacking on OOF. Expected weight distribution for this task: LGBM ensemble 0.55–0.70, NHITS/N-BEATS 0.10–0.20, Prophet 0.05–0.15, SARIMA 0.05, seasonal-naïve 0.05 as a safety anchor.

### 5.4 Why diverse bases matter

Prophet and SARIMA contribute **explicit trend/seasonality decomposition** that trees struggle to represent smoothly across 10+ years. NHITS/N-BEATS add non-linear long-horizon patterns. Seasonal-naïve prevents the blend from overreacting to noise by providing a stable anchor.

---

## 6. SHAP explainability integration

Use `shap.TreeExplainer(model, feature_perturbation='tree_path_dependent')` — this invokes the exact Linear-TreeSHAP algorithm compiled into LightGBM's C++ core, requires no background dataset, and runs in well under a second for 548 test rows.

```python
explainer = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent',
                               model_output='raw')
shap_vals = explainer(X_te)
shap_arr  = shap_vals.values
base_val  = float(shap_vals.base_values[0])

shap.plots.bar(shap_vals, max_display=25)          # global importance
shap.plots.beeswarm(shap_vals, max_display=25)     # distribution + direction
for f in ['lag_7','days_to_tet','max_discount','sessions_rmean_7','is_dd_11_11']:
    shap.dependence_plot(f, shap_arr, X_te)        # per-feature dependence
```

**Local explanations for business-critical days** (Tết 2024, 11.11 2023, Reunification Day 2023) are the single most impactful deliverable for a BI-facing project:

```python
def explain_date(date_str):
    loc = test_df.index[test_df[DATE]==pd.Timestamp(date_str)][0]
    expl = shap.Explanation(values=shap_arr[loc], base_values=base_val,
                            data=X_te.iloc[loc].values, feature_names=X_te.columns.tolist())
    shap.plots.waterfall(expl, max_display=15)
```

Ship a **top-N drivers side-table** alongside every forecast: for each test day, store the 5 features with largest `|SHAP|` and their direction. Aggregate by event-period (Tết, 11.11, payday window) for executive summaries. This transforms forecasts from black-box numbers into auditable decisions.

**SHAP as a leak-detector** is equally valuable. Three automated checks to run after every feature-engineering change:

```python
# (1) Any feature owning > 60% of mean |SHAP| is suspect
share = np.abs(shap_arr).mean(0) / np.abs(shap_arr).mean(0).sum()
print(pd.Series(share, index=X_te.columns).sort_values(ascending=False).head(10))

# (2) Train-vs-test importance drift — explosion on test often means leakage
shap_tr = explainer(shap.utils.sample(X_tr, 3000)).values
drift = (np.abs(shap_arr).mean(0)/(np.abs(shap_tr).mean(0)+1e-9))

# (3) Unshifted cross-table columns topping the chart = definite leak
```

---

## 7. Preventing data leakage — the seven non-negotiables

The single biggest source of false CV optimism in time-series is leakage. Enforce these rules programmatically, not by discipline alone:

1. **Rolling statistics must always be `shifted` before `.rolling()`**; assert at test time that `corr(feature_t, y_t)` does not exceed `corr(feature_t, y_{t-1})` by more than a few percent.
2. **Lag features must satisfy `lag ≥ forecast_horizon`**; a `make_lag_features(df, horizon=H, lags=...)` helper should filter automatically.
3. **Target encoding** of categoricals uses *past-only* folds via a time-aware OOF encoder with smoothing (`k=20`); never fit on the full dataset.
4. **Scalers fit on training portion of each fold only** — even though LightGBM itself doesn't need scaling, Prophet/linear meta-learners in the blend do.
5. **Cross-table joins use `pd.merge_asof(direction='backward', allow_exact_matches=False)`** for strictly-past semantics, and shift inventory/web_traffic columns by +1 day before the join.
6. **Run the shuffled-target sanity check** — train on `np.random.permutation(y_tr)`; if the shuffled-target model's R² exceeds ~0.05 you're leaking even index information.
7. **Treat 2023-01-01 → 2024-07-01 as radioactive**: it never enters feature-selection decisions, hyperparameter tuning, blend weights, or residual-correction calibration. Touch it only for final reporting.

Additional leak symptoms to monitor: CV RMSE dropping >40% after a single feature is added (real features rarely give this), feature importance dominated by one engineered feature (>50%), or the gap between CV and holdout scores exceeding 2×.

---

## 8. Executing the full pipeline end-to-end

```python
# 1. Build features
df_feat = (df_rev.pipe(add_basic_calendar).pipe(add_cyclical)
                  .pipe(add_tet_features).pipe(add_events)
                  .pipe(add_lag_roll, horizon=1)
                  .pipe(add_web_traffic, wt)
                  .pipe(add_promos, promos)
                  .pipe(add_inventory, inv))

# 2. Split (only now)
train_df = df_feat[df_feat[DATE] < TEST_START].dropna(subset=['lag_365'])
test_df  = df_feat[(df_feat[DATE]>=TEST_START)&(df_feat[DATE]<=TEST_END)]

# 3. Optuna tune on walk-forward CV (section 3)
# 4. Collect OOF predictions from base learners (section 5)
# 5. Residual diagnostics → SARIMA/LGBM residual model if Ljung-Box significant (4.2)
# 6. Event multipliers from OOF (4.3); isotonic/quantile calibration (4.4)
# 7. Blend weights via SLSQP on OOF (5.3)
# 8. Refit all base learners on FULL train; predict test; apply residual + event + calibration + blend
# 9. SHAP on final LGBM, ship top-N drivers table (section 6)
# 10. Report MAE/RMSE/R² + MASE vs seasonal-naïve-365 on test
```

**Final-model refit:** `n_estimators = median(best_iters_across_folds) × 1.1`. This accounts for the slightly longer convergence expected on the full training set.

---

## Conclusion — what this pipeline actually buys you

The most important lesson from M5 and subsequent retail-forecasting literature is that **feature engineering dominates model choice**, and within features the lunar-relative Tết encoding and event-distance features are the single largest levers for Vietnamese e-commerce. The LightGBM / Optuna / walk-forward spine is the industrial workhorse; its marginal wins over a well-tuned baseline are typically 5–10%, while correct Tết handling plus event calibration can be 15–25% by itself.

Residual correction and ensembling are **multiplicative, not additive** — their benefits compound only when the primary model is already well-specified and leak-free. Run the leakage audits first, Optuna second, residual correction third, ensembling last. Reversing that order wastes compute on a leaky model that will collapse in production.

Finally, **ship SHAP-based driver tables with every forecast**. In a fashion e-commerce context the business will always ask "why is 2024-02-09 predicted so high?" — answering that question programmatically (via top-5 SHAP features per day) converts the forecasting pipeline from a model into a decision-support product, and it doubles as a continuous leak-detector for the next feature batch.