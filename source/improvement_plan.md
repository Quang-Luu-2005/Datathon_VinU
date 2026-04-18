# 🎯 Plan Cải Tiến Modeling - Datathon VinU 2026

> **Baseline hiện tại:** Revenue R²=0.791, MAE=587K, RMSE=785K | COGS R²=0.745, MAE=544K, RMSE=738K
>
> **Mục tiêu:** Revenue R² ≥ 0.87, MAE ≤ 450K | Cải thiện 15-25% tổng thể

---

## 📊 Đánh giá hiện trạng

### Pipeline hiện tại đang làm đúng
- ✅ Expanding-window walk-forward CV với gap_days=28 (chống leakage đúng)
- ✅ Shift trước rolling (tránh target leak)
- ✅ XGBoost L1 (log1p) + Tweedie + Prophet blend
- ✅ Event multiplier với shrinkage Bayesian
- ✅ SARIMA residual correction có điều kiện (Ljung-Box)
- ✅ Multi-GPU parallel (Revenue + COGS)
- ✅ Tết features (days_to_tet, tet_proximity)

### Điểm yếu cần fix ngay
- ❌ **Chỉ dùng 1/15 files** — bỏ phí web_traffic, promotions, inventory, orders
- ❌ **Không tune hyperparameters** — dùng fixed params
- ❌ **CV chỉ 4 folds** — variance cao, blend weights không ổn định
- ❌ **Không có multi-seed bagging** — bỏ phí 1-3% RMSE miễn phí
- ❌ **Event multiplier bị COVID 2020-2021 làm nhiễu**
- ❌ **Recursive forecast 548 ngày compound error**
- ❌ **Feature importance thay vì SHAP** — đề bài bắt buộc SHAP
- ❌ **Thiếu features quan trọng:** payday, 8/3, interaction features
- ❌ **Chưa check MASE vs seasonal naive** — không biết model có thực sự tốt hơn baseline không

---

## 🗓️ Roadmap cải tiến (theo ROI)

| # | Cải tiến | Kỳ vọng giảm RMSE | Effort | Ưu tiên |
|---|---|---|---|---|
| 1 | Cross-table features (web, promo, inv) | 5-10% | Trung bình | 🔥🔥🔥 |
| 2 | Optuna hyperparameter tuning | 3-5% | Thấp | 🔥🔥🔥 |
| 3 | SHAP explainability | 0% (+8đ report) | Thấp | 🔥🔥🔥 BẮT BUỘC |
| 4 | Tăng CV folds + fix COVID multiplier | 2-4% | Rất thấp | 🔥🔥 |
| 5 | Multi-seed bagging | 1-3% | Rất thấp | 🔥🔥 |
| 6 | Thêm missing events (payday, 8/3, Valentine) | 1-3% | Rất thấp | 🔥🔥 |
| 7 | MASE benchmark vs seasonal naive | 0% (sanity check) | Rất thấp | 🔥 |
| 8 | Direct multi-step (thay recursive) | 3-8% | Cao | 🔥 |
| **Tổng** | | **15-30%** | | |

---

## 🔥 Priority 1: Cross-table features (lớn nhất)

**Vấn đề:** Đề bài có 15 files nhưng code chỉ đọc `sales.csv`. Các bảng khác chứa **leading indicators** cực mạnh.

### 1.1 Load thêm các bảng

```python
# Thêm vào cell setup
web = pd.read_csv(DATA_DIR / "web_traffic.csv", parse_dates=['date'])
promos = pd.read_csv(DATA_DIR / "promotions.csv", parse_dates=['start_date','end_date'])
inv = pd.read_csv(DATA_DIR / "inventory.csv", parse_dates=['snapshot_date'])
orders = pd.read_csv(DATA_DIR / "orders.csv", parse_dates=['order_date'])
order_items = pd.read_csv(DATA_DIR / "order_items.csv")
returns = pd.read_csv(DATA_DIR / "returns.csv", parse_dates=['return_date'])

print("Web:", web.shape, "Promos:", promos.shape, "Inv:", inv.shape)
```

### 1.2 Web traffic features (LUÔN shift để tránh leak)

```python
def add_web_features(det, web):
    # Aggregate daily (nếu web_traffic có multiple rows/day)
    web_daily = web.groupby('date').agg(
        sessions=('sessions','sum'),
        visitors=('unique_visitors','sum'),
        page_views=('page_views','sum'),
        bounce_rate=('bounce_rate','mean'),
        session_duration=('avg_session_duration_sec','mean')
    ).reset_index()

    det = det.reset_index().merge(web_daily, on='date', how='left').set_index('date')

    # Lag features - web traffic cùng ngày = leak nên luôn shift
    for col in ['sessions','visitors','page_views','bounce_rate']:
        for L in [1, 2, 3, 7, 14, 28]:
            det[f'{col}_lag_{L}'] = det[col].shift(L)
        # Rolling stats - shift(1) để cùng ngày không leak
        det[f'{col}_rmean_7'] = det[col].shift(1).rolling(7, min_periods=3).mean()
        det[f'{col}_rmean_28'] = det[col].shift(1).rolling(28, min_periods=3).mean()
        det[f'{col}_rstd_7'] = det[col].shift(1).rolling(7, min_periods=3).std()
        # YoY
        det[f'{col}_yoy'] = det[col].shift(365)
        det[f'{col}_yoy_ratio'] = det[col].shift(1) / (det[col].shift(365) + 1)

    # Conversion rate proxy (dùng lag để tránh leak)
    # Lưu ý: đây là feature engineering, không phải target leak
    det['sessions_momentum_7d'] = det['sessions'].shift(1).pct_change(7)

    # BỎ các cột same-day sau khi tạo xong features
    det = det.drop(columns=['sessions','visitors','page_views','bounce_rate','session_duration'])
    return det
```

### 1.3 Promotion features (same-day OK vì biết trước từ kế hoạch marketing)

```python
def add_promo_features(det, promos):
    # Expand mỗi promo thành từng ngày active
    daily_promos = []
    for _, r in promos.iterrows():
        for dt in pd.date_range(r.start_date, r.end_date):
            daily_promos.append({
                'date': dt,
                'promo_id': r.promo_id,
                'discount_value': r.discount_value,
                'promo_type': r.promo_type,
                'category': r.applicable_category,
                'channel': r.promo_channel,
                'stackable': r.stackable_flag,
                'min_order': r.min_order_value
            })
    dp = pd.DataFrame(daily_promos)

    agg = dp.groupby('date').agg(
        n_active_promos=('promo_id','nunique'),
        max_discount=('discount_value','max'),
        mean_discount=('discount_value','mean'),
        sum_discount=('discount_value','sum'),
        n_stackable=('stackable','sum'),
        n_channels=('channel','nunique'),
        n_categories=('category','nunique')
    ).reset_index()

    det = det.reset_index().merge(agg, on='date', how='left').set_index('date')
    fill_cols = ['n_active_promos','max_discount','mean_discount','sum_discount',
                 'n_stackable','n_channels','n_categories']
    det[fill_cols] = det[fill_cols].fillna(0)
    det['is_active_promo'] = (det['n_active_promos'] > 0).astype('int8')

    # Rolling promo intensity (ngày trước có nhiều promo → khách "để dành" cho hôm nay)
    for W in [7, 14, 30]:
        det[f'promo_days_last_{W}'] = det['is_active_promo'].shift(1).rolling(W, min_periods=1).sum()
        det[f'promo_depth_last_{W}'] = det['max_discount'].shift(1).rolling(W, min_periods=1).mean()

    # Interaction features (ĐIỂM SÁNG TẠO)
    det['promo_x_tet'] = det['is_active_promo'] * det['is_tet_week']
    det['promo_x_1111'] = det['is_active_promo'] * det['is_dd_11_11']
    det['promo_x_1212'] = det['is_active_promo'] * det['is_dd_12_12']
    det['promo_x_weekend'] = det['is_active_promo'] * det['is_weekend']
    det['depth_x_tetprox'] = det['max_discount'] * det['tet_proximity']

    return det
```

### 1.4 Inventory features (snapshot cuối tháng)

```python
def add_inventory_features(det, inv):
    # Aggregate inventory per day
    daily_inv = inv.groupby('snapshot_date').agg(
        total_stock=('stock_on_hand','sum'),
        total_sold=('units_sold','sum'),
        sku_count=('product_id','nunique'),
        n_stockout_skus=('stockout_flag','sum'),
        n_overstock_skus=('overstock_flag','sum'),
        mean_fill_rate=('fill_rate','mean'),
        mean_sell_through=('sell_through_rate','mean')
    ).reset_index().rename(columns={'snapshot_date':'date'})

    daily_inv['stockout_ratio'] = daily_inv['n_stockout_skus'] / daily_inv['sku_count']
    daily_inv['overstock_ratio'] = daily_inv['n_overstock_skus'] / daily_inv['sku_count']

    # Forward-fill vì snapshot chỉ có cuối tháng
    det = det.reset_index().merge(daily_inv, on='date', how='left').set_index('date')
    fill_cols = ['total_stock','total_sold','sku_count','stockout_ratio',
                 'overstock_ratio','mean_fill_rate','mean_sell_through']
    det[fill_cols] = det[fill_cols].ffill()

    # Shift 1 ngày để tránh leak (inventory snapshot cuối tháng biết vào đầu tháng sau)
    for col in fill_cols:
        det[f'{col}_lag1'] = det[col].shift(1)
        det[f'{col}_rmean_28'] = det[col].shift(1).rolling(28, min_periods=3).mean()

    # Interaction
    det['lowstock_x_promo'] = ((det['stockout_ratio_lag1'] > 0.3).astype(int) * det['is_active_promo'])

    # Drop same-day (giữ lại _lag1)
    det = det.drop(columns=fill_cols)
    return det
```

### 1.5 Orders features (aggregated signal)

```python
def add_order_features(det, orders):
    # Count/value orders per day - shift để tránh leak
    orders_daily = orders.groupby('order_date').agg(
        n_orders=('order_id','count'),
        n_customers=('customer_id','nunique'),
        n_cancelled=('order_status', lambda s: (s=='cancelled').sum()),
        n_delivered=('order_status', lambda s: (s=='delivered').sum())
    ).reset_index().rename(columns={'order_date':'date'})

    orders_daily['cancel_rate'] = orders_daily['n_cancelled'] / orders_daily['n_orders']

    det = det.reset_index().merge(orders_daily, on='date', how='left').set_index('date')
    cols = ['n_orders','n_customers','cancel_rate']
    det[cols] = det[cols].fillna(0)

    # Shift để không leak (orders cùng ngày = target leak trực tiếp)
    for col in cols:
        det[f'{col}_lag_1'] = det[col].shift(1)
        det[f'{col}_lag_7'] = det[col].shift(7)
        det[f'{col}_rmean_7'] = det[col].shift(1).rolling(7, min_periods=3).mean()
        det[f'{col}_rmean_28'] = det[col].shift(1).rolling(28, min_periods=3).mean()

    det = det.drop(columns=cols)  # bỏ same-day để tránh leak
    return det
```

### 1.6 Tích hợp vào pipeline

```python
# Trong cell build `det` DataFrame
det = pd.DataFrame({DATE_COL: pd.date_range(start_date, end_date, freq='D')})
det = add_event_features(add_tet_features(add_cyclical(add_basic_calendar(det))))
det = det.set_index(DATE_COL)

# Thêm các bảng mới
det = add_web_features(det, web)
det = add_promo_features(det, promos)
det = add_inventory_features(det, inv)
det = add_order_features(det, orders)

print(f"Total features sau khi merge: {det.shape[1]}")
```

**Lưu ý về leakage:**
- Web traffic, orders → **phải shift** vì cùng ngày = leak
- Promotions → same-day OK (biết trước từ marketing plan)
- Inventory → snapshot cuối tháng, shift(1) để biến thành "cuối tháng trước"

---

## 🔥 Priority 2: Optuna hyperparameter tuning

### 2.1 Thay hardcoded params bằng Optuna

```python
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

def optuna_objective(trial, train_df, feature_cols, target_col, splitter, gpu_id=0):
    params = {
        'n_estimators': 3000,
        'learning_rate': trial.suggest_float('lr', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 5, 12),
        'min_child_weight': trial.suggest_int('mcw', 5, 50),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('alpha', 1e-4, 10.0, log=True),
        'reg_lambda': trial.suggest_float('lambda', 1e-4, 10.0, log=True),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',
        'tree_method': 'hist',
        'device': f'cuda:{gpu_id}',
        'random_state': SEED,
        'early_stopping_rounds': 100,
    }

    maes = []
    for i, (tr, va) in enumerate(splitter.split(train_df)):
        X_tr = train_df.loc[tr, feature_cols]
        X_va = train_df.loc[va, feature_cols]
        y_tr = np.log1p(np.clip(train_df.loc[tr, target_col].values, 0, None))
        y_va_log = np.log1p(np.clip(train_df.loc[va, target_col].values, 0, None))
        y_va = train_df.loc[va, target_col].values

        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va_log)], verbose=False)
        pred = np.expm1(model.predict(X_va))
        maes.append(mean_absolute_error(y_va, pred))

        # Pruning ở fold level (vì CV)
        trial.report(np.mean(maes), step=i)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(maes))

# Chạy tuning cho Revenue (1 lần, dùng lại params cho COGS hoặc tune riêng)
study = optuna.create_study(
    direction='minimize',
    sampler=TPESampler(seed=SEED, multivariate=True, n_startup_trials=15),
    pruner=HyperbandPruner(min_resource=1, max_resource=CV_MAX_SPLITS, reduction_factor=3)
)
study.optimize(
    lambda t: optuna_objective(t, train_df, feature_cols, 'revenue', splitter, gpu_id=0),
    n_trials=50,  # tăng lên 100 nếu có thời gian
    show_progress_bar=True,
    gc_after_trial=True
)

best_params = study.best_params
print("Best MAE:", study.best_value)
print("Best params:", best_params)

# Lưu params để tái lập
import json
with open(OUTPUT_DIR / 'best_params_revenue.json', 'w') as f:
    json.dump(best_params, f, indent=2)
```

### 2.2 Áp dụng best_params vào final model

```python
# Thay XGB_CV_ESTIMATORS hardcoded
l1_final = xgb.XGBRegressor(
    n_estimators=2000,
    learning_rate=best_params['lr'],
    max_depth=best_params['max_depth'],
    min_child_weight=best_params['mcw'],
    subsample=best_params['subsample'],
    colsample_bytree=best_params['colsample'],
    reg_alpha=best_params['alpha'],
    reg_lambda=best_params['lambda'],
    gamma=best_params['gamma'],
    objective='reg:squarederror',
    tree_method='hist',
    device=f'cuda:{gpu_id}',
    random_state=SEED,
)
```

**Budget:** 50 trials × 4 folds × ~20s = ~60 phút cho Revenue. Trên Kaggle 2xT4 là OK.

---

## 🔥 Priority 3: SHAP explainability (BẮT BUỘC cho report)

Đề bài trang 12 ghi rõ: *"Khả năng giải thích (Explainability): bao gồm mục giải thích bằng SHAP values, feature importances, hoặc partial dependence plots"*.

### 3.1 Cài đặt và sinh SHAP values

```python
import shap
import matplotlib.pyplot as plt

# Sau khi có l1_final
explainer = shap.TreeExplainer(
    l1_final,
    feature_perturbation='tree_path_dependent'
)

# Sample để tăng tốc (full data 3800 rows OK nhưng sample 1000 đủ cho visualization)
X_sample = train_df[feature_cols].sample(n=min(1000, len(train_df)), random_state=SEED)
shap_values = explainer.shap_values(X_sample)

# 1. Global importance (beeswarm - thay thế feature_importances_)
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_sample, max_display=25, show=False)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'shap_summary_revenue.png', bbox_inches='tight', dpi=150)
plt.close()

# 2. Bar plot (đơn giản hơn cho slide)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_sample, plot_type='bar', max_display=20, show=False)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'shap_bar_revenue.png', bbox_inches='tight', dpi=150)
plt.close()
```

### 3.2 Local explanations cho các ngày business-critical

```python
def explain_specific_date(date_str, model, X_all, dates, explainer, save_path):
    """Giải thích prediction cho 1 ngày cụ thể"""
    mask = dates == pd.Timestamp(date_str)
    if not mask.any():
        print(f"Không tìm thấy {date_str}")
        return

    X_day = X_all[mask].iloc[[0]]
    shap_vals_day = explainer.shap_values(X_day)

    expl = shap.Explanation(
        values=shap_vals_day[0],
        base_values=explainer.expected_value,
        data=X_day.iloc[0].values,
        feature_names=X_day.columns.tolist()
    )
    plt.figure()
    shap.waterfall_plot(expl, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved {save_path}")

# Giải thích 3 ngày quan trọng
critical_dates = ['2022-01-31', '2022-11-11', '2022-12-12']  # Tết 2022, 11.11, 12.12
for dt in critical_dates:
    explain_specific_date(
        dt, l1_final, train_df[feature_cols], train_df[DATE_COL],
        explainer, OUTPUT_DIR / f'shap_waterfall_{dt}.png'
    )
```

### 3.3 Dependence plots cho top features

```python
# Lấy top 5 features theo |SHAP|
mean_abs_shap = np.abs(shap_values).mean(axis=0)
top5_features = X_sample.columns[np.argsort(mean_abs_shap)[-5:][::-1]].tolist()

for feat in top5_features:
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(feat, shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'shap_dep_{feat}.png', bbox_inches='tight', dpi=150)
    plt.close()
```

### 3.4 Bảng driver cho business (cho phần Prescriptive trong report)

```python
# Bảng top drivers per event period
def drivers_table(shap_values, X_sample, period_mask, top_n=10):
    period_shap = shap_values[period_mask]
    mean_shap = period_shap.mean(axis=0)
    abs_shap = np.abs(period_shap).mean(axis=0)
    df = pd.DataFrame({
        'feature': X_sample.columns,
        'mean_shap': mean_shap,
        'abs_shap': abs_shap,
        'direction': ['↑ tăng revenue' if m > 0 else '↓ giảm revenue' for m in mean_shap]
    }).sort_values('abs_shap', ascending=False).head(top_n)
    return df

# Ví dụ: driver cho Tết week
tet_mask = X_sample['is_tet_week'].values == 1
if tet_mask.any():
    tet_drivers = drivers_table(shap_values, X_sample, tet_mask)
    tet_drivers.to_csv(OUTPUT_DIR / 'drivers_tet_week.csv', index=False)
    print("Top drivers cho Tết week:")
    print(tet_drivers)
```

---

## 🔥 Priority 4: Tăng CV + Fix event multiplier

### 4.1 Tăng số folds

```python
# Cell cấu hình
CV_MAX_SPLITS = 8  # thay 4

splitter = ExpandingWindowWalkForward(
    initial_train_days=365*4,  # 4 năm initial (thay 5)
    val_days=120,               # 4 tháng val (thay 180)
    step_days=120,              # step 4 tháng
    gap_days=28,
    max_splits=8
)
```

Lý do: 8 folds × 120 ngày cover từ ~2017 đến 2022, có đủ mẫu Tết/11.11/Black Friday.

### 4.2 Exclude COVID khi tính event multiplier

```python
# Thay đoạn tính event_multipliers trong train_target_on_gpu
event_multipliers = {}

# LOẠI COVID years khỏi calibration
oof_for_mult = oof[
    ~oof[DATE_COL].dt.year.isin([2020, 2021])  # Outlier years
].copy()

for ev in EVENT_COLS:
    sub = oof_for_mult[(oof_for_mult[ev] == 1) & oof_for_mult['pred_blend'].notna()]
    if len(sub) >= 3 and sub['pred_blend'].sum() > 0:
        ratio = float(sub['actual'].sum() / sub['pred_blend'].sum())
        n = float(len(sub))
        k = 10.0  # shrinkage strength
        event_multipliers[ev] = (n*ratio + k*1.0) / (n + k)
    else:
        event_multipliers[ev] = 1.0

print("Event multipliers (excl. COVID):", event_multipliers)
```

### 4.3 Cap multipliers để tránh extreme values

```python
# Clip multipliers trong khoảng reasonable
for ev in event_multipliers:
    event_multipliers[ev] = float(np.clip(event_multipliers[ev], 0.5, 3.0))
```

---

## 🔥 Priority 5: Multi-seed bagging

### 5.1 Sửa function train để bag nhiều seeds

```python
def train_xgb_bagged(X_tr, y_tr, X_va, params, seeds=(42, 101, 202, 303, 404)):
    """Train XGB với nhiều seeds và average predictions"""
    preds_va = np.zeros(len(X_va))
    for s in seeds:
        p = params.copy()
        p['random_state'] = s
        p['subsample_seed'] = s
        model = xgb.XGBRegressor(**p)
        model.fit(X_tr, y_tr, verbose=False)
        preds_va += model.predict(X_va) / len(seeds)
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return preds_va

# Dùng trong fold loop
pred_l1_bagged = train_xgb_bagged(
    X_tr, np.log1p(np.clip(y_tr, 0, None)), X_va,
    params_l1, seeds=(42, 101, 202, 303, 404)
)
oof.loc[va_idx, 'pred_l1'] = np.clip(np.expm1(pred_l1_bagged), 0.0, None)
```

**Lưu ý:** Bagging nhân compute ×N seeds. Nên dùng 3-5 seeds cân bằng time/accuracy.

---

## 🔥 Priority 6: Thêm features còn thiếu

### 6.1 Payday features (Việt Nam lương 15 + cuối tháng)

```python
def add_payday_features(det):
    idx = det.index
    det['is_mid_month_pay'] = (idx.day == 15).astype('int8')
    det['is_eom_pay'] = idx.is_month_end.astype('int8')
    det['is_payday_window'] = (
        idx.day.isin([14, 15, 16, 17]) |
        idx.is_month_end |
        (idx + pd.Timedelta(days=1)).is_month_end |
        (idx + pd.Timedelta(days=2)).is_month_end
    ).astype('int8')
    return det
```

### 6.2 Women's Day, Valentine, Mid-Autumn

```python
def add_missing_events(det):
    idx = det.index
    # Women's Day VN 8/3 (rất lớn cho fashion)
    det['is_womens_day'] = ((idx.month == 3) & (idx.day == 8)).astype('int8')
    det['is_pre_womens_day'] = ((idx.month == 3) & idx.day.between(1, 7)).astype('int8')

    # Valentine 14/2
    det['is_valentine'] = ((idx.month == 2) & (idx.day == 14)).astype('int8')
    det['is_pre_valentine'] = ((idx.month == 2) & idx.day.between(7, 13)).astype('int8')

    # 20/10 Phụ nữ VN
    det['is_vn_womens_day'] = ((idx.month == 10) & (idx.day == 20)).astype('int8')

    # 20/11 Nhà giáo
    det['is_teachers_day'] = ((idx.month == 11) & (idx.day == 20)).astype('int8')

    # Back-to-school (tháng 8-9)
    det['is_back_to_school'] = ((idx.month.isin([8, 9])) &
                                 (idx.day.between(15, 30))).astype('int8')

    return det
```

### 6.3 Days-to-event (khoảng cách, không chỉ flag)

```python
def add_days_to_events(det):
    """Khoảng cách đến các ngày sale lớn - mạnh hơn flag"""
    idx = det.index
    years = idx.year.unique()

    for name, (m, d) in [('1111',(11,11)), ('1212',(12,12)), ('99',(9,9)),
                          ('1010',(10,10)), ('33',(3,3)), ('66',(6,6))]:
        this_year = pd.to_datetime([f'{y}-{m:02d}-{d:02d}' for y in years])
        next_year = pd.to_datetime([f'{y+1}-{m:02d}-{d:02d}' for y in years])

        def days_to(date):
            future = [ev for ev in this_year if ev >= date]
            if future:
                return (min(future) - date).days
            # Nếu đã qua năm nay, dùng năm sau
            future_next = [ev for ev in next_year if ev >= date]
            return (min(future_next) - date).days if future_next else 365

        det[f'days_to_{name}'] = [days_to(d) for d in idx]
        det[f'days_to_{name}'] = det[f'days_to_{name}'].clip(0, 120)

    return det
```

---

## 🔥 Priority 7: MASE benchmark (sanity check)

Đây là bước không cải thiện model nhưng **giúp biết model có thực sự tốt không**.

```python
def seasonal_naive_baseline(train_series, test_dates, lag=365):
    """Baseline: doanh thu cùng ngày năm trước"""
    preds = []
    hist = train_series.copy()
    for dt in test_dates:
        ref = dt - pd.Timedelta(days=lag)
        if ref in hist.index:
            preds.append(float(hist.loc[ref]))
        else:
            preds.append(float(hist.iloc[-7:].mean()))
    return np.array(preds)

def mase(y_true, y_pred, y_naive):
    mae_model = mean_absolute_error(y_true, y_pred)
    mae_naive = mean_absolute_error(y_true, y_naive)
    return mae_model / (mae_naive + 1e-9)

# Tính MASE cho OOF
oof_clean = oof.dropna(subset=['pred_blend_event'])
y_naive = seasonal_naive_baseline(
    target_series,
    pd.DatetimeIndex(oof_clean[DATE_COL].values)
)
mase_score = mase(oof_clean['actual'].values, oof_clean['pred_blend_event'].values, y_naive)
print(f"MASE (OOF): {mase_score:.3f}")

# Interpret:
# MASE < 0.8:  model tốt, hơn naive đáng kể
# 0.8 ≤ MASE < 0.95: model OK
# MASE ≥ 0.95: model gần như không tốt hơn naive - cần review
if mase_score >= 0.95:
    print("⚠️  CẢNH BÁO: Model không tốt hơn seasonal naive đáng kể!")
```

---

## 🔥 Priority 8: Direct multi-step (optional, cho competition edge)

Recursive forecast 548 ngày compound error. Thay bằng direct forecasting:

```python
# Train riêng model cho từng horizon
horizons = [1, 7, 14, 30, 60, 90, 180, 365]
direct_models = {}

for h in horizons:
    # Target = revenue của h ngày sau
    y_shifted = target_series.shift(-h)
    train_h = train_df[train_df[DATE_COL].shift(-h).notna()].copy()

    # Lag features phải có lag >= h để không leak
    # ... (train model với y_shifted)
    direct_models[h] = train_xgb(...)

# Dự báo: dùng model phù hợp horizon
# Ngày test cách train_end h ngày → dùng direct_models[nearest_h]
```

Effort cao, nên chỉ làm nếu có thời gian. Priority 1-6 ưu tiên trước.

---

## 📋 Checklist thực thi

### Tuần 1 (Quick wins)
- [ ] Tính MASE vs seasonal naive (30 phút) → **biết đang ở đâu**
- [ ] Fix COVID exclusion trong event multiplier (15 phút)
- [ ] Tăng CV folds lên 8 (5 phút + train time)
- [ ] Thêm payday + Women's Day features (30 phút)
- [ ] Thêm SHAP visualization (1h) → **BẮT BUỘC cho report**

### Tuần 2 (Big features)
- [ ] Load & merge `web_traffic.csv` (1h)
- [ ] Load & merge `promotions.csv` + interaction (1.5h)
- [ ] Load & merge `inventory.csv` (1h)
- [ ] Load & merge `orders.csv` aggregated (1h)
- [ ] Test lại CV sau khi thêm features

### Tuần 3 (Tuning & polish)
- [ ] Chạy Optuna tuning 50 trials (1h train + setup)
- [ ] Multi-seed bagging cho model cuối (2h train)
- [ ] Generate SHAP drivers table cho từng event
- [ ] Viết phần Explainability vào report
- [ ] Final submit Kaggle

---

## 🛡️ Leakage audit - kiểm tra trước mỗi submission

Chạy các test này trước khi submit:

```python
# 1. Shuffled target test
y_shuffled = np.random.permutation(y_train)
model_shuffle = xgb.XGBRegressor(**params).fit(X_train, y_shuffled)
pred_shuffle = model_shuffle.predict(X_val)
r2_shuffle = r2_score(y_val, pred_shuffle)
assert r2_shuffle < 0.05, f"⚠️ LEAK! R² với shuffled target = {r2_shuffle}"

# 2. Feature dominance check
fi = pd.Series(l1_final.feature_importances_, index=feature_cols)
top_feat_share = fi.max() / fi.sum()
if top_feat_share > 0.5:
    print(f"⚠️ Cảnh báo: feature '{fi.idxmax()}' chiếm {top_feat_share:.1%} importance")

# 3. Train vs Test distribution check
for col in ['lag_1', 'rmean_7', 'sessions_lag_1']:
    if col in train_df.columns:
        train_mean = train_df[col].mean()
        # Check nếu có giá trị impossible (vd negative sessions)
        assert train_df[col].notna().mean() > 0.8, f"Too many NaN in {col}"

# 4. Gap đủ lớn
assert splitter.gap_days >= max(ROLL_WINDOWS), "Gap phải >= rolling window max"
```

---

## 🎯 Kỳ vọng sau khi hoàn thành

| Metric | Baseline | Target | Stretch |
|---|---|---|---|
| Revenue MAE | 587K | ≤ 450K | ≤ 400K |
| Revenue RMSE | 785K | ≤ 600K | ≤ 550K |
| Revenue R² | 0.791 | ≥ 0.87 | ≥ 0.90 |
| MASE | ? | < 0.80 | < 0.70 |
| Kaggle rank | ? | Top 30% | Top 15% |

---

## 📚 Tài liệu tham khảo cho report

Khi viết phần Explainability trong report NeurIPS, dùng các sections này:

1. **Feature Importance (Global):** `shap_summary_revenue.png` + `shap_bar_revenue.png`
2. **Local Explanation:** 3 waterfall plots cho Tết/11.11/12.12 2022
3. **Dependence Analysis:** Top 5 features dependence plots
4. **Business Drivers:** Bảng `drivers_tet_week.csv` — diễn giải bằng ngôn ngữ kinh doanh
5. **Ngôn ngữ kinh doanh:** "Mô hình học được rằng promotions × Tết proximity đẩy doanh thu lên X%, trong khi web traffic lag 7 ngày là leading indicator mạnh nhất với correlation 0.7+"

---

**Lời khuyên cuối:** Làm Priority 1-5 trước khi submit lần tiếp theo. Đừng tune Optuna trước khi thêm cross-table features — tuning trên feature set nghèo sẽ overfit và phải tune lại sau.

Good luck! 🚀
