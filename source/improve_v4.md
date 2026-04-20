# PLAN_v4.md — Kế hoạch nâng cấp pipeline XGBoost v3 → v4

**Dự án**: Datathon VinUni 2026 — Dự báo Revenue & COGS 548 ngày  
**Mục tiêu tổng thể**: Revenue R² từ **0.831 → 0.87–0.88**, MAE từ **506,818 → 420,440k**  
**Ngân sách compute**: 12 giờ Kaggle save-mode trên 2× NVIDIA T4 16GB  
**Phiên bản hiện tại**: v3 (XGBoost multi-head ensemble với 5 heads, Caruana blender, STL+ARIMA branch)

---

## 1. Tóm tắt điều hành (Executive Summary)

Pipeline v3 đã đạt Revenue R² = 0.831 với CI bootstrap [0.778, 0.867] — còn **headroom ~4–5 điểm R²** trước khi chạm trần lý thuyết khoảng 0.90 (giới hạn bởi nhiễu trong doanh thu e-commerce thời trang). Phân tích OOF cho thấy **ba nguồn leak hiệu suất chính**: (1) Huber head bị bug làm mất ~0.5% trọng số ensemble, (2) CV chỉ phủ tối đa horizon 180 ngày nên hai bucket dài nhất phải copy-paste trọng số, (3) mọi head đều thuộc cùng họ XGBoost nên đa dạng mô hình (model diversity) bằng 0 — vi phạm nguyên tắc stacking tiêu biểu của các winner Kaggle 2024–2026.

v4 sẽ triển khai theo **4 tier ưu tiên chi phí/lợi ích**:

| Tier | Mô tả | Thời gian | ΔR² dự kiến | Rủi ro |
|---|---|---|---|---|
| **Tier 0** | Hotfix lỗi cấu hình v3 | 1.0h | +0.005–0.020 | Thấp |
| **Tier 1** | Diversity (LightGBM + CatBoost) + feature engineering (Prophet, Fourier, change-point, interactions) | 4.0h | +0.020–0.040 | Trung bình |
| **Tier 2** | Direct multi-horizon (mlforecast), N-HiTS, Ridge meta-learner, conformal | 5.0h | +0.020–0.030 | Trung bình |
| **Tier 3** | Chronos-Bolt fine-tune, TFT, STL features | 2.0h (buffer) | +0.005–0.020 | Cao |

Các mục tiêu số cụ thể tại Phần 12 (Success Criteria). Toàn bộ task được thiết kế **đảo ngược được** (rollback-able) qua git tag sau mỗi tier, và mỗi bước có **A/B test OOF MAE tự động** trước khi merge vào pipeline chính.

---

## 2. Tier 0 — Hotfix (1.0 giờ, +0.005–0.020 R²)

### Task 1 — Sửa lỗi Huber head (0.25 giờ, +0.003–0.008 R²)

**Mục tiêu**: Loại bỏ head `pred_huber` MAE 49M / R² -863 / weight 0.002 đang làm nhiễu blender.

**File cần sửa**: `src/models/xgb_heads.py` (function `train_huber_head`) và `src/config/heads.yaml`.

**Option A — Giữ Huber, sửa transform (khuyến nghị)**:

```python
# src/models/xgb_heads.py
import numpy as np, xgboost as xgb

def train_huber_head(X_tr, y_tr, X_va, y_va, params, seed=42):
    # Log1p transform giống L1 head để Huber slope hợp lý với scale y
    y_tr_log = np.log1p(y_tr)
    y_va_log = np.log1p(y_va)

    # Huber slope nên ~ MAD(y_log), không phải 306k (giá trị raw)
    mad = np.median(np.abs(y_tr_log - np.median(y_tr_log)))
    huber_slope = max(0.05, 1.345 * mad)  # ≈ 0.1–0.3 trên log scale

    dtr = xgb.QuantileDMatrix(X_tr, label=y_tr_log)
    dva = xgb.QuantileDMatrix(X_va, label=y_va_log, ref=dtr)

    p = {**params,
         "objective": "reg:pseudohubererror",
         "huber_slope": huber_slope,
         "tree_method": "hist", "device": "cuda",
         "eval_metric": "mae", "seed": seed}
    booster = xgb.train(p, dtr, num_boost_round=3000,
                        evals=[(dva, "val")],
                        early_stopping_rounds=200, verbose_eval=False)

    # Duan smearing back-transform (giống L1)
    raw_va = booster.predict(dva)
    resid  = y_va_log - raw_va
    duan   = np.mean(np.exp(resid))
    def predict_fn(X):
        d = xgb.QuantileDMatrix(X, ref=dtr)
        return np.expm1(booster.predict(d)) * duan
    return booster, predict_fn
```

**Option B — Thay Huber bằng Quantile τ=0.5 (phương án phòng thủ)**:

```python
p = {**params,
     "objective": "reg:quantileerror",
     "quantile_alpha": 0.5,
     "tree_method": "hist", "device": "cuda",
     "eval_metric": "mae", "seed": seed}
# Train trên log1p(y), back-transform expm1. KHÔNG dùng Duan (median, không mean).
```

XGBoost 2.0+ hỗ trợ `reg:quantileerror` native trên GPU. Quantile τ=0.5 là median regression — robust hơn Huber với heavy tail của sales e-commerce (Rohlik v2 top solutions confirmed).

**Acceptance criteria**:
- [ ] Huber head OOF MAE xuống dưới 700k (so với 49M hiện tại)
- [ ] Huber head OOF R² ≥ 0.75 (so với -863)
- [ ] Caruana blender assign weight ≥ 0.05 cho Huber (hoặc Quantile) ở ít nhất 2 bucket
- [ ] Revenue final R² không giảm; kỳ vọng +0.005

**Delta kỳ vọng**: +0.005–0.008 R² từ việc thêm một head lành mạnh vào blender.

### Task 2 — Sửa CV horizon coverage (0.25 giờ, +0.005–0.012 R²)

**Mục tiêu**: CV hiện tại (`val_days=180`, `max_splits=5`, `step=120`) không bao giờ đánh giá bucket h181–365 và h366+, khiến blend weight copy-paste từ global — nguồn chính của lạc quan hóa.

**File cần sửa**: `src/cv/expanding_walk_forward.py` và `src/config/cv.yaml`.

```yaml
# src/config/cv.yaml — THAY THẾ HOÀN TOÀN
cv_mode: "hybrid"               # Mới: chạy cả 2 loại CV
short_cv:
  max_splits: 3                 # giảm từ 5
  initial_train: 730
  val_days: 365                 # tăng từ 180 → phủ h181–365
  step: 180
  gap: 28
realistic_probe:
  enabled: true
  max_splits: 1                 # 1 fold duy nhất để estimate test thật
  initial_train: 2555           # 7 năm đầu (2012-07-04 → 2019-07-04)
  val_days: 548                 # đúng horizon test
  step: 0
  gap: 28
```

```python
# src/cv/expanding_walk_forward.py — thêm realistic_probe
class ExpandingWalkForward:
    def split(self, df):
        # Short CV: 3 folds × 365 days cho tuning Optuna + Caruana
        for i in range(self.max_splits):
            ...
        # Realistic probe: 1 fold 548 days — CHỈ để báo cáo, KHÔNG để tuning
        if self.realistic_probe:
            tr_end = df.index[-self.val_days - self.gap]
            yield (df.index[:tr_end], df.index[tr_end+self.gap:tr_end+self.gap+self.val_days],
                   {"fold_type": "realistic_probe"})
```

**Acceptance criteria**:
- [ ] CV sinh ra OOF cho cả 5 bucket horizon (h01_030, h031_090, h091_180, **h181_365**, **h366_plus**)
- [ ] Bucket h181_365 và h366_plus có ≥ 365 samples OOF mỗi bucket
- [ ] `realistic_probe` fold báo cáo riêng, không trộn vào training OOF
- [ ] Kỳ vọng: R² realistic_probe thấp hơn R² short CV khoảng 0.03–0.05 điểm (đây là con số "thật" để dự đoán private LB)

**Delta kỳ vọng**: Không tăng trực tiếp R² OOF, nhưng **tăng R² thực tế trên Kaggle test** +0.005–0.012 nhờ trọng số blend cho long-horizon được học đúng thay vì copy-paste.

### Task 3 — Gộp bucket horizon dài (0.15 giờ, +0.002–0.005 R²)

**Mục tiêu**: Dù Task 2 tạo được OOF cho h181_365 và h366_plus, số samples vẫn ít (≤ 548 mỗi bucket). Gộp thành **một bucket h181_plus** để Caruana có đủ data học trọng số.

**File cần sửa**: `src/blend/bucket_config.py`.

```python
# Trước (v3)
HORIZON_BUCKETS = {
    "h01_030":   (1, 30),
    "h031_090":  (31, 90),
    "h091_180":  (91, 180),
    "h181_365":  (181, 365),    # fallback, copy-paste
    "h366_plus": (366, 9999),   # fallback, copy-paste
}

# Sau (v4) — 4 bucket thực thi, 1 bucket dài duy nhất
HORIZON_BUCKETS = {
    "h01_030":   (1, 30),
    "h031_090":  (31, 90),
    "h091_180":  (91, 180),
    "h181_plus": (181, 9999),   # gộp để Caruana có ≥ 368 samples
}
```

**Acceptance criteria**:
- [ ] Không còn bucket nào fallback "copy-paste global"
- [ ] Caruana weights ở `h181_plus` khác với global blend ít nhất trên 1 head

**Delta kỳ vọng**: +0.002–0.005 R² (nhỏ vì tỷ trọng long-horizon trong 548-day test là 368/548 = 67%, nhưng nguồn lợi chính đã nằm ở Task 2).

### Task 4 — Verify/fix lunar TET features (0.35 giờ, +0.003–0.010 R²)

**Mục tiêu**: Xác minh `days_to_tet` / `days_since_tet` hiện tại dùng đúng âm lịch Việt Nam (UTC+7), không phải ngày cố định Gregorian hay âm lịch Trung Quốc (UTC+8).

**File cần sửa**: `src/features/holidays.py`.

**Tin tức kiểm chứng**: `lunardate` 0.2.2 (Dec 2023, mới nhất) khớp 100% ngày TET Việt Nam cho 2012–2026. Không có chênh lệch 1 ngày trong cửa sổ training/test. Chỉ các năm 2007, 2030, 2053 bị lệch — không ảnh hưởng.

```python
# src/features/holidays.py
# pip install lunardate==0.2.2
from lunardate import LunarDate
from datetime import date, timedelta
import numpy as np, pandas as pd

def tet_gregorian(y: int) -> date:
    """Ngày mùng 1 Tết Âm lịch theo giờ Việt Nam."""
    return LunarDate(y, 1, 1).toSolarDate()

# TET 2012..2027 (kiểm chứng với Wikipedia Tết article)
TET = {y: tet_gregorian(y) for y in range(2011, 2028)}
assert TET[2023] == date(2023, 1, 22), "TET 2023 phải là 22-01"
assert TET[2024] == date(2024, 2, 10), "TET 2024 phải là 10-02"

def build_tet_features(ds: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for d in ds:
        dd = d.date()
        next_tet = min((t for t in TET.values() if t >= dd), default=None)
        prev_tet = max((t for t in TET.values() if t <  dd), default=None)
        dt = (next_tet - dd).days if next_tet else 365
        ds_ = (dd - prev_tet).days if prev_tet else 365
        signed = dt if dt <= ds_ else -ds_
        rows.append({
            "days_to_tet":      dt,
            "days_since_tet":   ds_,
            "signed_days_tet":  signed,                       # MỚI: feature dấu có ích
            "is_tet_day":       int(dd in TET.values()),
            "is_tet_week":      int(min(dt, ds_) <= 7),
            "is_pre_tet_15d":   int(0 < dt <= 15),            # MỚI: peak shopping
            "is_post_tet_7d":   int(0 < ds_ <= 7),            # MỚI: dead period
            "tet_proximity":    float(np.exp(-min(dt, ds_)/14)),  # MỚI: decay
        })
    return pd.DataFrame(rows, index=ds)

# Thêm holidays khác
def build_vn_holiday_features(ds: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for d in ds:
        y, m, dd = d.year, d.month, d.day
        hung_kings = LunarDate(y, 3, 10).toSolarDate()
        mid_autumn = LunarDate(y, 8, 15).toSolarDate()
        ghost_start= LunarDate(y, 7, 1).toSolarDate()
        ghost_end  = LunarDate(y, 8, 1).toSolarDate() - timedelta(days=1)
        rows.append({
            "is_reunification":  int((m,dd)==(4,30)),
            "is_labour":         int((m,dd)==(5,1)),
            "is_national":       int((m,dd)==(9,2)),
            "is_hung_kings":     int(d.date()==hung_kings),
            "is_mid_autumn":     int(d.date()==mid_autumn),
            "days_to_mid_autumn":(mid_autumn - d.date()).days,
            "is_ghost_month":    int(ghost_start <= d.date() <= ghost_end),
            "is_apr30_bridge":   int((m,dd) in {(4,29),(4,30),(5,1),(5,2)}),
        })
    return pd.DataFrame(rows, index=ds)
```

**Acceptance criteria**:
- [ ] `assert TET[2023] == 2023-01-22` và `TET[2024] == 2024-02-10` đều pass
- [ ] So với v3, feature `days_to_tet` tại ngày 2024-01-15 phải ra 26 (= 10 Feb − 15 Jan) chứ không phải ~340
- [ ] Thêm 6 feature mới: `signed_days_tet`, `is_pre_tet_15d`, `is_post_tet_7d`, `tet_proximity`, `is_ghost_month`, `is_mid_autumn`, `days_to_mid_autumn`
- [ ] SHAP importance `signed_days_tet` ≥ `days_to_tet` v3 (25k) sau re-train

**Delta kỳ vọng**: +0.003–0.010 R² — đặc biệt trên bucket h031_090 và h091_180 phủ cả TET 2023 + TET 2024.

---

## 3. Tier 1 — Diversity & Feature Engineering (4.0 giờ, +0.020–0.040 R²)

### Task 5 — Thêm LightGBM Tweedie head (1.0 giờ, +0.008–0.015 R²)

**Mục tiêu**: Thêm library đa dạng đầu tiên. M5 1st place (YJ_STU) dùng LightGBM Tweedie ensemble 220 mô hình, 3rd–4th place cũng LGBM Tweedie — pattern nổi tiếng nhất trong retail forecasting.

**File mới**: `src/models/lgb_tweedie_head.py`.

**Lưu ý versioning**: LightGBM 4.6.0 (Feb 2025) là stable tháng 4/2026. Pip wheel trên Kaggle chỉ build với `device_type='gpu'` (OpenCL). `device_type='cuda'` cần source build — KHÔNG dùng trên Kaggle.

**Gotchas quan trọng**:
- `use_quantized_grad=True` + Tweedie → crash (issue #6134). **Phải để `False`**.
- `max_bin=63` thay vì 255 để có tăng tốc GPU thật sự.
- Tweedie dùng log-link nội bộ → **KHÔNG log1p y** trước khi train, prediction trả về đã ở μ-space.
- Target phải ≥ 0 (revenue thỏa mãn).

```python
# src/models/lgb_tweedie_head.py
import os, numpy as np, lightgbm as lgb

LGB_GPU_PARAMS = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.15,    # sweet spot retail (M5: 1.1–1.3)
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 255,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 0.1,
    "max_bin": 63,                     # BẮT BUỘC cho GPU speedup
    "device_type": "gpu",              # OpenCL, pip wheel hỗ trợ
    "gpu_platform_id": 0,
    "gpu_device_id": 0,                # pin GPU 0; parallel fold dùng GPU 1
    "gpu_use_dp": False,               # fp32, T4 fp64 yếu
    "use_quantized_grad": False,       # BẮT BUỘC False với tweedie
    "num_threads": 4,
    "verbose": -1,
}

def train_lgb_tweedie(X_tr, y_tr, X_va, y_va, cat_cols, seed=42, gpu_id=0):
    params = {**LGB_GPU_PARAMS, "seed": seed, "gpu_device_id": gpu_id}
    dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols, free_raw_data=False)
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr, categorical_feature=cat_cols)
    model = lgb.train(params, dtr, num_boost_round=5000,
                      valid_sets=[dva], valid_names=["val"],
                      callbacks=[lgb.early_stopping(200), lgb.log_evaluation(500)])
    return model  # model.predict trả về μ (không cần exp)
```

**Fold parallelism trên 2× T4** (tận dụng cả 2 GPU):

```python
import multiprocessing as mp
def _fold_worker(fold_id, gpu_id, X_tr, y_tr, X_va, y_va, cat_cols, out_q):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # CUDA
    # LightGBM OpenCL: set gpu_device_id trực tiếp
    m = train_lgb_tweedie(X_tr, y_tr, X_va, y_va, cat_cols, gpu_id=gpu_id)
    out_q.put((fold_id, m.predict(X_va)))

# Chạy 2 fold đồng thời (1 fold/GPU)
```

**Integration vào pipeline**: Thêm `pred_lgb_tweedie` vào danh sách heads ở `src/blend/caruana.py`, features list từ `src/features/combined_feature_set.py` (share với XGBoost heads để fair comparison).

**Acceptance criteria**:
- [ ] LGB Tweedie head OOF Revenue R² ≥ 0.80 trên short CV
- [ ] OOF correlation với `pred_l2` (XGB raw) ≤ 0.96 — tức là đem lại diversity thực sự
- [ ] Caruana weight ≥ 0.15 ở ít nhất 2/4 bucket
- [ ] Pipeline không OOM trên T4 (monitor `nvidia-smi` mỗi 30s)

**Delta kỳ vọng**: +0.008–0.015 R² — confirmed pattern từ Rohlik v2 top notebooks và M5.

### Task 6 — Thêm CatBoost head (Poisson/log1p-RMSE) (1.0 giờ, +0.005–0.010 R²)

**Mục tiêu**: Thêm library đa dạng thứ hai. **Quan trọng**: CatBoost Tweedie trên GPU vẫn có bug (issue #2812 open tính đến v1.2.10, Feb 2026). Dùng **Poisson** (GPU-OK) hoặc **RMSE trên log1p(y)** làm phương án phòng thủ.

**File mới**: `src/models/cat_head.py`.

```python
# src/models/cat_head.py
import os, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"   # set TRƯỚC khi import catboost
from catboost import CatBoostRegressor, Pool

CAT_GPU_PARAMS = {
    "iterations": 10000,
    "learning_rate": 0.05,
    "depth": 8,                        # GPU symmetric tree tối đa ~10
    "l2_leaf_reg": 3.0,
    "loss_function": "RMSE",           # huấn luyện trên log1p(y)
    "eval_metric": "RMSE",
    "task_type": "GPU",
    "devices": "0:1",                  # dùng CẢ 2 T4 (data parallel)
    "gpu_ram_part": 0.85,
    "border_count": 128,               # = max_bin; 128 optimal cho T4
    "grow_policy": "SymmetricTree",    # nhanh nhất GPU
    "bootstrap_type": "Bayesian",      # GPU-safe (MVS chỉ CPU)
    "bagging_temperature": 1.0,
    "od_type": "Iter",
    "od_wait": 300,
    "use_best_model": True,
    "verbose": 500,
    "allow_writing_files": False,
    "random_seed": 42,
}

def train_catboost(X_tr, y_tr, X_va, y_va, cat_cols):
    # log1p target trick
    ytr = np.log1p(y_tr); yva = np.log1p(y_va)
    tr_pool = Pool(X_tr, label=ytr, cat_features=cat_cols)
    va_pool = Pool(X_va, label=yva, cat_features=cat_cols)
    m = CatBoostRegressor(**CAT_GPU_PARAMS)
    m.fit(tr_pool, eval_set=va_pool)
    def predict_fn(X):
        return np.expm1(m.predict(X))  # back to μ
    return m, predict_fn
```

**Biến thể Poisson** (chạy song song, làm head thứ 3):

```python
CAT_POISSON = {**CAT_GPU_PARAMS,
               "loss_function": "Poisson",
               "boost_from_average": False}
# KHÔNG log1p y với Poisson — CatBoost output đã ở μ-space
```

**Acceptance criteria**:
- [ ] CatBoost head OOF Revenue R² ≥ 0.78
- [ ] OOF correlation với XGB và LGB cả hai ≤ 0.95 (diversity tốt)
- [ ] Tận dụng được cả 2 GPU (verify bằng `nvidia-smi` — cả 2 card có util > 50%)
- [ ] Caruana weight ≥ 0.08 ở ít nhất 1 bucket
- [ ] Fallback CPU tự động nếu GPU OOM

**Delta kỳ vọng**: +0.005–0.010 R² bổ sung trên Task 5.

### Task 7 — Prophet components làm features (0.5 giờ, +0.003–0.008 R²)

**Mục tiêu**: Thay thế `pred_prophet` (seasonal-naive parallel forecaster, weight 0.07–0.17) bằng **trend + yearly + weekly components** đưa thẳng vào feature matrix. Pattern này được xác nhận trong Rohlik 2024 top solutions và gold kernels của Store Sales Kaggle.

**File mới**: `src/features/prophet_components.py`.

**Version**: `prophet==1.3.0` (Jan 2026, mới nhất).

```python
# pip install prophet==1.3.0
from prophet import Prophet
import pandas as pd

def fit_prophet_components(df_train, df_full, holidays_df=None):
    """
    df_train: DataFrame(ds, y) — CHỈ training window để tránh leakage
    df_full : DataFrame(ds)    — toàn bộ timeline cần components (train + test)
    """
    m = Prophet(
        growth="linear",
        yearly_seasonality=10,
        weekly_seasonality=3,
        daily_seasonality=False,
        holidays=holidays_df,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
    )
    m.add_country_holidays(country_name="VN")  # prophet 1.1+ hỗ trợ
    m.fit(df_train[["ds","y"]])
    fcst = m.predict(df_full[["ds"]])
    return fcst[["ds","trend","yearly","weekly","holidays","yhat"]].rename(
        columns={"trend":"pf_trend", "yearly":"pf_yearly",
                 "weekly":"pf_weekly", "holidays":"pf_holidays", "yhat":"pf_yhat"})

# Tại mỗi CV fold, refit Prophet trên training window của fold đó
# Tại final submission, refit Prophet trên toàn bộ train 2012–2022
```

**Tích hợp**: Ngưng head `pred_prophet` hiện tại, xóa member seasonal-naive khỏi blender. Thêm 5 cột `pf_*` vào feature matrix dùng chung cho XGB/LGB/CAT.

**Leakage safety**: Prophet fit CHỈ trên training window, sau đó extrapolate deterministic (trend linear, yearly/weekly là hàm deterministic của date) → leak-free.

**Acceptance criteria**:
- [ ] Head `pred_prophet` bị loại khỏi blender
- [ ] 5 cột `pf_trend`, `pf_yearly`, `pf_weekly`, `pf_holidays`, `pf_yhat` xuất hiện trong SHAP top-50
- [ ] Tổng `|SHAP|` của 5 cột pf_* ≥ tổng `|SHAP|` của nhóm sin/cos cyclical hiện tại
- [ ] Overall Revenue R² OOF không giảm (kỳ vọng +0.003)

**Delta kỳ vọng**: +0.003–0.008 R².

### Task 8 — Fourier yearly K=8 + weekly K=3 (0.25 giờ, +0.002–0.005 R²)

**Mục tiêu**: Thay sin/cos đơn (K=1) hiện tại bằng Fourier order cao hơn. K=8 yearly / K=3 weekly là default Prophet và M5 winners.

**File cần sửa**: `src/features/cyclical.py`.

```python
# src/features/cyclical.py
import numpy as np, pandas as pd

def fourier_terms(dates, period, K, prefix):
    t = (dates - pd.Timestamp("2012-01-01")).days.values.astype(float)
    cols = {}
    for k in range(1, K+1):
        cols[f"{prefix}_sin_{k}"] = np.sin(2*np.pi*k*t/period)
        cols[f"{prefix}_cos_{k}"] = np.cos(2*np.pi*k*t/period)
    return pd.DataFrame(cols, index=dates)

def build_fourier(ds: pd.DatetimeIndex):
    yearly  = fourier_terms(ds, 365.25, K=8, prefix="fy")   # 16 cols
    weekly  = fourier_terms(ds, 7,      K=3, prefix="fw")   # 6 cols
    # Thêm lunar cycle (29.53 ngày synodic) — gắn với payday VN + full moon shopping
    lunar   = fourier_terms(ds, 29.53,  K=2, prefix="fl")   # 4 cols
    return pd.concat([yearly, weekly, lunar], axis=1)
```

Loại bỏ các cột cũ `dow_sin`, `dow_cos`, `month_sin`, `month_cos` để tránh redundancy — giữ `day_of_month` và `week_of_year` làm integer.

**Acceptance criteria**:
- [ ] Thêm 26 cột Fourier, xóa 4 cột sin/cos cũ
- [ ] SHAP top-30 có ≥ 3 cột `fy_*` hoặc `fw_*`
- [ ] R² OOF không giảm

**Delta kỳ vọng**: +0.002–0.005 R².

### Task 9 — Change-point features với `ruptures` (0.5 giờ, +0.002–0.007 R²)

**Mục tiêu**: Tự động detect các mean-shift lịch sử (VD: COVID 2020, hồi phục 2021, boom 2022) và encode khoảng cách tới change-point gần nhất làm feature.

**Version**: `ruptures==1.1.10` (Sep 2025).

**File mới**: `src/features/changepoints.py`.

```python
# pip install ruptures==1.1.10
import ruptures as rpt, numpy as np, pandas as pd

def changepoint_features(y_train: pd.Series, pen=None, min_size=14):
    """y_train: daily revenue (log-scale khuyến nghị). KHÔNG leak vì fit chỉ trên train."""
    y = np.log1p(y_train.values)  # de-scale trước PELT
    sigma2 = np.var(np.diff(y))
    if pen is None:
        pen = np.log(len(y)) * sigma2 * 2   # BIC heuristic
    algo = rpt.Pelt(model="l2", min_size=min_size, jump=1).fit(y)
    bkps = algo.predict(pen=pen)           # list indices (1-based end)
    cp_dates = [y_train.index[i-1] for i in bkps[:-1]]
    return cp_dates

def build_cp_features(ds: pd.DatetimeIndex, cp_dates):
    out = pd.DataFrame(index=ds)
    cp_set = set(cp_dates)
    # Days since last CP, days to next CP (relative to LAST training CP for test rows)
    last_cp = min(cp_dates) if cp_dates else ds[0]
    cp_sorted = sorted(cp_dates)
    out["days_since_last_cp"] = [(d - max([c for c in cp_sorted if c <= d] or [cp_sorted[0]])).days for d in ds]
    out["is_cp_window_7d"] = [int(min([abs((d - c).days) for c in cp_sorted]) <= 7) for d in ds]
    out["segment_id"] = [sum(1 for c in cp_sorted if c <= d) for d in ds]
    return out

# Usage: refit CP detection trong mỗi CV fold
```

**Cost**: PELT-L2 trên 3,500 ngày ~150ms. Refit 4 folds CV + 1 final = 5 runs ≈ 1 giây tổng. Không ảnh hưởng ngân sách.

**Acceptance criteria**:
- [ ] Detect được ≥ 3 change-points trong 2012–2022 (kỳ vọng: ~Q1 2020 COVID, ~Q3 2020 hồi phục, ~2022 boom)
- [ ] 3 cột `days_since_last_cp`, `is_cp_window_7d`, `segment_id` có SHAP importance > 0
- [ ] CP detection chạy trong mọi CV fold (không dùng toàn bộ data → leak-free)

**Delta kỳ vọng**: +0.002–0.007 R² — khiêm tốn vì COVID đã cách test set 3 năm, nhưng `segment_id` giúp model tránh dùng lag từ segment khác.

### Task 10 — Feature interactions (0.75 giờ, +0.003–0.010 R²)

**Mục tiêu**: Mã hóa tương tác numeric × numeric và categorical × categorical — trees không tự tạo được phép nhân continuous.

**File cần sửa**: `src/features/interactions.py` (tạo mới).

```python
# src/features/interactions.py
import pandas as pd

def build_interactions(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    # Continuous × continuous (trees không tự làm được phép nhân)
    out["tet_x_disc"]      = df["signed_days_tet"]  * df["max_discount"]
    out["ghost_x_disc"]    = df["is_ghost_month"]   * df["max_discount"]
    out["weekend_x_promo"] = df["is_weekend"]       * df["promo_days_last_7"]
    out["payday_x_disc"]   = df["is_payday"]        * df["max_discount"]
    out["pretet_x_traffic"]= df["is_pre_tet_15d"]   * df["page_views_lag_1"]
    out["traffic_x_disc"]  = df["sessions_lag_1"].clip(0) * df["max_discount"]
    # Cat × cat via concat (for sklearn TargetEncoder with CV)
    out["dow_x_month"]  = df["day_of_week"].astype(str) + "_" + df["month"].astype(str)
    out["tetbucket"]    = pd.cut(df["signed_days_tet"],
                                 bins=[-999,-30,-15,-7,-1,0,1,7,15,30,999],
                                 labels=False).astype(str)
    return out
```

**Target encoding cho cat×cat** (sklearn 1.3+ có native CV):

```python
from sklearn.preprocessing import TargetEncoder
enc = TargetEncoder(smooth="auto", cv=5, target_type="continuous", random_state=42)
X_enc = enc.fit_transform(df[["dow_x_month","tetbucket"]], y_train)
# KHÔNG fit trên test — chỉ transform
```

**Acceptance criteria**:
- [ ] Thêm 6 cột continuous interaction + 2 cột target-encoded
- [ ] Target encoder fit với `cv=5` (built-in KFold) — không leak
- [ ] Ít nhất 2 trong 6 continuous interactions lọt top-40 SHAP
- [ ] Revenue R² OOF không giảm

**Delta kỳ vọng**: +0.003–0.010 R², đặc biệt ở bucket h01_030 và quanh TET.

---

## 4. Tier 2 — Direct Multi-horizon Forecasting (5.0 giờ, +0.020–0.030 R²)

### Task 11 — Direct multi-horizon với mlforecast (3.0 giờ, +0.010–0.018 R²)

**Mục tiêu**: Thay recursive inference (error compound O(h^1.2–1.5) qua 548 bước) bằng **direct forecasting** — train một LGB Tweedie model riêng cho mỗi horizon bucket. Đây là pattern gốc từ Corporación Favorita 1st place (Shixuan Li, 2017) và được khẳng định trong review Kaggle forecasting (Bojer 2021): "one model per forecast horizon" là một trong các innovation chính của grand-prize-winning solutions.

**Version**: `mlforecast==1.0.2` (Feb 2025, stable).

**File mới**: `src/models/mlforecast_direct.py`.

```python
# pip install mlforecast==1.0.2 lightgbm==4.6.0
import lightgbm as lgb
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean, ExponentiallyWeightedMean
from mlforecast.target_transforms import Differences

BUCKETS = [(1,30), (31,90), (91,180), (181,548)]   # khớp với Task 3

def build_mlforecast(bucket_hi: int, gpu_id: int = 0):
    return MLForecast(
        models={"lgb_tw": lgb.LGBMRegressor(
            objective="tweedie", tweedie_variance_power=1.15,
            n_estimators=3000, learning_rate=0.03, num_leaves=255,
            min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8,
            bagging_freq=1, device_type="gpu",
            gpu_platform_id=0, gpu_device_id=gpu_id,
            max_bin=63, use_quantized_grad=False,
            verbosity=-1, random_state=42)},
        freq="D",
        lags=[1, 7, 14, 28, 56, 91, 182, 364, 728],
        lag_transforms={
            1:  [ExponentiallyWeightedMean(alpha=0.3)],
            7:  [RollingMean(window_size=7),  RollingMean(window_size=28)],
            28: [RollingMean(window_size=28), RollingMean(window_size=91)],
            91: [RollingMean(window_size=91), RollingMean(window_size=182)],
        },
        date_features=["dayofweek","day","dayofyear","week","month","quarter"],
        target_transforms=[Differences([1, 7])],
        num_threads=4,
    )

def direct_forecast(df_long, future_exog_df):
    """df_long: DataFrame(unique_id, ds, y, <exog cols>)"""
    all_preds = []
    for (lo, hi), gpu in zip(BUCKETS, [0, 1, 0, 1]):  # interleave GPUs
        fc = build_mlforecast(hi, gpu_id=gpu)
        fc.fit(df_long, id_col="unique_id", time_col="ds", target_col="y",
               static_features=[], max_horizon=hi, dropna=True)
        p = fc.predict(h=hi, X_df=future_exog_df)
        p = p[(p["ds"] >= df_long["ds"].max() + pd.Timedelta(days=lo)) &
              (p["ds"] <= df_long["ds"].max() + pd.Timedelta(days=hi))]
        p["horizon_bucket"] = f"h{lo:03d}_{hi:03d}"
        all_preds.append(p)
    return pd.concat(all_preds).rename(columns={"lgb_tw": "pred_mlf_direct"})
```

**Caveat quan trọng**: mlforecast giả định mọi exogenous đều là **future-known**. Với features kiểu `sessions_lag_1` (chỉ biết quá khứ), cần lag thêm ≥ horizon để an toàn, VD dùng `sessions_lag_30` cho bucket h01_030.

**Tích hợp blender**: Thêm `pred_mlf_direct` làm member mới trong Caruana. Với bucket h181_plus, đây thường là head có weight cao nhất (giảm error compound).

**Acceptance criteria**:
- [ ] 4 MLForecast objects train xong trong ≤ 2.5 giờ (fold parallel sang 2 GPU)
- [ ] OOF R² của `pred_mlf_direct` ở `h181_plus` ≥ R² của `pred_l2` (XGB raw recursive) — chứng minh direct win trên long horizon
- [ ] Caruana assign weight ≥ 0.3 cho `pred_mlf_direct` ở bucket h181_plus
- [ ] Không OOM

**Delta kỳ vọng**: +0.010–0.018 R² — đóng góp lớn nhất của Tier 2, vì bucket h181_plus chiếm 67% test samples.

### Task 12 — N-HiTS neural member (1.5 giờ, +0.005–0.010 R²)

**Mục tiêu**: Thêm một neural member để tăng diversity vượt ngoài tree-based. N-HiTS là MLP hierarchical — rất rẻ so với TFT/Chronos, thường đạt top-3 zero-shot trên benchmarks dài hạn.

**Version**: `neuralforecast==3.1.7` (Apr 2026).

**File mới**: `src/models/nhits_head.py`.

**Quyết định DDP**: Kaggle notebook không hỗ trợ `strategy="ddp"` (Jupyter-based) — phải dùng `ddp_spawn` hoặc single-GPU. Với 1 series, DDP tăng tốc không đáng kể — **khuyến nghị single GPU**, dùng GPU thứ 2 cho task khác song song (VD CatBoost hoặc Chronos).

```python
# pip install neuralforecast==3.1.7
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MQLoss
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # pin T4 #0

H = 548
L = 3 * H       # input_size = 1644 (≈ 4.5 năm lookback)

model = NHITS(
    h=H, input_size=L,
    futr_exog_list=["is_tet_week","is_pre_tet_15d","is_ghost_month",
                    "dayofweek","month","is_weekend","max_discount"],
    hist_exog_list=["sessions_lag_1","rating_mean_lag_1"],
    stat_exog_list=[],
    stack_types=["identity","identity","identity"],
    n_blocks=[1,1,1],
    mlp_units=3*[[512, 512]],
    n_pool_kernel_size=[16, 8, 1],
    n_freq_downsample=[168, 24, 1],
    interpolation_mode="linear",
    pooling_mode="MaxPool1d",
    dropout_prob_theta=0.1,
    loss=MQLoss(level=[80, 90]),
    max_steps=3000,
    learning_rate=1e-3,
    num_lr_decays=3,
    early_stop_patience_steps=10,
    val_check_steps=100,
    batch_size=32,
    windows_batch_size=256,
    scaler_type="robust",
    random_seed=42,
    accelerator="gpu",
    devices=1,
    precision="16-mixed",
)

nf = NeuralForecast(models=[model], freq="D")
nf.fit(Y_train_df, static_df=None, val_size=H)
Y_hat = nf.predict(futr_df=future_exog_df)
# Lấy quantile 0.5 (median) làm point prediction
pred_nhits = Y_hat["NHITS-median"].values
```

**Training time**: ~15–40 phút trên T4 cho 2000 steps. Memory ~2–4 GB.

**Caveats**:
- Issue #1037 của neuralforecast: `ddp_spawn` predict() bị lỗi vstack None — tránh bằng cách predict trên single GPU sau khi train.
- `futr_df` phải chứa tất cả `futr_exog_list` features ở toàn bộ 548 ngày test.

**Acceptance criteria**:
- [ ] N-HiTS train xong trong ≤ 45 phút trên T4
- [ ] OOF Revenue R² ≥ 0.75 (chấp nhận thấp hơn LGB/XGB — role của nó là diversity)
- [ ] Correlation với tree-heads ≤ 0.90 (diversity thật sự)
- [ ] Caruana weight ≥ 0.05 ở ít nhất 1 bucket long-horizon

**Delta kỳ vọng**: +0.005–0.010 R² khi blend với tree members.

### Task 13 — Ridge meta-learner per bucket (0.5 giờ, +0.003–0.007 R²)

**Mục tiêu**: Caruana forward selection có điểm yếu trên bucket có ít OOF samples (h181_plus ~368 samples). Ridge regression với positive constraint ổn định hơn. Kaggle Grandmaster Chris Deotte (winner April 2025 Playground) đã pattern stacking với level-2 Ridge cho chính lý do này.

**File cần sửa**: `src/blend/meta_learner.py`.

```python
# src/blend/meta_learner.py
from sklearn.linear_model import Ridge
from scipy.optimize import nnls
import numpy as np

def fit_ridge_meta_per_bucket(oof_preds_by_bucket, y_by_bucket, alpha=1.0):
    """
    oof_preds_by_bucket: {bucket_name: (N, K) array of OOF predictions from K heads}
    y_by_bucket       : {bucket_name: (N,) array of true target}
    Trả về: {bucket_name: weights (K,), all non-negative, sum-to-1 normalized}
    """
    weights = {}
    for b, P in oof_preds_by_bucket.items():
        y = y_by_bucket[b]
        # NNLS đảm bảo non-negative weights
        w, _ = nnls(P, y)
        # Ridge shrinkage để tránh overfit
        ridge = Ridge(alpha=alpha, positive=True, fit_intercept=False)
        ridge.fit(P, y)
        w_ridge = ridge.coef_
        # Blend NNLS (tight fit) và Ridge (regularized)
        w_final = 0.5 * w + 0.5 * w_ridge
        w_final = np.clip(w_final, 0, None)
        if w_final.sum() > 0:
            w_final /= w_final.sum()
        weights[b] = w_final
    return weights
```

**A/B test bắt buộc**: Chạy cả Caruana và Ridge, giữ cái cho OOF MAE thấp hơn trên realistic_probe (Task 2).

**Acceptance criteria**:
- [ ] Ridge weights cho bucket `h181_plus` có ít nhất 3/7 heads với weight > 0.05 (diversity)
- [ ] Realistic probe R² của Ridge meta ≥ Caruana — nếu không, giữ Caruana

**Delta kỳ vọng**: +0.003–0.007 R², chủ yếu giảm variance trên long-horizon.

### Task 14 — Split-conformal calibration (0.5 giờ, +0.002–0.005 R²)

**Mục tiêu**: Thay quantile mapping (strength=0.55) hiện tại bằng **split-conformal prediction** — cho guarantee coverage chính xác trên test, thường cải thiện MAE vì tránh calibration drift.

**File cần sửa**: `src/calibrate/conformal.py` (tạo mới, thay `quantile_mapping.py`).

```python
# src/calibrate/conformal.py
import numpy as np

def split_conformal_calibrate(oof_pred, y_oof, test_pred, alpha=0.1):
    """
    Multiplicative conformal: scale correction q sao cho pred * q khớp median OOF.
    Ít overfitting hơn additive với heavy-tailed data (revenue).
    """
    ratio = y_oof / np.clip(oof_pred, 1e-6, None)
    # Huber-like scale: trimmed mean để resist outliers
    lo, hi = np.quantile(ratio, [alpha, 1-alpha])
    trimmed = ratio[(ratio >= lo) & (ratio <= hi)]
    q = np.median(trimmed)          # calibration factor
    return test_pred * q, q

def additive_conformal_interval(oof_pred, y_oof, test_pred, alpha=0.1):
    """Trả về interval cho risk reporting (không dùng cho submission main)."""
    resid = y_oof - oof_pred
    q = np.quantile(np.abs(resid), 1 - alpha)
    return test_pred - q, test_pred + q
```

**Acceptance criteria**:
- [ ] Conformal factor `q` trong khoảng [0.9, 1.1] (nếu ngoài → có leak hoặc bug)
- [ ] OOF MAE sau conformal ≤ OOF MAE v3 quantile mapping
- [ ] Interval 90% coverage thực tế ∈ [85%, 95%] trên realistic probe

**Delta kỳ vọng**: +0.002–0.005 R², chủ yếu qua MAE giảm.

---

## 5. Tier 3 — Ambitious / Nếu còn thời gian (2.0 giờ, +0.005–0.020 R²)

### Task 15 — Chronos-Bolt Small fine-tune (1.0 giờ, +0.003–0.010 R²) ⚠️ RISK

**Mục tiêu**: Foundation-model anchor. Chronos-Bolt được huấn luyện trên 100B observations — có thể đem lại zero-shot diversity.

**⚠️ DISCLOSURE RISK**: Kaggle Competition Rules thường yêu cầu **disclose** mọi pretrained model trong forum ít nhất 7 ngày trước deadline. Chronos-Bolt là **Apache 2.0** (legal OK) nhưng **cần kiểm tra rule thread Datathon VinUni 2026 để xác nhận**. Nếu rules cấm external pretrained weights → **SKIP task này**. Upload weights làm Kaggle Dataset trước (notebook save-mode không có internet).

**⚠️ TET RISK**: Chronos pre-training corpus là `autogluon/chronos_datasets` + synthetic KernelSynth — chủ yếu lịch phương Tây. **TET là lunisolar, di chuyển 20–50 ngày Gregorian year-over-year** → model **sẽ không học được pattern TET** zero-shot. Mitigation bắt buộc:

1. **Fine-tune** trên 10 TET trong lịch sử (2013–2022) — đủ signal.
2. Dùng **Chronos-2** (Oct 2025, 120M params) vì model này **hỗ trợ native future covariates** — có thể truyền `days_to_tet` trực tiếp.
3. Kẹp bằng AutoGluon **covariate regressor** (CAT/GBM) để xử lý residual với calendar features.

**Version**: `chronos-forecasting==1.5.1` + `autogluon.timeseries>=1.5.1`.

**File mới**: `src/models/chronos_finetune.py`.

```python
# pip install -U autogluon.timeseries chronos-forecasting==1.5.1
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

train_data = TimeSeriesDataFrame.from_data_frame(
    df_long, id_column="unique_id", timestamp_column="ds")

predictor = TimeSeriesPredictor(
    prediction_length=548,
    eval_metric="MAE",
    target="y",
    known_covariates_names=["is_tet_week","is_pre_tet_15d","is_ghost_month",
                            "dayofweek","max_discount"],
).fit(
    train_data,
    hyperparameters={
        "Chronos": [
            {"model_path": "bolt_small",         # 48M, fit T4 16GB thoải mái
             "fine_tune": True,
             "fine_tune_lr": 1e-4,
             "fine_tune_steps": 3000,
             "context_length": 2048,
             "ag_args": {"name_suffix": "FineTuned"}},
        ]
    },
    time_limit=3000,   # 50 phút
    enable_ensemble=False,
)
pred_chronos = predictor.predict(train_data, known_covariates=future_cov_df)
```

**Memory**: `bolt_small` 48M + gradients + AdamW ≈ 3–4 GB trên T4 — thoải mái. `bolt_base` 205M sát trần 16GB, tránh.

**Acceptance criteria**:
- [ ] Fine-tune xong trong ≤ 60 phút
- [ ] OOF Revenue R² ≥ 0.70 (threshold thấp — role là diversity)
- [ ] Correlation với tree heads < 0.85
- [ ] Caruana weight ≥ 0.03 (nếu 0 → drop, không dùng)
- [ ] **KIỂM TRA RULES TRƯỚC KHI ENABLE**

**Delta kỳ vọng**: +0.003–0.010 R² (có điều kiện), hoặc 0 nếu rules cấm.

### Task 16 — TFT từ pytorch-forecasting (1.0 giờ, +0.003–0.010 R²) — OPTIONAL

**Mục tiêu**: Alternative neural architecture với interpretability native (variable selection, attention). Chỉ chạy nếu Task 15 skip và còn thời gian.

**Version**: `pytorch-forecasting==1.6.1` (Jan 2026), giờ maintained bởi sktime org (`github.com/sktime/pytorch-forecasting`).

**File mới**: `src/models/tft_head.py` (xem template đầy đủ ở Phần 11).

**Training time**: ~20–60 phút trên T4 cho 40 epochs với hidden_size=128.

**Acceptance criteria**:
- [ ] Train xong ≤ 60 phút
- [ ] OOF R² ≥ 0.72
- [ ] Diversity correlation ≤ 0.88 với N-HiTS
- [ ] Caruana weight ≥ 0.03

**Delta kỳ vọng**: +0.003–0.010 R² (mutually exclusive với Task 15 trong ngân sách thời gian).

### Task 17 — STL decomposition as features (0.5 giờ, +0.002–0.005 R²)

**Mục tiêu**: Khác với STL+ARIMA branch hiện tại (dùng như forecaster). Task này dùng MSTL để decompose target thành `trend`, `seasonal_7`, `seasonal_365`, `resid` và truyền **các components** làm features cho XGB/LGB/CAT.

**Version**: `statsmodels==0.14.5`.

```python
# pip install statsmodels==0.14.5
from statsmodels.tsa.seasonal import MSTL
import pandas as pd

def stl_features(y: pd.Series, test_start: str):
    cutoff = pd.Timestamp(test_start) - pd.Timedelta(days=1)
    res = MSTL(y.loc[:cutoff], periods=(7, 365)).fit()
    out = pd.DataFrame(index=y.index,
                       columns=["stl_trend","stl_s7","stl_s365","stl_resid"],
                       dtype=float)
    out.loc[:cutoff,"stl_trend"]= res.trend.values
    out.loc[:cutoff,"stl_s7"]   = res.seasonal["seasonal_7"].values
    out.loc[:cutoff,"stl_s365"] = res.seasonal["seasonal_365"].values
    out.loc[:cutoff,"stl_resid"]= res.resid.values
    # Carry forward cho test (leak-free nếu fit chỉ trên train)
    last_w  = res.seasonal["seasonal_7"].iloc[-7:].values
    last_y  = res.seasonal["seasonal_365"].iloc[-365:].values
    last_tr = res.trend.iloc[-1]
    test_idx= y.loc[test_start:].index
    for i, d in enumerate(test_idx):
        out.loc[d,"stl_trend"] = last_tr
        out.loc[d,"stl_s7"]    = last_w[i % 7]
        out.loc[d,"stl_s365"]  = last_y[i % 365]
        out.loc[d,"stl_resid"] = 0.0
    return out
```

**Cost**: MSTL 3500 điểm ~1–3s. Negligible.

**Acceptance criteria**:
- [ ] 4 cột STL trong SHAP top-60
- [ ] Loại bỏ STL+ARIMA branch cũ khỏi blender (redundant)
- [ ] R² không giảm

**Delta kỳ vọng**: +0.002–0.005 R².

---

## 6. Execution Schedule — 12 Giờ Kaggle Save Mode

| Khung giờ | Task | Chi tiết | GPU |
|---|---|---|---|
| H0:00–H0:15 | Setup | `pip install` tất cả libs, import test, download Kaggle datasets cho Chronos weights | — |
| H0:15–H0:30 | Task 1 | Fix Huber → Quantile τ=0.5 hoặc Huber+log1p | — |
| H0:30–H0:45 | Task 2 + 3 | Sửa CV config, gộp buckets, test smoke | — |
| H0:45–H1:00 | Task 4 | Lunar TET features + VN holidays, verify TET 2023/2024 | — |
| H1:00–H2:00 | Task 5 | LGB Tweedie head, train 4 folds (2 parallel/GPU) | T4 #0 + #1 |
| H2:00–H3:00 | Task 6 | CatBoost head (Poisson + log1p-RMSE), dual GPU | T4 #0:#1 |
| H3:00–H3:30 | Task 7 | Prophet components → features | CPU |
| H3:30–H3:45 | Task 8 | Fourier K=8/K=3/lunar K=2 | CPU |
| H3:45–H4:15 | Task 9 | Change-points với ruptures | CPU |
| H4:15–H5:00 | Task 10 | Feature interactions + target encoding, re-train XGB heads | T4 #0 |
| H5:00–H8:00 | Task 11 | mlforecast direct multi-horizon, 4 buckets | T4 #0 + #1 interleaved |
| H8:00–H9:30 | Task 12 | N-HiTS neural member (single GPU) | T4 #0 |
| H9:30–H10:00 | Task 13 | Ridge meta-learner per bucket, A/B vs Caruana | CPU |
| H10:00–H10:30 | Task 14 | Split-conformal calibration | CPU |
| H10:30–H11:00 | Final ensemble | Gen submission_safe, submission_main, submission_experimental | — |
| H11:00–H11:30 | Tier 3 (opt) | Task 15 (Chronos fine-tune) **NẾU rules cho phép**, else Task 16 (TFT) | T4 #1 |
| H11:30–H11:50 | Validation Protocol | Chạy smoke + realistic probe + adversarial AUC | — |
| H11:50–H12:00 | Buffer | SHAP regeneration, plot report, final sanity check | — |

**Checkpoint rule**: Mỗi 2 giờ, save model + OOF predictions ra `/kaggle/working/checkpoints/tier_<n>_<hhmm>.pkl`. Nếu hết giờ đột ngột, có thể rebuild submission từ checkpoint gần nhất.

**Parallel execution gợi ý**:
- H1:00–H3:00: LGB trên GPU#0 song song CatBoost trên GPU#1 (CatBoost native dual-GPU)
- H8:00–H11:00: N-HiTS/TFT trên GPU#0, Chronos fine-tune trên GPU#1

---

## 7. Validation Protocol (bắt buộc trước submit)

Chạy **theo thứ tự**, không bỏ bước:

**Bước 1 — Smoke test** (10 phút):

```python
# Chạy pipeline với cv_max_splits=2, n_estimators=500, windows=100 — chỉ để verify không crash
CONFIG["cv"]["short_cv"]["max_splits"] = 2
CONFIG["lgb"]["n_estimators"] = 500
run_pipeline(smoke=True)
assert all heads return non-null predictions
```

**Bước 2 — Realistic CV probe** (30 phút):

Chạy `realistic_probe` fold (Task 2: 7 năm train, 548 ngày val). Đây là estimate **gần nhất** với Kaggle private LB.

```python
oof_probe = pipeline.fit_predict(realistic_probe_fold_only=True)
probe_r2  = r2_score(y_probe, oof_probe)
probe_mae = mean_absolute_error(y_probe, oof_probe)
print(f"Realistic probe: R²={probe_r2:.4f}, MAE={probe_mae:,.0f}")
# Expect: R² thấp hơn short CV R² khoảng 0.03–0.05
```

**Bước 3 — Adversarial validation AUC** (5 phút):

```python
# Stack train (2012-2022) vs test (2023-01 → 2024-07)
X_all = pd.concat([X_train, X_test]); y_adv = [0]*len(X_train) + [1]*len(X_test)
auc = cross_val_score(LGBMClassifier(n_estimators=200), X_all, y_adv,
                      cv=5, scoring="roc_auc").mean()
print(f"Adversarial AUC: {auc:.3f}")
# AUC > 0.8 → distribution shift lớn, cần drop top drift features
# AUC < 0.65 → an toàn
```

**Bước 4 — Block bootstrap CI** (15 phút):

```python
# 500 runs × 30-day blocks
cis = block_bootstrap(y_oof, oof_preds, n_runs=500, block_size=30)
print(f"R² CI 95%: [{cis['r2'][0]:.3f}, {cis['r2'][1]:.3f}]")
# Expect: [0.84, 0.90] nếu Tier 0+1+2 đi đúng kế hoạch
```

**Bước 5 — Sanity check predictions** (2 phút):

```python
assert test_pred.min() >= 0,             "Negative prediction — bug back-transform"
assert test_pred.max() <= y_train.max()*2,"Outlier prediction — clip winsorize"
assert abs(test_pred.mean()/y_train.mean() - 1) < 0.5, "Mean drift >50%"
assert test_pred.std() > y_train.std() * 0.3, "Under-dispersion"
```

**Nếu bước nào fail** → không submit, rollback Tier tương ứng (Phần 9).

---

## 8. Submission Strategy

Luôn tạo **ba submission files**:

| File | Nội dung | Khi nào chọn |
|---|---|---|
| `submission_safe.csv` | Pipeline v3 baseline (chỉ Tier 0 hotfix) | Default nếu Tier 1+ làm tệ đi |
| `submission_main.csv` | Tier 0+1+2 full ensemble + Ridge meta + conformal | **Submission chính** nếu realistic probe R² ≥ 0.83 |
| `submission_experimental.csv` | Tier 0+1+2+3 (thêm Chronos/TFT/STL) | Chỉ nếu tất cả acceptance criteria Tier 3 pass VÀ realistic probe không giảm |

**Rule 3-tier fallback nếu có head fail**:

```python
def safe_ensemble(preds_dict, weights_dict, bucket):
    available = [k for k, v in preds_dict.items() if v is not None and np.isfinite(v).all()]
    if len(available) < 2:
        return preds_dict["pred_l2"]          # fallback hardest: XGB raw head
    w = np.array([weights_dict[bucket][k] if k in available else 0 for k in available])
    w = w / w.sum() if w.sum() > 0 else np.ones(len(available))/len(available)
    return sum(w[i] * preds_dict[k] for i, k in enumerate(available))
```

**Public vs Private LB strategy**:

- **Public LB** thường là subset ngẫu nhiên hoặc first N days. Kaggle forecasting truyền thống: public = 30% test, private = 70%. Với Datathon VinUni 2026, **giả định tương tự**.
- **Shake-up rủi ro cao**: Long-horizon test (548 ngày) có distribution shift post-COVID. **KHÔNG** trust blindly public LB.
- **Final submission rule**: Chọn submission có **realistic_probe R² cao nhất**, không phải public LB MAE thấp nhất. Nếu trust realistic probe < 0.85, submit `submission_safe.csv` như 1 trong 2 final submissions (nếu được 2 final), còn lại là `submission_main.csv`.

---

## 9. Rollback Strategy

Git workflow với tag sau mỗi tier:

```bash
git tag v4_tier0_done    # sau H1:00
git tag v4_tier1_done    # sau H5:00
git tag v4_tier2_done    # sau H10:30
git tag v4_tier3_done    # sau H11:30
```

**A/B test tự động sau mỗi task**:

```python
# Chạy mỗi khi thêm feature hoặc head mới
before_mae = load_checkpoint_metric("latest")
after_mae  = compute_oof_mae(pipeline_current)
if after_mae > before_mae * 1.003:      # tệ đi >0.3%
    git_revert_task()
    log.warning(f"Task X rolled back: {after_mae:,.0f} > {before_mae:,.0f}")
```

**Rollback cụ thể**:

| Task fail | Hành động |
|---|---|
| Task 1 (Huber fix) | Revert head, dùng 4-head blender (không có huber/quantile) |
| Task 5 (LGB) | Giữ pipeline XGB-only, re-tune Caruana với 5 XGB heads |
| Task 6 (CatBoost) | Skip, giữ LGB + XGB |
| Task 7 (Prophet feats) | Revert, giữ Prophet head cũ (seasonal-naive) |
| Task 11 (mlforecast) | Giữ recursive, tăng weight `pred_l2` ở bucket dài |
| Task 12 (N-HiTS) | Skip neural member |
| Task 15/16/17 (Tier 3) | Skip toàn bộ Tier 3, submit `submission_main.csv` |

**Red flag threshold**: Nếu realistic_probe R² giảm > 0.01 so với sau Tier 0, **dừng** và rollback tới `v4_tier0_done`.

---

## 10. Risk Matrix

| Rủi ro | Xác suất | Tác động | Mitigation |
|---|---|---|---|
| **Distribution shift COVID/post-COVID** ảnh hưởng 2023–2024 test | Cao (0.7) | Cao | Adversarial AUC screening (Bước 3); drop top-8 drift features; thêm `segment_id` từ change-point (Task 9); thêm `year` × `post_covid` interaction |
| **Leakage qua inventory snapshots** (stock cuối tháng biết trước) | Trung (0.4) | Cao | Audit mọi cột `*_inv_*`: chỉ dùng `lag_1` trở lên. Assert `inv_feature.max_date < target.min_date` ở mỗi fold |
| **Overfitting Caruana blender** (bucket h181_plus ít samples) | Trung (0.5) | Trung | Task 13 Ridge meta-learner làm default, Caruana làm fallback |
| **OOM trên T4 16GB** với full feature set (~200 cols) | Trung (0.4) | Trung | `max_bin=63` LGB; `border_count=128` CatBoost; `QuantileDMatrix` XGB; monitor `nvidia-smi`; fallback CPU cho CatBoost |
| **Recursive error compound ở h366+** | Cao (0.8) | Cao | Task 11 direct forecasting cho h181_plus bucket (weight ≥ 0.3) |
| **Chronos rules violation** | Trung (0.3) | Rất cao (DQ) | KIỂM TRA rules thread trước H11:00; nếu mơ hồ → SKIP task 15 |
| **mlforecast not-known covariates bug** | Trung (0.4) | Trung | Chỉ truyền future-known features (calendar, TET, Prophet components). Lag ≥ horizon các features quá khứ |
| **N-HiTS DDP predict fail** (issue #1037) | Trung (0.5) | Thấp | Dùng `devices=1` cho N-HiTS (không DDP) |
| **CatBoost Tweedie GPU crash** (issue #2812) | Cao (0.7) | Thấp | Dùng Poisson hoặc RMSE-log1p thay Tweedie |
| **TET lunar drift** (2030+) | Rất thấp (0.02) | Thấp | `lunardate` 0.2.2 đúng 2012–2026, không ảnh hưởng |
| **Kaggle 12h timeout đột ngột** | Trung (0.3) | Cao | Checkpoint mỗi 2h, submit `submission_safe.csv` nếu không kịp |

---

## 11. Code Templates sẵn sàng copy-paste

### 11.1 Complete LGBMRegressor config T4 Tweedie

```python
import lightgbm as lgb

def make_lgb_tweedie(gpu_id=0, seed=42, tweedie_vp=1.15):
    return lgb.LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=tweedie_vp,
        n_estimators=5000,
        learning_rate=0.03,
        num_leaves=255,
        min_data_in_leaf=100,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=0.1,
        max_bin=63,
        device_type="gpu",
        gpu_platform_id=0,
        gpu_device_id=gpu_id,
        gpu_use_dp=False,
        use_quantized_grad=False,
        num_threads=4,
        verbosity=-1,
        random_state=seed,
    )
```

### 11.2 Complete CatBoostRegressor config T4

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from catboost import CatBoostRegressor

def make_catboost(loss="RMSE", seed=42):
    return CatBoostRegressor(
        iterations=10000,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3.0,
        loss_function=loss,   # "RMSE" (log1p y) hoặc "Poisson" (raw y)
        eval_metric=loss,
        task_type="GPU",
        devices="0:1",
        gpu_ram_part=0.85,
        border_count=128,
        grow_policy="SymmetricTree",
        bootstrap_type="Bayesian",
        bagging_temperature=1.0,
        od_type="Iter",
        od_wait=300,
        use_best_model=True,
        verbose=500,
        allow_writing_files=False,
        random_seed=seed,
    )
```

### 11.3 mlforecast pipeline hoàn chỉnh

```python
import pandas as pd, lightgbm as lgb
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean, ExponentiallyWeightedMean
from mlforecast.target_transforms import Differences

BUCKETS = [(1,30), (31,90), (91,180), (181,548)]

def run_direct_mlforecast(df_long, future_exog_df):
    preds = []
    for i, (lo, hi) in enumerate(BUCKETS):
        gpu = i % 2
        fc = MLForecast(
            models={"lgb": lgb.LGBMRegressor(
                objective="tweedie", tweedie_variance_power=1.15,
                n_estimators=3000, learning_rate=0.03, num_leaves=255,
                device_type="gpu", gpu_device_id=gpu,
                max_bin=63, use_quantized_grad=False,
                verbosity=-1, random_state=42)},
            freq="D",
            lags=[1,7,14,28,56,91,182,364,728],
            lag_transforms={
                1:  [ExponentiallyWeightedMean(alpha=0.3)],
                7:  [RollingMean(7), RollingMean(28)],
                28: [RollingMean(28), RollingMean(91)],
                91: [RollingMean(91), RollingMean(182)],
            },
            date_features=["dayofweek","day","dayofyear","week","month","quarter"],
            target_transforms=[Differences([1, 7])],
            num_threads=4,
        )
        fc.fit(df_long, id_col="unique_id", time_col="ds", target_col="y",
               max_horizon=hi, dropna=True)
        p = fc.predict(h=hi, X_df=future_exog_df)
        cutoff = df_long["ds"].max()
        p = p[(p["ds"] >= cutoff + pd.Timedelta(days=lo)) &
              (p["ds"] <= cutoff + pd.Timedelta(days=hi))]
        p["bucket"] = f"h{lo:03d}_{hi:03d}"
        preds.append(p)
    return pd.concat(preds).rename(columns={"lgb": "pred_mlf_direct"})
```

### 11.4 neuralforecast N-HiTS config T4 (single GPU)

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MQLoss

def make_nhits(H=548, L=None, futr_cols=None, hist_cols=None):
    L = L or 3*H
    return NHITS(
        h=H, input_size=L,
        futr_exog_list=futr_cols or [],
        hist_exog_list=hist_cols or [],
        stack_types=["identity"]*3, n_blocks=[1,1,1],
        mlp_units=3*[[512,512]],
        n_pool_kernel_size=[16,8,1],
        n_freq_downsample=[168,24,1],
        interpolation_mode="linear", pooling_mode="MaxPool1d",
        dropout_prob_theta=0.1,
        loss=MQLoss(level=[80,90]),
        max_steps=3000, learning_rate=1e-3, num_lr_decays=3,
        early_stop_patience_steps=10, val_check_steps=100,
        batch_size=32, windows_batch_size=256,
        scaler_type="robust", random_seed=42,
        accelerator="gpu", devices=1, precision="16-mixed",
    )
```

### 11.5 Prophet component extraction helper

```python
from prophet import Prophet
import pandas as pd

def extract_prophet_components(train_df, full_df, country="VN"):
    m = Prophet(growth="linear", yearly_seasonality=10, weekly_seasonality=3,
                daily_seasonality=False, seasonality_mode="multiplicative",
                changepoint_prior_scale=0.05)
    m.add_country_holidays(country_name=country)
    m.fit(train_df[["ds","y"]])
    fcst = m.predict(full_df[["ds"]])
    return fcst[["ds","trend","yearly","weekly","holidays","yhat"]].rename(columns={
        "trend":"pf_trend","yearly":"pf_yearly","weekly":"pf_weekly",
        "holidays":"pf_holidays","yhat":"pf_yhat"})
```

### 11.6 Ridge meta-learner per bucket

```python
from sklearn.linear_model import Ridge
from scipy.optimize import nnls
import numpy as np

def fit_ridge_meta(oof_preds_by_bucket, y_by_bucket, alpha=1.0):
    weights = {}
    for b, P in oof_preds_by_bucket.items():
        y = y_by_bucket[b]
        w_nnls, _ = nnls(P, y)
        ridge = Ridge(alpha=alpha, positive=True, fit_intercept=False)
        ridge.fit(P, y)
        w = 0.5*w_nnls + 0.5*ridge.coef_
        w = np.clip(w, 0, None)
        w = w/w.sum() if w.sum() > 0 else np.ones(len(w))/len(w)
        weights[b] = w
    return weights

def apply_ridge_meta(test_preds_by_bucket, weights_by_bucket):
    out = {}
    for b, P in test_preds_by_bucket.items():
        out[b] = P @ weights_by_bucket[b]
    return out
```

### 11.7 Lunar TET feature helper

```python
# pip install lunardate==0.2.2
from lunardate import LunarDate
from datetime import date, timedelta
import numpy as np, pandas as pd

TET = {y: LunarDate(y, 1, 1).toSolarDate() for y in range(2011, 2028)}
assert TET[2023] == date(2023, 1, 22)
assert TET[2024] == date(2024, 2, 10)

def build_tet_features(ds: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for d in ds:
        dd = d.date()
        nxt = min((t for t in TET.values() if t >= dd), default=None)
        prv = max((t for t in TET.values() if t <  dd), default=None)
        dt = (nxt-dd).days if nxt else 365
        ds_= (dd-prv).days if prv else 365
        rows.append({
            "days_to_tet": dt, "days_since_tet": ds_,
            "signed_days_tet": dt if dt<=ds_ else -ds_,
            "is_tet_day":    int(dd in TET.values()),
            "is_tet_week":   int(min(dt,ds_)<=7),
            "is_pre_tet_15d":int(0<dt<=15),
            "is_post_tet_7d":int(0<ds_<=7),
            "tet_proximity": float(np.exp(-min(dt,ds_)/14)),
        })
    return pd.DataFrame(rows, index=ds)

def build_vn_holidays(ds: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for d in ds:
        y, m, dd = d.year, d.month, d.day
        mid_autumn = LunarDate(y,8,15).toSolarDate()
        ghost_s    = LunarDate(y,7,1).toSolarDate()
        ghost_e    = LunarDate(y,8,1).toSolarDate() - timedelta(days=1)
        rows.append({
            "is_reunif":    int((m,dd)==(4,30)),
            "is_labour":    int((m,dd)==(5,1)),
            "is_national":  int((m,dd)==(9,2)),
            "is_mid_autumn":int(d.date()==mid_autumn),
            "days_to_mid_autumn": (mid_autumn - d.date()).days,
            "is_ghost_month": int(ghost_s <= d.date() <= ghost_e),
            "is_apr30_bridge": int((m,dd) in {(4,29),(4,30),(5,1),(5,2)}),
        })
    return pd.DataFrame(rows, index=ds)
```

### 11.8 requirements.txt (pin versions April 2026)

```
python>=3.11,<3.13
xgboost>=3.0,<3.3
lightgbm==4.6.0
catboost==1.2.10
mlforecast==1.0.2
neuralforecast==3.1.7
pytorch-forecasting==1.6.1
chronos-forecasting==1.5.1
autogluon.timeseries>=1.5.1
pytorch-lightning>=2.5
torch>=2.1
prophet==1.3.0
lunardate==0.2.2
ruptures==1.1.10
statsmodels==0.14.5
sktime==0.36.0
category_encoders==2.6.4
holidays==0.55
scikit-learn>=1.3
scipy
optuna
pandas>=2.2,<3.0
numpy>=1.26,<2.4
shap
```

---

## 12. Tiêu chí Thành công (Success Criteria)

| Mức | Revenue R² | Revenue MAE | COGS R² | Mô tả |
|---|---|---|---|---|
| **Minimum** | ≥ 0.85 | ≤ 470,000 | ≥ 0.85 | Tier 0+1 thành công, +2% improvement vs v3. Submit `submission_main.csv`. |
| **Target** | ≥ 0.87 | ≤ 440,000 | ≥ 0.87 | Tier 0+1+2 đầy đủ, Ridge meta, conformal. Mục tiêu chính của kế hoạch. |
| **Stretch** | ≥ 0.89 | ≤ 410,000 | ≥ 0.89 | Tier 3 cũng đóng góp (Chronos hoặc TFT blend thành công). |

**Red flags — dừng và rollback**:
- Realistic probe R² giảm > 0.01 so với sau Tier 0
- Bất kỳ OOF MAE tăng > 5% sau một task
- Adversarial AUC > 0.85 (distribution shift nặng — dành thời gian fix trước khi thêm model)
- Kaggle public LB MAE chênh lệch > 20% so với realistic probe MAE
- Submission bị NaN / negative / outlier > 3σ

**Quyết định cuối** (H11:50):

```python
best_submission = (
    "submission_experimental.csv" if (probe_r2_exp >= probe_r2_main + 0.005
                                       and not any_red_flag()) else
    "submission_main.csv"          if (probe_r2_main >= 0.83
                                       and probe_r2_main >= probe_r2_safe + 0.005) else
    "submission_safe.csv"
)
```

---

**Kết thúc PLAN_v4.md.** Tổng thời gian triển khai: 12 giờ. Mục tiêu chính: **Revenue R² 0.87, MAE ≤ 440k**. Ba submission files được tạo để phòng ngừa rủi ro, với rule chọn cuối cùng dựa trên realistic_probe R² (1 fold val_days=548) — con số đáng tin cậy nhất cho private LB.