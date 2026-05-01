# Datathon VinU

Repository này chứa mã nguồn, notebook, báo cáo và file submission cho bài Datathon.

## Cấu trúc thư mục

```text
.
├── MCQ/
│   ├── Datathon.ipynb
│   ├── sales.csv
│   ├── sample_submission.csv
│   └── các file dữ liệu gốc khác
├── source/
│   └── datathon-v5-hybrid-upgrade-best.ipynb
├── output/
│   └── submission_v5_recommended.csv
├── flow/
│   ├── forecast_pipeline.mmd
│   └── forecast_pipeline.svg
├── Power Pi/
│   └── [VINUNI] - DASHBOARD.pbix
└── docs/
    └── neurips/
```

Trong đó:

- `source/datathon-v5-hybrid-upgrade-best.ipynb`: notebook chính để train model và sinh file submission.
- `MCQ/`: chứa dữ liệu đầu vào và notebook phân tích dữ liệu ban đầu.
- `output/submission_v5_recommended.csv`: file submission cuối cùng được dùng để nộp.
- `flow/`: sơ đồ pipeline mô hình.
- `Power Pi/`: dashboard Power BI.
- `docs/neurips/`: source báo cáo dạng LaTeX.

## Mô tả source

Notebook chính nằm tại:

```text
source/datathon-v5-hybrid-upgrade-best.ipynb
```

Pipeline trong notebook gồm các bước chính:

1. Đọc dữ liệu `sales.csv`, `sample_submission.csv` và `promotions.csv` nếu có.
2. Chuẩn hóa ngày tháng, tạo target `Revenue`, `COGS` và các đặc trưng thời gian.
3. Tạo đặc trưng calendar, Fourier, lag, rolling và promotion theo nguyên tắc không dùng dữ liệu tương lai.
4. Chia validation theo rolling-origin / expanding-window.
5. Train các mô hình XGBoost cho `Revenue`, `COGS_direct` và `COGS_ratio`.
6. Blend, calibration, reconcile `COGS`, sau đó ghi file submission.

## Cách chạy lại kết quả trên Kaggle

Khuyến nghị chạy notebook trên Kaggle để có môi trường ổn định và GPU P100.

1. Vào Kaggle và tạo một Notebook mới.
2. Upload notebook `source/datathon-v5-hybrid-upgrade-best.ipynb` lên Kaggle.
3. Upload dữ liệu trong thư mục `MCQ/` lên Kaggle Dataset hoặc add dataset vào Notebook. Cần đảm bảo có tối thiểu các file:
   - `sales.csv`
   - `sample_submission.csv`
   - `promotions.csv` nếu sử dụng
4. Trong phần Notebook Settings của Kaggle, bật accelerator:
   - `GPU`: `P100`
5. Chạy toàn bộ notebook bằng `Run All`.
6. Sau khi chạy xong, notebook sẽ ghi kết quả vào thư mục làm việc của Kaggle:

```text
/kaggle/working/submission_v5_recommended.csv
```

File cần tải về và dùng để nộp là:

```text
submission_v5_recommended.csv
```

Trong repository này, file kết quả tương ứng đã được lưu tại:

```text
output/submission_v5_recommended.csv
```

## Kết quả đầu ra

File submission cuối cùng:

```text
output/submission_v5_recommended.csv
```

Đây là file được sinh ra từ notebook chính và là file cần nộp lên hệ thống chấm.
