# Parallel Battery Analytics Dashboard

Ứng dụng Streamlit để phân tích **parallel-connected lithium-ion battery modules** cho nghiên cứu học thuật và phát triển mô hình data-driven / explainable AI.

## Mục tiêu chính

- Forecast **current imbalance** của từng cell hoặc các chỉ số imbalance tổng hợp.
- Forecast **thermal / temperature-related responses** như `sigma_T_mean`, `delta_T_max`, `temp_peak`, `TTSB`.
- Dự báo **SoH proxy / degradation risk** và **relative lifetime index** khi dataset chưa có long-term SoH/RUL chuẩn.
- Hỗ trợ **explainability** bằng feature importance, SHAP, PDP.
- Hỗ trợ **scenario simulation** cho các biến như operating temperature, interconnection resistance, chemistry mix, aged/unaged mismatch.

---

## Cấu trúc project

```text
parallel_battery_app/
├── app.py
├── requirements.txt
├── README.md
└── src/
    ├── __init__.py
    ├── data_loader.py
    ├── preprocessing.py
    ├── feature_engineering.py
    ├── modeling.py
    ├── inference.py
    ├── explainability.py
    ├── visualization.py
    └── utils.py
```

---

## Yêu cầu môi trường

- Python 3.10+
- Khuyến nghị tạo virtual environment riêng

## Cài đặt

```bash
cd parallel_battery_app
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate         # Windows
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Chạy app

```bash
streamlit run app.py
```

Sau đó nhập đường dẫn dataset thật ở sidebar, ví dụ:

- thư mục chứa nhiều file `.xlsx`, `.csv`, `.mat`
- hoặc một file cụ thể

Ví dụ:

```text
/data/parallel_cells_dataset/
```

hoặc

```text
/data/parallel_cells_dataset/module_timeseries.xlsx
```

---

## Dữ liệu đầu vào mong đợi

App được thiết kế để làm việc với các file:

- CSV
- XLSX / XLS
- MATLAB `.mat`

### 1. Timeseries table mong đợi

Nên có tối thiểu:

- cột thời gian: `time`, `timestamp`, `elapsed_time`, ...
- dòng từng cell: ví dụ `i_cell_1`, `i_cell_2`, `i_cell_3`, `i_cell_4`
- nhiệt độ từng cell: ví dụ `t_cell_1`, `t_cell_2`, `t_cell_3`, `t_cell_4`
- module current / voltage nếu có
- metadata nếu có:
  - `test_id`
  - `module_id`
  - `chemistry`
  - `ageing`
  - `operating_temperature`
  - `interconnection_resistance`

### 2. Characterization table mong đợi

Có thể chứa:

- `capacity`
- `ohmic_resistance`
- `r0_soc_10`, `r0_soc_20`, ...
- OCV-related columns
- `chemistry`
- `ageing`
- `cell_id`

### 3. Naming flexibility

App **không yêu cầu tên cột cố định tuyệt đối**. Nó dùng heuristic để suy luận:

- current columns
- temperature columns
- metadata columns
- capacity / resistance / OCV-related columns

Tuy vậy, càng đặt tên rõ ràng thì app càng map chính xác.

---

## Tính năng chính của UI

### Tab 1 — Overview
- catalog file/tables
- preview dataset
- missing values summary
- distribution của chemistry, ageing, operating temperature, interconnection resistance

### Tab 2 — Cell Characterization
- capacity distribution
- resistance distribution
- OCV-related curves nếu có
- so sánh aged/unaged và NMC/NCA nếu metadata có sẵn

### Tab 3 — Current Imbalance Analysis
- current time-series từng cell
- temperature time-series từng cell
- các metric như:
  - `sigma_I_start`
  - `sigma_I_mid`
  - `sigma_I_end`
  - `delta_SoC_max`
  - `delta_T_max`
  - `sigma_T_mean`
  - `TTSB`
- heatmap correlation

### Tab 4 — Forecast Temperature / Thermal
- train model cho thermal target
- target ví dụ:
  - `sigma_T_mean`
  - `delta_T_max`
  - `temp_peak`
  - `TTSB`
- hỗ trợ Linear / Ridge / RF / XGBoost

### Tab 5 — Forecast Current Imbalance
- train model cho imbalance target
- metric: MAE, RMSE, R²
- actual vs predicted
- residual plot
- feature importance

### Tab 6 — SoH / Degradation Risk
- nếu có nhãn SoH/RUL thì train regression
- nếu **không có nhãn SoH tuyệt đối**, app dùng:
  - `degradation_risk_score`
  - `relative_lifetime_index`
  - `estimated_cycle_life_band`

### Tab 7 — Explainability
- feature importance
- SHAP summary
- SHAP dependence
- PDP
- auto-generated text explanation

### Tab 8 — Scenario Simulator
Cho thay đổi:
- operating temperature
- interconnection resistance
- chemistry
- aged / unaged
- capacity dispersion proxy
- resistance dispersion proxy

Sau đó app sẽ:
- tính degradation risk
- ước lượng relative lifetime index
- nếu có model đã train, dự đoán thêm target tương ứng
- đưa recommendation

### Tab 9 — Export
- export engineered features CSV
- export HTML report
- hỗ trợ PNG nếu môi trường có `kaleido`

---

## Feature engineering đã implement

Từ raw time-series, app sinh ra các nhóm feature:

### Current features
- mean / std / max / min / range / slope / AUC cho từng cell current
- pairwise current differences
- `sigma_i_start`, `sigma_i_mid`, `sigma_i_end`
- `current_spread_mean`, `current_spread_max`

### Thermal features
- `sigma_t_start`, `sigma_t_mid`, `sigma_t_end`
- `sigma_t_mean`
- `delta_t_start`, `delta_t_mid`, `delta_t_end`
- `delta_t_max`
- `temp_peak`, `temp_rise`
- module temperature gradient AUC
- rolling statistics

### SoC proxy features
- Coulomb-counting based `delta_soc_max`, `delta_soc_end`, `sigma_soc_mean`

### Module-level features
- module current / voltage summary
- metadata features như chemistry, ageing, operating temperature, interconnection resistance

### Risk / lifetime proxy
- `degradation_risk_score`
- `relative_lifetime_index`
- `estimated_cycle_life_band`

---

## Ghi chú quan trọng về SoH và tuổi thọ pin

Dataset parallel-module thực nghiệm kiểu này thường rất mạnh cho:

- **current imbalance forecasting**
- **thermal gradient forecasting**
- **stress/risk estimation**

Nhưng thường **không đủ long-term cycle labels** để forecast **SoH tuyệt đối** hoặc **RUL tuyệt đối** một cách nghiêm ngặt.

Vì vậy app implement hai chế độ:

### Chế độ 1 — Có nhãn SoH/RUL
App sẽ train supervised regression.

### Chế độ 2 — Không có nhãn SoH/RUL
App sẽ dùng **proxy-based degradation risk** và **relative lifetime index**.

Điều này phù hợp với hướng nghiên cứu thực tế: dùng imbalance + nhiệt độ + nội trở + mismatch chemistry / ageing để suy ra rủi ro suy thoái và mức tuổi thọ tương đối.

---

## Cách thay dataset thật

1. Chuẩn bị thư mục chứa file thật.
2. Chạy app.
3. Dán path vào sidebar.
4. Chọn đúng bảng `timeseries` và `characterization`.
5. App sẽ tự sinh feature table.
6. Train model theo target mong muốn.

Nếu dataset thật dùng tên cột khác nhiều so với giả định heuristic:
- chỉnh file `src/utils.py` trong phần alias/regex
- hoặc chuẩn hóa tên cột trước khi nạp vào app

---

## Hướng mở rộng nên làm tiếp

- thêm **LSTM / GRU / TCN** cho sequence forecasting thật sự
- thêm **digital twin coupling** để sinh nhãn tuổi thọ dài hạn
- thêm **cell arrangement / regrouping optimizer**
- thêm **uncertainty quantification**
- thêm **group-aware benchmarking** theo điều kiện DoE
- thêm export report PDF hoàn chỉnh

---

## Troubleshooting

### 1. App báo không tìm thấy time/current columns
- Kiểm tra tên cột trong file timeseries
- Chuẩn hóa về dạng như `time`, `i_cell_1`, `i_cell_2`, ...

### 2. MAT file đọc không trọn vẹn
- Một số `.mat` phức tạp cần map thủ công
- Có thể convert `.mat` sang `.csv/.xlsx` rồi nạp lại

### 3. XGBoost / SHAP lỗi cài đặt
- tạm thời dùng Ridge hoặc Random Forest
- SHAP không bắt buộc để app chạy

### 4. Export PNG lỗi
- cài `kaleido`
- hoặc dùng export HTML thay thế

---

## Cách chạy nhanh

```bash
pip install -r requirements.txt
streamlit run app.py
```

