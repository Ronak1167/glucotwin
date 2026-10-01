# GlucoTwin — DATASET FORENSIC AUDIT REPORT

**Generated**: 2026-09-30 02:07:20  
**Dataset**: HUPA-UCM Diabetes Dataset  
**DOI**: 10.17632/3hbcscwz44.1  
**License**: CC BY 4.0  

---

## ⚡ Executive Summary

| Metric | Value |
|--------|-------|
| Total patient files found | 25 |
| Successfully parsed | 25 |
| Glucose (CGM) available | 25/25 |
| Timestamp available | 25/25 |
| Insulin bolus available | 25/25 |
| Insulin basal available | 25/25 |
| Meals/Carbs available | 25/25 |
| Steps available | 25/25 |
| Heart rate available | 25/25 |
| Sleep available | 0/25 |

**CGM Median Sampling Interval**: 5.0 min (target: 15 min)

---

## 📂 File Structure

All patient files follow naming convention: `HUPA{ID}P.csv`

### All columns found across dataset:
```
basal_rate
bolus_volume_delivered
calories
carb_input
glucose
heart_rate
steps
time
```

---

## 📊 Per-Patient Summary

| Patient | Rows | CGM Readings | Mean Glucose (mg/dL) | % <70 mg/dL | % <54 mg/dL |
|---------|------|-------------|----------------------|-------------|-------------|
| HUPA0001P.csv | — | 4096 | 181.4 | 1.44% | 0.27% |
| HUPA0002P.csv | — | 3181 | 113.5 | 23.86% | 12.92% |
| HUPA0003P.csv | — | 3770 | 143.3 | 7.03% | 0.48% |
| HUPA0004P.csv | — | 3184 | 180.4 | 9.7% | 4.71% |
| HUPA0005P.csv | — | 3858 | 147.7 | 2.95% | 0.49% |
| HUPA0006P.csv | — | 2290 | 165.2 | 8.65% | 1.53% |
| HUPA0007P.csv | — | 3857 | 173.2 | 8.94% | 3.99% |
| HUPA0009P.csv | — | 3812 | 193.7 | 0.0% | 0.0% |
| HUPA0010P.csv | — | 2976 | 135.9 | 8.33% | 0.81% |
| HUPA0011P.csv | — | 3839 | 159.1 | 2.68% | 0.13% |
| HUPA0014P.csv | — | 3829 | 188.1 | 4.05% | 1.41% |
| HUPA0015P.csv | — | 3792 | 163.5 | 4.17% | 1.03% |
| HUPA0016P.csv | — | 3835 | 150.2 | 15.15% | 6.02% |
| HUPA0017P.csv | — | 3599 | 200.8 | 2.39% | 0.61% |
| HUPA0018P.csv | — | 3895 | 143.7 | 17.59% | 10.94% |
| HUPA0019P.csv | — | 3711 | 162.9 | 2.69% | 0.05% |
| HUPA0020P.csv | — | 2862 | 194.3 | 3.56% | 0.31% |
| HUPA0021P.csv | — | 2343 | 137.3 | 4.74% | 0.3% |
| HUPA0022P.csv | — | 4023 | 112.6 | 15.39% | 5.92% |
| HUPA0023P.csv | — | 3919 | 133.1 | 4.31% | 0.92% |
| HUPA0024P.csv | — | 2902 | 166.9 | 7.79% | 2.17% |
| HUPA0025P.csv | — | 4006 | 113.8 | 13.38% | 4.92% |
| HUPA0026P.csv | — | 40605 | 163.0 | 7.71% | 3.09% |
| HUPA0027P.csv | — | 165306 | 130.8 | 6.36% | 0.64% |
| HUPA0028P.csv | — | 25902 | 128.4 | 3.08% | 0.63% |

---

## 🔬 Signal Details

### Glucose (CGM — FreeStyle Libre 2)
- **Unit**: Determined from column name / sample values (see per-patient data)
- **Nominal interval**: 15 minutes (FreeStyle Libre 2 standard)
- **Actual median interval**: 5.0 min

### Insulin
- Bolus insulin available in 25/25 patients
- Basal insulin available in 25/25 patients

### Meals (Carbohydrates)
- Available in 25/25 patients

### Activity (Steps, HR)
- Steps available in 25/25 patients
- Heart rate available in 25/25 patients

### Sleep
- Sleep data available in 0/25 patients
- Sleep stage resolution: **See per-patient audit** (may be quality/quantity only)

---

## ⚠️ Data Quality Observations

> [TODO: Updated after running full audit — add specific findings here]

---

## ✅ Column Mapping (for config/data.yaml)

> [TODO: Populate after confirming column names below]

```yaml
column_map:
  timestamp: "FILL_AFTER_AUDIT"
  glucose_mgdl: "FILL_AFTER_AUDIT"
  insulin_basal: "FILL_AFTER_AUDIT"
  insulin_bolus: "FILL_AFTER_AUDIT"
  carbs_g: "FILL_AFTER_AUDIT"
  steps: "FILL_AFTER_AUDIT"
  calories: "FILL_AFTER_AUDIT"
  heart_rate: "FILL_AFTER_AUDIT"
  sleep_seconds: "FILL_AFTER_AUDIT"
  sleep_stage: "FILL_AFTER_AUDIT"
```

---

## 📋 Detailed Per-Patient Data

### ✅ HUPA0001P.csv
- Rows: 4096
- Columns: 8
- Column names: `time, glucose, calories, heart_rate, steps, basal_rate, bolus_volume_delivered, carb_input`
- Signals detected:
  - **glucose**: column=`glucose`, n=4096, mean=181.4 mg/dL, <70: 1.44%
  - **timestamp**: column=`time`, interval=5.0 min
  - **insulin_basal**: column=`basal_rate`, n_valid=4096
  - **insulin_bolus**: column=`bolus_volume_delivered`, n_valid=4096
  - **carbs**: column=`carb_input`, n_valid=4096
  - **steps**: column=`steps`, n_valid=4096
  - **heart_rate**: column=`heart_rate`, n_valid=4096

### ✅ HUPA0002P.csv
- Rows: 3181
- Columns: 8
- Column names: `time, glucose, calories, heart_rate, steps, basal_rate, bolus_volume_delivered, carb_input`
- Signals detected:
  - **glucose**: column=`glucose`, n=3181, mean=113.5 mg/dL, <70: 23.86%
  - **timestamp**: column=`time`, interval=5.0 min
  - **insulin_basal**: column=`basal_rate`, n_valid=3181
  - **insulin_bolus**: column=`bolus_volume_delivered`, n_valid=3181
  - **carbs**: column=`carb_input`, n_valid=3181
  - **steps**: column=`steps`, n_valid=3181
  - **heart_rate**: column=`heart_rate`, n_valid=3181

### ✅ HUPA0003P.csv
- Rows: 3770
- Columns: 8
- Column names: `time, glucose, calories, heart_rate, steps, basal_rate, bolus_volume_delivered, carb_input`
- Signals detected:
  - **glucose**: column=`glucose`, n=3770, mean=143.3 mg/dL, <70: 7.03%
  - **timestamp**: column=`time`, interval=5.0 min
  - **insulin_basal**: column=`basal_rate`, n_valid=3770
  - **insulin_bolus**: column=`bolus_volume_delivered`, n_valid=3770
  - **carbs**: column=`carb_input`, n_valid=3770
  - **steps**: column=`steps`, n_valid=3770
  - **heart_rate**: column=`heart_rate`, n_valid=3770

### ✅ HUPA0004P.csv
- Rows: 3184
- Columns: 8
- Column names: `time, glucose, calories, heart_rate, steps, basal_rate, bolus_volume_delivered, carb_input`
- Signals detected:
  - **glucose**: column=`glucose`, n=3184, mean=180.4 mg/dL, <70: 9.7%
  - **timestamp**: column=`time`, interval=5.0 min
  - **insulin_basal**: column=`basal_rate`, n_valid=3184
  - **insulin_bolus**: column=`bolus_volume_delivered`, n_valid=3184
  - **carbs**: column=`carb_input`, n_valid=3184
  - **steps**: column=`steps`, n_valid=3184
  - **heart_rate**: column=`heart_rate`, n_valid=3184

### ✅ HUPA0005P.csv
- Rows: 3858
- Columns: 8
- Column names: `time, glucose, calories, heart_rate, steps, basal_rate, bolus_volume_delivered, carb_input`
- Signals detected:
  - **glucose**: column=`glucose`, n=3858, mean=147.7 mg/dL, <70: 2.95%
  - **timestamp**: column=`time`, interval=5.0 min
  - **insulin_basal**: column=`basal_rate`, n_valid=3858
  - **insulin_bolus**: column=`bolus_volume_delivered`, n_valid=3858
  - **carbs**: column=`carb_input`, n_valid=3858
  - **steps**: column=`steps`, n_valid=3858
  - **heart_rate**: column=`heart_rate`, n_valid=3858

### ✅ HUPA0006P.csv
- Rows: 2290
- Columns: 8
- Column names: `time, glucose, calories, heart_rate, steps, basal_rate, bolus_volume_delivered, carb_input`
- Signals detected:
  - **glucose**: column=`glucose`, n=2290, mean=165.2 mg/dL, <70: 8.65%
  - **timestamp**: column=`time`, interval=5.0 min
  - **insulin_basal**: column=`basal_rate`, n_valid=2290
  - **insulin_bolus**: column=`bolus_volume_delivered`, n_valid=2290
  - **carbs**: column=`carb_input`, n_valid=2290
  - **steps**: column=`steps`, n_valid=2290
  - **heart_rate**: column=`heart_rate`, n_valid=2290

### ✅ HUPA0007P.csv
- Rows: 3857
- Columns: 8
- Column names: `time, glucose, calories, heart_rate, steps, basal_rate, bolus_volume_delivered, carb_input`
- Signals detected:
  - **glucose**: column=`glucose`, n=3857, mean=173.2 mg/dL, <70: 8.94%
  - **timestamp**: column=`time`, interval=5.0 min
  - **insulin_basal**: column=`basal_rate`, n_valid=3857
  - **insulin_bolus**: column=`bolus_volume_delivered`, n_valid=3857
  - **carbs**: column=`carb_input`, n_valid=3857
  - **steps**: column=`steps`, n_valid=3857
  - **heart_rate**: column=`heart_rate`, n_valid=3857

### ✅ HUPA0009P.csv
- Rows: 3812
- Columns: 8
- Column names: `time, glucose, calories, heart_rate, steps, basal_rate, bolus_volume_delivered, carb_input`
- Signals detected:
  - **glucose**: column=`glucose`, n=3812, mean=193.7 mg/dL, <70: 0.0%
  - **timestamp**: column=`time`, interval=5.0 min
  - **insulin_basal**: column=`basal_rate`, n_valid=3812
  - **insulin_bolus**: column=`bolus_volume_delivered`, n_valid=3812
  - **carbs**: column=`carb_input`, n_valid=3812
  - **steps**: column=`steps`, n_valid=3812
  - **heart_rate**: column=`heart_rate`, n_valid=3812

### ✅ HUPA0010P.csv
- Rows: 2976
- Columns: 8
- Column names: `time, glucose, calories, heart_rate, steps, basal_rate, bolus_volume_delivered, carb_input`
- Signals detected:
  - **glucose**: column=`glucose`, n=2976, mean=135.9 mg/dL, <70: 8.33%
  - **timestamp**: column=`time`, interval=5.0 min
  - **insulin_basal**: column=`basal_rate`, n_valid=2976
  - **insulin_bolus**: column=`bolus_volume_delivered`, n_valid=2976
  - **carbs**: column=`carb_input`, n_valid=2976
  - **steps**: column=`steps`, n_valid=2976
  - **heart_rate**: column=`heart_rate`, n_valid=2976

### ✅ HUPA0011P.csv
- Rows: 3839
- Columns: 8
- Column names: `time, glucose, calories, heart_rate, steps, basal_rate, bolus_volume_delivered, carb_input`
- Signals detected:
  - **glucose**: column=`glucose`, n=3839, mean=159.1 mg/dL, <70: 2.68%
  - **timestamp**: column=`time`, interval=5.0 min
  - **insulin_basal**: column=`basal_rate`, n_valid=3839
  - **insulin_bolus**: column=`bolus_volume_delivered`, n_valid=3839
  - **carbs**: column=`carb_input`, n_valid=3839
  - **steps**: column=`steps`, n_valid=3839
  - **heart_rate**: column=`heart_rate`, n_valid=3839

### ✅ HUPA0014P.csv
- Rows: 3829
- Columns: 8
- Column names: `time, glucose, calories, heart_rate, steps, basal_rate, bolus_volume_delivered, carb_input`
- Signals detected:
  - **glucose**: column=`glucose`, n=3829, mean=188.1 mg/dL, <70: 4.05%
  - **timestamp**: column=`time`, interval=5.0 min
  - **insulin_basal**: column=`basal_rate`, n_valid=3829
  - **insulin_bolus**: column=`bolus_volume_delivered`, n_valid=3829
  - **carbs**: column=`carb_input`, n_valid=3829
  - **steps**: column=`steps`, n_valid=3829
  - **heart_rate**: column=`heart_rate`, n_valid=3829

### ✅ HUPA0015P.csv
- Rows: 3792
- Columns: 8
- Column names: `time, glucose, calories, heart_rate, steps, basal_rate, bolus_volume_delivered, carb_input`
- Signals detected:
  - **glucose**: column=`glucose`, n=3792, mean=163.5 mg/dL, <70: 4.17%
  - **timestamp**: column=`time`, interval=5.0 min
  - **insulin_basal**: column=`basal_rate`, n_valid=3792
  - **insulin_bolus**: column=`bolus_volume_delivered`, n_valid=3792
  - **carbs**: column=`carb_input`, n_valid=3792
  - **steps**: column=`steps`, n_valid=3792
  - **heart_rate**: column=`heart_rate`, n_valid=3792

### ✅ HUPA0016P.csv
- Rows: 3835
- Columns: 8
- Column names: `time, glucose, calories, heart_rate, steps, basal_rate, bolus_volume_delivered, carb_input`
- Signals detected:
  - **glucose**: column=`glucose`, n=3835, mean=150.2 mg/dL, <70: 15.15%
  - **timestamp**: column=`time`, interval=5.0 min
  - **insulin_basal**: column=`basal_rate`, n_valid=3835
  - **insulin_bolus**: column=`bolus_volume_delivered`, n_valid=3835
  - **carbs**: column=`carb_input`, n_valid=3835
  - **steps**: column=`steps`, n_valid=3835
  - **heart_rate**: column=`heart_rate`, n_valid=3835

### ✅ HUPA0017P.csv
- Rows: 3599
- Columns: 8
- Column names: `time, glucose, calories, heart_rate, steps, basal_rate, bolus_volume_delivered, carb_input`
- Signals detected:
  - **glucose**: column=`glucose`, n=3599, mean=200.8 mg/dL, <70: 2.39%
  - **timestamp**: column=`time`, interval=5.0 min
  - **insulin_basal**: column=`basal_rate`, n_valid=3599
  - **insulin_bolus**: column=`bolus_volume_delivered`, n_valid=3599
  - **carbs**: column=`carb_input`, n_valid=3599
  - **steps**: column=`steps`, n_valid=3599
  - **heart_rate**: column=`heart_rate`, n_valid=3599

### ✅ HUPA0018P.csv
- Rows: 3895
- Columns: 8
- Column names: `time, glucose, calories, heart_rate, steps, basal_rate, bolus_volume_delivered, carb_input`
- Signals detected:
  - **glucose**: column=`glucose`, n=3895, mean=143.7 mg/dL, <70: 17.59%
  - **timestamp**: column=`time`, interval=5.0 min
  - **insulin_basal**: column=`basal_rate`, n_valid=3895
  - **insulin_bolus**: column=`bolus_volume_delivered`, n_valid=3895
  - **carbs**: column=`carb_input`, n_valid=3895
  - **steps**: column=`steps`, n_valid=3895
  - **heart_rate**: column=`heart_rate`, n_valid=3895

### ✅ HUPA0019P.csv
- Rows: 3711
- Columns: 8
- Column names: `time, glucose, calories, heart_rate, steps, basal_rate, bolus_volume_delivered, carb_input`
- Signals detected:
  - **glucose**: column=`glucose`, n=3711, mean=162.9 mg/dL, <70: 2.69%
  - **timestamp**: column=`time`, interval=5.0 min
  - **insulin_basal**: column=`basal_rate`, n_valid=3711
  - **insulin_bolus**: column=`bolus_volume_delivered`, n_valid=3711
  - **carbs**: column=`carb_input`, n_valid=3711
  - **steps**: column=`steps`, n_valid=3711
  - **heart_rate**: column=`heart_rate`, n_valid=3711

### ✅ HUPA0020P.csv
- Rows: 2862
- Columns: 8
- Column names: `time, glucose, calories, heart_rate, steps, basal_rate, bolus_volume_delivered, carb_input`
- Signals detected:
  - **glucose**: column=`glucose`, n=2862, mean=194.3 mg/dL, <70: 3.56%
  - **timestamp**: column=`time`, interval=5.0 min
  - **insulin_basal**: column=`basal_rate`, n_valid=2862
  - **insulin_bolus**: column=`bolus_volume_delivered`, n_valid=2862
  - **carbs**: column=`carb_input`, n_valid=2862
  - **steps**: column=`steps`, n_valid=2862
  - **heart_rate**: column=`heart_rate`, n_valid=2862

### ✅ HUPA0021P.csv
- Rows: 2343
- Columns: 8
- Column names: `time, glucose, calories, heart_rate, steps, basal_rate, bolus_volume_delivered, carb_input`
- Signals detected:
  - **glucose**: column=`glucose`, n=2343, mean=137.3 mg/dL, <70: 4.74%
  - **timestamp**: column=`time`, interval=5.0 min
  - **insulin_basal**: column=`basal_rate`, n_valid=2343
  - **insulin_bolus**: column=`bolus_volume_delivered`, n_valid=2343
  - **carbs**: column=`carb_input`, n_valid=2343
  - **steps**: column=`steps`, n_valid=2343
  - **heart_rate**: column=`heart_rate`, n_valid=2343

### ✅ HUPA0022P.csv
- Rows: 4023
- Columns: 8
- Column names: `time, glucose, calories, heart_rate, steps, basal_rate, bolus_volume_delivered, carb_input`
- Signals detected:
  - **glucose**: column=`glucose`, n=4023, mean=112.6 mg/dL, <70: 15.39%
  - **timestamp**: column=`time`, interval=5.0 min
  - **insulin_basal**: column=`basal_rate`, n_valid=4023
  - **insulin_bolus**: column=`bolus_volume_delivered`, n_valid=4023
  - **carbs**: column=`carb_input`, n_valid=4023
  - **steps**: column=`steps`, n_valid=4023
  - **heart_rate**: column=`heart_rate`, n_valid=4023

### ✅ HUPA0023P.csv
- Rows: 3919
- Columns: 8
- Column names: `time, glucose, calories, heart_rate, steps, basal_rate, bolus_volume_delivered, carb_input`
- Signals detected:
  - **glucose**: column=`glucose`, n=3919, mean=133.1 mg/dL, <70: 4.31%
  - **timestamp**: column=`time`, interval=5.0 min
  - **insulin_basal**: column=`basal_rate`, n_valid=3919
  - **insulin_bolus**: column=`bolus_volume_delivered`, n_valid=3919
  - **carbs**: column=`carb_input`, n_valid=3919
  - **steps**: column=`steps`, n_valid=3919
  - **heart_rate**: column=`heart_rate`, n_valid=3919

### ✅ HUPA0024P.csv
- Rows: 2902
- Columns: 8
- Column names: `time, glucose, calories, heart_rate, steps, basal_rate, bolus_volume_delivered, carb_input`
- Signals detected:
  - **glucose**: column=`glucose`, n=2902, mean=166.9 mg/dL, <70: 7.79%
  - **timestamp**: column=`time`, interval=5.0 min
  - **insulin_basal**: column=`basal_rate`, n_valid=2902
  - **insulin_bolus**: column=`bolus_volume_delivered`, n_valid=2902
  - **carbs**: column=`carb_input`, n_valid=2902
  - **steps**: column=`steps`, n_valid=2902
  - **heart_rate**: column=`heart_rate`, n_valid=2902

### ✅ HUPA0025P.csv
- Rows: 4006
- Columns: 8
- Column names: `time, glucose, calories, heart_rate, steps, basal_rate, bolus_volume_delivered, carb_input`
- Signals detected:
  - **glucose**: column=`glucose`, n=4006, mean=113.8 mg/dL, <70: 13.38%
  - **timestamp**: column=`time`, interval=5.0 min
  - **insulin_basal**: column=`basal_rate`, n_valid=4006
  - **insulin_bolus**: column=`bolus_volume_delivered`, n_valid=4006
  - **carbs**: column=`carb_input`, n_valid=4006
  - **steps**: column=`steps`, n_valid=4006
  - **heart_rate**: column=`heart_rate`, n_valid=4006

### ✅ HUPA0026P.csv
- Rows: 40605
- Columns: 8
- Column names: `time, glucose, calories, heart_rate, steps, basal_rate, bolus_volume_delivered, carb_input`
- Signals detected:
  - **glucose**: column=`glucose`, n=40605, mean=163.0 mg/dL, <70: 7.71%
  - **timestamp**: column=`time`, interval=5.0 min
  - **insulin_basal**: column=`basal_rate`, n_valid=40605
  - **insulin_bolus**: column=`bolus_volume_delivered`, n_valid=40605
  - **carbs**: column=`carb_input`, n_valid=40605
  - **steps**: column=`steps`, n_valid=40605
  - **heart_rate**: column=`heart_rate`, n_valid=40605

### ✅ HUPA0027P.csv
- Rows: 165306
- Columns: 8
- Column names: `time, glucose, calories, heart_rate, steps, basal_rate, bolus_volume_delivered, carb_input`
- Signals detected:
  - **glucose**: column=`glucose`, n=165306, mean=130.8 mg/dL, <70: 6.36%
  - **timestamp**: column=`time`, interval=5.0 min
  - **insulin_basal**: column=`basal_rate`, n_valid=165306
  - **insulin_bolus**: column=`bolus_volume_delivered`, n_valid=165306
  - **carbs**: column=`carb_input`, n_valid=165306
  - **steps**: column=`steps`, n_valid=165306
  - **heart_rate**: column=`heart_rate`, n_valid=165306

### ✅ HUPA0028P.csv
- Rows: 25902
- Columns: 8
- Column names: `time, glucose, calories, heart_rate, steps, basal_rate, bolus_volume_delivered, carb_input`
- Signals detected:
  - **glucose**: column=`glucose`, n=25902, mean=128.4 mg/dL, <70: 3.08%
  - **timestamp**: column=`time`, interval=5.0 min
  - **insulin_basal**: column=`basal_rate`, n_valid=25902
  - **insulin_bolus**: column=`bolus_volume_delivered`, n_valid=25902
  - **carbs**: column=`carb_input`, n_valid=25902
  - **steps**: column=`steps`, n_valid=25902
  - **heart_rate**: column=`heart_rate`, n_valid=25902

---

*Generated by GlucoTwin dataset_audit.py*  
*DO NOT commit this file with patient-identifiable data — patient IDs are already anonymized in HUPA-UCM.*
