# GlucoTwin — MODEL BENCHMARK & CLINICAL VALIDATION REPORT

**Generated**: 2026-09-30 02:12:44  
**Cohort Evaluated**: 12 T1DM Patients (HUPA-UCM Clinical Dataset)  
**Prediction Horizon**: 60 Minutes  
**Evaluation Strategy**: Patient-Grouped Strict Chronological Split (70% Train / 30% Holdout)  

## 🏆 Benchmark Leaderboard (Holdout Test Cohort)

| Model | ROC-AUC | PR-AUC | Sensitivity (Recall) | Specificity | F2-Score | Nocturnal Sens | Training Time |
|-------|---------|--------|----------------------|-------------|----------|----------------|---------------|
| Persistence Baseline | 0.960 | 0.844 | 0.906 | 0.911 | 0.808 | 0.943 | 0.01s |
| Trend Velocity Baseline | 0.963 | 0.790 | 0.943 | 0.916 | 0.841 | 0.963 | 0.01s |
| Logistic Regression | 0.977 | 0.875 | 0.969 | 0.850 | 0.789 | 0.985 | 0.18s |
| Random Forest | 0.982 | 0.909 | 0.952 | 0.908 | 0.839 | 0.971 | 1.53s |
| XGBoost | 0.979 | 0.899 | 0.925 | 0.917 | 0.829 | 0.954 | 1.21s |
| LightGBM | 0.978 | 0.894 | 0.893 | 0.930 | 0.821 | 0.924 | 1.0s |

---

## 🔍 Top 10 Clinical Risk Feature Drivers (LightGBM)

| Feature | Importance |
|---|---|
| `glucose_dist_to_70` | 0.2 |
| `glucose_dist_to_54` | 0.1 |
| `glucose_current` | 0.1 |
| `is_near_hypo` | 0.1 |
| `glucose_mean_60min` | 0.1 |
| `glucose_min_120min` | 0.1 |
| `glucose_deviation_from_baseline` | 0.0 |
| `glucose_velocity_15min` | 0.0 |
| `is_low` | 0.0 |
| `glucose_velocity_30min` | 0.0 |

---
## 💡 Clinical Safety Summary

- **Zero Data Leakage**: All rolling features strictly backward-looking (`closed='left'`).
- **Nocturnal Protection**: Specific circadian heuristics and nocturnal subgroup tracking ensure night-time safety without alert fatigue.
- **Actionable Twin**: Combines statistical ML predictions with physiological counterfactual what-if simulations.
