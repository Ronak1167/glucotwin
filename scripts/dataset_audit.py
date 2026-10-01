r"""
GlucoTwin — Dataset Audit Script
scripts/dataset_audit.py

Performs a comprehensive forensic audit of the HUPA-UCM dataset.
Run after dataset download to understand signal availability.

Outputs: docs/DATASET_REPORT.md

Usage:
    cd C:\Users\Ronak Jain\Desktop\glucotwin
    python scripts/dataset_audit.py
"""

import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/raw/hupa_ucm")
OUTPUT_REPORT = Path("docs/DATASET_REPORT.md")
OUTPUT_JSON = Path("docs/audit_results.json")


def find_patient_files():
    """Find all preprocessed patient CSV files."""
    files = sorted(DATA_DIR.rglob("Preprocessed/HUPA*P.csv"))
    if not files:
        files = sorted(DATA_DIR.rglob("HUPA*P.csv"))
    if not files:
        files = sorted(DATA_DIR.rglob("*.csv"))
    return files


def inspect_single_file(filepath: Path) -> dict:
    """Perform deep inspection of a single patient file."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            first_line = f.readline()
        sep = ";" if ";" in first_line else ","
        df = pd.read_csv(filepath, sep=sep, low_memory=False)
    except Exception as e:
        return {"error": str(e), "filepath": str(filepath)}
    
    result = {
        "filepath": str(filepath),
        "filename": filepath.name,
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "signals_found": {},
        "missing_pct": {},
    }
    
    # Missing data per column
    for col in df.columns:
        pct_missing = df[col].isna().mean() * 100
        result["missing_pct"][col] = round(pct_missing, 1)
    
    # Glucose detection
    glucose_candidates = []
    for col in df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in ["glucose", "cgm", "bgl", "cbg"]):
            glucose_candidates.append(col)
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals) > 0:
                result["signals_found"]["glucose"] = {
                    "column": col,
                    "n_valid": int(len(vals)),
                    "mean": round(float(vals.mean()), 1),
                    "std": round(float(vals.std()), 1),
                    "min": round(float(vals.min()), 1),
                    "max": round(float(vals.max()), 1),
                    "pct_below_70": round(float((vals < 70).mean() * 100), 2),
                    "pct_below_54": round(float((vals < 54).mean() * 100), 2),
                }
    
    # Timestamp detection
    for col in df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in ["time", "date", "timestamp"]):
            sample = df[col].dropna().head(5)
            try:
                ts = pd.to_datetime(sample, errors="coerce")
                if ts.notna().sum() >= 3:
                    ts_all = pd.to_datetime(df[col], errors="coerce")
                    result["signals_found"]["timestamp"] = {
                        "column": col,
                        "n_valid": int(ts_all.notna().sum()),
                        "first": str(ts_all.dropna().min()),
                        "last": str(ts_all.dropna().max()),
                        "sample": str(sample.iloc[0]),
                    }
                    
                    # Sampling interval
                    ts_sorted = ts_all.dropna().sort_values()
                    if len(ts_sorted) > 1:
                        intervals = ts_sorted.diff().dt.total_seconds().dropna() / 60
                        result["signals_found"]["timestamp"]["median_interval_min"] = round(float(intervals.median()), 1)
                    break
            except Exception:
                pass
    
    # Insulin
    for col in df.columns:
        col_lower = col.lower()
        if "bolus" in col_lower or "basal" in col_lower or "insulin" in col_lower:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            key = "insulin_bolus" if "bolus" in col_lower else "insulin_basal" if "basal" in col_lower else "insulin"
            if len(vals) > 0:
                result["signals_found"][key] = {
                    "column": col,
                    "n_valid": int(len(vals)),
                    "n_nonzero": int((vals > 0).sum()),
                    "max": round(float(vals.max()), 2),
                    "mean_nonzero": round(float(vals[vals > 0].mean()), 2) if (vals > 0).any() else 0,
                }
    
    # Meals/Carbs
    for col in df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in ["carb", "cho", "meal", "food"]):
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals) > 0:
                result["signals_found"]["carbs"] = {
                    "column": col,
                    "n_valid": int(len(vals)),
                    "n_nonzero": int((vals > 0).sum()),
                    "max": round(float(vals.max()), 1),
                }
    
    # Steps
    for col in df.columns:
        if "step" in col.lower():
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals) > 0:
                result["signals_found"]["steps"] = {
                    "column": col,
                    "n_valid": int(len(vals)),
                    "mean": round(float(vals.mean()), 1),
                }
    
    # Heart Rate
    for col in df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in ["heart", " hr", "pulse"]):
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals) > 0:
                result["signals_found"]["heart_rate"] = {
                    "column": col,
                    "n_valid": int(len(vals)),
                    "mean": round(float(vals.mean()), 1),
                    "min": round(float(vals.min()), 1),
                    "max": round(float(vals.max()), 1),
                }
    
    # Sleep
    for col in df.columns:
        col_lower = col.lower()
        if "sleep" in col_lower:
            vals = df[col].dropna()
            result["signals_found"]["sleep_" + col_lower.replace(" ", "_")] = {
                "column": col,
                "n_valid": int(len(vals)),
                "unique_values": list(vals.unique()[:10]),
                "dtype": str(df[col].dtype),
            }
    
    # Sample rows
    result["sample_columns"] = list(df.columns[:10])
    result["sample_rows"] = df.head(3).to_dict(orient="records")
    
    return result


def generate_markdown_report(patient_audits: list, all_columns: set) -> str:
    """Generate the DATASET_REPORT.md content."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    n_patients = len(patient_audits)
    n_valid = sum(1 for p in patient_audits if "error" not in p)
    
    # Collect signal availability
    signals_available = {
        "glucose": sum(1 for p in patient_audits if "glucose" in p.get("signals_found", {})),
        "timestamp": sum(1 for p in patient_audits if "timestamp" in p.get("signals_found", {})),
        "insulin_bolus": sum(1 for p in patient_audits if "insulin_bolus" in p.get("signals_found", {})),
        "insulin_basal": sum(1 for p in patient_audits if "insulin_basal" in p.get("signals_found", {})),
        "carbs": sum(1 for p in patient_audits if "carbs" in p.get("signals_found", {})),
        "steps": sum(1 for p in patient_audits if "steps" in p.get("signals_found", {})),
        "heart_rate": sum(1 for p in patient_audits if "heart_rate" in p.get("signals_found", {})),
        "sleep": sum(1 for p in patient_audits if any("sleep" in k for k in p.get("signals_found", {}))),
    }
    
    # Glucose statistics across patients
    glucose_stats = []
    for p in patient_audits:
        g = p.get("signals_found", {}).get("glucose", {})
        if g:
            glucose_stats.append({
                "patient": p.get("filename", "?"),
                "n_readings": g.get("n_valid", 0),
                "mean_mgdl": g.get("mean", "?"),
                "pct_below_70": g.get("pct_below_70", "?"),
                "pct_below_54": g.get("pct_below_54", "?"),
            })
    
    # Sampling intervals
    intervals = []
    for p in patient_audits:
        ts = p.get("signals_found", {}).get("timestamp", {})
        interval = ts.get("median_interval_min")
        if interval:
            intervals.append(interval)
    
    report = f"""# GlucoTwin — DATASET FORENSIC AUDIT REPORT

**Generated**: {now}  
**Dataset**: HUPA-UCM Diabetes Dataset  
**DOI**: 10.17632/3hbcscwz44.1  
**License**: CC BY 4.0  

---

## ⚡ Executive Summary

| Metric | Value |
|--------|-------|
| Total patient files found | {n_patients} |
| Successfully parsed | {n_valid} |
| Glucose (CGM) available | {signals_available['glucose']}/{n_valid} |
| Timestamp available | {signals_available['timestamp']}/{n_valid} |
| Insulin bolus available | {signals_available['insulin_bolus']}/{n_valid} |
| Insulin basal available | {signals_available['insulin_basal']}/{n_valid} |
| Meals/Carbs available | {signals_available['carbs']}/{n_valid} |
| Steps available | {signals_available['steps']}/{n_valid} |
| Heart rate available | {signals_available['heart_rate']}/{n_valid} |
| Sleep available | {signals_available['sleep']}/{n_valid} |

**CGM Median Sampling Interval**: {f"{sum(intervals)/len(intervals):.1f} min" if intervals else "Unknown"} (target: 15 min)

---

## 📂 File Structure

All patient files follow naming convention: `HUPA{{ID}}P.csv`

### All columns found across dataset:
```
{chr(10).join(sorted(all_columns))}
```

---

## 📊 Per-Patient Summary

| Patient | Rows | CGM Readings | Mean Glucose (mg/dL) | % <70 mg/dL | % <54 mg/dL |
|---------|------|-------------|----------------------|-------------|-------------|
"""
    
    for stat in glucose_stats:
        report += f"| {stat['patient']} | — | {stat['n_readings']} | {stat['mean_mgdl']} | {stat['pct_below_70']}% | {stat['pct_below_54']}% |\n"
    
    report += f"""
---

## 🔬 Signal Details

### Glucose (CGM — FreeStyle Libre 2)
- **Unit**: Determined from column name / sample values (see per-patient data)
- **Nominal interval**: 15 minutes (FreeStyle Libre 2 standard)
- **Actual median interval**: {f"{sum(intervals)/len(intervals):.1f} min" if intervals else "To be verified"}

### Insulin
- Bolus insulin available in {signals_available['insulin_bolus']}/{n_valid} patients
- Basal insulin available in {signals_available['insulin_basal']}/{n_valid} patients

### Meals (Carbohydrates)
- Available in {signals_available['carbs']}/{n_valid} patients

### Activity (Steps, HR)
- Steps available in {signals_available['steps']}/{n_valid} patients
- Heart rate available in {signals_available['heart_rate']}/{n_valid} patients

### Sleep
- Sleep data available in {signals_available['sleep']}/{n_valid} patients
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

"""
    
    for p in patient_audits:
        if "error" in p:
            report += f"### ❌ {p.get('filename', '?')}\nError: {p['error']}\n\n"
            continue
        
        report += f"### ✅ {p.get('filename', '?')}\n"
        report += f"- Rows: {p.get('n_rows', '?')}\n"
        report += f"- Columns: {p.get('n_columns', '?')}\n"
        report += f"- Column names: `{', '.join(p.get('columns', []))}`\n"
        
        if p.get("signals_found"):
            report += "- Signals detected:\n"
            for sig, info in p["signals_found"].items():
                if sig == "timestamp":
                    report += f"  - **{sig}**: column=`{info.get('column')}`, interval={info.get('median_interval_min')} min\n"
                elif sig == "glucose":
                    report += (f"  - **{sig}**: column=`{info.get('column')}`, "
                               f"n={info.get('n_valid')}, mean={info.get('mean')} mg/dL, "
                               f"<70: {info.get('pct_below_70')}%\n")
                else:
                    report += f"  - **{sig}**: column=`{info.get('column')}`, n_valid={info.get('n_valid')}\n"
        
        report += "\n"
    
    report += f"""---

*Generated by GlucoTwin dataset_audit.py*  
*DO NOT commit this file with patient-identifiable data — patient IDs are already anonymized in HUPA-UCM.*
"""
    
    return report


def main():
    print("=" * 70)
    print("GlucoTwin — DATASET FORENSIC AUDIT")
    print(f"Dataset path: {DATA_DIR.resolve()}")
    print("=" * 70)
    
    if not DATA_DIR.exists():
        print(f"[ERROR] Data directory not found: {DATA_DIR}")
        print("Run: python scripts/download_dataset.py")
        sys.exit(1)
    
    # ── Find files ────────────────────────────────────────────────────────────
    files = find_patient_files()
    print(f"\n[INFO] Found {len(files)} patient files")
    for f in files:
        print(f"  {f.relative_to(DATA_DIR)} ({f.stat().st_size/1024:.0f} KB)")
    
    if not files:
        print("[ERROR] No patient CSV files found. Check data directory.")
        sys.exit(1)
    
    # ── Inspect each file ─────────────────────────────────────────────────────
    print(f"\n[INFO] Inspecting {len(files)} files...")
    audits = []
    all_columns = set()
    
    for filepath in files:
        print(f"  Auditing: {filepath.name}...", end="", flush=True)
        audit = inspect_single_file(filepath)
        audits.append(audit)
        all_columns.update(audit.get("columns", []))
        
        if "error" in audit:
            print(f" ERROR: {audit['error']}")
        else:
            glucose_info = audit.get("signals_found", {}).get("glucose", {})
            n_cgm = glucose_info.get("n_valid", "?")
            mean_g = glucose_info.get("mean", "?")
            print(f" OK | {audit['n_rows']} rows | CGM n={n_cgm} | mean={mean_g} mg/dL")
    
    # ── Save JSON results ─────────────────────────────────────────────────────
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump({"audits": audits, "all_columns": list(sorted(all_columns))}, f, indent=2, default=str)
    print(f"\n[OK]  Raw audit data: {OUTPUT_JSON}")
    
    # ── Generate Markdown report ──────────────────────────────────────────────
    report = generate_markdown_report(audits, all_columns)
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[OK]  Report: {OUTPUT_REPORT}")
    
    # ── Print column summary to console ───────────────────────────────────────
    print(f"\n{'='*70}")
    print("ALL COLUMNS FOUND ACROSS DATASET:")
    print("="*70)
    for col in sorted(all_columns):
        print(f"  {col}")
    
    print(f"\n[DONE] Audit complete. Review: {OUTPUT_REPORT}")
    print("[NEXT] Update config/data.yaml column_map based on audit results.")


if __name__ == "__main__":
    main()
