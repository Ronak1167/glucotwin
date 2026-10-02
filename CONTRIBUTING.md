# Contributing to GlucoTwin

Welcome to the **GlucoTwin** project! This repository contains the complete metabolic digital twin engine for the CRP Digital Twin Challenge 2026.

- **Backend / Digital Twin ML Research**: @Ronak1167
- **Frontend / Clinical Interface & Dashboard**: @wildchord (Aishni Rathore)

---

## 🛡️ Repository Protection & Branching Rules

> [!IMPORTANT]
> **Direct pushes to the `main` branch are blocked.**
> To ensure code quality and prevent breaking the verified 34-test ML engine, the `main` branch is protected by GitHub Branch Protection rules.
> 
> **Never commit or push directly to `main`.** Always work on a feature branch and submit a Pull Request (PR).

### Branching Policy
1. **Protected Branch**: `main` (Production release branch, requires PR and passing CI tests).
2. **Feature Branches**: All frontend and UI development must take place on dedicated branches, for example:
   - `feat/frontend-dashboard`
   - `feat/ui-components`
   - `feat/patient-view`
   - `fix/...`

---

## 🚀 Step-by-Step Contribution Workflow

### Step 1: Clone the Repository
```bash
git clone https://github.com/Ronak1167/glucotwin.git
cd glucotwin
```

### Step 2: Create a Feature Branch
Always start your work by branching off an updated `main`:
```bash
git checkout main
git pull origin main
git checkout -b feat/frontend-dashboard
```

### Step 3: Set Up Your Python Environment
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Linux/macOS
# or: .venv\Scripts\activate  # On Windows

# Install project dependencies
pip install -r requirements.txt
```

### Step 4: Verify Existing Tests Pass
Run the automated test suite locally to verify your setup:
```bash
pytest
```
*Expected: 34 tests passing.*

---

## 💻 Frontend Development & Integration Guide

The GlucoTwin backend exposes rich clinical digital twin states and simulation endpoints designed specifically for the frontend dashboard:

### 1. Instant Mock Data (Zero Setup)
A production-ready demo snapshot is committed and ready for immediate frontend consumption:
- File path: `data/processed/demo_twin_state.json`
- Contains:
  - Live glucose trajectory and velocity
  - Multi-horizon calibrated risk predictions ($30$, $60$, $90$, $120$ min)
  - Stateful alert status (`NORMAL`, `WATCH`, `ALARM`) with hysteresis
  - Continuous patient memory precedents
  - Counterfactual "What-If" rescue carb simulation curves ($+15\text{g}$ fast carbs, $+25\text{g}$ complex snack)
  - Model self-awareness and OOD confidence indicators

### 2. Programmatic Python API
You can query the digital twin programmatically in `src/interface/api.py`:
```python
from src.interface.api import GlucoTwinAPI

api = GlucoTwinAPI()

# Run What-If simulation
simulation = api.simulate_intervention(
    current_glucose=78.0,
    current_velocity=-1.2,
    carb_grams=15.0,
    minutes_ahead=120
)
print("Simulation result:", simulation)
```

---

## 🔄 Submitting Your Changes

### 1. Verify Everything Works
Before pushing, ensure all unit tests pass:
```bash
pytest tests/
```

### 2. Commit and Push Your Feature Branch
```bash
git add .
git commit -m "feat(ui): implement clinical dashboard layout and risk gauges"
git push -u origin feat/frontend-dashboard
```

### 3. Open a Pull Request (PR)
1. Go to the repository on GitHub: `https://github.com/Ronak1167/glucotwin`
2. You will see a banner prompting: **"Compare & pull request"**.
3. Set base repository: `Ronak1167/glucotwin`, base: `main` $\leftarrow$ compare: `feat/frontend-dashboard`.
4. Provide a clear title and description of your frontend changes or screenshots.
5. GitHub Actions CI will automatically run the test suite on your PR.
6. Once the CI check passes and the review is approved, your PR will be merged into `main`!
