"""
GlucoTwin Submission Assets Builder
Generates: Architecture Diagram PDF + Presentation PPTX
"""
import os, sys
from pathlib import Path

ROOT = Path(r"C:\Users\Ronak Jain\Desktop\glucotwin")
ASSETS_DIR = ROOT / "docs" / "submission"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# 1. ARCHITECTURE DIAGRAM PDF (via reportlab)
# ─────────────────────────────────────────────
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

W, H = landscape(A4)
arch_pdf = ASSETS_DIR / "GlucoTwin_Architecture_Diagram.pdf"
doc = SimpleDocTemplate(str(arch_pdf), pagesize=landscape(A4),
                        leftMargin=1.5*cm, rightMargin=1.5*cm,
                        topMargin=1.5*cm, bottomMargin=1.5*cm)

styles = getSampleStyleSheet()
DARK  = colors.HexColor("#1a1a2e")
BLUE  = colors.HexColor("#2563eb")
GREEN = colors.HexColor("#16a34a")
GOLD  = colors.HexColor("#d97706")
TEAL  = colors.HexColor("#0891b2")
GRAY  = colors.HexColor("#f1f5f9")
WHITE = colors.white

title_style  = ParagraphStyle("title",  fontName="Helvetica-Bold", fontSize=20, textColor=DARK,  alignment=TA_CENTER, spaceAfter=6)
sub_style    = ParagraphStyle("sub",    fontName="Helvetica",      fontSize=11, textColor=BLUE,  alignment=TA_CENTER, spaceAfter=18)
header_style = ParagraphStyle("hdr",    fontName="Helvetica-Bold", fontSize=9,  textColor=WHITE, alignment=TA_CENTER)
cell_style   = ParagraphStyle("cell",   fontName="Helvetica",      fontSize=8,  textColor=DARK,  alignment=TA_CENTER, leading=11)
metric_style = ParagraphStyle("metric", fontName="Helvetica-Bold", fontSize=9,  textColor=GREEN, alignment=TA_CENTER)
note_style   = ParagraphStyle("note",   fontName="Helvetica-Oblique", fontSize=7, textColor=colors.grey, alignment=TA_CENTER)

story = []
story.append(Paragraph("GlucoTwin — Predictive Metabolic Digital Twin", title_style))
story.append(Paragraph("System Architecture &amp; Component Design | CRP Digital Twin Challenge 2026", sub_style))
story.append(HRFlowable(width="100%", thickness=2, color=BLUE))
story.append(Spacer(1, 0.3*cm))

# --- Layer table -------
layer_data = [
    [Paragraph("DATA INGESTION", header_style),
     Paragraph("PREPROCESSING", header_style),
     Paragraph("FEATURE ENGINEERING", header_style),
     Paragraph("ML ENGINE", header_style),
     Paragraph("SIMULATOR", header_style)],
    [Paragraph("HUPA-UCM Dataset\nDOI: 10.17632/3hbcscwz44.1\n25 T1D patients\nAbbott FreeStyle Libre 2 CGM\nCSII + MDI Insulin Pumps\nFitbit Ionic Wristbands", cell_style),
     Paragraph("5-min CGM Interpolation\nLeakage-Safe Rolling Windows\n(closed='left' guard)\nForward Label Generation\n(T → T+60 min)\nTimestamp Temporal Sort\nZero Data Leakage Guarantee", cell_style),
     Paragraph("50-Feature Stack\nCGM Rolling Stats (mean/std/min)\nGlucose Velocity (15/30 min)\nIOB & COB Approximations\nCircadian Phase Vectors (sin/cos)\nExpanding Baseline Mean/EWM\nNocturnal Context Flags", cell_style),
     Paragraph("LightGBM Primary Classifier\n+ RandomForest Ensemble\nIsotonic Calibrator\n(Brier: 0.069 → 0.039)\nLOPO Cross-Validation\n24 patients | AUC 0.970±0.027\nDynamic Fallback Gate", cell_style),
     Paragraph("MetabolicSimulator v2\nDamped-Velocity ODE\n(k=0.015 min⁻¹)\n2-Compartment Bergman Model\nRescue Carb Scenarios\n15g fast / 25g snack\nCustom bolus & exercise", cell_style)],
]
layer_table = Table(layer_data, colWidths=[4.9*cm]*5, rowHeights=[0.7*cm, 4.5*cm])
layer_table.setStyle(TableStyle([
    ("BACKGROUND",    (0,0), (-1,0), BLUE),
    ("BACKGROUND",    (0,1), (0,1), GRAY),
    ("BACKGROUND",    (1,1), (1,1), colors.HexColor("#e0f2fe")),
    ("BACKGROUND",    (2,1), (2,1), GRAY),
    ("BACKGROUND",    (3,1), (3,1), colors.HexColor("#dcfce7")),
    ("BACKGROUND",    (4,1), (4,1), colors.HexColor("#fef9c3")),
    ("BOX",           (0,0), (-1,-1), 1.5, BLUE),
    ("INNERGRID",     (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
    ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
    ("TOPPADDING",    (0,0), (-1,-1), 4),
    ("BOTTOMPADDING", (0,0), (-1,-1), 4),
]))
story.append(layer_table)
story.append(Spacer(1, 0.35*cm))

# Arrow label
story.append(Paragraph("▼  TWIN CORE  ▼", ParagraphStyle("arrow", fontName="Helvetica-Bold", fontSize=11, textColor=BLUE, alignment=TA_CENTER)))
story.append(Spacer(1, 0.2*cm))

# --- Core twin components ----
core_data = [
    [Paragraph("ALERT STATE MACHINE", header_style),
     Paragraph("PATIENT MEMORY", header_style),
     Paragraph("OOD DETECTOR", header_style),
     Paragraph("TwinState v6 API", header_style)],
    [Paragraph("3-State FSM: NORMAL→WATCH→ALARM\nEntry threshold: 0.13 (calibrated)\nExit threshold: 0.08 (hysteresis)\n15-min cooldown buffer\n4.39 alert episodes/day\nvs 29.21 raw point alarms/day", cell_style),
     Paragraph("Persistent per-patient episode log\nStrict backward-looking guard (< T)\nSimilar episode retrieval (Top-3)\nAlert transition timeline\nBaseline mean deviation tracking\nExpanding + EWM personalisation", cell_style),
     Paragraph("Mahalanobis feature-space distance\nCGM gap continuity check\nConfidence score: 0.0–1.0\nOOD flag + component breakdown\nFlagged: 0.00% on HUPA0023P\nTraining distribution boundary", cell_style),
     Paragraph("Multi-Horizon: 30/60/90/120 min\nSHAP top-K risk drivers\nWhat-If simulation payloads\nModel self-awareness block\nJSON-serializable full snapshot\nTwinState contract v6 sealed", cell_style)],
]
core_table = Table(core_data, colWidths=[6.125*cm]*4, rowHeights=[0.7*cm, 3.8*cm])
core_table.setStyle(TableStyle([
    ("BACKGROUND",    (0,0), (-1,0), TEAL),
    ("BACKGROUND",    (0,1), (0,1), colors.HexColor("#fff7ed")),
    ("BACKGROUND",    (1,1), (1,1), colors.HexColor("#f0fdf4")),
    ("BACKGROUND",    (2,1), (2,1), colors.HexColor("#faf5ff")),
    ("BACKGROUND",    (3,1), (3,1), colors.HexColor("#eff6ff")),
    ("BOX",           (0,0), (-1,-1), 1.5, TEAL),
    ("INNERGRID",     (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
    ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
    ("TOPPADDING",    (0,0), (-1,-1), 4),
    ("BOTTOMPADDING", (0,0), (-1,-1), 4),
]))
story.append(core_table)
story.append(Spacer(1, 0.3*cm))

# --- Key metrics row ----
metric_data = [[
    Paragraph("LOPO ROC-AUC\n0.970 ± 0.027\n(24 patients)", metric_style),
    Paragraph("TTT MAE\n32.55 min\n(11,471 pairs)", metric_style),
    Paragraph("Brier Reduction\n0.069 → 0.039\n(−43.5%)", metric_style),
    Paragraph("ECE Reduction\n0.1006 → 0.0055\n(−94.5%)", metric_style),
    Paragraph("Episode Detection\n300/300 (100%)\nRetrospective", metric_style),
    Paragraph("Lead Time\n47.4 min mean\n48.8 min median", metric_style),
]]
metric_table = Table(metric_data, colWidths=[4.08*cm]*6, rowHeights=[1.5*cm])
metric_table.setStyle(TableStyle([
    ("BACKGROUND",    (0,0), (-1,-1), colors.HexColor("#f0fdf4")),
    ("BOX",           (0,0), (-1,-1), 1.5, GREEN),
    ("INNERGRID",     (0,0), (-1,-1), 0.5, GREEN),
    ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
]))
story.append(metric_table)
story.append(Spacer(1, 0.2*cm))
story.append(Paragraph("Jaypee Institute of Information Technology, Noida-62  |  CRP Digital Twin Challenge 2026 (Happiest Health)  |  github.com/Ronak1167/glucotwin", note_style))

doc.build(story)
print("[OK] Architecture PDF -> " + str(arch_pdf))

# ─────────────────────────────────────────────
# 2. PRESENTATION PPTX (via python-pptx)
# ─────────────────────────────────────────────
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

pptx_path = ASSETS_DIR / "GlucoTwin_Presentation.pptx"
prs = Presentation()
prs.slide_width  = Inches(13.33)
prs.slide_height = Inches(7.5)

DARK_RGB   = RGBColor(0x1a, 0x1a, 0x2e)
BLUE_RGB   = RGBColor(0x25, 0x63, 0xeb)
GREEN_RGB  = RGBColor(0x16, 0xa3, 0x4a)
TEAL_RGB   = RGBColor(0x08, 0x91, 0xb2)
WHITE_RGB  = RGBColor(0xff, 0xff, 0xff)
GOLD_RGB   = RGBColor(0xd9, 0x77, 0x06)
GRAY_RGB   = RGBColor(0xf1, 0xf5, 0xf9)

BLANK = prs.slide_layouts[6]  # completely blank

def add_rect(slide, l, t, w, h, fill_rgb, alpha=None):
    shape = slide.shapes.add_shape(1, Inches(l), Inches(t), Inches(w), Inches(h))
    shape.line.fill.background()
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_rgb
    return shape

def add_text(slide, text, l, t, w, h, size, bold=False, color=WHITE_RGB, align=PP_ALIGN.LEFT, wrap=True):
    txBox = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = txBox.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    return txBox

slides_content = [
    # Slide 1 — Title
    {
        "type": "title",
        "title": "GlucoTwin",
        "subtitle": "Predictive Metabolic Digital Twin for Hypoglycemia Risk",
        "team": "Ronak Jain · Aishni Rathore",
        "org": "Jaypee Institute of Information Technology, Noida-62",
        "event": "CRP Digital Twin Challenge 2026 · Happiest Health",
    },
    # Slide 2 — Problem
    {
        "type": "content",
        "header_color": BLUE_RGB,
        "heading": "Problem Statement",
        "bullets": [
            "Hypoglycemia (≤ 70 mg/dL) is the most critical acute hazard for Type 1 Diabetes patients",
            "Risk is highest during sleep — when symptom perception is absent",
            "Current CGM threshold alerts: 29.21 raw false alarms/day causing severe alert fatigue",
            "Static alarms ignore physiological context: IOB, COB, circadian vulnerability",
            "No counterfactual simulation: cannot ask 'what if the patient takes 15g carbs?'",
            "No continuous patient memory: every moment evaluated in isolation",
        ],
    },
    # Slide 3 — Healthcare Use Case
    {
        "type": "content",
        "header_color": TEAL_RGB,
        "heading": "Healthcare Use Case",
        "bullets": [
            "Target Users: Type 1 Diabetes patients on CSII/MDI insulin therapy with CGM sensors",
            "Clinical Setting: Nocturnal hypoglycemia prevention and early-warning decision support",
            "Clinician Decision Support: Calibrated 60-min risk alerts before threshold crossing",
            "Patient Empowerment: Interactive What-If simulator for rescue carb intervention planning",
            "Hospital/Clinic Use: Multi-patient dashboard with per-patient personalized twin states",
            "Regulatory Scope: Research prototype — clinician decision support only (not FDA-cleared)",
        ],
    },
    # Slide 4 — Technical Stack
    {
        "type": "two_col",
        "heading": "Technical Stack",
        "col1_title": "ML & Backend",
        "col1": [
            "Python 3.10/3.11",
            "LightGBM + RandomForest ensemble",
            "SHAP (explainability)",
            "Scikit-learn (isotonic calibration)",
            "NumPy / Pandas / SciPy",
            "Joblib (model serialization)",
            "pytest (38 tests, CI verified)",
        ],
        "col2_title": "Frontend & Infrastructure",
        "col2": [
            "Streamlit (clinical dashboard)",
            "Plotly (interactive charts)",
            "GitHub Actions CI/CD",
            "Python 3.10 + 3.11 CI matrix",
            "edge-tts / moviepy (demo video)",
            "python-pptx / reportlab (docs)",
            "Dataset: HUPA-UCM (Mendeley DOI)",
        ],
    },
    # Slide 5 — AI/ML Model Details
    {
        "type": "content",
        "header_color": GREEN_RGB,
        "heading": "AI/ML Model & Framework Details",
        "bullets": [
            "Primary Model: LightGBM 60-min hypoglycemia classifier (50 features)",
            "Validation: Leave-One-Patient-Out (LOPO) on 24 patients — ROC-AUC 0.970 ± 0.027",
            "Calibration: Isotonic regression — Brier score 0.069 → 0.039 (−43.5%), ECE −94.5%",
            "Explainability: SHAP TreeExplainer — real-time top-K risk factor attribution",
            "Trajectory: Damped-velocity ODE (k=0.015 min⁻¹) — TTT MAE 32.55 min over 11,471 pairs",
            "Alert FSM: Hysteresis state machine — 4.39 episodes/day vs 29.21 raw point alarms",
            "OOD Detection: Mahalanobis distance + CGM continuity → twin_confidence 0.0–1.0",
        ],
    },
    # Slide 6 — Five Digital Twin Differentiators
    {
        "type": "content",
        "header_color": GOLD_RGB,
        "heading": "5 Core Digital Twin Differentiators",
        "bullets": [
            "① Continuous Patient Memory — persistent episode log with strict backward temporal guards",
            "② Trajectory Modeling with Validated Uncertainty — damped ODE + empirical MAE tiers",
            "③ Counterfactual What-If Simulation — in-silico rescue carb & bolus scenario engine",
            "④ Stateful Alert Engine with Hysteresis — 4.39 episodes/day (vs 29.21 raw false alarms)",
            "⑤ Model Self-Awareness & OOD Guardrails — TwinState v6 exposes full empirical metrics",
        ],
    },
    # Slide 7 — Benchmark Results
    {
        "type": "content",
        "header_color": GREEN_RGB,
        "heading": "Authoritative Empirical Results",
        "bullets": [
            "LOPO ROC-AUC: 0.970 ± 0.027  |  PR-AUC: 0.858 ± 0.099  (24 patients, patient-independent)",
            "LOPO Sensitivity: 0.958 ± 0.041  |  Specificity: 0.821 ± 0.088  |  F2: 0.754 ± 0.065",
            "TTT MAE: 32.55 min  |  Median AE: 17.60 min  |  Within 15 min: 46.0%  (11,471 pairs)",
            "Brier Score: 0.0690 (raw) → 0.0390 (calibrated)  |  ECE: 0.1006 → 0.0055",
            "Full-Cohort Ablation AUC: 0.9468  |  PR-AUC: 0.8132  |  Sensitivity: 0.8628",
            "Episode Detection: 300/300 (100%)  |  Mean Lead Time: 47.4 min",
            "Alert Episodes: 4.39/day  (vs naive 29.21 raw point-level alarms/day)",
        ],
    },
    # Slide 8 — Dashboard Features
    {
        "type": "content",
        "header_color": TEAL_RGB,
        "heading": "Streamlit Clinical Dashboard (app.py)",
        "bullets": [
            "Live Inference Mode: real-time risk scoring from production LightGBM + IsotonicCalibrator",
            "Patient Navigator: Sidebar dropdown for all 25 HUPA-UCM cohort patients",
            "Current State Panel: glucose, 15-min velocity, risk score, alert status, trend",
            "What-If Simulator: carb intervention sliders (0–80g), bolus, exercise, 30–180 min horizon",
            "SHAP Explanations: top risk drivers computed live per inference call",
            "OOD Confidence Panel: twin_confidence score + Mahalanobis distance visualization",
            "Demo Fallback Mode: verified static payload if model artifacts are absent",
        ],
    },
    # Slide 9 — Architecture Overview
    {
        "type": "content",
        "header_color": DARK_RGB,
        "heading": "System Architecture Overview",
        "bullets": [
            "Data Layer: HUPA-UCM CGM + Bolus + Meal + Activity data ingestion",
            "Processing Layer: Leakage-safe preprocessing → 50-feature engineering pipeline",
            "Twin Core: LightGBM + Isotonic Calibrator → Alert FSM → Patient Memory → OOD",
            "Simulation Layer: MetabolicSimulator v2 (Bergman 2-compartment ODE)",
            "API Layer: GlucoTwinAPI → TwinState v6 JSON (full snapshot, SHAP, What-If)",
            "Presentation Layer: Streamlit + Plotly live dashboard (app.py)",
            "Full architecture diagram: docs/submission/GlucoTwin_Architecture_Diagram.pdf",
        ],
    },
    # Slide 10 — Limitations & Ethics
    {
        "type": "content",
        "header_color": GOLD_RGB,
        "heading": "Limitations & Ethical Considerations",
        "bullets": [
            "RESEARCH PROTOTYPE — NOT A MEDICAL DEVICE (not FDA/CE cleared)",
            "Observational cohort: 25 Spanish adults with T1D — generalizability unknown",
            "Sensor-specific: FreeStyle Libre 2 CGM; other devices may differ in factory calibration",
            "No Closed-Loop Control: open-loop decision support only — not integrated into AID systems",
            "What-If simulations are mathematical projections, NOT clinical prescriptions",
            "All predictions are statistical probabilities — never replace physician judgment",
        ],
    },
    # Slide 11 — Team & Repository
    {
        "type": "title",
        "title": "Team GlucoTwin",
        "subtitle": "CRP Digital Twin Challenge 2026 · Happiest Health",
        "team": "Ronak Jain (ML/Backend)  ·  Aishni Rathore (Dashboard/Frontend)",
        "org": "Jaypee Institute of Information Technology, Noida-62",
        "event": "github.com/Ronak1167/glucotwin  |  CI: ✅ 38/38 tests passing",
    },
]

for idx, sc in enumerate(slides_content):
    slide = prs.slides.add_slide(BLANK)

    if sc["type"] == "title":
        add_rect(slide, 0, 0, 13.33, 7.5, DARK_RGB)
        add_rect(slide, 0, 0, 13.33, 0.08, BLUE_RGB)
        add_rect(slide, 0, 7.42, 13.33, 0.08, BLUE_RGB)
        add_text(slide, sc["title"],    0.5, 1.8, 12.3, 1.4, 48, bold=True,  color=WHITE_RGB, align=PP_ALIGN.CENTER)
        add_text(slide, sc["subtitle"], 0.5, 3.3, 12.3, 0.8, 20, bold=False, color=RGBColor(0x93,0xc5,0xfd), align=PP_ALIGN.CENTER)
        add_text(slide, sc["team"],     0.5, 4.4, 12.3, 0.6, 16, bold=True,  color=WHITE_RGB, align=PP_ALIGN.CENTER)
        add_text(slide, sc["org"],      0.5, 5.1, 12.3, 0.5, 13, bold=False, color=RGBColor(0xcb,0xd5,0xe1), align=PP_ALIGN.CENTER)
        add_text(slide, sc["event"],    0.5, 5.7, 12.3, 0.5, 12, bold=False, color=RGBColor(0xcb,0xd5,0xe1), align=PP_ALIGN.CENTER)

    elif sc["type"] == "content":
        add_rect(slide, 0, 0, 13.33, 1.1, sc["header_color"])
        add_text(slide, sc["heading"], 0.4, 0.18, 12.5, 0.75, 26, bold=True, color=WHITE_RGB, align=PP_ALIGN.LEFT)
        add_rect(slide, 0, 1.1, 13.33, 6.4, GRAY_RGB)
        bullet_text = "\n".join(f"  •  {b}" for b in sc["bullets"])
        add_text(slide, bullet_text, 0.5, 1.25, 12.3, 5.9, 15, bold=False, color=DARK_RGB, align=PP_ALIGN.LEFT)
        # slide number
        add_text(slide, f"{idx+1}", 12.7, 7.1, 0.5, 0.35, 9, color=RGBColor(0x94,0xa3,0xb8), align=PP_ALIGN.RIGHT)

    elif sc["type"] == "two_col":
        add_rect(slide, 0, 0, 13.33, 1.1, BLUE_RGB)
        add_text(slide, sc["heading"], 0.4, 0.18, 12.5, 0.75, 26, bold=True, color=WHITE_RGB)
        add_rect(slide, 0, 1.1, 6.5, 6.4, GRAY_RGB)
        add_rect(slide, 6.5, 1.1, 6.83, 6.4, RGBColor(0xdb, 0xea, 0xfe))
        add_text(slide, sc["col1_title"], 0.3, 1.2, 6.0, 0.5, 14, bold=True, color=BLUE_RGB)
        add_text(slide, sc["col2_title"], 6.7, 1.2, 6.3, 0.5, 14, bold=True, color=TEAL_RGB)
        col1_text = "\n".join(f"  •  {b}" for b in sc["col1"])
        col2_text = "\n".join(f"  •  {b}" for b in sc["col2"])
        add_text(slide, col1_text, 0.3, 1.75, 6.0, 5.5, 14, color=DARK_RGB)
        add_text(slide, col2_text, 6.7, 1.75, 6.3, 5.5, 14, color=DARK_RGB)
        add_text(slide, f"{idx+1}", 12.7, 7.1, 0.5, 0.35, 9, color=RGBColor(0x94,0xa3,0xb8), align=PP_ALIGN.RIGHT)

prs.save(str(pptx_path))
print("[OK] Presentation PPTX -> " + str(pptx_path))
print("[DONE] All submission assets built successfully.")
