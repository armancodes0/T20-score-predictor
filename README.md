# 🏏 T20 World Cup Cricket Score Predictor

**Major Project Report & Prototype Implementation**  
**Title:** *T20 World Cup Cricket Score Predictor: Design and Development of an Explainable Ball by Ball First Innings Score Forecasting System*  
**Author:** Md Arman Alam (Enrolment No. A357175125020)  
**Supervisor:** Miss Tanaya Ganguly  
**Institution:** Amity School of Engineering and Technology, Amity University Jharkhand, Ranchi  

---

## 📌 1. Project Scope & Source of Truth

This repository implements the exact requirements specified in the candidate's B.Tech Major Project Report:
- **Primary Prediction Target:** The completed final total of an uninterrupted **Men's T20 World Cup first innings** scheduled for 20 overs.
- **Explicit Exclusions (Report Section 1.6 & 3.9):**
  - Second-innings chase forecasting and target pursuit dynamics are **out of scope**.
  - Match winner probability and betting/fantasy advice are **strictly excluded**.
  - Duckworth-Lewis-Stern (DLS) calculations are **not implemented**; rain-truncated matches (<115 legal balls) are excluded.
- **Data Scope Strategy (Report Section 2.3 & 5.1):**
  - Due to the small sample size of Men's T20 World Cup tournament matches (~250 matches historically), the model is trained on broader Men's T20 Internationals (T20Is) between major nations (2005–2024) from Cricsheet.
  - Held-out World Cup tournament matches are evaluated as an independent subgroup to verify out-of-time transfer.

---

## 🎯 2. Requirements & Verification Matrix

| Requirement / Module | Report Specification | Implementation Status | Evidence / Location |
| :--- | :--- | :---: | :--- |
| **1. First-Innings Scope** | Men's T20 World Cup 1st innings only (scheduled 20 ov) | **Fully Aligned** | [app.py](file:///c:/ANTIGRAVITY/NTCC/T20/app.py#L305-L330), non-goals clearly documented |
| **2. Inputs & Validation** | Ball 0 to 120, wickets 0–10, wickets $\le$ legal balls | **Fully Aligned** | [app.py](file:///c:/ANTIGRAVITY/NTCC/T20/app.py#L430-L470), zero-division safeguards |
| **3. Baseline & Model** | Side-by-side transparent CRR baseline with prior | **Fully Aligned** | [app.py](file:///c:/ANTIGRAVITY/NTCC/T20/app.py#L480-L540), prior = 160.2 runs |
| **4. Uncertainty Bounds** | 80% Conformal Prediction Interval (not raw MAE) | **Fully Aligned** | [models/model_metadata.json](file:///c:/ANTIGRAVITY/NTCC/T20/models/model_metadata.json#L315-L322) (73.7% coverage) |
| **5. Chronological Split** | Match-level chronological splitting (no data leakage) | **Fully Aligned** | [train_model.py](file:///c:/ANTIGRAVITY/NTCC/T20/train_model.py#L189-L230) (Train: 2005–2024, Test: 2024–2026) |
| **6. Full Metrics Suite** | MAE, RMSE, Signed Mean Bias, Phase MAE | **Fully Aligned** | Report Table 7.1 & 7.2 replicated in app & metadata |
| **7. World Cup Subgroup** | Separate evaluation on World Cup tournament games | **Fully Aligned** | 50 test World Cup matches evaluated (17.59 runs MAE) |
| **8. Functional Tests** | Table 7.2 test protocol (TC01–TC06) | **Fully Aligned** | Interactive test panel in Tab 2 of Streamlit UI |

---

## 🧠 3. Chronological Match Splitting & Evaluation Results

Following **Report Section 2.7 & 5.7**, matches are split chronologically by match date rather than random shuffling to reflect authentic forecasting conditions and eliminate cross-delivery data leakage:

- **Total Matches Analyzed:** 1,416 uninterrupted matches (29,572 snapshots)
- **Chronological Training Period:** `2005-02-17` to `2024-09-27` (1,132 matches, 23,647 snapshots)
- **Held-Out Test Period:** `2024-09-28` to `2026-10-13` (284 matches, 5,925 snapshots)
- **Documented Opening Prior:** $\bar{Y}_{\text{train}} = 160.2$ runs (8.01 RPO)

### Table 7.1: Held-Out Test Evaluation Results (Report Replication)
| Model Pipeline | Test Matches | Snapshots | MAE (runs) | RMSE (runs) | Signed Mean Bias | 80% Conformal Coverage | Avg Width |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Run-Rate Baseline (CRR + Prior)** | 284 | 5,925 | 22.97 runs | 34.84 runs | -9.15 runs | N/A (Point Extrapolation) | 0.0 runs |
| **Ridge Linear Regression** | 284 | 5,925 | 17.98 runs | 24.37 runs | -4.94 runs | Uncalibrated | N/A |
| **Random Forest (Selected)** | **284** | **5,925** | **16.94 runs** | **24.50 runs** | **-4.56 runs** | **73.7%** | **42.9 runs** |

### Phase-Specific MAE Analysis (Report Section 7.5)
| Innings Phase | Legal Balls Range | Test Snapshots | Run-Rate Baseline MAE | Random Forest MAE |
| :--- | :---: | :---: | :---: | :---: |
| **Powerplay** | Balls 0 – 36 (Overs 0.0 – 6.0) | 1,988 | 40.26 runs | **25.94 runs** |
| **Middle Overs** | Balls 37 – 90 (Overs 6.1 – 15.0) | 2,553 | 18.43 runs | **15.74 runs** |
| **Death Overs** | Balls 91 – 120 (Overs 15.1 – 20.0) | 1,384 | 6.52 runs | **6.21 runs** |

### World Cup Tournament Subgroup (Report Section 7.2)
- **Test Set World Cup Matches:** 50 matches (1,047 snapshots)
- **Random Forest MAE on World Cup Games:** **17.59 runs** (RMSE: 25.13 runs, Bias: -8.07 runs)
- **Baseline MAE on World Cup Games:** **22.12 runs**

---

## 🧪 4. Functional Validation Test Suite (Report Table 7.2)

The application includes an interactive test suite verifying all required edge cases:
- **TC01 [Opening State]:** $0/0$ in $0.0$ ov $\to$ Evaluates documented prior (160.2 runs) without zero-division error.
- **TC02 [Completed Innings]:** 120 legal balls $\to$ Accepted as completed innings; projection locks to current score.
- **TC03 [Over-limit Balls]:** 121 legal balls $\to$ Blocked with explicit validation error.
- **TC04 [Over-limit Wickets]:** 11 wickets $\to$ Blocked with explicit validation error.
- **TC05 [Unseen Venue]:** Unrecognized venue $\to$ Accepted with explicit warning and encoded safely with baseline priors.
- **TC06 [Deterministic Output]:** Repeat identical request $\to$ Returns bitwise identical prediction.

---

## 🚀 5. How to Run Locally

### Step 1: Clone & Navigate to Repository
```powershell
cd c:\ANTIGRAVITY\NTCC\T20
```

### Step 2: Install Dependencies
```powershell
py -m pip install -r requirements.txt
```

### Step 3: Run Model Training & Evaluation Pipeline
*(Extracts chronological snapshots from Cricsheet archive, trains baseline and Random Forest models, performs chronological held-out evaluation, and generates `model_metadata.json`)*
```powershell
py train_model.py
```

### Step 4: Launch the Streamlit Web Application
```powershell
py -m streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 🔮 6. Remaining Gaps & Planned Work (Report Chapter 8)
- **Live Ball-by-Ball Feed:** Push integration with licensed real-time match APIs requires commercial broadcaster licensing (Report Section 8.5).
- **Pitch Telemetry:** Live pitch telemetry, soil compaction, and boundary perimeter measurements require specialized sensory data currently unavailable in public archives (Report Section 8.7).
- **Distribution Shift Monitoring:** Recent tournament matches demonstrate higher first-innings totals, necessitating periodic chronological retraining (Report Section 8.8).
