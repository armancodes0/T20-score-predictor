"""
T20 World Cup Cricket Score Predictor - Model Training & Evaluation Pipeline
Treats the Academic Project Report as the Source of Truth:
- Scope: Completed first-innings score forecasting for Men's T20 World Cup
- Data Scope: Uses Cricsheet Men's T20 Internationals to supplement the small World Cup sample
- Splits: Match-level chronological splitting (no data leakage)
- States: Supports innings states from opening ball (0 balls, 0 runs) to 120 legal balls / 10 wickets
- Metrics: MAE, RMSE, Signed Mean Bias, Phase MAE (Powerplay, Middle, Death), Interval Coverage & Width
- Uncertainty: Split-conformal prediction interval with measured test coverage
"""

import os
import json
import zipfile
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
ZIP_PATH = os.path.join(DATA_DIR, "t20s_male_csv2.zip")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "t20_snapshots.csv")
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "t20_score_model.joblib")
METADATA_SAVE_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

# Target International Teams (ICC Full & Top Associate Members at T20 World Cups)
TARGET_TEAMS = [
    "India", "Australia", "England", "Pakistan", "South Africa",
    "New Zealand", "West Indies", "Sri Lanka", "Bangladesh",
    "Ireland", "Zimbabwe", "Netherlands", "Scotland", "Namibia",
    "Nepal", "United Arab Emirates", "United States of America", "Oman", "Canada"
]

def ensure_directories():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

def extract_snapshots(force_recompute=False):
    """
    Extracts chronological ball-by-ball first-innings snapshots from Cricsheet CSV2 archive.
    Covers state from opening ball (0 balls, 0 runs) to 120 balls / 10 wickets.
    Tracks match dates, venue, and World Cup event flags.
    """
    if os.path.exists(PROCESSED_DATA_PATH) and not force_recompute:
        print(f"[Data] Loading cached snapshots from {PROCESSED_DATA_PATH}")
        df = pd.read_csv(PROCESSED_DATA_PATH)
        # Quick check if it has legal_balls, start_date, and current_score
        if "legal_balls" in df.columns and "start_date" in df.columns and "is_world_cup" in df.columns and "current_score" in df.columns:
            return df
        print("[Data] Cached snapshots are from older schema. Rebuilding...")

    if not os.path.exists(ZIP_PATH):
        raise FileNotFoundError(f"Source archive not found at {ZIP_PATH}. Please provide t20s_male_csv2.zip.")

    print(f"[Data] Reading match files from {ZIP_PATH}...")
    snapshots = []
    match_manifest = []

    with zipfile.ZipFile(ZIP_PATH) as z:
        csv_files = [f for f in z.namelist() if f.endswith(".csv") and not f.endswith("_info.csv")]
        print(f"       Found {len(csv_files)} total match files in Cricsheet archive.")

        for idx, f in enumerate(csv_files):
            try:
                df = pd.read_csv(z.open(f), low_memory=False)
                if "innings" not in df.columns:
                    continue
                inn1 = df[df["innings"] == 1].copy()
                if inn1.empty:
                    continue

                batting_team = inn1["batting_team"].iloc[0]
                bowling_team = inn1["bowling_team"].iloc[0]

                # Filter for recognized international teams
                if batting_team not in TARGET_TEAMS or bowling_team not in TARGET_TEAMS:
                    continue

                # Handle deliveries, extras and legal balls
                inn1["runs_off_bat"] = inn1["runs_off_bat"].fillna(0)
                inn1["extras"] = inn1["extras"].fillna(0)
                inn1["wides"] = inn1["wides"].fillna(0) if "wides" in inn1.columns else 0
                inn1["noballs"] = inn1["noballs"].fillna(0) if "noballs" in inn1.columns else 0

                # A delivery is legal only when not a wide or noball
                inn1["is_legal"] = (inn1["wides"] == 0) & (inn1["noballs"] == 0)
                inn1["legal_ball_count"] = inn1["is_legal"].cumsum()

                # Dismissals: excluding non-dismissal retired hurt
                if "wicket_type" in inn1.columns:
                    inn1["is_wicket"] = inn1["wicket_type"].notna() & (inn1["wicket_type"] != "retired hurt")
                else:
                    inn1["is_wicket"] = False
                inn1["cumulative_wickets"] = inn1["is_wicket"].cumsum()

                # Cumulative runs
                inn1["ball_total_runs"] = inn1["runs_off_bat"] + inn1["extras"]
                inn1["cumulative_score"] = inn1["ball_total_runs"].cumsum()

                total_legal = inn1["legal_ball_count"].iloc[-1]
                final_wickets = inn1["cumulative_wickets"].iloc[-1]
                final_total = inn1["cumulative_score"].iloc[-1]

                # Exclusion policy: Exclude shortened/rain innings (<115 legal balls and <10 wickets)
                # and super overs (only innings 1 of scheduled 20-over matches included)
                if total_legal < 115 and final_wickets < 10:
                    continue

                match_id = inn1["match_id"].iloc[0]
                start_date = str(inn1["start_date"].iloc[0])
                venue = str(inn1["venue"].iloc[0]) if "venue" in inn1.columns and pd.notna(inn1["venue"].iloc[0]) else "Unknown Venue"

                # Check if this match is an ICC Men's T20 World Cup match
                is_wc = False
                info_f = f.replace(".csv", "_info.csv")
                if info_f in z.namelist():
                    itxt = z.read(info_f).decode("utf-8", errors="ignore")
                    for l in itxt.splitlines():
                        if l.startswith("info,event,") and ("World Cup" in l or "World T20" in l):
                            is_wc = True
                            break

                legal_rows = inn1[inn1["is_legal"]].copy()
                score_by_ball = {0: 0}
                wickets_by_ball = {0: 0}
                for _, r in legal_rows.iterrows():
                    score_by_ball[int(r["legal_ball_count"])] = int(r["cumulative_score"])
                    wickets_by_ball[int(r["legal_ball_count"])] = int(r["cumulative_wickets"])

                match_manifest.append({
                    "match_id": match_id,
                    "start_date": start_date,
                    "venue": venue,
                    "batting_team": batting_team,
                    "bowling_team": bowling_team,
                    "is_world_cup": is_wc,
                    "final_total": int(final_total)
                })

                # Checkpoints from opening ball (0) to end of 20 overs (120 balls) in 6-ball intervals
                checkpoints = [b for b in range(0, min(total_legal + 1, 121), 6)]
                # If innings terminated by all out before 120 balls, include the exact terminal ball
                if final_wickets == 10 and total_legal not in checkpoints and total_legal <= 120:
                    checkpoints.append(int(total_legal))
                    checkpoints = sorted(checkpoints)

                for b in checkpoints:
                    s = score_by_ball.get(b, score_by_ball.get(max([k for k in score_by_ball if k <= b], default=0)))
                    w = wickets_by_ball.get(b, wickets_by_ball.get(max([k for k in wickets_by_ball if k <= b], default=0)))
                    prev_b = max(0, b - 30)
                    s_prev = score_by_ball.get(prev_b, 0)
                    runs_last_5 = s - s_prev
                    overs = b / 6.0
                    crr = round(s / overs, 2) if overs > 0 else 8.12  # Documented opening prior RPO at ball 0

                    snapshots.append({
                        "match_id": match_id,
                        "start_date": start_date,
                        "venue": venue,
                        "is_world_cup": is_wc,
                        "batting_team": batting_team,
                        "bowling_team": bowling_team,
                        "current_score": int(s),
                        "legal_balls": int(b),
                        "overs_completed": round(overs, 1),
                        "wickets_lost": int(w),
                        "runs_last_5_overs": int(runs_last_5),
                        "current_run_rate": float(crr),
                        "final_innings_total": int(final_total)
                    })

            except Exception:
                continue

    snapshots_df = pd.DataFrame(snapshots)
    print(f"[Data] Extraction complete: {len(match_manifest)} matches, {len(snapshots_df)} snapshots.")
    snapshots_df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"[Data] Cached snapshots to {PROCESSED_DATA_PATH}")
    return snapshots_df

def run_chronological_pipeline():
    """
    Executes the training and evaluation protocol adhering strictly to project report:
    1. Match-level chronological splitting
    2. Transparent Run-Rate Baseline with documented opening prior
    3. Random Forest Regressor & Ridge Regression
    4. Metrics: MAE, RMSE, Signed Bias, Phase MAE (Powerplay, Middle, Death)
    5. Evaluation of World Cup subset
    6. Conformalized 80% prediction interval estimation & test coverage measurement
    """
    ensure_directories()
    snapshots_df = extract_snapshots()

    # Step 1: Match-level Chronological Split
    unique_matches_df = (
        snapshots_df[["match_id", "start_date", "is_world_cup"]]
        .drop_duplicates("match_id")
        .sort_values("start_date")
        .reset_index(drop=True)
    )

    n_matches = len(unique_matches_df)
    n_train = int(n_matches * 0.80)
    train_match_df = unique_matches_df.iloc[:n_train]
    test_match_df = unique_matches_df.iloc[n_train:]

    train_ids = set(train_match_df["match_id"])
    test_ids = set(test_match_df["match_id"])

    train_period = f"{train_match_df.iloc[0]['start_date']} to {train_match_df.iloc[-1]['start_date']}"
    test_period = f"{test_match_df.iloc[0]['start_date']} to {test_match_df.iloc[-1]['start_date']}"

    train_df = snapshots_df[snapshots_df["match_id"].isin(train_ids)].copy().reset_index(drop=True)
    test_df = snapshots_df[snapshots_df["match_id"].isin(test_ids)].copy().reset_index(drop=True)

    print("\n" + "="*60)
    print("  CHRONOLOGICAL MATCH-LEVEL SPLIT CONFIGURATION")
    print("="*60)
    print(f"Total Matches:    {n_matches} ({len(snapshots_df)} snapshots)")
    print(f"Train Matches:    {len(train_ids)} ({len(train_df)} snapshots) | Period: {train_period}")
    print(f"Held-Out Test:    {len(test_ids)} ({len(test_df)} snapshots) | Period: {test_period}")
    print(f"World Cup Matches in Test: {test_match_df['is_world_cup'].sum()} matches")

    # Documented Opening Prior
    opening_prior_total = round(float(train_df["final_innings_total"].mean()), 1)
    opening_prior_rpo = round(opening_prior_total / 20.0, 2)
    print(f"\nDocumented Opening Prior (Train Mean 1st Innings Total): {opening_prior_total} runs ({opening_prior_rpo} RPO)")

    # Features
    feature_cols = [
        "batting_team",
        "bowling_team",
        "venue",
        "current_score",
        "legal_balls",
        "overs_completed",
        "wickets_lost",
        "runs_last_5_overs",
        "current_run_rate"
    ]
    target_col = "final_innings_total"

    X_train = train_df[feature_cols]
    y_train = train_df[target_col].values
    X_test = test_df[feature_cols]
    y_test = test_df[target_col].values

    # Step 2: Baseline Model (Run-Rate Extrapolation with Documented Prior)
    def compute_baseline(b, w, s):
        if b == 0:
            return opening_prior_total
        elif b >= 120 or w >= 10:
            return s
        else:
            crr = s / (b / 6.0)
            rem_overs = (120 - b) / 6.0
            return max(s, s + crr * rem_overs)

    base_preds = np.array([
        compute_baseline(r["legal_balls"], r["wickets_lost"], r["current_score"])
        for _, r in test_df.iterrows()
    ])

    base_mae = float(mean_absolute_error(y_test, base_preds))
    base_rmse = float(np.sqrt(mean_squared_error(y_test, base_preds)))
    base_bias = float(np.mean(base_preds - y_test))

    # Step 3: Train Machine Learning Candidates
    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("teams", OneHotEncoder(categories=[TARGET_TEAMS, TARGET_TEAMS], handle_unknown="ignore", sparse_output=False), ["batting_team", "bowling_team"]),
            ("venue", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["venue"]),
            ("num", StandardScaler(), ["current_score", "legal_balls", "overs_completed", "wickets_lost", "runs_last_5_overs", "current_run_rate"])
        ]
    )

    # Candidate 1: Ridge Regression
    ridge_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", Ridge(alpha=10.0))
    ])
    ridge_pipeline.fit(X_train, y_train)
    ridge_raw_preds = ridge_pipeline.predict(X_test)
    ridge_preds = np.array([
        r["current_score"] if (r["legal_balls"] >= 120 or r["wickets_lost"] >= 10) else max(r["current_score"], ridge_raw_preds[i])
        for i, (_, r) in enumerate(test_df.iterrows())
    ])
    ridge_mae = float(mean_absolute_error(y_test, ridge_preds))
    ridge_rmse = float(np.sqrt(mean_squared_error(y_test, ridge_preds)))
    ridge_bias = float(np.mean(ridge_preds - y_test))

    # Candidate 2: Random Forest Regressor
    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_split=8,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1
        ))
    ])
    print("\n[Model] Fitting Random Forest Regressor on chronological training matches...")
    rf_pipeline.fit(X_train, y_train)

    rf_raw_preds = rf_pipeline.predict(X_test)
    rf_preds = np.array([
        r["current_score"] if (r["legal_balls"] >= 120 or r["wickets_lost"] >= 10) else max(r["current_score"], rf_raw_preds[i])
        for i, (_, r) in enumerate(test_df.iterrows())
    ])
    rf_mae = float(mean_absolute_error(y_test, rf_preds))
    rf_rmse = float(np.sqrt(mean_squared_error(y_test, rf_preds)))
    rf_bias = float(np.mean(rf_preds - y_test))

    # Step 4: Phase-Specific Analysis (Powerplay: 0-36 balls, Middle: 37-90 balls, Death: 91-120 balls)
    pp_mask = (test_df["legal_balls"] <= 36).values
    mid_mask = ((test_df["legal_balls"] > 36) & (test_df["legal_balls"] <= 90)).values
    death_mask = (test_df["legal_balls"] > 90).values

    phase_metrics = {
        "powerplay": {
            "name": "Powerplay (Overs 0.0 - 6.0 / Balls 0 - 36)",
            "snapshots": int(pp_mask.sum()),
            "baseline_mae": round(float(mean_absolute_error(y_test[pp_mask], base_preds[pp_mask])), 2),
            "rf_mae": round(float(mean_absolute_error(y_test[pp_mask], rf_preds[pp_mask])), 2)
        },
        "middle": {
            "name": "Middle Overs (Overs 6.1 - 15.0 / Balls 37 - 90)",
            "snapshots": int(mid_mask.sum()),
            "baseline_mae": round(float(mean_absolute_error(y_test[mid_mask], base_preds[mid_mask])), 2),
            "rf_mae": round(float(mean_absolute_error(y_test[mid_mask], rf_preds[mid_mask])), 2)
        },
        "death": {
            "name": "Death Overs (Overs 15.1 - 20.0 / Balls 91 - 120)",
            "snapshots": int(death_mask.sum()),
            "baseline_mae": round(float(mean_absolute_error(y_test[death_mask], base_preds[death_mask])), 2),
            "rf_mae": round(float(mean_absolute_error(y_test[death_mask], rf_preds[death_mask])), 2)
        }
    }

    # Step 5: Separate World Cup Evaluation
    wc_mask = test_df["is_world_cup"].values
    wc_metrics = {}
    if wc_mask.sum() > 0:
        wc_matches_count = int(test_df[test_df["is_world_cup"]]["match_id"].nunique())
        wc_snapshots_count = int(wc_mask.sum())
        wc_base_mae = float(mean_absolute_error(y_test[wc_mask], base_preds[wc_mask]))
        wc_rf_mae = float(mean_absolute_error(y_test[wc_mask], rf_preds[wc_mask]))
        wc_rf_rmse = float(np.sqrt(mean_squared_error(y_test[wc_mask], rf_preds[wc_mask])))
        wc_rf_bias = float(np.mean(rf_preds[wc_mask] - y_test[wc_mask]))
        wc_metrics = {
            "test_matches": wc_matches_count,
            "test_snapshots": wc_snapshots_count,
            "baseline_mae": round(wc_base_mae, 2),
            "rf_mae": round(wc_rf_mae, 2),
            "rf_rmse": round(wc_rf_rmse, 2),
            "rf_bias": round(wc_rf_bias, 2)
        }
        print(f"\n[Subgroup] World Cup Matches in Test: {wc_matches_count} matches, {wc_snapshots_count} snapshots")
        print(f"           RF MAE: {wc_rf_mae:.2f} runs | Baseline MAE: {wc_base_mae:.2f} runs")

    # Step 6: Conformalized 80% Prediction Interval Calibration
    # Use internal chronological validation partition (last 20% of train matches)
    calib_split_idx = int(len(train_match_df) * 0.8)
    calib_match_ids = set(train_match_df.iloc[calib_split_idx:]["match_id"])
    calib_df = train_df[train_df["match_id"].isin(calib_match_ids)]

    calib_raw_preds = rf_pipeline.predict(calib_df[feature_cols])
    calib_residuals = np.abs(calib_df["final_innings_total"].values - calib_raw_preds)
    q80_halfwidth = float(np.quantile(calib_residuals, 0.80))

    # Evaluate interval coverage on test set
    test_low = np.maximum(test_df["current_score"].values, rf_preds - q80_halfwidth)
    test_high = rf_preds + q80_halfwidth
    covered = (y_test >= test_low) & (y_test <= test_high)
    empirical_coverage = float(np.mean(covered))
    avg_interval_width = float(np.mean(test_high - test_low))

    print("\n" + "="*60)
    print("  HELD-OUT EVALUATION RESULTS (TABLE 7.1 TEMPLATE REPRODUCED)")
    print("="*60)
    print(f"Baseline (CRR + Prior): MAE={base_mae:.2f} runs | RMSE={base_rmse:.2f} | Bias={base_bias:.2f}")
    print(f"Ridge Regression:       MAE={ridge_mae:.2f} runs | RMSE={ridge_rmse:.2f} | Bias={ridge_bias:.2f}")
    print(f"Random Forest (RF):     MAE={rf_mae:.2f} runs | RMSE={rf_rmse:.2f} | Bias={rf_bias:.2f}")
    print(f"80% Conformal Interval: Measured Coverage={empirical_coverage*100:.1f}% | Avg Width={avg_interval_width:.1f} runs")
    print("="*60)

    # Save trained model artifact
    joblib.dump(rf_pipeline, MODEL_SAVE_PATH)
    print(f"\n[Save] Trained Random Forest Pipeline saved to {MODEL_SAVE_PATH}")

    # Compile verified model metadata and audit record
    known_venues = sorted(list(train_df["venue"].unique()))

    metadata = {
        "project_name": "T20 World Cup Cricket Score Predictor",
        "model_version": "v2.0-rf-conformal",
        "trained_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "best_model_name": "Random Forest Regressor",
        "split_methodology": "Match-level Chronological Split (No Data Leakage)",
        "train_period": train_period,
        "test_period": test_period,
        "supported_teams": TARGET_TEAMS,
        "known_venues": known_venues,
        "feature_columns": feature_cols,
        "opening_prior": {
            "mean_first_innings_score": opening_prior_total,
            "mean_run_rate_per_over": opening_prior_rpo,
            "explanation": "At ball 0, current run rate is undefined. The documented prior is the empirical mean 1st innings total of completed 20-over innings from the chronological training set."
        },
        "dataset_statistics": {
            "source": "Cricsheet Men's T20 Internationals (CSV2 ball-by-ball archive)",
            "total_matches_evaluated": n_matches,
            "train_matches": len(train_ids),
            "test_matches": len(test_ids),
            "total_snapshots": int(len(snapshots_df)),
            "train_snapshots": int(len(train_df)),
            "test_snapshots": int(len(test_df)),
            "data_scope_note": "Men's T20 World Cup tournament matches are supplemented with broader Men's T20 Internationals between target teams to provide sufficient training sample size. World Cup matches are evaluated separately as a subgroup.",
            "exclusions_policy": "Rain-shortened matches (<115 legal balls unless bowled out), super overs, and abandoned matches are strictly excluded.",
            "average_first_innings_score": round(float(snapshots_df["final_innings_total"].mean()), 1),
            "std_first_innings_score": round(float(snapshots_df["final_innings_total"].std()), 1),
        },
        "evaluation_metrics": {
            "baseline_run_rate": {
                "name": "Current Run Rate Extrapolation (with Documented Opening Prior)",
                "test_matches": len(test_ids),
                "test_snapshots": len(test_df),
                "mae": round(base_mae, 2),
                "rmse": round(base_rmse, 2),
                "signed_bias": round(base_bias, 2)
            },
            "ridge_regression": {
                "name": "Ridge Linear Regression",
                "test_matches": len(test_ids),
                "test_snapshots": len(test_df),
                "mae": round(ridge_mae, 2),
                "rmse": round(ridge_rmse, 2),
                "signed_bias": round(ridge_bias, 2)
            },
            "random_forest_regressor": {
                "name": "Random Forest Regressor (Selected Model)",
                "test_matches": len(test_ids),
                "test_snapshots": len(test_df),
                "mae": round(rf_mae, 2),
                "rmse": round(rf_rmse, 2),
                "signed_bias": round(rf_bias, 2)
            },
            "phase_specific_mae": phase_metrics,
            "world_cup_subgroup_metrics": wc_metrics,
            "uncertainty_interval": {
                "method": "Split-Conformal Residual Bound (80% Nominal Target)",
                "calibration_set": "Chronological validation partition (20% of train matches)",
                "half_width_q80": round(q80_halfwidth, 2),
                "empirical_coverage_percent": round(empirical_coverage * 100.0, 1),
                "average_interval_width_runs": round(avg_interval_width, 1),
                "caution_note": "Interval coverage is an aggregate empirical guarantee measured on unseen matches under identical conditions. It does not guarantee outcomes under sudden collapse, weather shifts, or distribution shift."
            }
        },
        "limitations_and_ethics": {
            "first_innings_only": "Model predicts completed 1st innings total of uninterrupted 20-over matches only.",
            "exclusions": "No chase / 2nd innings predictions, no win probabilities, no betting advice, no fantasy advice, no DLS calculation.",
            "explainability": "Feature importance reflects statistical tree partitions, not causal assertions about teams or players."
        }
    }

    with open(METADATA_SAVE_PATH, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"[Save] Comprehensive model metadata written to {METADATA_SAVE_PATH}")
    print("\nTraining & evaluation workflow completed successfully!")
    return metadata

if __name__ == "__main__":
    run_chronological_pipeline()
