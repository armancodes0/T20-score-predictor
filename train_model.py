"""
T20 World Cup Cricket Score Predictor - Model Training Pipeline
NTCC College Project
Author: Machine Learning Student
Description:
    Downloads (or loads) real Cricsheet T20 International ball-by-ball data,
    cleans and structures innings snapshots, splits by unique matches to prevent
    data leakage, trains Baseline (Linear Regression) and Random Forest models,
    evaluates on held-out matches using MAE, and saves the best model & pipeline.
"""

import os
import io
import json
import zipfile
# pyrefly: ignore [missing-import]
import requests
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import pandas as pd
from datetime import datetime

# pyrefly: ignore [missing-import]
from sklearn.model_selection import train_test_split
# pyrefly: ignore [missing-import]
from sklearn.compose import ColumnTransformer
# pyrefly: ignore [missing-import]
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# pyrefly: ignore [missing-import]
from sklearn.pipeline import Pipeline
# pyrefly: ignore [missing-import]
from sklearn.linear_model import LinearRegression
# pyrefly: ignore [missing-import]
from sklearn.ensemble import RandomForestRegressor
# pyrefly: ignore [missing-import]
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# pyrefly: ignore [missing-import]
import joblib

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
ZIP_PATH = os.path.join(DATA_DIR, "t20s_male_csv2.zip")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "t20_snapshots.csv")
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "t20_score_model.joblib")
METADATA_SAVE_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

# URL for Men's T20 Internationals (Cricsheet CSV2 format)
CRICSHEET_URL = "https://cricsheet.org/downloads/t20s_male_csv2.zip"

# Major International / T20 World Cup Teams
TARGET_TEAMS = [
    "India", "Australia", "England", "Pakistan", "South Africa",
    "New Zealand", "West Indies", "Sri Lanka", "Bangladesh",
    "Ireland", "Zimbabwe", "Netherlands", "Scotland", "Namibia",
    "Nepal", "United Arab Emirates", "United States of America", "Oman", "Canada"
]


def ensure_directories():
    """Ensure data and models directories exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)


def download_dataset_if_needed():
    """Download Cricsheet dataset zip if not already present."""
    ensure_directories()
    if os.path.exists(ZIP_PATH):
        print(f"[1/5] Dataset found locally at: {ZIP_PATH}")
        return

    print(f"[1/5] Dataset not found. Downloading men's T20 data from Cricsheet...")
    print(f"      URL: {CRICSHEET_URL}")
    try:
        response = requests.get(CRICSHEET_URL, stream=True, timeout=120)
        response.raise_for_status()
        total_len = response.headers.get("content-length")

        with open(ZIP_PATH, "wb") as f:
            if total_len is None:
                f.write(response.content)
            else:
                downloaded = 0
                total_len = int(total_len)
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        pct = (downloaded / total_len) * 100
                        print(f"      Downloaded {downloaded / (1024*1024):.1f} MB / {total_len / (1024*1024):.1f} MB ({pct:.1f}%)", end="\r")
        print(f"\n      Download complete! Saved to {ZIP_PATH}")
    except Exception as e:
        print(f"\n[ERROR] Failed to download dataset: {e}")
        print("Please manually download 't20s_male_csv2.zip' from https://cricsheet.org/downloads/")
        print(f"and place it inside the '{DATA_DIR}' folder.")
        raise


def extract_and_prepare_snapshots():
    """
    Parses ball-by-ball match CSVs, filters for 1st innings between target teams,
    handles extras and legal deliveries, and creates realistic mid-innings snapshots
    at over intervals (overs 5.0 to 19.0).
    """
    if os.path.exists(PROCESSED_DATA_PATH):
        print(f"[2/5] Loading preprocessed snapshots from cache: {PROCESSED_DATA_PATH}")
        return pd.read_csv(PROCESSED_DATA_PATH)

    print(f"[2/5] Processing ball-by-ball matches from {ZIP_PATH}...")
    if not os.path.exists(ZIP_PATH):
        download_dataset_if_needed()

    with zipfile.ZipFile(ZIP_PATH) as zf:
        csv_files = [f for f in zf.namelist() if f.endswith(".csv") and not f.endswith("_info.csv")]
        print(f"      Found {len(csv_files)} match files in archive.")

        rows = []
        valid_match_count = 0

        for idx, fname in enumerate(csv_files):
            try:
                df = pd.read_csv(zf.open(fname), low_memory=False)
            except Exception:
                continue

            # Check required columns
            required_cols = {"match_id", "innings", "batting_team", "bowling_team", "runs_off_bat", "extras"}
            if not required_cols.issubset(df.columns):
                continue

            # Only analyze 1st innings
            inn1 = df[df["innings"] == 1].copy()
            if inn1.empty:
                continue

            batting_team = inn1["batting_team"].iloc[0]
            bowling_team = inn1["bowling_team"].iloc[0]

            # Filter for matches between recognized target teams
            if batting_team not in TARGET_TEAMS or bowling_team not in TARGET_TEAMS:
                continue

            # Handle missing values
            inn1["runs_off_bat"] = inn1["runs_off_bat"].fillna(0)
            inn1["extras"] = inn1["extras"].fillna(0)
            inn1["wides"] = inn1["wides"].fillna(0) if "wides" in inn1.columns else 0
            inn1["noballs"] = inn1["noballs"].fillna(0) if "noballs" in inn1.columns else 0

            # Legal ball definition (wides and noballs are illegal deliveries requiring a re-bowl)
            inn1["is_legal"] = (inn1["wides"] == 0) & (inn1["noballs"] == 0)
            inn1["legal_ball_count"] = inn1["is_legal"].cumsum()

            # Wickets handling (exclude non-dismissal retired hurt)
            if "wicket_type" in inn1.columns:
                inn1["is_wicket"] = inn1["wicket_type"].notna() & (inn1["wicket_type"] != "retired hurt")
            else:
                inn1["is_wicket"] = False
            inn1["cumulative_wickets"] = inn1["is_wicket"].cumsum()

            # Cumulative runs off bat + extras
            inn1["ball_total_runs"] = inn1["runs_off_bat"] + inn1["extras"]
            inn1["cumulative_score"] = inn1["ball_total_runs"].cumsum()

            total_legal_balls = inn1["legal_ball_count"].iloc[-1]
            final_wickets = inn1["cumulative_wickets"].iloc[-1]
            final_total = inn1["cumulative_score"].iloc[-1]

            # Ensure this was a full standard T20 innings (reached >= 115 legal balls or bowled all out)
            # This discards rain-shortened DLS matches (e.g. 5-10 overs) that skew predictions
            if total_legal_balls < 115 and final_wickets < 10:
                continue

            legal_rows = inn1[inn1["is_legal"]].copy()
            if len(legal_rows) < 30:
                continue

            valid_match_count += 1
            match_id = inn1["match_id"].iloc[0]

            # Map legal ball count to cumulative stats
            score_by_ball = dict(zip(legal_rows["legal_ball_count"], legal_rows["cumulative_score"]))
            wickets_by_ball = dict(zip(legal_rows["legal_ball_count"], legal_rows["cumulative_wickets"]))

            # Sample snapshots at over ends: overs 5.0, 6.0, ..., 19.0 (legal balls 30, 36, ..., 114)
            max_legal = legal_rows["legal_ball_count"].iloc[-1]
            for b in range(30, min(max_legal, 115), 6):
                curr_score = score_by_ball.get(b)
                curr_wickets = wickets_by_ball.get(b)
                # Runs in last 5 overs (previous 30 legal balls)
                score_5_overs_ago = score_by_ball.get(b - 30, 0)
                runs_last_5 = curr_score - score_5_overs_ago
                overs = b / 6.0
                crr = round(curr_score / overs, 2)

                rows.append({
                    "match_id": match_id,
                    "batting_team": batting_team,
                    "bowling_team": bowling_team,
                    "current_score": int(curr_score),
                    "overs_completed": round(overs, 1),
                    "wickets_lost": int(curr_wickets),
                    "runs_last_5_overs": int(runs_last_5),
                    "current_run_rate": crr,
                    "final_innings_total": int(final_total)
                })

            if (idx + 1) % 500 == 0 or (idx + 1) == len(csv_files):
                print(f"      Scanned {idx + 1}/{len(csv_files)} files... Valid matches: {valid_match_count}")

    snapshots_df = pd.DataFrame(rows)
    print(f"      Extraction finished: {valid_match_count} matches -> {len(snapshots_df)} snapshot examples.")

    # Cache preprocessed data
    snapshots_df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"      Saved processed snapshots cache to: {PROCESSED_DATA_PATH}")
    return snapshots_df


def train_and_evaluate(snapshots_df):
    """
    Splits data by match_id to eliminate data leakage across identical matches,
    trains Linear Regression (Baseline) and Random Forest, compares MAE, and saves best model.
    """
    print("\n[3/5] Splitting dataset by match_id to prevent data leakage...")
    unique_matches = snapshots_df["match_id"].unique()
    train_matches, test_matches = train_test_split(unique_matches, test_size=0.20, random_state=42)

    train_df = snapshots_df[snapshots_df["match_id"].isin(train_matches)].copy()
    test_df = snapshots_df[snapshots_df["match_id"].isin(test_matches)].copy()

    print(f"      Total unique matches: {len(unique_matches)}")
    print(f"      Train matches: {len(train_matches)} ({len(train_df)} snapshots)")
    print(f"      Test matches:  {len(test_matches)} ({len(test_df)} snapshots)")

    feature_cols = [
        "batting_team",
        "bowling_team",
        "current_score",
        "overs_completed",
        "wickets_lost",
        "runs_last_5_overs",
        "current_run_rate"
    ]
    target_col = "final_innings_total"

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    # Preprocessing Pipeline
    cat_features = ["batting_team", "bowling_team"]
    num_features = ["current_score", "overs_completed", "wickets_lost", "runs_last_5_overs", "current_run_rate"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(categories=[TARGET_TEAMS, TARGET_TEAMS], handle_unknown="ignore", sparse_output=False), cat_features),
            ("num", StandardScaler(), num_features)
        ]
    )

    print("\n[4/5] Training models and computing evaluation metrics...")

    # 1. Baseline Model: Linear Regression
    print("      Training Baseline Model (Linear Regression)...")
    baseline_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])
    baseline_pipeline.fit(X_train, y_train)
    y_pred_base = baseline_pipeline.predict(X_test)

    base_mae = mean_absolute_error(y_test, y_pred_base)
    base_rmse = np.sqrt(mean_squared_error(y_test, y_pred_base))
    base_r2 = r2_score(y_test, y_pred_base)

    # 2. Main Model: Random Forest Regressor
    print("      Training Random Forest Regressor (100 estimators)...")
    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=8,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1
        ))
    ])
    rf_pipeline.fit(X_train, y_train)
    y_pred_rf = rf_pipeline.predict(X_test)

    rf_mae = mean_absolute_error(y_test, y_pred_rf)
    rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    rf_r2 = r2_score(y_test, y_pred_rf)

    print("\n========================================================")
    print("           MODEL EVALUATION RESULTS (TEST SET)          ")
    print("========================================================")
    print(f" Baseline (Linear Regression):")
    print(f"   - Mean Absolute Error (MAE):  {base_mae:.2f} runs")
    print(f"   - Root Mean Sq Error (RMSE):  {base_rmse:.2f} runs")
    print(f"   - R-squared (R2 Score):       {base_r2:.4f}")
    print("--------------------------------------------------------")
    print(f" Random Forest Regressor:")
    print(f"   - Mean Absolute Error (MAE):  {rf_mae:.2f} runs")
    print(f"   - Root Mean Sq Error (RMSE):  {rf_rmse:.2f} runs")
    print(f"   - R-squared (R2 Score):       {rf_r2:.4f}")
    print("========================================================")

    # Comparison and Selection
    if rf_mae <= base_mae:
        best_model_name = "Random Forest Regressor"
        best_pipeline = rf_pipeline
        best_mae = rf_mae
        best_rmse = rf_rmse
        best_r2 = rf_r2
    else:
        best_model_name = "Linear Regression"
        best_pipeline = baseline_pipeline
        best_mae = base_mae
        best_rmse = base_rmse
        best_r2 = base_r2

    print(f"\n[5/5] Saving superior model ({best_model_name}) and metadata...")
    joblib.dump(best_pipeline, MODEL_SAVE_PATH)
    print(f"      Model pipeline saved to: {MODEL_SAVE_PATH}")

    # Compile genuine metadata for viva and app display
    metadata = {
        "project_name": "T20 World Cup Cricket Score Predictor",
        "trained_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "best_model_name": best_model_name,
        "supported_teams": TARGET_TEAMS,
        "feature_columns": feature_cols,
        "dataset_statistics": {
            "total_matches_evaluated": int(len(unique_matches)),
            "train_matches": int(len(train_matches)),
            "test_matches": int(len(test_matches)),
            "total_snapshots": int(len(snapshots_df)),
            "average_first_innings_score": round(float(snapshots_df["final_innings_total"].mean()), 1),
            "min_first_innings_score": int(snapshots_df["final_innings_total"].min()),
            "max_first_innings_score": int(snapshots_df["final_innings_total"].max()),
            "std_first_innings_score": round(float(snapshots_df["final_innings_total"].std()), 1),
        },
        "evaluation_metrics": {
            "baseline_linear_regression": {
                "mae": round(float(base_mae), 2),
                "rmse": round(float(base_rmse), 2),
                "r2_score": round(float(base_r2), 4)
            },
            "random_forest_regressor": {
                "mae": round(float(rf_mae), 2),
                "rmse": round(float(rf_rmse), 2),
                "r2_score": round(float(rf_r2), 4)
            },
            "best_model_mae": round(float(best_mae), 2),
            "best_model_rmse": round(float(best_rmse), 2),
            "best_model_r2": round(float(best_r2), 4),
            "prediction_interval_margin": round(float(best_mae), 1)
        }
    }

    with open(METADATA_SAVE_PATH, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"      Model metadata saved to: {METADATA_SAVE_PATH}")
    print("\nTraining workflow completed successfully!")
    return metadata


def main():
    print("=" * 60)
    print(" T20 World Cup Cricket Score Predictor - Training Pipeline ")
    print("=" * 60)
    ensure_directories()
    download_dataset_if_needed()
    snapshots_df = extract_and_prepare_snapshots()
    train_and_evaluate(snapshots_df)


if __name__ == "__main__":
    main()
