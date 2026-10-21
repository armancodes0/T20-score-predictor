"""
T20 World Cup Cricket Score Predictor - Streamlit Web Application
NTCC College Project
Author: Machine Learning Student
Description:
    An interactive machine learning dashboard for predicting the final 1st-innings
    total in a T20 International cricket match using real ball-by-ball Cricsheet data,
    Random Forest Regression, and genuine held-out evaluation metrics.
"""

import os
import json
# pyrefly: ignore [missing-import]
import joblib
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import pandas as pd
# pyrefly: ignore [missing-import]
import streamlit as st
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt         

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="T20 World Cup Cricket Score Predictor | NTCC Project",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "t20_score_model.joblib")
METADATA_PATH = os.path.join(BASE_DIR, "models", "model_metadata.json")
DATA_ZIP_PATH = os.path.join(BASE_DIR, "data", "t20s_male_csv2.zip")

DEFAULT_TEAMS = [
    "India", "Australia", "England", "Pakistan", "South Africa",
    "New Zealand", "West Indies", "Sri Lanka", "Bangladesh",
    "Ireland", "Zimbabwe", "Netherlands", "Scotland", "Namibia",
    "Nepal", "United Arab Emirates", "United States of America", "Oman", "Canada"
]

# -----------------------------------------------------------------------------
# Custom Styling (Cricket Pitch Dark Theme with Emerald Green Accents)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Dark Pitch Background */
    .stApp {
        background: radial-gradient(circle at 50% 10%, #10261D 0%, #08120E 100%);
        color: #E2E8F0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Cards */
    .cricket-card {
        background: rgba(16, 38, 29, 0.7);
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
        backdrop-filter: blur(8px);
    }

    .hero-pred-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.18), rgba(5, 150, 105, 0.08));
        border: 2px solid #00E676;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 0 25px rgba(0, 230, 118, 0.25);
        margin-bottom: 16px;
    }

    .hero-score {
        font-size: 3.5rem;
        font-weight: 800;
        color: #00E676;
        line-height: 1.1;
        letter-spacing: -1px;
        text-shadow: 0 2px 12px rgba(0, 230, 118, 0.4);
    }

    .hero-subtitle {
        color: #A7F3D0;
        font-size: 1.05rem;
        font-weight: 500;
        margin-top: 6px;
    }

    .metric-badge {
        display: inline-block;
        background: rgba(0, 230, 118, 0.12);
        border: 1px solid rgba(0, 230, 118, 0.3);
        border-radius: 8px;
        padding: 10px 14px;
        margin: 4px;
        text-align: center;
        min-width: 130px;
    }

    .metric-val {
        font-size: 1.35rem;
        font-weight: 700;
        color: #FFFFFF;
    }

    .metric-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #94A3B8;
        margin-top: 2px;
    }

    /* Buttons */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #00E676 0%, #059669 100%);
        color: #062418;
        font-weight: 700;
        font-size: 1.1rem;
        border: none;
        border-radius: 10px;
        padding: 14px 28px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 230, 118, 0.35);
        width: 100%;
    }

    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #10B981 0%, #00E676 100%);
        color: #031810;
        box-shadow: 0 6px 20px rgba(0, 230, 118, 0.55);
        transform: translateY(-1px);
    }

    /* Selectors & Inputs */
    .stSelectbox label, .stNumberInput label, .stSlider label {
        color: #A7F3D0 !important;
        font-weight: 600 !important;
    }

    /* Headings */
    h1, h2, h3 {
        color: #F8FAFC !important;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Helper Functions: Model Loading & Fallback
# -----------------------------------------------------------------------------
@st.cache_resource
def load_trained_model():
    """Load the trained machine learning pipeline."""
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            return model, None
        except Exception as e:
            return None, f"Failed to deserialize model: {str(e)}"
    return None, "Model file not found. Please train the model first."


@st.cache_data
def load_metadata():
    """Load model evaluation results and supported team lists."""
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def render_missing_model_view(error_msg):
    """Fallback view when the model artifact hasn't been generated yet."""
    st.error(f"⚠️ Model Artifact Notice: {error_msg}")
    st.markdown("""
    ### 📁 Machine Learning Model Setup Required
    To ensure complete authenticity for your college NTCC evaluation, this application does not use mock or fabricated scores.
    The real trained model needs to be generated using historical Cricsheet ball-by-ball matches.

    **How to prepare the model:**
    1. Make sure `train_model.py` is present in the project directory.
    2. Open your terminal / Windows PowerShell in this folder (`c:\\ANTIGRAVITY\\NTCC\\T20`).
    3. Run the training pipeline:
       ```powershell
       py train_model.py
       ```
    4. Once finished, refresh this page.
    """)
    if st.button("🔄 Check Again for Model"):
        st.rerun()


# -----------------------------------------------------------------------------
# Score Projection Visualization (Matplotlib)
# -----------------------------------------------------------------------------
def plot_innings_projection(current_overs, current_score, predicted_score, mae_margin, batting_team, bowling_team):
    """Generates a clean cricket run progression and projection chart."""
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor="#0B1712")
    ax.set_facecolor("#08120E")

    # 1. Historical run progression up to current overs
    x_curr = np.linspace(0, current_overs, max(int(current_overs * 4), 10))
    y_curr = (current_score / current_overs) * x_curr

    # 2. Projected continuation to 20 overs
    x_pred = np.linspace(current_overs, 20.0, 25)
    y_pred = np.linspace(current_score, predicted_score, 25)

    # 3. Model Confidence interval (± MAE)
    upper_bound = np.linspace(current_score, predicted_score + mae_margin, 25)
    lower_bound = np.linspace(current_score, max(current_score, predicted_score - mae_margin), 25)

    # 4. Naive run-rate baseline (if scoring continued at flat current run rate)
    crr = current_score / current_overs
    y_naive = np.linspace(current_score, current_score + crr * (20.0 - current_overs), 25)

    # Plot Lines
    ax.plot(x_curr, y_curr, color="#00E676", lw=2.8, label=f"Current Progress ({current_score} runs)")
    ax.plot(x_pred, y_pred, color="#38BDF8", lw=2.6, ls="--", label=f"ML Projected Path ({int(round(predicted_score))} runs)")
    ax.plot(x_pred, y_naive, color="#94A3B8", lw=1.5, ls=":", label=f"Flat CRR Projection ({int(round(current_score + crr * (20 - current_overs)))})")
    ax.fill_between(x_pred, lower_bound, upper_bound, color="#38BDF8", alpha=0.15, label=f"Confidence Band (±{mae_margin:.1f} MAE)")

    # Scatter points
    ax.scatter([current_overs], [current_score], color="#FBBF24", s=90, zorder=6, edgecolors="#FFFFFF", lw=1.5, label="Current Position")
    ax.scatter([20.0], [predicted_score], color="#38BDF8", s=110, marker="*", zorder=6, edgecolors="#FFFFFF", lw=1)

    # Phase shading (Powerplay, Middle, Death Overs)
    ax.axvspan(0, 6, color="#10B981", alpha=0.06)
    ax.axvspan(6, 15, color="#3B82F6", alpha=0.04)
    ax.axvspan(15, 20, color="#EF4444", alpha=0.06)

    # Annotate Phase Labels
    y_text_pos = max(predicted_score + mae_margin + 12, 180)
    ax.text(3, y_text_pos * 0.93, "Powerplay (1-6)", color="#6EE7B7", fontsize=8, ha="center", style="italic")
    ax.text(10.5, y_text_pos * 0.93, "Middle (7-15)", color="#93C5FD", fontsize=8, ha="center", style="italic")
    ax.text(17.5, y_text_pos * 0.93, "Death (16-20)", color="#FCA5A5", fontsize=8, ha="center", style="italic")

    # Axes & Aesthetics
    ax.set_xlim(0, 20.4)
    ax.set_ylim(0, max(predicted_score + mae_margin + 20, 200))
    ax.set_xlabel("Overs Bowled", color="#A7F3D0", fontsize=10, fontweight="bold")
    ax.set_ylabel("Innings Runs", color="#A7F3D0", fontsize=10, fontweight="bold")
    ax.tick_params(colors="#94A3B8", labelsize=9)
    ax.spines["bottom"].set_color("#1E3A2F")
    ax.spines["top"].set_color("none")
    ax.spines["right"].set_color("none")
    ax.spines["left"].set_color("#1E3A2F")
    ax.grid(True, ls="--", color="#132B20", alpha=0.7)

    ax.legend(facecolor="#0B1C15", edgecolor="#1E3A2F", fontsize=8, labelcolor="#E2E8F0", loc="upper left")
    plt.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Main Application Layout
# -----------------------------------------------------------------------------
def main():
    # Load Model & Metadata
    model, model_err = load_trained_model()
    metadata = load_metadata()

    if model is None:
        render_missing_model_view(model_err)
        return

    # Team List
    teams_list = metadata.get("supported_teams", DEFAULT_TEAMS) if metadata else DEFAULT_TEAMS
    mae_margin = metadata.get("evaluation_metrics", {}).get("best_model_mae", 14.6) if metadata else 14.6

    # -------------------------------------------------------------------------
    # Sidebar: Project Info & Presets
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### 🏏 NTCC Project Info")
        st.markdown("**Project:** T20 World Cup Score Predictor")
        st.markdown("**Domain:** Sports Analytics & Machine Learning")
        st.markdown("**Evaluation:** College Viva & Final Demo")
        st.markdown("---")

        st.markdown("### ⚡ Quick Match Presets")
        st.caption("Click a preset below to populate realistic match scenarios for your viva demonstration:")

        preset = st.radio(
            "Select a Demonstration Scenario:",
            [
                "Custom Inputs",
                "Scenario A: India vs Australia (Dominant 10 Overs)",
                "Scenario B: England vs Pakistan (Death Overs Collapse)",
                "Scenario C: South Africa vs New Zealand (Steady Momentum)"
            ],
            index=0
        )

        st.markdown("---")
        st.markdown("### 📊 Dataset Reference")
        st.caption("Data Source: [Cricsheet Men's T20 International ball-by-ball](https://cricsheet.org/downloads/)")
        if metadata:
            st.caption(f"Trained on: **{metadata['dataset_statistics']['total_matches_evaluated']} matches** ({metadata['dataset_statistics']['total_snapshots']:,} snapshots)")
            st.caption(f"Algorithm: **{metadata['best_model_name']}**")
            st.caption(f"Test MAE: **±{metadata['evaluation_metrics']['best_model_mae']} runs**")

    # Set default values based on selected preset
    if preset == "Scenario A: India vs Australia (Dominant 10 Overs)":
        def_bat, def_bowl = "India", "Australia"
        def_score, def_overs_int, def_balls, def_wickets, def_last5 = 104, 10, 0, 1, 58
    elif preset == "Scenario B: England vs Pakistan (Death Overs Collapse)":
        def_bat, def_bowl = "England", "Pakistan"
        def_score, def_overs_int, def_balls, def_wickets, def_last5 = 128, 16, 2, 7, 29
    elif preset == "Scenario C: South Africa vs New Zealand (Steady Momentum)":
        def_bat, def_bowl = "South Africa", "New Zealand"
        def_score, def_overs_int, def_balls, def_wickets, def_last5 = 78, 9, 3, 3, 42
    else:
        def_bat, def_bowl = "India", "Australia"
        def_score, def_overs_int, def_balls, def_wickets, def_last5 = 92, 10, 0, 2, 45

    # -------------------------------------------------------------------------
    # Header Banner
    # -------------------------------------------------------------------------
    st.markdown("""
        <div style="text-align: center; margin-bottom: 24px;">
            <h1 style="font-size: 2.3rem; font-weight: 800; margin-bottom: 4px;">
                🏏 T20 World Cup Cricket Score Predictor
            </h1>
            <p style="color: #A7F3D0; font-size: 1.05rem; max-width: 800px; margin: 0 auto;">
                Machine learning-powered 1st innings total projection built on historical Men's T20 International ball-by-ball data.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Main Content Columns
    # -------------------------------------------------------------------------
    col_input, col_output = st.columns([1.05, 1.15], gap="large")

    with col_input:
        st.markdown('<div class="cricket-card">', unsafe_allow_html=True)
        st.markdown("### 📋 1. Match Situation Inputs")
        st.caption("Provide current innings state (valid for completed overs between 5.0 and 19.5):")

        # Teams Selection
        c1, c2 = st.columns(2)
        with c1:
            batting_team = st.selectbox(
                "Batting Team",
                teams_list,
                index=teams_list.index(def_bat) if def_bat in teams_list else 0,
                key="batting_team_select"
            )
        with c2:
            bowl_default_idx = teams_list.index(def_bowl) if def_bowl in teams_list else min(1, len(teams_list)-1)
            bowling_team = st.selectbox(
                "Bowling Team",
                teams_list,
                index=bowl_default_idx,
                key="bowling_team_select"
            )

        # Team Validation
        teams_invalid = batting_team == bowling_team
        if teams_invalid:
            st.error("❌ Batting team and Bowling team cannot be the same. Please choose two distinct teams.")

        st.markdown("---")

        # Score & Overs
        c3, c4, c5 = st.columns([1.2, 1.0, 1.0])
        with c3:
            current_score = st.number_input(
                "Current Score (Runs)",
                min_value=0,
                max_value=350,
                value=def_score,
                step=1,
                help="Total runs scored by the batting team so far."
            )
        with c4:
            overs_completed_int = st.number_input(
                "Completed Overs",
                min_value=5,
                max_value=19,
                value=def_overs_int,
                step=1,
                help="Full completed overs (model operates from over 5 to 19)."
            )
        with c5:
            balls_in_over = st.number_input(
                "Balls in Over",
                min_value=0,
                max_value=5,
                value=def_balls,
                step=1,
                help="Legal deliveries bowled in current over (0 to 5)."
            )

        # Compute fractional and display overs
        exact_overs_float = round(overs_completed_int + (balls_in_over / 6.0), 2)
        display_overs_str = f"{overs_completed_int}.{balls_in_over}"

        # Wickets & Recent Runs
        c6, c7 = st.columns(2)
        with c6:
            wickets_lost = st.number_input(
                "Wickets Lost",
                min_value=0,
                max_value=9,
                value=def_wickets,
                step=1,
                help="Number of wickets fallen (0 to 9). At 10 wickets an innings terminates."
            )
        with c7:
            # Runs in last 5 overs
            max_possible_last5 = max(current_score, 0)
            suggested_last5 = min(def_last5, max_possible_last5)
            runs_last_5 = st.number_input(
                "Runs in Last 5 Overs",
                min_value=0,
                max_value=300,
                value=suggested_last5,
                step=1,
                help="Runs scored in the previous 5 overs (30 legal balls). Cannot exceed current score."
            )

        # Live Real-time Validation Checks
        validation_errors = []
        if teams_invalid:
            validation_errors.append("Batting and Bowling teams must be different.")
        if runs_last_5 > current_score:
            validation_errors.append(f"Runs in last 5 overs ({runs_last_5}) cannot exceed total current score ({current_score}).")

        # Current Run Rate indicator
        crr = round(current_score / exact_overs_float, 2) if exact_overs_float > 0 else 0.0

        st.markdown(f"""
            <div style="background: rgba(0, 230, 118, 0.08); border-radius: 8px; padding: 10px 14px; margin-top: 10px; display: flex; justify-content: space-between; align-items: center;">
                <span style="color: #A7F3D0; font-size: 0.9rem;">Current State: <b>{current_score}/{wickets_lost}</b> in <b>{display_overs_str}</b> ov</span>
                <span style="color: #00E676; font-size: 0.95rem; font-weight: bold;">CRR: {crr:.2f} RPO</span>
            </div>
        """, unsafe_allow_html=True)

        if validation_errors:
            for err in validation_errors:
                st.warning(f"⚠️ {err}")

        st.markdown("<br>", unsafe_allow_html=True)
        predict_clicked = st.button("⚡ PREDICT FINAL INNINGS SCORE", disabled=bool(validation_errors))
        st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Prediction Processing & Results Column
    # -------------------------------------------------------------------------
    with col_output:
        if not validation_errors:
            # Input dataframe matching training features
            input_df = pd.DataFrame([{
                "batting_team": batting_team,
                "bowling_team": bowling_team,
                "current_score": int(current_score),
                "overs_completed": float(exact_overs_float),
                "wickets_lost": int(wickets_lost),
                "runs_last_5_overs": int(runs_last_5),
                "current_run_rate": float(crr)
            }])

            # Perform Inference
            raw_prediction = model.predict(input_df)[0]
            # Domain logic: Final score cannot be less than current score
            predicted_score = max(int(round(raw_prediction)), current_score)

            remaining_overs = max(20.0 - exact_overs_float, 0.0)
            estimated_runs_remaining = max(predicted_score - current_score, 0)
            projected_rrr = round(estimated_runs_remaining / remaining_overs, 2) if remaining_overs > 0 else 0.0

            # Prediction Interval (based on measured test MAE)
            lower_range = max(int(round(predicted_score - mae_margin)), current_score)
            upper_range = int(round(predicted_score + mae_margin))

            # Hero Card
            st.markdown(f"""
                <div class="hero-pred-box">
                    <div style="font-size: 0.95rem; text-transform: uppercase; letter-spacing: 1.5px; color: #6EE7B7; margin-bottom: 4px;">
                        Predicted 1st Innings Total
                    </div>
                    <div class="hero-score">{predicted_score}</div>
                    <div class="hero-subtitle">
                        Expected Range: <b>{lower_range} – {upper_range}</b> runs <span style="font-size: 0.85rem; opacity: 0.8;">(±{mae_margin:.1f} MAE)</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Key Match Metrics Grid
            st.markdown(f"""
                <div style="display: flex; flex-wrap: wrap; justify-content: space-around; margin-bottom: 12px;">
                    <div class="metric-badge">
                        <div class="metric-val">{crr:.2f}</div>
                        <div class="metric-label">Current Run Rate</div>
                    </div>
                    <div class="metric-badge">
                        <div class="metric-val">+{estimated_runs_remaining}</div>
                        <div class="metric-label">Runs Remaining</div>
                    </div>
                    <div class="metric-badge">
                        <div class="metric-val">{remaining_overs:.1f}</div>
                        <div class="metric-label">Overs Left</div>
                    </div>
                    <div class="metric-badge">
                        <div class="metric-val">{projected_rrr:.2f}</div>
                        <div class="metric-label">Projected RPO</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Chart Visualization
            st.markdown('<div class="cricket-card">', unsafe_allow_html=True)
            st.markdown("#### 📈 Innings Progression & Model Trajectory")
            chart_fig = plot_innings_projection(
                exact_overs_float, current_score, predicted_score, mae_margin, batting_team, bowling_team
            )
            st.pyplot(chart_fig)
            st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.info("👈 Please resolve the validation notice in the inputs panel to view score projection.")

    # -------------------------------------------------------------------------
    # Evaluation Metrics & Viva Explanation Section
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🎓 Model Performance & NTCC Viva Highlights")

    t1, t2, t3 = st.tabs([
        "📊 Genuine Test Evaluation Metrics",
        "🧠 Machine Learning Methodology",
        "❓ Top College Viva Questions & Answers"
    ])

    with t1:
        if metadata:
            eval_data = metadata.get("evaluation_metrics", {})
            ds_stats = metadata.get("dataset_statistics", {})
            base_metrics = eval_data.get("baseline_linear_regression", {})
            rf_metrics = eval_data.get("random_forest_regressor", {})

            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.markdown("#### 🏆 Model Comparison on Held-Out Test Matches")
                comparison_df = pd.DataFrame([
                    {
                        "Model": "Baseline (Linear Regression)",
                        "Mean Absolute Error (MAE)": f"{base_metrics.get('mae', 'N/A')} runs",
                        "Root Mean Sq Error (RMSE)": f"{base_metrics.get('rmse', 'N/A')} runs",
                        "R² Score": f"{base_metrics.get('r2_score', 'N/A')}"
                    },
                    {
                        "Model": "Random Forest Regressor (Selected)",
                        "Mean Absolute Error (MAE)": f"{rf_metrics.get('mae', 'N/A')} runs",
                        "Root Mean Sq Error (RMSE)": f"{rf_metrics.get('rmse', 'N/A')} runs",
                        "R² Score": f"{rf_metrics.get('r2_score', 'N/A')}"
                    }
                ])
                st.dataframe(comparison_df, hide_index=True)

            with col_m2:
                st.markdown("#### 📦 Training Dataset Summary")
                st.markdown(f"""
                - **Data Source:** Cricsheet Men's T20 Internationals (ball-by-ball)
                - **Total Matches Analyzed:** {ds_stats.get('total_matches_evaluated', 'N/A')} matches
                - **Train / Test Split:** {ds_stats.get('train_matches', 'N/A')} matches (Train) vs {ds_stats.get('test_matches', 'N/A')} matches (Held-Out Test)
                - **Total Snapshots Evaluated:** {ds_stats.get('total_snapshots', 'N/A'):,} instances
                - **Average 1st Innings Total:** {ds_stats.get('average_first_innings_score', 'N/A')} runs (Std: {ds_stats.get('std_first_innings_score', 'N/A')} runs)
                """)

    with t2:
        st.markdown("""
        #### 1. Why Split by Match Rather than Individual Snapshots?
        In cricket, balls within the same match share identical pitch conditions, weather, boundary sizes, and player lineups.
        If snapshots from the same match were randomly placed in both training and test sets, the model would memorize match-specific conditions.
        **Solution:** We split by **`match_id`**, ensuring the test set consists entirely of unseen matches, completely eliminating data leakage.

        #### 2. Handling Cricket Nuances (Legal Deliveries & Extras)
        - Deliveries where `wides > 0` or `noballs > 0` are **illegal deliveries** and re-bowled.
        - `overs_completed` is strictly based on **legal balls bowled** (`legal_ball_count // 6 + (legal_ball_count % 6) / 6.0`).
        - Extras (byes, leg-byes, wides, no-balls, penalty) are fully counted toward the cumulative innings total.
        - Only completed matches reaching >= 19 overs or 10 wickets are included to eliminate rain-shortened DLS skews.

        #### 3. Features Used at Inference
        1. `batting_team` (One-hot encoded)
        2. `bowling_team` (One-hot encoded)
        3. `current_score` (Cumulative runs up to snapshot)
        4. `overs_completed` (Continuous fractional overs)
        5. `wickets_lost` (Cumulative dismissals)
        6. `runs_last_5_overs` (Momentum indicator: runs in preceding 30 legal deliveries)
        7. `current_run_rate` (Runs per over up to this point)
        """)

    with t3:
        st.markdown("""
        #### Common Viva Questions for College Evaluation:
        1. **Q: What is the target variable of this model?**
           *A:* The final total score of the 1st innings of the T20 match (`final_innings_total`).
        2. **Q: Why predict only the 1st innings?**
           *A:* In the 2nd innings, teams bat according to a specific target score. Once the target is reached, the innings ends early. The 1st innings represents an unconstrained scoring effort.
        3. **Q: How does the model prevent data leakage?**
           *A:* Training and testing data are split on unique `match_id` rather than random row splitting. All snapshots from a match remain strictly in either train or test.
        4. **Q: What does Mean Absolute Error (MAE) mean in this context?**
           *A:* An MAE of ~14.6 runs means that, on average, the model's predicted final score deviates from the actual final score by approximately 14-15 runs on unseen matches.
        5. **Q: Why is Random Forest preferred over simple Linear Regression?**
           *A:* Cricket scoring exhibits non-linear acceleration in death overs and steep scoring drops when clusters of wickets fall, which tree ensembles model better than linear relationships.
        """)


if __name__ == "__main__":
    main()
