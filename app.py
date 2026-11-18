from pathlib import Path
from html import escape

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Student Social Media Impact Predictor",
    page_icon=":iphone:",
    layout="wide",
    initial_sidebar_state="expanded",
)

BUNDLE_FILENAME = "social_media_impact_pipeline.joblib"

GENDER_OPTIONS = ["Female", "Male", "Non-Binary", "Prefer not to say"]
ACADEMIC_LEVEL_OPTIONS = ["High School", "Undergraduate", "Postgraduate"]
PLATFORM_OPTIONS = ["Instagram", "TikTok", "YouTube", "Snapchat", "Reddit", "LinkedIn", "X (Twitter)"]
DEVICE_OPTIONS = ["Smartphone", "Laptop/PC", "Tablet"]
SOCIAL_OPTIONS = ["Never", "Rarely", "Sometimes", "Frequently", "Always"]

TEAM_MEMBERS = [
    {"name": "BOUALEM ABDEL AZIZ", "university_id": "443058333"},
    {"name": "ABDEL MONEM", "university_id": "443058333"},
    {"name": "ABDUL HASIB ZAHID", "university_id": "443058333"},
]

DEFAULT_NOMINAL_COLS = ["Gender", "Primary_Platform", "Device_Type"]
DEFAULT_NUM_COLS = [
    "Age",
    "Daily_Usage_Hours",
    "Weekend_Extra_Hours",
    "Sleep_Duration_Hours",
    "Sleep_Quality_Score",
    "Perceived_Stress_Score",
    "Mental_Health_Index",
    "Academic_Performance_GPA",
]

RAW_COLUMNS = [
    "Age",
    "Gender",
    "Academic_Level",
    "Academic_Performance_GPA",
    "Primary_Platform",
    "Device_Type",
    "Daily_Usage_Hours",
    "Weekend_Extra_Hours",
    "Late_Night_Usage",
    "Sleep_Duration_Hours",
    "Sleep_Quality_Score",
    "Social_Comparison_Frequency",
    "Perceived_Stress_Score",
    "Mental_Health_Index",
]

PRESETS = {
    "Balanced Student": {
        "Age": 19,
        "Gender": "Female",
        "Academic_Level": "Undergraduate",
        "Academic_Performance_GPA": 3.72,
        "Primary_Platform": "YouTube",
        "Device_Type": "Laptop/PC",
        "Daily_Usage_Hours": 2.4,
        "Weekend_Extra_Hours": 0.8,
        "Late_Night_Usage": False,
        "Sleep_Duration_Hours": 8.2,
        "Sleep_Quality_Score": 5,
        "Social_Comparison_Frequency": "Rarely",
        "Perceived_Stress_Score": 7.0,
        "Mental_Health_Index": 91,
    },
    "Heavy Late-Night User": {
        "Age": 20,
        "Gender": "Male",
        "Academic_Level": "Undergraduate",
        "Academic_Performance_GPA": 2.75,
        "Primary_Platform": "TikTok",
        "Device_Type": "Smartphone",
        "Daily_Usage_Hours": 11.5,
        "Weekend_Extra_Hours": 4.2,
        "Late_Night_Usage": True,
        "Sleep_Duration_Hours": 4.4,
        "Sleep_Quality_Score": 2,
        "Social_Comparison_Frequency": "Frequently",
        "Perceived_Stress_Score": 29.0,
        "Mental_Health_Index": 51,
    },
    "High-Stress Exam Week": {
        "Age": 21,
        "Gender": "Non-Binary",
        "Academic_Level": "Postgraduate",
        "Academic_Performance_GPA": 3.18,
        "Primary_Platform": "Instagram",
        "Device_Type": "Smartphone",
        "Daily_Usage_Hours": 7.6,
        "Weekend_Extra_Hours": 3.1,
        "Late_Night_Usage": True,
        "Sleep_Duration_Hours": 5.4,
        "Sleep_Quality_Score": 3,
        "Social_Comparison_Frequency": "Always",
        "Perceived_Stress_Score": 35.0,
        "Mental_Health_Index": 47,
    },
}

STYLES = """
<style>
    .hero {
        background: linear-gradient(120deg, #0f172a 0%, #1e3a8a 55%, #0e7490 100%);
        border-radius: 18px;
        padding: 1.6rem 2rem 1.4rem;
        color: #f8fafc;
        margin-bottom: 0.9rem;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.18);
    }
    .hero .eyebrow {
        font-size: 0.76rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: #7dd3fc;
        font-weight: 700;
    }
    .hero h1 {
        margin: 0.35rem 0 0.4rem;
        font-size: 2.2rem;
        line-height: 1.15;
        color: #ffffff;
    }
    .hero p {
        margin: 0;
        max-width: 76ch;
        color: #dbeafe;
        font-size: 1rem;
    }
    .model-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        background: #ecfdf5;
        color: #065f46;
        border: 1px solid #6ee7b7;
        border-radius: 999px;
        padding: 0.35rem 0.85rem;
        font-size: 0.85rem;
        font-weight: 700;
    }
    .model-pill .dot {
        width: 0.5rem;
        height: 0.5rem;
        border-radius: 50%;
        background: #10b981;
        box-shadow: 0 0 0 4px rgba(16, 185, 129, 0.18);
    }
    .section-label {
        font-size: 0.74rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        font-weight: 800;
        color: #2563eb;
        margin-bottom: 0.2rem;
    }
    .input-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.1rem 1.1rem 0.7rem;
        height: 100%;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
    }
    .form-intro {
        background: #f1f5f9;
        border-left: 4px solid #2563eb;
        border-radius: 10px;
        padding: 0.7rem 0.9rem;
        margin-bottom: 0.9rem;
        color: #334155;
        font-size: 0.9rem;
        line-height: 1.4;
    }
    .preset-note {
        font-size: 0.85rem;
        color: #475569;
        line-height: 1.35;
    }
    .team-card {
        background: linear-gradient(160deg, #f8fafc, #eef2ff);
        border: 1px solid #dbe3f0;
        border-radius: 12px;
        padding: 0.7rem 0.9rem;
        height: 100%;
        box-shadow: 0 6px 16px rgba(15, 23, 42, 0.05);
    }
    .team-name {
        display: block;
        color: #0f172a;
        font-weight: 700;
        font-size: 0.85rem;
        line-height: 1.25;
    }
    .team-label {
        display: block;
        margin-top: 0.45rem;
        font-size: 0.65rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #64748b;
        font-weight: 800;
    }
    .team-id {
        display: block;
        margin-top: 0.1rem;
        color: #1d4ed8;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 0.78rem;
    }
    .tip {
        border-left: 4px solid #94a3b8;
        background: #f8fafc;
        border-radius: 8px;
        padding: 0.65rem 0.85rem;
        margin-bottom: 0.55rem;
        font-size: 0.92rem;
        line-height: 1.4;
    }
    .tip strong { display: block; color: #0f172a; margin-bottom: 0.1rem; }
    .tip-high { border-left-color: #ef4444; background: #fef2f2; }
    .tip-medium { border-left-color: #f59e0b; background: #fffbeb; }
    .tip-good { border-left-color: #10b981; background: #ecfdf5; }
    .result-caption {
        color: #64748b;
        font-size: 0.87rem;
        margin-top: -0.4rem;
        margin-bottom: 0.9rem;
    }
    [data-testid="stSidebar"] .stButton button {
        text-align: left;
        font-weight: 600;
    }
    [data-testid="stMetricValue"] { font-size: 1.5rem; }
    .footer {
        margin-top: 2rem;
        padding-top: 0.8rem;
        border-top: 1px solid #e2e8f0;
        color: #94a3b8;
        font-size: 0.8rem;
    }
</style>
"""

st.markdown(STYLES, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading trained model...")
def load_pipeline(path: str):
    return joblib.load(path)


bundle_path = Path(__file__).resolve().with_name(BUNDLE_FILENAME)
try:
    bundle = load_pipeline(str(bundle_path))
except FileNotFoundError:
    st.error(
        f"Model bundle not found: `{BUNDLE_FILENAME}`. Place the file next to `app.py` "
        "(it is generated by the final export cell of the training notebook), then reload this page."
    )
    st.stop()
except Exception as exc:
    st.error(f"Could not load `{BUNDLE_FILENAME}`: {exc}")
    st.stop()

required_keys = [
    "model",
    "model_name",
    "stress_imputer",
    "gpa_imputer",
    "social_encoder",
    "academic_encoder",
    "ohe",
    "scaler",
    "label_encoder",
    "feature_columns",
]
missing_keys = [key for key in required_keys if key not in bundle]
if missing_keys:
    st.error(f"The model bundle is missing required objects: {', '.join(missing_keys)}.")
    st.stop()

model = bundle["model"]
model_name = str(bundle["model_name"])
stress_imputer = bundle["stress_imputer"]
gpa_imputer = bundle["gpa_imputer"]
for fitted_imputer in (stress_imputer, gpa_imputer):
    if not hasattr(fitted_imputer, "_fill_dtype") and hasattr(fitted_imputer, "_fit_dtype"):
        fitted_imputer._fill_dtype = fitted_imputer._fit_dtype
social_encoder = bundle["social_encoder"]
academic_encoder = bundle["academic_encoder"]
ohe = bundle["ohe"]
scaler = bundle["scaler"]
label_encoder = bundle["label_encoder"]
feature_columns = list(bundle["feature_columns"])
nominal_cols = list(bundle.get("nominal_cols", DEFAULT_NOMINAL_COLS))
num_cols = list(bundle.get("num_cols", DEFAULT_NUM_COLS))


def current(key: str, default):
    return st.session_state[key] if key in st.session_state else default


def apply_preset(name: str) -> None:
    for key, value in PRESETS[name].items():
        st.session_state[key] = value
    st.session_state.pop("last_prediction", None)


def preprocess(raw: pd.DataFrame) -> pd.DataFrame:
    processed = raw.copy()
    processed[["Perceived_Stress_Score"]] = stress_imputer.transform(
        processed[["Perceived_Stress_Score"]]
    )
    processed[["Academic_Performance_GPA"]] = gpa_imputer.transform(
        processed[["Academic_Performance_GPA"]]
    )
    processed[["Social_Comparison_Frequency"]] = social_encoder.transform(
        processed[["Social_Comparison_Frequency"]]
    )
    processed[["Academic_Level"]] = academic_encoder.transform(
        processed[["Academic_Level"]]
    )

    encoded = ohe.transform(processed[nominal_cols])
    if hasattr(encoded, "toarray"):
        encoded = encoded.toarray()
    encoded_df = pd.DataFrame(
        encoded,
        columns=ohe.get_feature_names_out(nominal_cols),
        index=processed.index,
    )
    processed = pd.concat([processed.drop(columns=nominal_cols), encoded_df], axis=1)
    processed[num_cols] = scaler.transform(processed[num_cols])

    missing_features = [column for column in feature_columns if column not in processed.columns]
    if missing_features:
        raise ValueError(f"Processed features are missing: {', '.join(missing_features)}")
    return processed[feature_columns]


def wellness_tips(row: pd.Series) -> list:
    tips = []
    usage = float(row["Daily_Usage_Hours"])
    weekend = float(row["Weekend_Extra_Hours"])
    sleep = float(row["Sleep_Duration_Hours"])
    quality = int(row["Sleep_Quality_Score"])
    stress = float(row["Perceived_Stress_Score"])
    mental_health = int(row["Mental_Health_Index"])
    gpa = float(row["Academic_Performance_GPA"])
    comparison = row["Social_Comparison_Frequency"]

    if usage >= 6.0 or weekend >= 3.0:
        tips.append(
            ("high", "Set a daily screen-time ceiling",
             f"You are at {usage:.1f} h/day plus {weekend:.1f} h on weekends. "
             "Try a 4-5 h weekday cap and schedule one screen-free block each evening.")
        )
    if bool(row["Late_Night_Usage"]):
        tips.append(
            ("high", "Create a wind-down buffer",
             "Late-night scrolling delays sleep onset. Keep phones out of the bedroom "
             "or switch on a grayscale / do-not-disturb schedule 60 minutes before bed.")
        )
    if sleep < 7.0 or quality <= 2:
        tips.append(
            ("high", "Protect your sleep",
             f"Current sleep is {sleep:.1f} h with quality {quality}/5. "
             "Aim for 7-9 hours and a consistent wake time, even on weekends.")
        )
    if stress >= 20.0:
        tips.append(
            ("high", "Add stress-recovery breaks",
             f"Your stress score is {stress:.0f}/40. Take short offline breaks, "
             "use breathing exercises, and prioritise tasks during exam periods.")
        )
    if comparison in ("Frequently", "Always"):
        tips.append(
            ("medium", "Curate your feed",
             "Frequent comparison is linked to lower wellbeing. Unfollow accounts that trigger "
             "comparison and follow educational or supportive communities instead.")
        )
    if mental_health < 65:
        tips.append(
            ("medium", "Check in with your wellbeing",
             f"Mental health index is {mental_health}/100. Consider talking with a counsellor "
             "or trusted peer, and keep social media use intentional rather than automatic.")
        )
    if gpa < 3.0:
        tips.append(
            ("medium", "Build a study-first routine",
             f"GPA is {gpa:.2f}. Try focused study blocks with the phone in another room, "
             "then reward yourself with a deliberate, time-boxed scroll.")
        )
    if not tips:
        tips.append(
            ("good", "Healthy digital balance",
             "Your usage, sleep, stress, and comparison habits look well balanced. "
             "Keep the routine going and review it during high-pressure weeks.")
        )
    return tips


st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Student wellbeing analytics</div>
        <h1>Student Social Media Impact Predictor</h1>
        <p>Explore how usage habits, sleep, stress, and social comparison combine into an overall
        digital-wellbeing outcome. Adjust a profile below and run the prediction to see the
        model's confidence and personalised recommendations.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

badge_col, guide_col = st.columns([1.05, 2.4])
with badge_col:
    st.markdown(
        f'<div class="model-pill"><span class="dot"></span>Active model: '
        f'{escape(model_name)}</div>',
        unsafe_allow_html=True,
    )
with guide_col:
    st.caption("Three outcome classes: Beneficial, Neutral, Negative. Predictions are estimates, not medical advice.")

st.markdown('<div class="section-label">Project team</div>', unsafe_allow_html=True)
team_cols = st.columns(len(TEAM_MEMBERS))
for team_col, member in zip(team_cols, TEAM_MEMBERS):
    with team_col:
        st.markdown(
            f'<div class="team-card"><span class="team-name">{escape(member["name"])}</span>'
            f'<span class="team-label">University ID</span>'
            f'<span class="team-id">{escape(member["university_id"])}</span></div>',
            unsafe_allow_html=True,
        )

st.divider()

with st.sidebar:
    st.header("Quick Sample Presets")
    st.markdown(
        '<div class="preset-note">Load a one-click student profile, then press '
        '<strong>Predict Impact</strong>.</div>',
        unsafe_allow_html=True,
    )
    for preset_name in PRESETS:
        if st.button(preset_name, key=f"preset_{preset_name}", width="stretch"):
            apply_preset(preset_name)
            st.rerun()
    st.divider()
    st.subheader("How it works")
    st.markdown(
        """
        1. Inputs are encoded with the fitted imputers, encoders, and scaler from training.
        2. The classifier returns one of three impact categories.
        3. Confidence scores and tailored wellness tips are shown instantly.
        """
    )
    st.caption(f"Bundle: {BUNDLE_FILENAME}")

st.subheader("Student profile form")
st.markdown(
    '<div class="form-intro">Complete the 14 fields below, or load a sample preset from the '
    'sidebar. Click <strong>Predict Impact</strong> to see the result, confidence breakdown, '
    'and a personalised wellness plan.</div>',
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Demographics & academics</div>', unsafe_allow_html=True)
    age = st.slider("Age", 15, 26, current("Age", 19), key="Age")
    gender = st.selectbox("Gender", GENDER_OPTIONS, index=GENDER_OPTIONS.index(current("Gender", "Female")), key="Gender")
    academic_level = st.selectbox(
        "Academic Level",
        ACADEMIC_LEVEL_OPTIONS,
        index=ACADEMIC_LEVEL_OPTIONS.index(current("Academic_Level", "Undergraduate")),
        key="Academic_Level",
    )
    gpa = st.slider(
        "Academic Performance GPA", 1.50, 4.00, current("Academic_Performance_GPA", 3.40), 0.01,
        key="Academic_Performance_GPA",
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Social media habits</div>', unsafe_allow_html=True)
    primary_platform = st.selectbox(
        "Primary Platform",
        PLATFORM_OPTIONS,
        index=PLATFORM_OPTIONS.index(current("Primary_Platform", "Instagram")),
        key="Primary_Platform",
    )
    device_type = st.selectbox(
        "Device Type",
        DEVICE_OPTIONS,
        index=DEVICE_OPTIONS.index(current("Device_Type", "Smartphone")),
        key="Device_Type",
    )
    daily_usage = st.slider(
        "Daily Usage Hours", 0.5, 14.0, current("Daily_Usage_Hours", 5.0), 0.1,
        key="Daily_Usage_Hours",
    )
    weekend_extra = st.slider(
        "Weekend Extra Hours", 0.0, 5.0, current("Weekend_Extra_Hours", 1.8), 0.1,
        key="Weekend_Extra_Hours",
    )
    late_night = st.toggle(
        "Late-Night Usage", current("Late_Night_Usage", False), key="Late_Night_Usage"
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Sleep & wellbeing</div>', unsafe_allow_html=True)
    sleep_duration = st.slider(
        "Sleep Duration Hours", 3.0, 11.0, current("Sleep_Duration_Hours", 7.0), 0.1,
        key="Sleep_Duration_Hours",
    )
    sleep_quality = st.slider(
        "Sleep Quality Score", 1, 5, current("Sleep_Quality_Score", 3), 1,
        key="Sleep_Quality_Score",
    )
    social_comparison = st.select_slider(
        "Social Comparison Frequency",
        SOCIAL_OPTIONS,
        value=current("Social_Comparison_Frequency", "Sometimes"),
        key="Social_Comparison_Frequency",
    )
    stress_score = st.slider(
        "Perceived Stress Score", 0.0, 40.0, current("Perceived_Stress_Score", 14.0), 1.0,
        key="Perceived_Stress_Score",
    )
    mental_health = st.slider(
        "Mental Health Index", 30, 100, current("Mental_Health_Index", 80), 1,
        key="Mental_Health_Index",
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.divider()

action_col, auto_col, note_col = st.columns([1.1, 1.1, 3])
with action_col:
    predict_clicked = st.button("Predict Impact", type="primary", width="stretch")
with auto_col:
    auto_predict = st.toggle("Auto-predict", value=False, key="auto_predict")
with note_col:
    st.caption("Auto-predict recomputes the result whenever an input changes.")

if predict_clicked or auto_predict:
    raw = pd.DataFrame(
        [
            {
                "Age": age,
                "Gender": gender,
                "Academic_Level": academic_level,
                "Academic_Performance_GPA": gpa,
                "Primary_Platform": primary_platform,
                "Device_Type": device_type,
                "Daily_Usage_Hours": daily_usage,
                "Weekend_Extra_Hours": weekend_extra,
                "Late_Night_Usage": int(late_night),
                "Sleep_Duration_Hours": sleep_duration,
                "Sleep_Quality_Score": sleep_quality,
                "Social_Comparison_Frequency": social_comparison,
                "Perceived_Stress_Score": stress_score,
                "Mental_Health_Index": mental_health,
            }
        ],
        columns=RAW_COLUMNS,
    )
    try:
        processed = preprocess(raw.copy())
        prediction = int(model.predict(processed)[0])
        label = str(label_encoder.inverse_transform([prediction])[0])
        probabilities = None
        if hasattr(model, "predict_proba"):
            probabilities = np.asarray(model.predict_proba(processed))[0]
        st.session_state["last_prediction"] = {
            "label": label,
            "probabilities": probabilities,
            "raw": raw,
        }
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

if "last_prediction" in st.session_state:
    result = st.session_state["last_prediction"]
    label = result["label"]
    probabilities = result["probabilities"]
    raw = result["raw"]
    row = raw.iloc[0]

    snapshot_cols = st.columns(4)
    snapshot_cols[0].metric("Daily usage", f"{float(row['Daily_Usage_Hours']):.1f} h")
    snapshot_cols[1].metric("Sleep", f"{float(row['Sleep_Duration_Hours']):.1f} h")
    snapshot_cols[2].metric("Stress score", f"{float(row['Perceived_Stress_Score']):.0f}/40")
    snapshot_cols[3].metric("Mental health", f"{int(row['Mental_Health_Index'])}/100")

    st.subheader("Prediction result")
    st.markdown(
        '<div class="result-caption">Based on the submitted 14-field student profile.</div>',
        unsafe_allow_html=True,
    )

    result_left, result_right = st.columns([1.15, 1])
    with result_left:
        if label == "Beneficial":
            st.success(
                f"**Overall impact: {label}**  \n"
                "Social media habits appear to support this student's wellbeing and academics."
            )
        elif label == "Neutral":
            st.warning(
                f"**Overall impact: {label}**  \n"
                "Effects are mixed; small changes in sleep or screen time could shift the outcome."
            )
        else:
            st.error(
                f"**Overall impact: {label}**  \n"
                "Current habits show signs of strain on wellbeing or academic performance."
            )

        st.markdown("#### Personalised digital wellness plan")
        for severity, title, text in wellness_tips(row):
            st.markdown(
                f'<div class="tip tip-{severity}"><strong>{escape(title)}</strong>'
                f'{escape(text)}</div>',
                unsafe_allow_html=True,
            )

    with result_right:
        st.markdown("#### Confidence breakdown")
        if probabilities is not None and len(probabilities) == len(label_encoder.classes_):
            confidence = pd.DataFrame(
                {
                    "Impact Category": label_encoder.classes_,
                    "Confidence (%)": np.round(probabilities * 100, 2),
                }
            ).set_index("Impact Category")
            st.bar_chart(confidence)
            st.dataframe(
                confidence.sort_values("Confidence (%)", ascending=False),
                width="stretch",
            )
        else:
            st.info("This classifier does not expose probability estimates.")

st.markdown(
    '<div class="footer">Estimates from a trained classifier. Use them as a reflection aid, '
    'not as a diagnosis or substitute for professional advice.</div>',
    unsafe_allow_html=True,
)
