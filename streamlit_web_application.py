import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page Configuration
st.set_page_config(
    page_title="Student Social Media Impact Predictor",
    page_icon="📱",
    layout="wide"
)

# Load the saved deployment bundle
@st.cache_resource
def load_pipeline():
    return joblib.load("social_media_impact_pipeline.joblib")

try:
    bundle = load_pipeline()
    model = bundle["model"]
    model_name = bundle["model_name"]
    stress_imputer = bundle["stress_imputer"]
    gpa_imputer = bundle["gpa_imputer"]
    social_encoder = bundle["social_encoder"]
    academic_encoder = bundle["academic_encoder"]
    ohe = bundle["ohe"]
    scaler = bundle["scaler"]
    label_encoder = bundle["label_encoder"]
    nominal_cols = bundle["nominal_cols"]
    num_cols = bundle["num_cols"]
    feature_columns = bundle["feature_columns"]
except FileNotFoundError:
    st.error("❌ `social_media_impact_pipeline.joblib` not found! Please run Section 12 of your Jupyter Notebook first and place the `.joblib` file in the same folder as `app.py`.")
    st.stop()

# Header
st.title("📱 Student Social Media Impact Predictor")
st.markdown(
    f"Predict whether social media has a **Beneficial**, **Neutral**, or **Negative** overall impact on a student's life "
    f"using our trained **{model_name}** model."
)
st.divider()

# Input Form Layout (3 Columns)
st.subheader("📋 Enter Student Profile & Social Media Habits")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**1. Demographics & Academics**")
    age = st.slider("Age (Years)", min_value=14, max_value=30, value=19, step=1)
    gender = st.selectbox("Gender", options=["Female", "Male", "Non-Binary", "Prefer not to say"])
    academic_level = st.selectbox("Academic Level", options=["High School", "Undergraduate", "Postgraduate"])
    gpa = st.slider("Academic Performance (GPA)", min_value=1.0, max_value=4.0, value=3.40, step=0.01)

with col2:
    st.markdown("**2. Social Media Usage**")
    primary_platform = st.selectbox(
        "Primary Platform",
        options=["Instagram", "TikTok", "YouTube", "Snapchat", "Reddit", "LinkedIn", "X (Twitter)"]
    )
    device_type = st.selectbox("Primary Device Type", options=["Smartphone", "Laptop/PC", "Tablet"])
    daily_usage = st.slider("Daily Usage Hours", min_value=0.0, max_value=16.0, value=4.5, step=0.1)
    weekend_extra = st.slider("Weekend Extra Hours", min_value=0.0, max_value=8.0, value=1.8, step=0.1)
    late_night = st.toggle("Regular Late-Night Usage", value=False)

with col3:
    st.markdown("**3. Sleep & Mental Wellbeing**")
    sleep_duration = st.slider("Sleep Duration (Hours/Night)", min_value=3.0, max_value=12.0, value=7.0, step=0.1)
    sleep_quality = st.slider("Sleep Quality Score (1 = Poor, 5 = Excellent)", min_value=1, max_value=5, value=3, step=1)
    social_comparison = st.selectbox(
        "Social Comparison Frequency",
        options=["Never", "Rarely", "Sometimes", "Frequently", "Always"]
    )
    stress_score = st.slider("Perceived Stress Score (0 - 40)", min_value=0.0, max_value=40.0, value=14.0, step=1.0)
    mental_health = st.slider("Mental Health Index (0 - 100)", min_value=20, max_value=100, value=80, step=1)

st.divider()

# Prediction Button
if st.button("🔍 Predict Overall Impact", type="primary", use_container_width=True):
    # 1. Construct raw input DataFrame matching training columns
    input_df = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Academic_Level": academic_level,
        "Primary_Platform": primary_platform,
        "Daily_Usage_Hours": daily_usage,
        "Weekend_Extra_Hours": weekend_extra,
        "Device_Type": device_type,
        "Sleep_Duration_Hours": sleep_duration,
        "Sleep_Quality_Score": sleep_quality,
        "Late_Night_Usage": int(late_night),
        "Social_Comparison_Frequency": social_comparison,
        "Perceived_Stress_Score": stress_score,
        "Mental_Health_Index": mental_health,
        "Academic_Performance_GPA": gpa
    }])

    # 2. Apply exact preprocessing steps from the notebook
    input_df[["Perceived_Stress_Score"]] = stress_imputer.transform(input_df[["Perceived_Stress_Score"]])
    input_df[["Academic_Performance_GPA"]] = gpa_imputer.transform(input_df[["Academic_Performance_GPA"]])

    input_df[["Social_Comparison_Frequency"]] = social_encoder.transform(input_df[["Social_Comparison_Frequency"]])
    input_df[["Academic_Level"]] = academic_encoder.transform(input_df[["Academic_Level"]])

    ohe_encoded = pd.DataFrame(
        ohe.transform(input_df[nominal_cols]),
        columns=ohe.get_feature_names_out(nominal_cols),
        index=input_df.index
    )
    processed_df = pd.concat([input_df.drop(columns=nominal_cols), ohe_encoded], axis=1)

    processed_df[num_cols] = scaler.transform(processed_df[num_cols])

    # Ensure exact column order
    processed_df = processed_df[feature_columns]

    # 3. Run Prediction
    pred_idx = model.predict(processed_df)[0]
    pred_label = label_encoder.inverse_transform([pred_idx])[0]

    # 4. Display Results
    res_col1, res_col2 = st.columns([1.2, 1])

    with res_col1:
        st.subheader("📊 Prediction Result")
        if pred_label == "Beneficial":
            st.success(f"### Overall Impact: **{pred_label}** 🌟")
            st.write("This student maintains a healthy balance between social media usage, sleep, and academic wellbeing.")
        elif pred_label == "Neutral":
            st.warning(f"### Overall Impact: **{pred_label}** ⚖️")
            st.write("Social media usage shows mixed effects. Minor improvements in sleep or screen time could shift this toward Beneficial.")
        else:
            st.error(f"### Overall Impact: **{pred_label}** ⚠️")
            st.write("High screen time, stress, or disrupted sleep patterns indicate an adverse impact on wellbeing and academic performance.")

        # Personalized Recommendations
        st.markdown("#### 💡 Actionable Wellness Insights")
        tips = []
        if daily_usage > 6.0:
            tips.append(f"- **Reduce Daily Screen Time:** Currently at `{daily_usage} hrs/day`. Aim for under 4–5 hours.")
        if late_night:
            tips.append("- **Avoid Late-Night Usage:** Turning off screens 1 hour before bed improves sleep quality.")
        if sleep_duration < 7.0:
            tips.append(f"- **Increase Sleep Duration:** Currently getting `{sleep_duration} hrs`. Target 7–8 hours per night.")
        if stress_score > 20:
            tips.append(f"- **Manage Perceived Stress:** Stress score (`{int(stress_score)}/40`) is elevated; consider digital detox breaks.")
        if social_comparison in ["Frequently", "Always"]:
            tips.append("- **Curate Feeds:** Frequent social comparison lowers mental health index; unfollow stress-inducing accounts.")
        if not tips:
            tips.append("- **Keep It Up!** Current habits align well with positive academic and mental health outcomes.")

        for tip in tips:
            st.markdown(tip)

    with res_col2:
        st.subheader("📈 Class Confidence Probabilities")
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(processed_df)[0]
            prob_df = pd.DataFrame({
                "Impact Category": label_encoder.classes_,
                "Probability (%)": np.round(probs * 100, 2)
            }).set_index("Impact Category")
            st.bar_chart(prob_df)
            st.dataframe(prob_df, use_container_width=True)
        else:
            st.info("Probability breakdown is not available for this classifier (e.g., hinge-loss SGD). Switch loss to `log_loss` in tuning to enable probabilities.")