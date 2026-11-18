from functools import lru_cache
from html import escape
from pathlib import Path
import json
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Student Social Media Impact Predictor")

BASE_DIR = Path(__file__).resolve().parents[1]
BUNDLE_FILENAME = "social_media_impact_pipeline.joblib"

TEAM_MEMBERS = [
    {"name": "BOUALEM ABDEL AZIZ", "university_id": "443058333"},
    {"name": "ABDEL MONEM", "university_id": "443058333"},
    {"name": "ABDUL HASIB ZAHID", "university_id": "443058333"},
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

SUMMARY = {
    "Beneficial": "Social media habits appear to support this student's wellbeing and academics.",
    "Neutral": "Effects are mixed; small changes in sleep or screen time could shift the outcome.",
    "Negative": "Current habits show signs of strain on wellbeing or academic performance.",
}


class StudentProfile(BaseModel):
    Age: int = Field(19, ge=15, le=26)
    Gender: Literal["Female", "Male", "Non-Binary", "Prefer not to say"] = "Female"
    Academic_Level: Literal["High School", "Undergraduate", "Postgraduate"] = "Undergraduate"
    Academic_Performance_GPA: float = Field(3.4, ge=1.5, le=4.0)
    Primary_Platform: Literal[
        "Instagram", "TikTok", "YouTube", "Snapchat", "Reddit", "LinkedIn", "X (Twitter)"
    ] = "Instagram"
    Device_Type: Literal["Smartphone", "Laptop/PC", "Tablet"] = "Smartphone"
    Daily_Usage_Hours: float = Field(5.0, ge=0.5, le=14.0)
    Weekend_Extra_Hours: float = Field(1.8, ge=0.0, le=5.0)
    Late_Night_Usage: bool = False
    Sleep_Duration_Hours: float = Field(7.0, ge=3.0, le=11.0)
    Sleep_Quality_Score: int = Field(3, ge=1, le=5)
    Social_Comparison_Frequency: Literal[
        "Never", "Rarely", "Sometimes", "Frequently", "Always"
    ] = "Sometimes"
    Perceived_Stress_Score: float = Field(14.0, ge=0.0, le=40.0)
    Mental_Health_Index: int = Field(80, ge=30, le=100)


@lru_cache(maxsize=1)
def get_bundle() -> dict:
    path = BASE_DIR / BUNDLE_FILENAME
    if not path.exists():
        raise RuntimeError(
            f"{BUNDLE_FILENAME} is missing. Place it next to api/ in the project root."
        )
    bundle = joblib.load(str(path))
    for fitted_imputer in (bundle["stress_imputer"], bundle["gpa_imputer"]):
        if not hasattr(fitted_imputer, "_fill_dtype") and hasattr(fitted_imputer, "_fit_dtype"):
            fitted_imputer._fill_dtype = fitted_imputer._fit_dtype
    return bundle


def preprocess(bundle: dict, profile: dict) -> pd.DataFrame:
    processed = pd.DataFrame([profile], columns=RAW_COLUMNS)
    processed[["Perceived_Stress_Score"]] = bundle["stress_imputer"].transform(
        processed[["Perceived_Stress_Score"]]
    )
    processed[["Academic_Performance_GPA"]] = bundle["gpa_imputer"].transform(
        processed[["Academic_Performance_GPA"]]
    )
    processed[["Social_Comparison_Frequency"]] = bundle["social_encoder"].transform(
        processed[["Social_Comparison_Frequency"]]
    )
    processed[["Academic_Level"]] = bundle["academic_encoder"].transform(
        processed[["Academic_Level"]]
    )

    nominal_cols = list(bundle.get("nominal_cols", DEFAULT_NOMINAL_COLS))
    num_cols = list(bundle.get("num_cols", DEFAULT_NUM_COLS))
    encoded = bundle["ohe"].transform(processed[nominal_cols])
    if hasattr(encoded, "toarray"):
        encoded = encoded.toarray()
    encoded_df = pd.DataFrame(
        encoded,
        columns=bundle["ohe"].get_feature_names_out(nominal_cols),
        index=processed.index,
    )
    processed = pd.concat([processed.drop(columns=nominal_cols), encoded_df], axis=1)
    processed[num_cols] = bundle["scaler"].transform(processed[num_cols])

    feature_columns = list(bundle["feature_columns"])
    missing = [column for column in feature_columns if column not in processed.columns]
    if missing:
        raise ValueError(f"Processed features are missing: {', '.join(missing)}")
    return processed[feature_columns]


def wellness_tips(row: dict) -> list:
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
            {
                "severity": "high",
                "title": "Set a daily screen-time ceiling",
                "text": f"You are at {usage:.1f} h/day plus {weekend:.1f} h on weekends. "
                "Try a 4-5 h weekday cap and schedule one screen-free block each evening.",
            }
        )
    if bool(row["Late_Night_Usage"]):
        tips.append(
            {
                "severity": "high",
                "title": "Create a wind-down buffer",
                "text": "Late-night scrolling delays sleep onset. Keep phones out of the bedroom "
                "or use a do-not-disturb schedule 60 minutes before bed.",
            }
        )
    if sleep < 7.0 or quality <= 2:
        tips.append(
            {
                "severity": "high",
                "title": "Protect your sleep",
                "text": f"Current sleep is {sleep:.1f} h with quality {quality}/5. "
                "Aim for 7-9 hours and a consistent wake time, even on weekends.",
            }
        )
    if stress >= 20.0:
        tips.append(
            {
                "severity": "high",
                "title": "Add stress-recovery breaks",
                "text": f"Your stress score is {stress:.0f}/40. Take short offline breaks, "
                "use breathing exercises, and prioritise tasks during exam periods.",
            }
        )
    if comparison in ("Frequently", "Always"):
        tips.append(
            {
                "severity": "medium",
                "title": "Curate your feed",
                "text": "Frequent comparison is linked to lower wellbeing. Unfollow accounts "
                "that trigger comparison and follow supportive communities instead.",
            }
        )
    if mental_health < 65:
        tips.append(
            {
                "severity": "medium",
                "title": "Check in with your wellbeing",
                "text": f"Mental health index is {mental_health}/100. Consider talking with a "
                "counsellor or trusted peer, and keep social media use intentional.",
            }
        )
    if gpa < 3.0:
        tips.append(
            {
                "severity": "medium",
                "title": "Build a study-first routine",
                "text": f"GPA is {gpa:.2f}. Try focused study blocks with the phone in another "
                "room, then reward yourself with a time-boxed scroll.",
            }
        )
    if not tips:
        tips.append(
            {
                "severity": "good",
                "title": "Healthy digital balance",
                "text": "Your usage, sleep, stress, and comparison habits look well balanced. "
                "Keep the routine going and review it during high-pressure weeks.",
            }
        )
    return tips


HTML_TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Student Social Media Impact Predictor</title>
<style>
:root { --ink:#0f172a; --muted:#64748b; --blue:#2563eb; --line:#e2e8f0; }
* { box-sizing:border-box; }
body { margin:0; font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; color:var(--ink); background:#f1f5f9; }
.wrap { max-width:1240px; margin:0 auto; padding:22px; }
.hero { background:linear-gradient(120deg,#0f172a 0%,#1e3a8a 55%,#0e7490 100%); color:#f8fafc; border-radius:18px; padding:28px 32px; box-shadow:0 12px 30px rgba(15,23,42,.18); }
.hero .eyebrow { font-size:.75rem; letter-spacing:.16em; text-transform:uppercase; color:#7dd3fc; font-weight:700; }
.hero h1 { margin:8px 0 10px; font-size:2.15rem; line-height:1.15; }
.hero p { margin:0; max-width:78ch; color:#dbeafe; line-height:1.55; }
.pill { display:inline-flex; gap:.5rem; align-items:center; margin-top:16px; background:rgba(236,253,245,.95); color:#065f46; border:1px solid #6ee7b7; border-radius:999px; padding:.35rem .85rem; font-size:.85rem; font-weight:700; }
.dot { width:.5rem; height:.5rem; border-radius:50%; background:#10b981; box-shadow:0 0 0 4px rgba(16,185,129,.18); }
.team-label { font-size:.72rem; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); font-weight:800; margin:22px 0 8px; }
.team { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; }
.team-card { background:linear-gradient(160deg,#fff,#eef2ff); border:1px solid #dbe3f0; border-radius:12px; padding:12px 14px; box-shadow:0 6px 16px rgba(15,23,42,.05); }
.team-card strong { display:block; font-size:.88rem; }
.team-card span { display:block; margin-top:6px; color:#1d4ed8; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.78rem; }
.team-card small { display:block; margin-top:6px; color:var(--muted); font-size:.65rem; letter-spacing:.08em; text-transform:uppercase; font-weight:800; }
.panel { background:#fff; border:1px solid var(--line); border-radius:16px; padding:22px; margin-top:18px; box-shadow:0 8px 22px rgba(15,23,42,.05); }
.grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:18px; }
fieldset { border:0; padding:0; margin:0; }
legend { font-size:.72rem; letter-spacing:.12em; text-transform:uppercase; color:var(--blue); font-weight:800; margin-bottom:10px; }
label { display:block; font-size:.84rem; font-weight:600; margin:12px 0 5px; }
input[type=range] { width:100%; accent-color:var(--blue); }
select,input[type=number] { width:100%; padding:8px 10px; border:1px solid #cbd5e1; border-radius:8px; background:#fff; font:inherit; font-size:.88rem; }
.check { display:flex; gap:8px; align-items:center; margin-top:16px; }
.check input { width:18px; height:18px; accent-color:var(--blue); }
.check label { margin:0; }
.presets { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:14px; }
button { font:inherit; cursor:pointer; border-radius:9px; border:1px solid #cbd5e1; background:#fff; padding:9px 13px; font-weight:700; font-size:.86rem; }
button:hover { border-color:var(--blue); color:var(--blue); }
button.primary { background:var(--blue); border-color:var(--blue); color:#fff; padding:11px 18px; }
button.primary:hover { background:#1d4ed8; color:#fff; }
.actions { display:flex; flex-wrap:wrap; gap:12px; align-items:center; margin-top:18px; }
.auto { display:flex; gap:7px; align-items:center; color:var(--muted); font-size:.85rem; }
.result { margin-top:18px; }
.banner { border-radius:12px; padding:14px 16px; font-weight:700; margin-bottom:14px; }
.banner.beneficial { background:#ecfdf5; color:#065f46; border:1px solid #6ee7b7; }
.banner.neutral { background:#fffbeb; color:#92400e; border:1px solid #fcd34d; }
.banner.negative { background:#fef2f2; color:#991b1b; border:1px solid #fca5a5; }
.banner span { display:block; font-weight:500; font-size:.9rem; margin-top:4px; }
.metrics { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin-bottom:16px; }
.metric { background:#f8fafc; border:1px solid var(--line); border-radius:10px; padding:10px 12px; }
.metric small { display:block; color:var(--muted); font-size:.7rem; text-transform:uppercase; letter-spacing:.08em; }
.metric strong { display:block; margin-top:4px; font-size:1.2rem; }
.cols { display:grid; grid-template-columns:1.15fr 1fr; gap:18px; }
.bar-row { margin:10px 0; }
.bar-head { display:flex; justify-content:space-between; font-size:.82rem; margin-bottom:4px; }
.track { height:9px; background:#e2e8f0; border-radius:999px; overflow:hidden; }
.fill { height:100%; background:var(--blue); border-radius:999px; }
.tip { border-left:4px solid #94a3b8; border-radius:8px; padding:10px 12px; margin-bottom:8px; font-size:.88rem; line-height:1.45; }
.tip strong { display:block; margin-bottom:3px; }
.tip.high { border-color:#ef4444; background:#fef2f2; }
.tip.medium { border-color:#f59e0b; background:#fffbeb; }
.tip.good { border-color:#10b981; background:#ecfdf5; }
.error { background:#fef2f2; color:#991b1b; border:1px solid #fca5a5; border-radius:10px; padding:12px; margin-top:14px; }
.footer { color:#94a3b8; font-size:.8rem; text-align:center; padding:24px 0 8px; }
@media (max-width:900px) { .grid,.team,.cols,.metrics { grid-template-columns:1fr; } }
</style>
</head>
<body>
<div class="wrap">
  <div class="hero">
    <div class="eyebrow">Student wellbeing analytics</div>
    <h1>Student Social Media Impact Predictor</h1>
    <p>Explore how usage habits, sleep, stress, and social comparison combine into an overall digital-wellbeing outcome. Adjust a profile and run the prediction to see confidence and personalised recommendations.</p>
    <div class="pill"><span class="dot"></span>Active model: __MODEL_NAME__</div>
  </div>

  <div class="team-label">Project team</div>
  <div class="team">__TEAM_CARDS__</div>

  <div class="panel">
    <div class="presets">
      <button type="button" data-preset="Balanced Student">Balanced Student</button>
      <button type="button" data-preset="Heavy Late-Night User">Heavy Late-Night User</button>
      <button type="button" data-preset="High-Stress Exam Week">High-Stress Exam Week</button>
    </div>
    <form id="profile-form">
      <div class="grid">
        <fieldset>
          <legend>Demographics & academics</legend>
          <label>Age (<span id="age-out">19</span>)</label>
          <input type="range" name="Age" min="15" max="26" step="1" value="19" />
          <label>Gender</label>
          <select name="Gender"><option>Female</option><option>Male</option><option>Non-Binary</option><option>Prefer not to say</option></select>
          <label>Academic Level</label>
          <select name="Academic_Level"><option>High School</option><option selected>Undergraduate</option><option>Postgraduate</option></select>
          <label>Academic Performance GPA (<span id="gpa-out">3.40</span>)</label>
          <input type="range" name="Academic_Performance_GPA" min="1.50" max="4.00" step="0.01" value="3.40" />
        </fieldset>
        <fieldset>
          <legend>Social media habits</legend>
          <label>Primary Platform</label>
          <select name="Primary_Platform"><option>Instagram</option><option>TikTok</option><option>YouTube</option><option>Snapchat</option><option>Reddit</option><option>LinkedIn</option><option>X (Twitter)</option></select>
          <label>Device Type</label>
          <select name="Device_Type"><option>Smartphone</option><option>Laptop/PC</option><option>Tablet</option></select>
          <label>Daily Usage Hours (<span id="usage-out">5.0</span>)</label>
          <input type="range" name="Daily_Usage_Hours" min="0.5" max="14.0" step="0.1" value="5.0" />
          <label>Weekend Extra Hours (<span id="weekend-out">1.8</span>)</label>
          <input type="range" name="Weekend_Extra_Hours" min="0.0" max="5.0" step="0.1" value="1.8" />
          <div class="check"><input type="checkbox" name="Late_Night_Usage" id="late" /><label for="late">Late-Night Usage</label></div>
        </fieldset>
        <fieldset>
          <legend>Sleep & wellbeing</legend>
          <label>Sleep Duration Hours (<span id="sleep-out">7.0</span>)</label>
          <input type="range" name="Sleep_Duration_Hours" min="3.0" max="11.0" step="0.1" value="7.0" />
          <label>Sleep Quality Score (<span id="quality-out">3</span>)</label>
          <input type="range" name="Sleep_Quality_Score" min="1" max="5" step="1" value="3" />
          <label>Social Comparison Frequency</label>
          <select name="Social_Comparison_Frequency"><option>Never</option><option>Rarely</option><option selected>Sometimes</option><option>Frequently</option><option>Always</option></select>
          <label>Perceived Stress Score (<span id="stress-out">14</span>)</label>
          <input type="range" name="Perceived_Stress_Score" min="0" max="40" step="1" value="14" />
          <label>Mental Health Index (<span id="mental-out">80</span>)</label>
          <input type="range" name="Mental_Health_Index" min="30" max="100" step="1" value="80" />
        </fieldset>
      </div>
      <div class="actions">
        <button class="primary" type="submit">Predict Impact</button>
        <div class="auto"><input type="checkbox" id="auto" /><label for="auto">Auto-predict</label></div>
      </div>
    </form>
  </div>

  <div id="result" class="panel result" hidden>
    <div id="banner" class="banner"></div>
    <div class="metrics">
      <div class="metric"><small>Daily usage</small><strong id="m-usage"></strong></div>
      <div class="metric"><small>Sleep</small><strong id="m-sleep"></strong></div>
      <div class="metric"><small>Stress score</small><strong id="m-stress"></strong></div>
      <div class="metric"><small>Mental health</small><strong id="m-mental"></strong></div>
    </div>
    <div class="cols">
      <div><h3>Personalised digital wellness plan</h3><div id="tips"></div></div>
      <div><h3>Confidence breakdown</h3><div id="confidence"></div></div>
    </div>
  </div>
  <div id="error" class="error" hidden></div>

  <div class="footer">Estimates from a trained classifier. Use them as a reflection aid, not as a diagnosis or substitute for professional advice.</div>
</div>

<script>
const presets = __PRESETS__;
const form = document.getElementById("profile-form");
const result = document.getElementById("result");
const errorBox = document.getElementById("error");
const auto = document.getElementById("auto");
const outputs = {
  Age: ["age-out", v => String(v)],
  Academic_Performance_GPA: ["gpa-out", v => Number(v).toFixed(2)],
  Daily_Usage_Hours: ["usage-out", v => Number(v).toFixed(1)],
  Weekend_Extra_Hours: ["weekend-out", v => Number(v).toFixed(1)],
  Sleep_Duration_Hours: ["sleep-out", v => Number(v).toFixed(1)],
  Sleep_Quality_Score: ["quality-out", v => String(v)],
  Perceived_Stress_Score: ["stress-out", v => String(v)],
  Mental_Health_Index: ["mental-out", v => String(v)]
};

function syncOutputs() {
  for (const [key, [id, fmt]] of Object.entries(outputs)) {
    const el = form.elements[key];
    document.getElementById(id).textContent = fmt(el.value);
  }
}

function setPreset(name) {
  const preset = presets[name];
  for (const [key, value] of Object.entries(preset)) {
    const el = form.elements[key];
    if (!el) continue;
    if (el.type === "checkbox") el.checked = Boolean(value);
    else el.value = value;
  }
  syncOutputs();
  result.hidden = true;
  errorBox.hidden = true;
}

document.querySelectorAll("[data-preset]").forEach(btn => {
  btn.addEventListener("click", () => setPreset(btn.dataset.preset));
});
form.addEventListener("input", () => {
  syncOutputs();
  if (auto.checked) predict();
});
form.addEventListener("change", () => { if (auto.checked) predict(); });
form.addEventListener("submit", event => { event.preventDefault(); predict(); });
syncOutputs();

function payload() {
  const f = new FormData(form);
  return {
    Age: parseInt(f.get("Age")),
    Gender: f.get("Gender"),
    Academic_Level: f.get("Academic_Level"),
    Academic_Performance_GPA: parseFloat(f.get("Academic_Performance_GPA")),
    Primary_Platform: f.get("Primary_Platform"),
    Device_Type: f.get("Device_Type"),
    Daily_Usage_Hours: parseFloat(f.get("Daily_Usage_Hours")),
    Weekend_Extra_Hours: parseFloat(f.get("Weekend_Extra_Hours")),
    Late_Night_Usage: document.getElementById("late").checked,
    Sleep_Duration_Hours: parseFloat(f.get("Sleep_Duration_Hours")),
    Sleep_Quality_Score: parseInt(f.get("Sleep_Quality_Score")),
    Social_Comparison_Frequency: f.get("Social_Comparison_Frequency"),
    Perceived_Stress_Score: parseFloat(f.get("Perceived_Stress_Score")),
    Mental_Health_Index: parseInt(f.get("Mental_Health_Index"))
  };
}

async function predict() {
  errorBox.hidden = true;
  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload())
    });
    const data = await response.json();
    if (!response.ok) {
      errorBox.textContent = "Prediction failed: " + (data.detail || "unknown error");
      errorBox.hidden = false;
      return;
    }
    render(data);
  } catch (err) {
    errorBox.textContent = "Prediction failed: " + err.message;
    errorBox.hidden = false;
  }
}

function render(data) {
  result.hidden = false;
  const banner = document.getElementById("banner");
  banner.className = "banner " + data.label.toLowerCase();
  banner.textContent = "Overall impact: " + data.label;
  const summary = document.createElement("span");
  summary.textContent = data.summary;
  banner.appendChild(summary);

  document.getElementById("m-usage").textContent = data.metrics.Daily_Usage_Hours;
  document.getElementById("m-sleep").textContent = data.metrics.Sleep_Duration_Hours;
  document.getElementById("m-stress").textContent = data.metrics.Perceived_Stress_Score;
  document.getElementById("m-mental").textContent = data.metrics.Mental_Health_Index;

  document.getElementById("confidence").innerHTML = data.confidence.map(item => {
    const pct = Number(item.probability).toFixed(2);
    return '<div class="bar-row"><div class="bar-head"><span>' + item.label + '</span><span>' + pct + '%</span></div>' +
      '<div class="track"><div class="fill" style="width:' + pct + '%"></div></div></div>';
  }).join("");

  document.getElementById("tips").innerHTML = data.tips.map(tip =>
    '<div class="tip ' + tip.severity + '"><strong>' + tip.title + '</strong>' + tip.text + '</div>'
  ).join("");
}
</script>
</body>
</html>
'''


def team_cards_html() -> str:
    return "".join(
        f'<div class="team-card"><strong>{escape(member["name"])}</strong>'
        f'<small>University ID</small><span>{escape(member["university_id"])}</span></div>'
        for member in TEAM_MEMBERS
    )


def build_html(model_name: str) -> str:
    return (
        HTML_TEMPLATE.replace("__MODEL_NAME__", escape(model_name))
        .replace("__TEAM_CARDS__", team_cards_html())
        .replace("__PRESETS__", json.dumps(PRESETS))
    )


@app.get("/", response_class=HTMLResponse)
@app.get("/api/index", response_class=HTMLResponse)
@app.get("/api/index/", response_class=HTMLResponse)
def index() -> str:
    bundle = get_bundle()
    return build_html(str(bundle["model_name"]))


@app.get("/health")
@app.get("/api/index/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
@app.post("/api/index/predict")
@app.post("/api/index")
def predict(profile: StudentProfile) -> dict:
    bundle = get_bundle()
    data = profile.model_dump()
    try:
        processed = preprocess(bundle, data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {exc}")

    prediction = int(bundle["model"].predict(processed)[0])
    label = str(bundle["label_encoder"].inverse_transform([prediction])[0])

    confidence = []
    if hasattr(bundle["model"], "predict_proba"):
        probabilities = np.asarray(bundle["model"].predict_proba(processed))[0]
        confidence = [
            {
                "label": str(class_label),
                "probability": round(float(probability) * 100, 2),
            }
            for class_label, probability in zip(bundle["label_encoder"].classes_, probabilities)
        ]

    return {
        "label": label,
        "summary": SUMMARY.get(label, "Prediction complete."),
        "confidence": confidence,
        "tips": wellness_tips(data),
        "metrics": {
            "Daily_Usage_Hours": f'{float(data["Daily_Usage_Hours"]):.1f} h',
            "Sleep_Duration_Hours": f'{float(data["Sleep_Duration_Hours"]):.1f} h',
            "Perceived_Stress_Score": f'{float(data["Perceived_Stress_Score"]):.0f}/40',
            "Mental_Health_Index": f'{int(data["Mental_Health_Index"])}/100',
        },
    }
