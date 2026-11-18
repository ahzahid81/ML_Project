# Student Social Media Impact Predictor

An interactive Streamlit dashboard that predicts whether a student's social-media habits have a **Beneficial**, **Neutral**, or **Negative** overall impact. The app loads a trained scikit-learn classifier together with its fitted preprocessing objects from `social_media_impact_pipeline.joblib`.

## Features

- 14-input profile form organized into demographics, social-media habits, and sleep/wellbeing
- One-click sample presets: **Balanced Student**, **Heavy Late-Night User**, and **High-Stress Exam Week**
- Button-triggered prediction with an optional auto-predict mode
- Color-coded outcome banner plus a confidence table and bar chart
- Personalized digital wellness recommendations based on the submitted profile
- Active model badge populated from the saved pipeline

## Project structure

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit dashboard application |
| `requirements.txt` | Python dependencies |
| `social_media_impact_pipeline.joblib` | Trained model and fitted preprocessors |
| `run_local.cmd` | Windows one-click setup and launch script |
| `Machine_Learning_Project_Complete.ipynb` | Training and model-export notebook |
| `streamlit_web_application.py` | Earlier standalone prototype |

## Run locally on Windows

### One-click

Double-click `run_local.cmd`. It will:

1. Create a local `.venv` virtual environment if needed.
2. Install/refresh packages from `requirements.txt`.
3. Start Streamlit at `http://localhost:8501` and open it in your browser automatically.

### Manual

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Or, without activating the environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Model bundle

The app expects `social_media_impact_pipeline.joblib` in the same folder as `app.py`. If it is missing, the dashboard shows a clear error and stops. Regenerate it by running the final export cell of `Machine_Learning_Project_Complete.ipynb`, then copy the file next to `app.py`.

`scikit-learn` is pinned to `1.6.1` in `requirements.txt` because that is the version used to create the bundle. Newer versions can raise errors such as `'SimpleImputer' object has no attribute '_fill_dtype'`. Re-run `run_local.cmd` after pulling changes to apply the pinned version.

The bundle must contain these keys:

- `model`, `model_name`
- `stress_imputer`, `gpa_imputer`
- `social_encoder`, `academic_encoder`, `ohe`, `scaler`
- `label_encoder`, `feature_columns`
- `nominal_cols`, `num_cols` (optional; the app has built-in defaults)

## Team

| Name | University ID |
| --- | --- |
| BOUALEM ABDEL AZIZ | 443058333 |
| ABDEL MONEM | 443058333 |
| ABDUL HASIB ZAHID | 443058333 |

The same details are shown in cards at the top of the app.

## Push to GitHub

```powershell
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin <your-repository-url>
git push -u origin main
```

The included `.gitignore` excludes Python bytecode, virtual environments, local secrets, editor files, and logs. The `.joblib` bundle is intentionally tracked.

## Troubleshooting

- **`Model bundle not found`** — keep `social_media_impact_pipeline.joblib` beside `app.py`.
- **`Python was not found on PATH`** — install Python 3.9+ and enable **Add Python to PATH** during setup.
- **Port 8501 already in use** — run `streamlit run app.py --server.port 8502`.
