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
| `api/index.py` | FastAPI app and Vercel entrypoint (`app`) |
| `vercel.json` | Routes Vercel requests to the FastAPI app |
| `app.py` | Streamlit dashboard for local use |
| `requirements.txt` | Dependencies for Vercel / FastAPI |
| `requirements-local.txt` | Local Streamlit dependencies (includes `requirements.txt`) |
| `social_media_impact_pipeline.joblib` | Trained model and fitted preprocessors |
| `run_local.cmd` | Windows one-click Streamlit setup and launch script |
| `Machine_Learning_Project_Complete.ipynb` | Training and model-export notebook |
| `streamlit_web_application.py` | Earlier standalone prototype |

## Run locally on Windows

### One-click

Double-click `run_local.cmd`. It will:

1. Create a local `.venv` virtual environment if needed.
2. Install/refresh packages from `requirements-local.txt`.
3. Start Streamlit at `http://localhost:8501` and open it in your browser automatically.

### Manual

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements-local.txt
streamlit run app.py
```

Or, without activating the environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-local.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Deploy to Vercel

Vercel uses `api/index.py`, which exports the FastAPI `app` variable expected by the Python runtime. `vercel.json` routes `/`, `/predict`, and `/health` to that function.

1. Import `https://github.com/ahzahid81/ML_Project.git` in Vercel.
2. Keep the default **Framework Preset** as **Other** and the **Root Directory** as the repository root.
3. Vercel installs `requirements.txt` and builds automatically from `main`.
4. Visit `/health`; it should return `{"status":"ok"}`.

The Vercel page provides the same 14-input form, three sample presets, team cards, confidence chart, and wellness recommendations as the Streamlit version.

To test the API locally:

```powershell
pip install -r requirements.txt
uvicorn api.index:app --reload
```

## Model bundle

Both apps expect `social_media_impact_pipeline.joblib` at the repository root: next to `app.py` for Streamlit, and one level above `api/index.py` for Vercel. If it is missing, the app shows a clear error and stops. Regenerate it by running the final export cell of `Machine_Learning_Project_Complete.ipynb`, then copy the file to the repository root.

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
- **Cloud URL redirects to a login page** — the Cloud app is not publicly reachable. Verify the exact URL from your workspace, then open **App settings -> Sharing** and set **Who can view this app** to **This app is public and searchable**. If it is already public, check **Manage app -> Logs** for a failed deploy.
