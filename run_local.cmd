@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "VENV_PY=.venv\Scripts\python.exe"

where py >nul 2>nul
if not errorlevel 1 (
    set "BOOTSTRAP=py -3"
) else (
    where python >nul 2>nul
    if errorlevel 1 goto :no_python
    set "BOOTSTRAP=python"
)

if not exist "%VENV_PY%" (
    echo Creating virtual environment...
    %BOOTSTRAP% -m venv .venv
    if errorlevel 1 goto :fail
)

echo Installing dependencies...
"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 goto :fail
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 goto :fail

if not exist "social_media_impact_pipeline.joblib" (
    echo Missing social_media_impact_pipeline.joblib next to this script.
    goto :fail
)

echo.
echo Starting Streamlit at http://localhost:8501
"%VENV_PY%" -m streamlit run app.py
if errorlevel 1 goto :fail

echo.
pause
exit /b 0

:no_python
echo Python was not found on PATH. Install Python 3.9+ and rerun.
goto :fail

:fail
echo.
echo Setup failed. Review the messages above.
pause
exit /b 1
