@echo off

REM Check if virtual environment already exists
if exist venv (
    echo Virtual environment already exists. Skipping setup.
) else (
    echo Creating virtual environment...
    python -m venv venv

    call venv\Scripts\activate

    python -m pip install --upgrade pip

    if exist requirements.txt (
        echo Installing dependencies from requirements.txt...
        pip install -r requirements.txt
    ) else (
        echo No requirements.txt found. Skipping dependency installation.
    )

    echo Setup complete.
)

REM Activate and run Python script
call venv\Scripts\activate

REM Run your script
streamlit run rag_app.py

pause
