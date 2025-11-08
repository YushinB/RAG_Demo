@echo off
echo Setting up Enhanced RAG Environment...

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing Python packages...
pip install -r requirements.txt

REM Install spaCy language model
echo Installing spaCy English model...
python -m spacy download en_core_web_sm

REM Install additional dependencies for fine-tuning
echo Installing additional dependencies...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo.
echo Setup complete! 
echo.
echo To run the enhanced RAG app:
echo 1. Activate the virtual environment: venv\Scripts\activate.bat
echo 2. Run the app: streamlit run enhanced_rag_app.py
echo.
echo Or simply run: run_enhanced_app.bat
echo.
pause