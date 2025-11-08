@echo off
echo Starting Enhanced RAG Application...

REM Check if virtual environment exists
if not exist "venv" (
    echo Virtual environment not found. Running setup...
    call setup_enhanced_env.bat
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Run the enhanced RAG app
echo Starting Streamlit app...
streamlit run enhanced_rag_app.py

pause