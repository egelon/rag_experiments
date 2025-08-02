@echo off
echo Activating virtual environment...
call venv\Scripts\activate.bat
mkdir processed_files

echo Starting RAG application with Streamlit...
streamlit run src/app.py

pause
