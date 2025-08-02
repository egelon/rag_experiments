@echo off
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Starting RAG application with Streamlit...
streamlit run src/app.py

pause
