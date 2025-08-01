@echo off
REM setup.bat ─ Set up Python virtual environment and install dependencies (Windows CMD)

REM --- 1. Create the virtual environment -------------------------------------
echo Creating virtual environment
python -m venv venv
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Could not create virtual environment. Ensure Python 3.9+ is on PATH.
    exit /b 1
)
echo [OK] Virtual environment 'venv' created.

REM --- 2. Activate the virtual environment ------------------------------------
echo Activating venv
call "%~dp0venv\Scripts\activate.bat"
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Could not activate the virtual environment.
    exit /b 1
)
echo [OK] Virtual environment activated.

REM --- 3. Upgrade pip ---------------------------------------------------------
echo Upgrading pip
python -m pip install --upgrade pip
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to upgrade pip.
    exit /b 1
)
echo [OK] pip upgraded.

REM --- 4. Install project dependencies ---------------------------------------
echo Installing dependencies
pip install -r requirements.txt
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Dependency installation failed. Check requirements.txt.
    exit /b 1
)
echo [OK] Dependencies installed from requirements.txt.

REM --- 5. Reminder ------------------------------------------------------------
echo.
echo Setup complete!  From now on, start your session with:
echo     call venv\Scripts\activate.bat
echo Then run the application (main.py).
echo.

pause