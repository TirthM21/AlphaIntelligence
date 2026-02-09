@echo off
setlocal EnableDelayedExpansion

echo ========================================================
echo   DAILY STOCK MARKET SCANNER
echo ========================================================
echo.

:: 1. Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found! Please install Python 3.9+
    pause
    exit /b 1
)

:: 2. Check/Create Virtual Environment
if not exist "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate.bat

:: 3. Install Requirements
if not exist "requirements.txt" (
    echo [ERROR] requirements.txt not found!
    pause
    exit /b 1
)
echo [INFO] Checking dependencies...
pip install -r requirements.txt >nul 2>&1

:: 4. Check for .env file
if not exist ".env" (
    echo [WARNING] .env file not found!
    echo [INFO] Creating template .env...
    echo FMP_API_KEY=your_key_here > .env
    echo Please edit .env and add your API keys.
    pause
)

:: 5. Run the Scan
echo.
echo [INFO] Starting Market Scan...
echo        - Workers: 3
echo        - Delay: 0.5s
echo        - FMP Enhanced Fundamentals: Enabled
echo        - SEC Filing Downloads: Enabled
echo.

python run_optimized_scan.py --workers 3 --delay 0.5 --use-fmp --download-sec --send-email

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Scan failed with error code %errorlevel%
    pause
    exit /b 1
)

echo.
echo ========================================================
echo   SCAN COMPLETE
echo ========================================================
echo.
echo Newsletter saved to: data\daily_newsletter.md
echo.
pause
