@echo off
echo ========================================
echo DeepCSAT GitHub Repository Setup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python from https://python.org
    pause
    exit /b 1
)

REM Check if Git is installed
git --version >nul 2>&1
if errorlevel 1 (
    echo Error: Git is not installed or not in PATH
    echo Please install Git from https://git-scm.com
    pause
    exit /b 1
)

echo Python and Git are installed. Starting setup...
echo.

REM Run the GitHub setup script
python github_setup.py

echo.
echo Setup completed! Press any key to exit.
pause
