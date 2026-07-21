@echo off

REM ===== CONFIG =====
set ENV_NAME=yd
set PYTHON_VERSION=3.10

REM ===== CREATE ENV =====
echo Creating conda environment %ENV_NAME%...
call conda create -y -n %ENV_NAME% python=%PYTHON_VERSION%

REM ===== ACTIVATE ENV =====
echo Activating environment...
call conda activate %ENV_NAME%

REM ===== INSTALL REQUIREMENTS =====
if exist requirements.txt (
    echo Installing dependencies from requirements.txt...
    pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126

    pip install -r requirements.txt

    pip install "accelerate>=1.1.0"
) else (
    echo requirements.txt not found!
)

echo Done!