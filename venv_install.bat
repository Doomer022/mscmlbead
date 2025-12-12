@echo off
set VENV_NAME=fruit_venv

echo --- PyTorch CNN Project Setup ---
echo.
echo 1. Checking for Python installation...

:: Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not found. Please install Python and ensure it's in your PATH.
    goto :eof
)

echo Python found.
echo.

:: 2. Create Local Virtual Environment
echo 2. Creating local virtual environment "%VENV_NAME%"...
python -m venv %VENV_NAME%

if not exist %VENV_NAME%\Scripts\activate.bat (
    echo Error: Failed to create virtual environment.
    goto :eof
)
echo Virtual environment created successfully in ./%VENV_NAME%/
echo.

:: 3. Define VENV Python Path
set VENV_PYTHON=%VENV_NAME%\Scripts\python.exe

:: 4. Install Required Packages (including PyTorch with CUDA)
echo 3. Installing packages into the virtual environment. This may take a few minutes...
echo (Torch with CUDA, Torchvision, Optuna, Scikit-learn, Matplotlib, Seaborn)
echo.

:: Upgrade pip first
%VENV_PYTHON% -m pip install --upgrade pip

:: Install all required packages
%VENV_PYTHON% -m pip install torch torchvision torchaudio optuna scikit-learn matplotlib seaborn

if errorlevel 1 (
    echo.
    echo Error: Failed to install one or more packages.
    echo This is often due to missing CUDA Toolkit or incompatible Python/Torch versions.
    echo Please check the error messages above and ensure you have a compatible system setup.
    goto :eof
)

echo.
echo --- SETUP COMPLETE ---
echo.
echo The local virtual environment "%VENV_NAME%" has been created and packages are installed.
echo.
echo NEXT STEPS:
echo 1. To activate this environment in your Command Prompt:
echo    Type: %VENV_NAME%\Scripts\activate.bat
echo 2. To register this environment for use in Jupyter Notebook/Lab (after activating):
echo    Run: pip install ipykernel
echo    Run: python -m ipykernel install --name=%VENV_NAME% --display-name "Python (Fruit CNN venv)"
echo    Then, manually switch the kernel in your notebook application.
echo.

pause