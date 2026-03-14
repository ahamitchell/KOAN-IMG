@echo off
echo ============================================
echo  KOAN.img Installer Build Script
echo ============================================
echo.

REM Step 1: Build the launcher exe
echo [1/2] Building launcher executable...
pip install pyinstaller >nul 2>&1
pyinstaller --onefile --name "KOAN.img" --console launcher.py
if errorlevel 1 (
    echo ERROR: PyInstaller build failed.
    pause
    exit /b 1
)
echo       Done. Output: dist\KOAN.img.exe

REM Step 2: Build the installer (requires Inno Setup installed)
echo.
echo [2/2] Building installer with Inno Setup...
where iscc >nul 2>&1
if errorlevel 1 (
    echo.
    echo Inno Setup not found in PATH.
    echo Download from: https://jrsoftware.org/isdl.php
    echo Then run:  iscc installer.iss
    echo.
    echo Alternatively, open installer.iss in the Inno Setup GUI and click Build.
    pause
    exit /b 0
)
iscc installer.iss
if errorlevel 1 (
    echo ERROR: Inno Setup build failed.
    pause
    exit /b 1
)

echo.
echo ============================================
echo  BUILD COMPLETE
echo  Installer: installer_output\KOAN.img_Setup_1.0.0.exe
echo ============================================
pause
