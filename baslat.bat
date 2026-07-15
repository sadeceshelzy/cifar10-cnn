@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Goruntu Siniflandirma Sistemi

echo ================================================
echo    GORUNTU SINIFLANDIRMA SISTEMI BASLATILIYOR
echo ================================================
echo.

REM --- 1) Sanal ortam (venv) var mi kontrol et, yoksa olustur ---
if not exist "venv\Scripts\python.exe" (
    echo [1/3] Sanal ortam bulunamadi, olusturuluyor...
    py -V:3.12 -m venv venv
    if errorlevel 1 (
        echo.
        echo HATA: Python 3.12 bulunamadi. Lutfen once Python 3.12 kurun.
        echo.
        pause
        exit /b 1
    )
    echo [2/3] Gerekli kutuphaneler kuruluyor ^(ilk seferde birkac dakika surebilir^)...
    venv\Scripts\python.exe -m pip install --upgrade pip
    venv\Scripts\python.exe -m pip install -r requirements.txt
) else (
    echo [OK] Sanal ortam hazir.
)

REM --- 2) Egitilmis model var mi kontrol et ---
if not exist "cifar10_cnn_model.keras" (
    echo.
    echo UYARI: Egitilmis model ^(cifar10_cnn_model.keras^) bulunamadi.
    echo Once modeli egitmelisiniz:
    echo     venv\Scripts\python.exe train_transfer.py
    echo.
    pause
    exit /b 1
)

REM --- 3) Web arayuzunu baslat ---
echo.
echo [3/3] Web arayuzu baslatiliyor...
echo Tarayicida otomatik acilacak: http://localhost:8501
echo Kapatmak icin bu pencereyi kapatin veya Ctrl+C yapin.
echo.

REM Sunucu hazir olunca tarayiciyi otomatik ac (6 sn bekleyip)
start "" /min cmd /c "timeout /t 6 >nul & start http://localhost:8501"

REM Arayuzu sessiz modda baslat (e-posta sorusu sorulmaz)
venv\Scripts\python.exe -m streamlit run app.py --server.headless true

pause
