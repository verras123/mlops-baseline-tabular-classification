@echo off
setlocal

echo [INFO] Running dataset generation...
python gerar_csv_grande.py
if errorlevel 1 exit /b 1

echo [INFO] Running ML pipeline...
python ml_pipeline.py
if errorlevel 1 exit /b 1

echo [OK] Done. Check outputs/ folder.
endlocal
