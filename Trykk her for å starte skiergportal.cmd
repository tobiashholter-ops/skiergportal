@echo off
setlocal
set SCRIPT_DIR=%~dp0
set SYS_DIR=%SCRIPT_DIR%data\system\

REM One entrypoint: start the GUI launcher
if exist "%SYS_DIR%.venv\Scripts\pythonw.exe" (
  start "" "%SYS_DIR%.venv\Scripts\pythonw.exe" "%SYS_DIR%Start-Treningshistorikk-GUI.pyw"
  goto :eof
)

where pyw >nul 2>nul
if %ERRORLEVEL%==0 (
  start "" pyw "%SYS_DIR%Start-Treningshistorikk-GUI.pyw"
  goto :eof
)

where py >nul 2>nul
if %ERRORLEVEL%==0 (
  start "" py "%SYS_DIR%Start-Treningshistorikk-GUI.pyw"
  goto :eof
)

powershell -NoProfile -Command "Add-Type -AssemblyName PresentationFramework; [System.Windows.MessageBox]::Show('Python 3.11 eller nyere mangler. Nettleseren apner na den offisielle nedlastingssiden. Installer Python, og start deretter Trykk her for å starte skiergportal.cmd pa nytt.', 'Python mangler', [System.Windows.MessageBoxButton]::OK, [System.Windows.MessageBoxImage]::Warning) | Out-Null; Start-Process 'https://www.python.org/downloads/windows/'"
endlocal & exit /b 1
