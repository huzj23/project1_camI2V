@echo off
setlocal
set "DEMO0_PYTHON=C:\Users\12447\.conda\envs\hajicar\python.exe"
if not exist "%DEMO0_PYTHON%" set "DEMO0_PYTHON=python"
"%DEMO0_PYTHON%" "%~dp0launch_viewer.py"
if errorlevel 1 (
  echo Demo0 failed to start. See data\demo0_v11\runtime\viewer_8767.log.
  pause
)
endlocal
