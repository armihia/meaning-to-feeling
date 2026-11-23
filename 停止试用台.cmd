@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "taskPython=%MTF_PYTHON%"
for /f "delims=" %%P in ('where python 2^>nul') do (
    echo %%P | findstr /i /c:"WindowsApps" >nul
    if errorlevel 1 if not defined taskPython set "taskPython=%%P"
)
for %%P in ("%USERPROFILE%\anaconda3\envs\meaning-to-feeling\python.exe" "%USERPROFILE%\miniconda3\envs\meaning-to-feeling\python.exe" "%ProgramData%\anaconda3\envs\meaning-to-feeling\python.exe") do (
    if not defined taskPython if exist "%%~P" set "taskPython=%%~P"
)
if not defined taskPython (
    echo 请在 Anaconda Prompt 中激活 Python 环境后运行此脚本。
    pause
    exit /b 1
)
"%taskPython%" -u -B -X utf8 -m meaning_to_feeling.testbench --stop %*
if errorlevel 1 pause
