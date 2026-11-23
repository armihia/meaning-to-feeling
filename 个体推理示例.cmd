@echo off
chcp 65001 >nul
cd /d "%~dp0"
python -B -X utf8 -m meaning_to_feeling profile --profile-strategy history_mean --input examples/history.json --output outputs/profile.json
if errorlevel 1 goto end
python -B -X utf8 -m meaning_to_feeling predict --input examples/input.json --profile outputs/profile.json --output outputs/personal.json
:end
pause
