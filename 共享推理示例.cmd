@echo off
chcp 65001 >nul
cd /d "%~dp0"
python -B -X utf8 -m meaning_to_feeling predict --input examples/input.json --output outputs/shared.json
:end
pause
