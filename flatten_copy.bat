@echo off
setlocal enabledelayedexpansion

REM Flattens all files under the current directory into .\temp with directory names prefixed.
REM Example: .\abc\q1\qwe.pdf -> .\temp\abc_q1_qwe.pdf

set "BASEDIR=%~dp0"
set "OUTDIR=%BASEDIR%temp"
if not exist "%OUTDIR%" mkdir "%OUTDIR%"

for /r "%BASEDIR%" %%F in (*.pdf) do (
    set "FULL=%%~fF"
    REM Skip anything already under the output folder
    if /i "!FULL:%OUTDIR%=!"=="!FULL!" (
        set "REL=!FULL:%BASEDIR%=!"
        set "FLAT=!REL:\=_!"
        copy /y "%%~fF" "%OUTDIR%\!FLAT!" >nul
    )
)

echo Done. Files copied to "%OUTDIR%".
