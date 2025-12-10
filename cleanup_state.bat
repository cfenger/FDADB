@echo off
setlocal

echo.
echo FDA Document Q^&A - Local Cleanup
echo This will delete:
echo   - data\metadata.json (document metadata and vector_store_id)
echo   - data\uploads\ (local uploaded copies and scratch files)
echo.
set /p CONFIRM=Type YES to continue: 
if /I not "%CONFIRM%"=="YES" (
  echo Aborted.
  goto :EOF
)

set "BASEDIR=%~dp0"
set "METADATA=%BASEDIR%data\metadata.json"
set "UPLOADS=%BASEDIR%data\uploads"

if exist "%METADATA%" (
  del /f /q "%METADATA%"
  echo Deleted "%METADATA%"
) else (
  echo No metadata file to delete.
)

if exist "%UPLOADS%" (
  rmdir /s /q "%UPLOADS%"
  echo Deleted "%UPLOADS%"
) else (
  echo No uploads directory to delete.
)

echo.
echo Cleanup complete. Restart the app and re-upload documents as needed.

endlocal

