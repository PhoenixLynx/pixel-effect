@echo off
setlocal
rem Render with a preset.   render.bat PRESET INPUT OUTPUT [draft]
rem   render.bat angelcore inputDancer.mp4 out.mp4 draft    (fast preview)
rem   render.bat angelcore inputDancer.mp4 final.mp4        (full quality)
rem Every group from presets_template.bat is passed; groups a preset leaves out stay empty.
if "%~3"=="" (
  echo Usage: render.bat PRESET INPUT OUTPUT [draft]
  echo Presets:
  for %%F in ("%~dp0presets_*.bat") do echo   %%~nF
  exit /b 1
)
rem Presets and the script are found next to this file (%~dp0), from any folder.
if not exist "%~dp0presets_%~1.bat" (
  echo No such preset: presets_%~1.bat
  exit /b 1
)
rem Clear every group first, so nothing leaks in from a preset loaded earlier.
for %%V in (SCRIPT LOOK HUD FONTS CHAT PIXEL IRID GLYPHS ECHO HALO DAMAGE CAMERA SIGIL BODY BG OVERLAY CROP FACE FULL DRAFT) do set "%%V="
call "%~dp0presets_%~1.bat" >nul
if not defined SCRIPT set SCRIPT=pixel_effect.py
if exist "%~dp0%SCRIPT%" set "SCRIPT=%~dp0%SCRIPT%"
if not defined FULL set FULL=--scale 1.0
set SPEED=%FULL%
if /i "%~4"=="draft" set SPEED=%DRAFT%
python "%SCRIPT%" %2 %3 %LOOK% %HUD% %FONTS% %CHAT% %PIXEL% %IRID% %GLYPHS% %ECHO% %HALO% %DAMAGE% %CAMERA% %SIGIL% %BODY% %BG% %OVERLAY% %CROP% %FACE% %SPEED%
endlocal
