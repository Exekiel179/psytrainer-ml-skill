@echo off
setlocal EnableExtensions
rem Install psytrainer-ml runtime on Windows (Skill install time).
rem Usage:
rem   scripts\install_runtime.cmd
rem   scripts\install_runtime.cmd --recreate
rem Globs are expanded by install_runtime.py (cmd.exe does not expand *.whl).

cd /d "%~dp0\.."
where py >nul 2>&1
if %ERRORLEVEL%==0 (
  goto run_py
)
where python >nul 2>&1
if %ERRORLEVEL%==0 (
  goto run_python
)
echo Python not found. Install CPython 3.12-3.14 and enable the py launcher.
exit /b 1

:run_py
py -3 scripts\install_runtime.py %*
exit /b %ERRORLEVEL%

:run_python
python scripts\install_runtime.py %*
exit /b %ERRORLEVEL%
