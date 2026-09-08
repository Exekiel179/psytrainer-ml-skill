@echo off
setlocal EnableExtensions
rem Install psytrainer-ml runtime on Windows (Skill install time).
rem Usage:
rem   scripts\install_runtime.cmd --wheel D:\wheels\PsyTrainer-0.2.0-cp313-none-any.whl
rem   set PSYTRAINER_WHEEL=D:\wheels\PsyTrainer-0.2.0-cp313-none-any.whl && scripts\install_runtime.cmd
rem Globs are expanded by install_runtime.py (cmd.exe does not expand *.whl).

cd /d "%~dp0\.."
where py >nul 2>&1
if %ERRORLEVEL%==0 (
  py -3 scripts\install_runtime.py %*
  exit /b %ERRORLEVEL%
)
where python >nul 2>&1
if %ERRORLEVEL%==0 (
  python scripts\install_runtime.py %*
  exit /b %ERRORLEVEL%
)
echo Python not found. Install Python 3.10+ from python.org and enable the py launcher.
exit /b 1
