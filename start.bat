@echo off
rem ============================================================
rem  可解释机器学习程序 启动脚本（Windows，双击即可运行）
rem  首次运行会自动用 uv 安装依赖（缓存指向项目内 .uv-cache，
rem  以规避本机默认 uv 缓存软链损坏的问题），之后直接用虚拟
rem  环境解释器启动，不再触碰 uv 缓存。
rem  注意：本文件需以 GBK/ANSI 编码保存，cmd 才能正确显示中文。
rem ============================================================
setlocal
cd /d "%~dp0"
set "UV_CACHE_DIR=%~dp0.uv-cache"
set "PYTHONIOENCODING=utf-8"
set "VENV_PY=%~dp0.venv\Scripts\python.exe"

rem —— 首次运行：虚拟环境不存在则安装依赖 ——
if not exist "%VENV_PY%" (
    echo [初始化] 未检测到虚拟环境，正在安装依赖（首次运行，可能需要数分钟）...
    uv sync
    if errorlevel 1 (
        echo.
        echo [错误] 依赖安装失败，请先手动执行： uv sync
        pause
        exit /b 1
    )
)

echo [启动] 可解释机器学习程序
echo        浏览器将自动打开 http://localhost:8080
echo        关闭本窗口即可停止程序。
echo.
"%VENV_PY%" -m app.main

echo.
echo [已退出] 程序已停止。
pause
endlocal
