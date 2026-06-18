@echo off
rem ============================================================
rem  构建 GPU 环境：Python 3.10 + torch 1.12.1+cu113 + TabPFN
rem  （照搬参考环境 F:\py\Multi_algorithm_comparison_basic_version\.venv）
rem  仅在需要 TabPFN / GPU 时运行；普通 9 个模型用 start.bat 即可。
rem  tabpfn / tabpfn-extensions 声明 torch>=2.1，与 torch 1.12.1 冲突，
rem  故用 --no-deps 绕过（与参考环境做法一致）。
rem  注意：本文件需以 GBK/ANSI 编码保存，cmd 才能正确显示中文。
rem ============================================================
setlocal
cd /d "%~dp0"
set "UV_CACHE_DIR=%~dp0.uv-cache"
set "PYTHONIOENCODING=utf-8"
set "REF_EXT=F:\py\Multi_algorithm_comparison_basic_version\tabpfn-extensions"

echo [1/5] 准备 tabpfn-extensions 源码 (vendor\，缺失时从 F: 复制)
if not exist "vendor\tabpfn-extensions\src\tabpfn_extensions" (
    mkdir "vendor\tabpfn-extensions" 2>nul
    xcopy /E /I /Y "%REF_EXT%\src" "vendor\tabpfn-extensions\src" >nul
    copy /Y "%REF_EXT%\pyproject.toml" "vendor\tabpfn-extensions\" >nul
    copy /Y "%REF_EXT%\setup.py" "vendor\tabpfn-extensions\" >nul
    copy /Y "%REF_EXT%\README.md" "vendor\tabpfn-extensions\" >nul
    copy /Y "%REF_EXT%\LICENSE" "vendor\tabpfn-extensions\" >nul
)

echo [2/5] 用 Python 3.10 重建虚拟环境 + 安装基础依赖
uv sync || (echo uv sync 失败 & pause & exit /b 1)

echo [3/5] 安装 torch 1.12.1 + cu113 (GPU 版)
uv pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1+cu113 --index-url https://download.pytorch.org/whl/cu113 || (echo torch 安装失败 & pause & exit /b 1)

echo [4/5] 安装 tabpfn 及其运行依赖 (--no-deps 绕过 torch>=2.1)
uv pip install --no-deps tabpfn==2.2.1 tabpfn-common-utils==0.2.2
uv pip install einops==0.8.1 huggingface-hub==0.35.3 safetensors==0.7.0

echo [5/5] 以源码方式安装 tabpfn-extensions (不走 PyPI)
uv pip install --no-deps -e ".\vendor\tabpfn-extensions"

echo.
echo [完成] GPU 环境就绪。
echo 提醒：之后请勿再单独运行 uv sync（会删除手动装的 torch/tabpfn）；用 start.bat 启动。
pause
endlocal
