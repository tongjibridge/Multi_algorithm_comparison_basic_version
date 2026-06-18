#!/usr/bin/env bash
# ============================================================
#  可解释机器学习程序 启动脚本（bash / git-bash）
#  - 首次运行自动用 uv 安装依赖（缓存指向项目内 .uv-cache，
#    规避本机默认 uv 缓存软链损坏的问题）
#  - 之后直接用虚拟环境解释器启动，不再触碰 uv 缓存
# ============================================================
set -e
cd "$(dirname "$0")"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$(pwd)/.uv-cache}"
export PYTHONIOENCODING=utf-8

# 兼容 Windows(.venv/Scripts) 与 *nix(.venv/bin) 两种虚拟环境布局
if [ -f ".venv/Scripts/python.exe" ]; then
    PY=".venv/Scripts/python.exe"
elif [ -f ".venv/bin/python" ]; then
    PY=".venv/bin/python"
else
    echo "[初始化] 未检测到虚拟环境，正在安装依赖（首次运行，可能需要数分钟）..."
    uv sync
    if [ -f ".venv/Scripts/python.exe" ]; then PY=".venv/Scripts/python.exe"; else PY=".venv/bin/python"; fi
fi

echo "[启动] 可解释机器学习程序 —— 浏览器将自动打开 http://localhost:8080"
echo "       按 Ctrl+C 可停止程序。"
exec "$PY" -m app.main
