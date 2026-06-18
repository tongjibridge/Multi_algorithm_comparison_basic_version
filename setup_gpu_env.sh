#!/usr/bin/env bash
# ============================================================
#  构建 GPU 环境：Python 3.10 + torch 1.12.1+cu113 + TabPFN
#  （照搬参考环境 F:\py\Multi_algorithm_comparison_basic_version\.venv）
#  仅在需要 TabPFN / GPU 时运行；普通 9 个模型用 ./start.sh 即可。
#
#  说明：tabpfn 与 tabpfn-extensions 都声明 torch>=2.1，与我们要用的
#  torch 1.12.1 冲突，故用 --no-deps 绕过（与参考环境做法一致）。
# ============================================================
set -e
cd "$(dirname "$0")"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$(pwd)/.uv-cache}"
export PYTHONIOENCODING=utf-8
REF_EXT="F:/py/Multi_algorithm_comparison_basic_version/tabpfn-extensions"

echo "[1/5] 准备 tabpfn-extensions 源码（vendor/，缺失时从 F: 复制）"
if [ ! -d "vendor/tabpfn-extensions/src/tabpfn_extensions" ]; then
    mkdir -p vendor/tabpfn-extensions
    for it in pyproject.toml setup.py README.md LICENSE mypy.ini src; do
        cp -r "$REF_EXT/$it" vendor/tabpfn-extensions/ 2>/dev/null || true
    done
fi

echo "[2/5] 用 Python 3.10 重建虚拟环境 + 安装基础依赖"
uv sync

echo "[3/5] 安装 torch 1.12.1 + cu113（GPU 版）"
uv pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1+cu113 \
    --index-url https://download.pytorch.org/whl/cu113

echo "[4/5] 安装 tabpfn 及其运行依赖（--no-deps 绕过 torch>=2.1）"
uv pip install --no-deps tabpfn==2.2.1 tabpfn-common-utils==0.2.2
uv pip install einops==0.8.1 huggingface-hub==0.35.3 safetensors==0.7.0

echo "[5/5] 以源码方式安装 tabpfn-extensions（不走 PyPI）"
uv pip install --no-deps -e ./vendor/tabpfn-extensions

echo
echo "[完成] GPU 环境就绪。"
echo "提醒：之后请勿再单独运行 'uv sync'（会删除手动安装的 torch/tabpfn）；用 ./start.sh 启动。"
