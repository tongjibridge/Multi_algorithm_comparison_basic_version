"""核心管道冒烟测试：用参考数据集跑通 训练 → 评估 → 出图。

运行：
    uv run python tests/smoke_core.py [excel路径]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Windows 控制台默认 GBK，强制 UTF-8 以输出中文与 ²/τ 等字符
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core import data as D  # noqa: E402
from app.core import explain as E  # noqa: E402
from app.core import optimize as O  # noqa: E402
from app.core import pipeline as P  # noqa: E402

EXCEL = sys.argv[1] if len(sys.argv) > 1 else str(
    ROOT / "Multi_algorithm_comparison_basic_version" / "database2.xlsx")
OUT = ROOT / "outputs" / "smoke"


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    df = D.load_excel(EXCEL)
    cols = D.column_names(df)
    target = cols[-1]
    features = cols[:-1]
    categorical = [c for c in features if str(df[c].dtype) == "object"]
    cfg = D.DataConfig(target=target, features=features, categorical=categorical,
                       non_standardize=categorical, test_size=0.2, random_state=42)
    log(f"数据 {df.shape}，目标='{target}'，分类列={categorical}")

    # 1) 树模型全流程（RF + Optuna 小预算）
    rf_opt = O.OptConfig(method="optuna_tpe", n_trials=8, cv_folds=3)
    t0 = time.time()
    rf = P.train_one(df, cfg, "random_forest", rf_opt, log)
    log(f"RF 训练耗时 {time.time() - t0:.1f}s，测试 R²={rf.test_metrics['R2']:.4f}")

    sel = ["regression_fit", "residuals", "importance_native", "importance_perm",
           "importance_shap", "shap_summary", "shap_waterfall", "pdp_1d", "pdp_2d",
           "results_table"]
    t0 = time.time()
    outs = E.generate(rf, sel, str(OUT), fmt="png", dpi=120, top_k=2,
                      scheme_key="material_teal", log=log)
    log(f"RF 生成 {len(outs)} 个输出，耗时 {time.time() - t0:.1f}s")
    for label, path in outs:
        assert Path(path).exists(), f"缺失输出: {path}"
        log(f"   ✓ {label} -> {Path(path).name}")

    # 2) 非树模型训练路径（KNN 手动参数 + 置换重要性）
    knn = P.train_one(df, cfg, "knn", O.OptConfig(method="manual"), log)
    log(f"KNN 测试 R²={knn.test_metrics['R2']:.4f}")
    outs2 = E.generate(knn, ["regression_fit", "importance_perm"], str(OUT),
                       fmt="png", dpi=120, top_k=2, log=log)
    for label, path in outs2:
        assert Path(path).exists(), f"缺失输出: {path}"
        log(f"   ✓ {label} -> {Path(path).name}")

    # 3) 多模型指标对比表
    table = P.metrics_table([rf, knn])
    log("\n指标对比：\n" + table.to_string(index=False))
    log("\n冒烟测试通过 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
