import os
import subprocess

python_path = "./.venv/Scripts/python.exe"
# 需要运行的算法脚本列表
scripts = [
    "bay.py",
    "DT.py",
    "KNN.py",
    "MLP.py",
    "RF.py",
    "SVR.py",
    "CatBoost.py",
    "elm.py",
    "GBR.py",
    "LightGBM-optuna.py",
    "tabm_reg.py",
    "XGBoost.py",
    "xrfm_reg.py",
]

# 遍历并依次运行每个算法脚本
for script in scripts:
    print(f"正在运行 {script} ...")
    try:
        subprocess.run([python_path, script], check=True)
        print(f"{script} 运行成功！")
    except subprocess.CalledProcessError as e:
        print(f"{script} 运行失败，错误信息：{e}")
