import os
import pickle
import glob
import joblib

# 设置savemodel文件夹路径
SAVEMODEL_DIR = "f:/py/Multi_algorithm_comparison_basic_version/savemodel"
# 设置输出文件路径
OUTPUT_FILE = "model_best_params.txt"

# 存储所有模型的参数
model_params = []


# 遍历savemodel文件夹下的所有子文件夹和文件，读取已训练模型的最佳参数
def extract_best_params():
    # 使用glob模式匹配所有_model_final.pkl文件
    pattern = os.path.join(SAVEMODEL_DIR, "**", "*_model_final.pkl")
    model_files = glob.glob(pattern, recursive=True)

    if not model_files:
        print(f"未找到任何符合条件的模型文件: {pattern}")
        return

    print(f"找到{len(model_files)}个模型文件")

    # 遍历所有找到的模型文件
    for file_path in model_files:
        try:
            # 获取模型名称（从文件路径中提取）
            model_name = os.path.basename(file_path).replace("_model_final.pkl", "")

            # 加载模型文件
            model = joblib.load(file_path)

            # 尝试获取best_params
            if hasattr(model, "best_params"):
                best_params = model.best_params
                model_params.append(f"{model_name}: {best_params}")
                print(f"成功提取{model_name}的best_params")
            elif hasattr(model, "get_params"):
                # 如果没有best_params_，尝试获取模型的所有参数
                params = model.get_params()
                model_params.append(f"{model_name} (get_params()): {params}")
                print(f"{model_name}没有best_params_属性，使用get_params()")
            else:
                model_params.append(f"{model_name}: 未找到best_params")
                print(f"警告: {model_name}没有best_params_或get_params()属性")

        except Exception as e:
            error_msg = f"处理文件{file_path}时出错: {str(e)}"
            model_params.append(error_msg)
            print(error_msg)

    # 将结果保存到txt文件
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for line in model_params:
            f.write(line + "\n")

    print(f"所有模型参数已保存到: {OUTPUT_FILE}")


if __name__ == "__main__":
    extract_best_params()
