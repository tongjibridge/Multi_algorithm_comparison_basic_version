import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class CorrelationAnalyzer:
    def __init__(self, data_path=None, data=None):
        """初始化分析器，可以通过文件路径或直接传入DataFrame"""
        if data_path:
            self.data = pd.read_csv(data_path)
        elif data is not None:
            self.data = data
        else:
            raise ValueError("请提供数据路径或DataFrame")
            
        # 假设最后一列是因变量
        self.dependent_var = self.data.columns[-1]
        # 第一列和第二列是离散变量
        self.categorical_vars = self.data.columns[:2].tolist()
        # 中间的是连续自变量
        self.continuous_vars = self.data.columns[2:-1].tolist()
        
        print(f"数据加载完成，共{self.data.shape[0]}行，{self.data.shape[1]}列")
        print(f"因变量: {self.dependent_var}")
        print(f"离散自变量: {self.categorical_vars}")
        print(f"连续自变量: {self.continuous_vars}")
    
    def check_normality(self, variables=None):
        """检验连续变量是否符合正态分布"""
        if variables is None:
            variables = self.continuous_vars + [self.dependent_var]
            
        normality_results = {}
        print("\n正态性检验结果 (Shapiro-Wilk):")
        print("变量名 | 统计量 | p值 | 是否正态分布(α=0.05)")
        print("-" * 50)
        
        for var in variables:
            stat, p = stats.shapiro(self.data[var].dropna())
            is_normal = p > 0.05
            normality_results[var] = is_normal
            print(f"{var:8} | {stat:.4f} | {p:.4f} | {is_normal}")
            
        return normality_results
    
    def analyze_continuous_correlation(self, normality_results=None):
        """分析连续变量之间以及与因变量的相关性"""
        if normality_results is None:
            normality_results = self.check_normality()
            
        # 所有连续变量（包括因变量）
        all_continuous = self.continuous_vars + [self.dependent_var]
        
        # 判断是否所有连续变量都符合正态分布
        all_normal = all(normality_results[var] for var in all_continuous)
        
        # 选择合适的相关系数方法
        method = 'pearson' if all_normal else 'spearman'
        print(f"\n连续变量相关性分析使用方法: {method}")
        
        # 计算相关系数
        corr_matrix = self.data[all_continuous].corr(method=method)
        
        # 打印与因变量的相关性
        print(f"\n与因变量 {self.dependent_var} 的相关性:")
        dep_corr = corr_matrix[self.dependent_var].sort_values(ascending=False)
        print(dep_corr.drop(self.dependent_var).to_string())
        
        # 可视化相关性矩阵
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   vmin=-1, vmax=1, center=0, fmt='.2f',
                   square=True, linewidths=.5, cbar_kws={"shrink": .8})
        plt.title(f'连续变量相关性矩阵 ({method}方法)', fontsize=15)
        plt.tight_layout()
        plt.show()
        
        return corr_matrix
    
    def analyze_categorical_continuous(self):
        """分析离散变量与连续因变量的相关性"""
        results = {}
        
        for cat_var in self.categorical_vars:
            print(f"\n分析离散变量: {cat_var} 与因变量 {self.dependent_var} 的关系")
            
            # 获取类别列表
            categories = self.data[cat_var].unique()
            print(f"类别: {categories}")
            
            # 按类别分组
            groups = [self.data[self.data[cat_var] == cat][self.dependent_var].dropna() 
                     for cat in categories]
            
            # 检验方差齐性
            stat, p_levene = stats.levene(*groups)
            equal_var = p_levene > 0.05
            print(f"方差齐性检验 (Levene): p值 = {p_levene:.4f}, {'齐性' if equal_var else '不齐性'}")
            
            # 检验因变量是否符合正态分布
            stat, p_shapiro = stats.shapiro(self.data[self.dependent_var].dropna())
            is_normal = p_shapiro > 0.05
            print(f"因变量正态性检验: p值 = {p_shapiro:.4f}, {'符合' if is_normal else '不符合'}正态分布")
            
            # 选择合适的检验方法
            if is_normal and equal_var and len(categories) > 2:
                # 多类别，正态分布且方差齐性，使用单因素方差分析
                stat, p = stats.f_oneway(*groups)
                test_name = "单因素方差分析 (ANOVA)"
            elif len(categories) > 2:
                # 多类别，非正态分布或方差不齐，使用Kruskal-Wallis检验
                stat, p = stats.kruskal(*groups)
                test_name = "Kruskal-Wallis检验"
            else:
                # 二分类变量，使用t检验或Mann-Whitney U检验
                if is_normal and equal_var:
                    stat, p = stats.ttest_ind(*groups)
                    test_name = "独立样本t检验"
                else:
                    stat, p = stats.mannwhitneyu(*groups)
                    test_name = "Mann-Whitney U检验"
            
            print(f"{test_name}结果: 统计量 = {stat:.4f}, p值 = {p:.4f}")
            print(f"结论: {'存在显著相关性' if p < 0.05 else '不存在显著相关性'} (α=0.05)")
            
            # 可视化：箱线图
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=cat_var, y=self.dependent_var, data=self.data)
            plt.title(f'{cat_var} 与 {self.dependent_var} 的关系', fontsize=15)
            plt.suptitle(f'{test_name} p值 = {p:.4f}', y=0.01)
            plt.tight_layout()
            plt.show()
            
            results[cat_var] = {
                'test_name': test_name,
                'statistic': stat,
                'p_value': p,
                'significant': p < 0.05
            }
        
        return results
    
    def check_multicollinearity(self):
        """检查连续自变量之间的多重共线性（VIF）"""
        print("\n多重共线性检验 (VIF):")
        print("变量名 | VIF值 | 共线性程度")
        print("-" * 40)
        
        # 准备数据
        X = self.data[self.continuous_vars].dropna()
        
        # 计算VIF
        vif_data = pd.DataFrame()
        vif_data["变量名"] = X.columns
        vif_data["VIF值"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        
        # 判断共线性程度
        vif_data["共线性程度"] = vif_data["VIF值"].apply(
            lambda x: "无" if x < 5 else "中等" if x < 10 else "严重"
        )
        
        # 按VIF值排序
        vif_data = vif_data.sort_values("VIF值", ascending=False)
        
        # 打印结果
        print(vif_data.to_string(index=False))
        
        return vif_data
    
    def analyze_discrete_relationship(self):
        """分析两个离散变量之间的关系"""
        if len(self.categorical_vars) < 2:
            print("\n只有一个离散变量，无需分析离散变量之间的关系")
            return None
            
        var1, var2 = self.categorical_vars[0], self.categorical_vars[1]
        print(f"\n分析两个离散变量: {var1} 与 {var2} 的关系")
        
        # 创建列联表
        contingency = pd.crosstab(self.data[var1], self.data[var2])
        print("\n列联表:")
        print(contingency)
        
        # 卡方检验
        chi2, p, dof, expected = stats.chi2_contingency(contingency)
        
        # 计算Cramer's V系数（衡量关联强度）
        n = contingency.sum().sum()
        min_dim = min(contingency.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim))
        
        print(f"\n卡方检验结果: 卡方值 = {chi2:.4f}, p值 = {p:.4f}, 自由度 = {dof}")
        print(f"Cramer's V系数: {cramers_v:.4f} (关联强度: {'弱' if cramers_v < 0.3 else '中' if cramers_v < 0.5 else '强'})")
        print(f"结论: {'存在显著关联' if p < 0.05 else '不存在显著关联'} (α=0.05)")
        
        # 可视化：马赛克图
        plt.figure(figsize=(10, 6))
        sns.heatmap(contingency, annot=True, fmt='d', cmap='YlGnBu')
        plt.title(f'{var1} 与 {var2} 的关系', fontsize=15)
        plt.suptitle(f'卡方检验 p值 = {p:.4f}, Cramer\'s V = {cramers_v:.4f}', y=0.01)
        plt.tight_layout()
        plt.show()
        
        return {
            'chi2': chi2,
            'p_value': p,
            'cramers_v': cramers_v,
            'significant': p < 0.05
        }
    
    def full_analysis(self):
        """执行完整的相关性分析流程"""
        print("=" * 60)
        print("开始执行完整相关性分析")
        print("=" * 60)
        
        # 1. 数据基本信息
        print("\n" + "=" * 40)
        print("1. 数据基本信息")
        print("=" * 40)
        print(self.data.info())
        print("\n描述性统计:")
        print(self.data.describe(include='all'))
        
        # 2. 正态性检验
        print("\n" + "=" * 40)
        print("2. 正态性检验")
        print("=" * 40)
        normality = self.check_normality()
        
        # 3. 连续变量相关性分析
        print("\n" + "=" * 40)
        print("3. 连续变量相关性分析")
        print("=" * 40)
        cont_corr = self.analyze_continuous_correlation(normality)
        
        # 4. 离散变量与因变量的关系分析
        print("\n" + "=" * 40)
        print("4. 离散变量与因变量的关系分析")
        print("=" * 40)
        cat_cont_rel = self.analyze_categorical_continuous()
        
        # 5. 离散变量之间的关系分析
        print("\n" + "=" * 40)
        print("5. 离散变量之间的关系分析")
        print("=" * 40)
        cat_cat_rel = self.analyze_discrete_relationship()
        
        # 6. 多重共线性检验
        print("\n" + "=" * 40)
        print("6. 多重共线性检验")
        print("=" * 40)
        vif_results = self.check_multicollinearity()
        
        print("\n" + "=" * 60)
        print("相关性分析完成")
        print("=" * 60)
        
        return {
            'normality': normality,
            'continuous_c': cont_corr,
            'categorical_continuous_relationship': cat_cont_rel,
            'categorical_categorical_relationship': cat_cat_rel,
            'multicollinearity': vif_results
        }

# 使用示例
if __name__ == "__main__":
    # 示例：使用随机生成的数据进行测试
    # 实际使用时，替换为自己的数据路径或DataFrame
    np.random.seed(42)  # 设置随机种子，保证结果可复现
    
    df = pd.read_excel("./FRP筋UHPC粘结锚固数据.xlsx",sheet_name="机器学习版")
    
    # 创建分析器实例并执行完整分析
    analyzer = CorrelationAnalyzer(data=df)
    results = analyzer.full_analysis()
