import { useState } from "react"
import { Check } from "lucide-react"

import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"

const steps = [
  {
    title: "导入 Excel",
    description: "指定目标、特征与分类列",
    detail: "上传你的 Excel 数据文件，系统会自动解析所有列。选择你的目标变量（预测对象）、特征列（输入变量），如果是分类问题，还需指定分类列。支持 .xlsx 和 .csv 格式。"
  },
  {
    title: "选择模型",
    description: "一次勾选多个回归模型",
    detail: "内置多种主流回归算法：线性回归、随机森林、梯度提升树（XGBoost/LightGBM）、支持向量机等。你可以同时勾选多个模型，系统会自动并行训练并生成对比结果。"
  },
  {
    title: "设置优化",
    description: "Optuna、mealpy 与自定义搜索空间",
    detail: "自动超参数调优，支持 Optuna（基于 TPE 算法）和 mealpy（元启发式算法）。你也可以手动定义每个参数的搜索范围，设置上下界、采样方式（线性/对数），让模型找到最优配置。"
  },
  {
    title: "选择输出",
    description: "SHAP、PDP、ALE 与结果表",
    detail: "选择模型解释性分析方式：SHAP 值展示每个特征的贡献度，PDP（部分依赖图）显示特征与预测的关系，ALE（累积局部效应）处理特征相关性。同时导出详细的性能指标结果表。"
  },
  {
    title: "运行与比较",
    description: "查看日志、指标和图像画廊",
    detail: "一键运行训练流程，实时查看优化日志。训练完成后，进入结果画廊对比各模型表现 — R²、RMSE、MAE 等指标一目了然，所有可解释性图表集中展示，方便决策。"
  },
]

export function WorkflowSection() {
  const [activeStep, setActiveStep] = useState(2)

  return (
    <section className="workflow-section" id="workflow">
      <div className="page-shell">
        <div className="section-heading">
          <h2>从数据到解释，只需五步</h2>
          <p>把原本分散的脚本操作，收束成可追踪、可重复的一条流程。</p>
        </div>
        <div
          className="workflow-rail"
          role="tablist"
          aria-label="ExplainableML 五步工作流"
        >
          {steps.map((step, index) => (
            <button
              className={
                activeStep === index ? "workflow-step active" : "workflow-step"
              }
              key={step.title}
              onClick={() => setActiveStep(index)}
              role="tab"
              aria-selected={activeStep === index}
            >
              <span className="step-number">
                {index < activeStep ? (
                  <Check />
                ) : (
                  String(index + 1).padStart(2, "0")
                )}
              </span>
              <strong>{step.title}</strong>
              <small>{step.description}</small>
            </button>
          ))}
        </div>
        <div className="workflow-detail">
          <Card className="parameter-card">
            <CardHeader>
              <CardTitle>{steps[activeStep].title}</CardTitle>
              <CardDescription>{steps[activeStep].description}</CardDescription>
              <CardAction>
                <span className="step-counter">0{activeStep + 1} / 05</span>
              </CardAction>
            </CardHeader>
            <CardContent>
              <p className="step-detail-text">{steps[activeStep].detail}</p>
            </CardContent>
          </Card>
        </div>
      </div>
    </section>
  )
}
