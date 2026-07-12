import { useState } from "react"
import { ArrowRight, Check } from "lucide-react"

import { Button } from "@/components/ui/button"
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Switch } from "@/components/ui/switch"

const BASE = import.meta.env.BASE_URL

const steps = [
  { title: "导入 Excel", description: "指定目标、特征与分类列" },
  { title: "选择模型", description: "一次勾选多个回归模型" },
  { title: "设置优化", description: "Optuna、mealpy 与自定义搜索空间", image: `${BASE}/images/params.png` },
  { title: "选择输出", description: "SHAP、PDP、ALE 与结果表", image: `${BASE}/images/gallery.png` },
  { title: "运行与比较", description: "查看日志、指标和图像画廊", image: `${BASE}/images/workbench.png` },
]

export function WorkflowSection() {
  const [activeStep, setActiveStep] = useState(2)

  const currentStep = steps[activeStep]

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
              <CardTitle>{currentStep.title}</CardTitle>
              <CardDescription>{currentStep.description}</CardDescription>
              <CardAction>
                <span className="step-counter">0{activeStep + 1} / 05</span>
              </CardAction>
            </CardHeader>
            <CardContent className="workflow-preview-content">
              {currentStep.image ? (
                <img
                  src={currentStep.image}
                  alt={`${currentStep.title} 预览`}
                  className="workflow-preview-image"
                />
              ) : (
                <>
                  <div className="parameter-head">
                    <span>参数</span>
                    <span>下限</span>
                    <span>上限</span>
                    <span>对数采样</span>
                  </div>
                  <div className="parameter-row">
                    <span>learning_rate</span>
                    <span>0.01</span>
                    <span>0.30</span>
                    <Switch defaultChecked aria-label="启用对数采样" />
                  </div>
                </>
              )}
            </CardContent>
            <CardFooter>
              <span>搜索空间仅作用于本次运行</span>
              <Button size="sm">
                保存 <ArrowRight data-icon="inline-end" />
              </Button>
            </CardFooter>
          </Card>
        </div>
      </div>
    </section>
  )
}
