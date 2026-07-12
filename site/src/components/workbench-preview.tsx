import {
  BarChart3,
  Check,
  Database,
  FileSpreadsheet,
  Play,
  SlidersHorizontal,
} from "lucide-react"

import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"

const models = ["XGBoost", "LightGBM", "CatBoost", "Random Forest"]

export function WorkbenchPreview() {
  return (
    <Card className="workbench" aria-label="ExplainableML 工作台预览">
      <CardHeader className="workbench-header">
        <CardTitle>ExplainableML 工作台</CardTitle>
        <CardDescription>数据、模型、优化与解释</CardDescription>
        <CardAction>
          <span className="window-dots" aria-hidden="true">
            <i />
            <i />
            <i />
          </span>
        </CardAction>
      </CardHeader>
      <CardContent className="workbench-content">
        <div className="step-tabs" aria-label="工作台步骤">
          {["数据导入", "模型选择", "参数优化", "解释与输出", "运行"].map(
            (label, index) => (
              <span className={index === 2 ? "active" : ""} key={label}>
                <b>{index + 1}</b>
                {label}
              </span>
            )
          )}
        </div>
        <Separator />
        <div className="workbench-grid">
          <section className="preview-panel data-panel">
            <div className="panel-title">
              <Database /> 数据集
            </div>
            <div className="file-row">
              <FileSpreadsheet />
              <span>data.xlsx</span>
              <Check />
            </div>
            <small>5,120 行 · 48 列 · 目标列 y</small>
            <div className="target-field">
              <span>目标列</span>
              <strong>y</strong>
            </div>
          </section>
          <section className="preview-panel model-panel">
            <div className="panel-title">
              <BarChart3 /> 模型选择 <small>4 / 10</small>
            </div>
            <div className="model-list">
              {models.map((model) => (
                <div key={model}>
                  <Check />
                  <span>{model}</span>
                  <SlidersHorizontal />
                </div>
              ))}
            </div>
          </section>
          <section className="preview-panel optimize-panel">
            <div className="panel-title">
              <SlidersHorizontal /> 参数优化
            </div>
            <label>
              优化器<strong>Optuna · TPE</strong>
            </label>
            <div className="preview-fields">
              <label>
                试验次数<strong>100</strong>
              </label>
              <label>
                CV 折数<strong>5</strong>
              </label>
            </div>
            <label>
              评分<strong>RMSE</strong>
            </label>
          </section>
          <section className="preview-panel metric-panel">
            <div className="panel-title">指标预览</div>
            <div className="metric-head">
              <span>模型</span>
              <span>R²</span>
              <span>RMSE</span>
            </div>
            {models.slice(0, 3).map((model, index) => (
              <div className="metric-row" key={model}>
                <span>{model}</span>
                <b>{[".892", ".887", ".884"][index]}</b>
                <span>{[".312", ".322", ".335"][index]}</span>
              </div>
            ))}
          </section>
        </div>
      </CardContent>
      <CardFooter className="workbench-footer">
        <span>配置将在本地运行</span>
        <span className="run-preview">
          <Play /> 开始运行
        </span>
      </CardFooter>
    </Card>
  )
}
