import {
  Card,
  CardAction,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"

export function WorkbenchPreview() {
  return (
    <Card className="workbench workbench-screenshot" aria-label="ExplainableML 工作台预览">
      <img
        src="/images/workbench.png"
        alt="ExplainableML 工作台界面"
        className="workbench-image"
      />
      <CardHeader className="workbench-header-overlay">
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
    </Card>
  )
}
