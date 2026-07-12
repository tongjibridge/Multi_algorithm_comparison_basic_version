import {
  BarChart3,
  FileSpreadsheet,
  LineChart,
  ScatterChart,
} from "lucide-react"

const plotItems = [
  { title: "SHAP Summary", kind: "shap" },
  { title: "PDP · ICE", kind: "pdp" },
  { title: "ALE", kind: "ale" },
  { title: "残差分析", kind: "residual" },
  { title: "特征重要性", kind: "importance" },
  { title: "预测结果表", kind: "table" },
]

function PlotThumbnail({ kind }: { kind: string }) {
  if (kind === "shap") {
    return (
      <div className="shap-plot">
        {[0, 1, 2, 3, 4, 5].map((row) => (
          <i key={row} style={{ width: `${82 - row * 7}%` }} />
        ))}
      </div>
    )
  }
  if (kind === "table") {
    return (
      <div className="table-plot">
        {Array.from({ length: 20 }).map((_, index) => (
          <i key={index} />
        ))}
      </div>
    )
  }
  if (kind === "importance") {
    return (
      <div className="bar-plot">
        {[88, 76, 68, 54, 41, 30].map((width) => (
          <i key={width} style={{ width: `${width}%` }} />
        ))}
      </div>
    )
  }
  if (kind === "residual") {
    return (
      <div className="residual-plot">
        {Array.from({ length: 28 }).map((_, index) => (
          <i
            key={index}
            style={{
              left: `${8 + ((index * 31) % 84)}%`,
              top: `${17 + ((index * 47) % 68)}%`,
            }}
          />
        ))}
      </div>
    )
  }
  return (
    <svg className="line-plot" viewBox="0 0 180 100" aria-hidden="true">
      {[0, 1, 2, 3].map((line) => (
        <path
          key={line}
          d={`M5 ${82 - line * 5} C35 ${30 + line * 7}, 55 ${88 - line * 3}, 82 ${55 + line * 3} S135 ${18 + line * 8}, 175 ${28 + line * 4}`}
        />
      ))}
      {kind === "ale" ? (
        <path
          className="line-main"
          d="M5 84 C32 78, 44 61, 67 65 S91 45, 113 48 S145 22, 175 18"
        />
      ) : null}
    </svg>
  )
}

const icons = [
  ScatterChart,
  LineChart,
  LineChart,
  ScatterChart,
  BarChart3,
  FileSpreadsheet,
]

export function OutputGallery() {
  return (
    <section className="outputs-section page-shell" id="outputs">
      <div className="section-heading output-heading">
        <h2>解释结果，不止一个分数</h2>
        <p>
          从全局重要性到单样本解释，从效应曲线到预测结果表，一次运行统一输出。
        </p>
      </div>
      <div className="plot-gallery">
        {plotItems.map((item, index) => {
          const Icon = icons[index]
          return (
            <article className="plot-item" key={item.title}>
              <div className="plot-canvas">
                <PlotThumbnail kind={item.kind} />
              </div>
              <div>
                <Icon />
                <strong>{item.title}</strong>
              </div>
            </article>
          )
        })}
      </div>
    </section>
  )
}
