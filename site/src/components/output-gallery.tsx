import {
  BarChart3,
  LineChart,
  ScatterChart,
} from "lucide-react"

const BASE = import.meta.env.BASE_URL

const plotItems = [
  { title: "SHAP Summary", kind: "shap", image: `${BASE}/images/shap-summary.png` },
  { title: "PDP · ICE", kind: "pdp", image: `${BASE}/images/pdp-ice.png` },
  { title: "ALE", kind: "ale", image: `${BASE}/images/ale.png` },
  { title: "残差分析", kind: "residual", image: `${BASE}/images/residuals.png` },
  { title: "特征重要性", kind: "importance", image: `${BASE}/images/importance-shap.png` },
  { title: "回归拟合", kind: "fit", image: `${BASE}/images/regression-fit.png` },
]

function PlotThumbnail({ image }: { image: string }) {
  return (
    <img
      src={image}
      alt="Chart output"
      loading="lazy"
      style={{ width: "100%", height: "100%", objectFit: "contain" }}
    />
  )
}

const icons = [
  ScatterChart,
  LineChart,
  LineChart,
  ScatterChart,
  BarChart3,
  LineChart,
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
                <PlotThumbnail image={item.image} />
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
