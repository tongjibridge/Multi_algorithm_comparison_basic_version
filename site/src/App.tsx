import {
  ArrowDownRight,
  ArrowRight,
  Code2,
  Palette,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
} from "lucide-react"

import { BrandMark } from "@/components/brand-mark"
import { OutputGallery } from "@/components/output-gallery"
import { WorkbenchPreview } from "@/components/workbench-preview"
import { WorkflowSection } from "@/components/workflow-section"
import { buttonVariants } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"
import { cn } from "@/lib/utils"

const repositoryUrl =
  "https://github.com/tongjibridge/Multi_algorithm_comparison_basic_version"

const capabilities = [
  { icon: Sparkles, value: "10", label: "个回归模型" },
  { icon: SlidersHorizontal, value: "Optuna", label: "+ mealpy" },
  { icon: ShieldCheck, value: "无泄漏", label: "交叉验证" },
  { icon: Palette, value: "5", label: "组期刊级配色" },
]

function SiteHeader() {
  return (
    <header className="site-header">
      <div className="page-shell header-inner">
        <a className="brand" href="#top" aria-label="ExplainableML 首页">
          <BrandMark />
          <span>ExplainableML</span>
        </a>
        <nav className="header-nav" aria-label="主导航">
          <a href="#capabilities">能力</a>
          <a href="#workflow">工作流</a>
          <a href="#outputs">输出</a>
        </nav>
        <a
          className={cn(
            buttonVariants({ variant: "outline", size: "sm" }),
            "header-action"
          )}
          href="#start"
        >
          开始使用
        </a>
      </div>
    </header>
  )
}

function Hero() {
  return (
    <section className="hero page-shell" id="top">
      <div className="hero-copy">
        <h1>把回归建模、调参与解释，收进一个工作台</h1>
        <p>
          导入 Excel，横向比较 10 个回归模型，用 Optuna 或 mealpy
          搜索参数，并生成 SHAP、PDP、ALE 与期刊级图表。
        </p>
        <div className="hero-actions">
          <a
            className={cn(buttonVariants({ size: "lg" }), "cta-button")}
            href="#workflow"
          >
            查看工作流
            <ArrowDownRight data-icon="inline-end" />
          </a>
          <a
            className={cn(
              buttonVariants({ variant: "outline", size: "lg" }),
              "cta-button"
            )}
            href={repositoryUrl}
            target="_blank"
            rel="noreferrer"
          >
            <Code2 data-icon="inline-start" />
            浏览源码
          </a>
        </div>
      </div>
      <WorkbenchPreview />
      <div className="hero-trace" aria-hidden="true">
        <svg viewBox="0 0 520 140" preserveAspectRatio="none">
          <path d="M4 126L70 112L126 117L190 75L248 92L312 43L374 58L430 21L516 40" />
          <path d="M4 136L70 132L126 91L190 110L248 65L312 83L374 47L430 72L516 31" />
          {[70, 126, 190, 248, 312, 374, 430].map((x, index) => (
            <circle
              key={x}
              cx={x}
              cy={[112, 117, 75, 92, 43, 58, 21][index]}
              r="3"
            />
          ))}
        </svg>
      </div>
    </section>
  )
}

function Capabilities() {
  return (
    <section
      className="capabilities page-shell"
      id="capabilities"
      aria-label="核心能力"
    >
      {capabilities.map(({ icon: Icon, value, label }, index) => (
        <div className="capability" key={label}>
          <Icon aria-hidden="true" />
          <div>
            <strong>{value}</strong>
            <span>{label}</span>
          </div>
          {index < capabilities.length - 1 ? (
            <Separator
              orientation="vertical"
              className="capability-separator"
            />
          ) : null}
        </div>
      ))}
    </section>
  )
}

function StartSection() {
  return (
    <section className="start-section page-shell" id="start">
      <div>
        <h2>让模型比较与解释回到同一条工作流</h2>
        <p>
          本地运行，数据留在你的电脑；开放源码，适合继续扩展模型与解释方法。
        </p>
      </div>
      <div className="start-actions">
        <a
          className={cn(buttonVariants({ size: "lg" }), "cta-button")}
          href={repositoryUrl}
          target="_blank"
          rel="noreferrer"
        >
          开始使用
          <ArrowRight data-icon="inline-end" />
        </a>
        <a
          className={cn(
            buttonVariants({ variant: "outline", size: "lg" }),
            "cta-button"
          )}
          href={repositoryUrl}
          target="_blank"
          rel="noreferrer"
        >
          <Code2 data-icon="inline-start" />
          GitHub
        </a>
      </div>
    </section>
  )
}

function Footer() {
  return (
    <footer className="footer page-shell">
      <div className="brand footer-brand">
        <BrandMark />
        <span>ExplainableML</span>
      </div>
      <p>Local-first · Open source · Built for interpretable regression</p>
      <a href="#top">返回顶部 ↑</a>
    </footer>
  )
}

export function App() {
  return (
    <div className="site-root">
      <SiteHeader />
      <main>
        <Hero />
        <Capabilities />
        <WorkflowSection />
        <OutputGallery />
        <StartSection />
      </main>
      <Footer />
    </div>
  )
}

export default App
