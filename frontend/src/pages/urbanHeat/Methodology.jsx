import { Brain, ChartNoAxesCombined, FlaskConical, Layers3, Map, ShieldCheck, Target, Trees } from 'lucide-react'
import PageHeader from '../../components/shared/PageHeader'

const stages = [
  { icon: Map, title: '1. Geospatial data preparation', text: 'Historical Kaduwela grid-level remote-sensing and GIS predictors are cleaned, validated and aligned to the stable grid framework.' },
  { icon: Layers3, title: '2. Leakage-aware feature engineering', text: 'Spatial, temporal, lag, rolling and interaction features are built with time-aware rules so future information is not intentionally leaked into model development.' },
  { icon: ChartNoAxesCombined, title: '3. Model benchmarking & validation', text: 'Baseline regression, Random Forest, XGBoost, LightGBM and NGBoost are compared using time-aware development and spatial/temporal validation stages.' },
  { icon: ShieldCheck, title: '4. Frozen production strategy', text: 'The finalized production strategy uses the selected LightGBM workflow, followed by a reserved final evaluation and production retraining for downstream forecasting.' },
  { icon: Trees, title: '5. Heat trajectory intelligence', text: 'Mann–Kendall and Sen-slope evidence supports a five-category relative spatial heat trajectory classification: Persistent Hot, Intensifying, Emerging, Diminishing and Persistent Cool.' },
  { icon: Brain, title: '6. Explainable forecasting', text: 'SHAP contributions are exported for future grid-year predictions so planners can inspect which model features pushed a prediction higher or lower.' },
  { icon: FlaskConical, title: '7. Planning scenarios & uncertainty', text: 'Green-cover, built-up intensity, combined green+built and blue-space sensitivity scenarios are evaluated alongside forecast and scenario uncertainty outputs.' },
  { icon: Target, title: '8. Cooling priority', text: 'Cooling indicators are objectively weighted using CRITIC and ranked with TOPSIS. Sensitivity validation is used to evaluate ranking stability before dashboard export.' },
]

export default function Methodology() {
  return <div>
    <PageHeader eyebrow="Research Workflow" title="Urban Heat Methodology" subtitle="A concise frontend summary of the finalized Urban Heat research pipeline represented by the supplied source scripts. The frontend does not rerun or alter any research step." badges={['Historical 2015–2025', 'Forecast 2026–2030', 'CRITIC + TOPSIS', 'SHAP + uncertainty']}/>
    <div className="grid gap-5 md:grid-cols-2">{stages.map(({ icon: Icon, title, text }) => <div key={title} className="card-pad"><div className="flex items-start gap-4"><div className="grid h-11 w-11 shrink-0 place-items-center rounded-2xl bg-teal-50 text-teal-700"><Icon size={21}/></div><div><h2 className="text-sm font-bold text-slate-900">{title}</h2><p className="mt-2 text-[12px] leading-6 text-slate-500">{text}</p></div></div></div>)}</div>

  </div>
}
