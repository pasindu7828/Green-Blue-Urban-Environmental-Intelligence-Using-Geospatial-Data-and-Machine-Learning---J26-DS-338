import { useEffect, useMemo, useState } from 'react'
import { Brain, Snowflake, Thermometer, TrendingDown, TrendingUp } from 'lucide-react'
import { Bar, BarChart, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import PageHeader from '../../components/shared/PageHeader'
import MetricCard from '../../components/shared/MetricCard'
import GridPicker from '../../features/urbanHeat/components/GridPicker'
import { EmptyState, ErrorState, LoadingState } from '../../features/urbanHeat/components/AsyncState'
import { useUrbanHeat } from '../../features/urbanHeat/context/UrbanHeatContext'
import { urbanHeatApi } from '../../features/urbanHeat/services/urbanHeatApi'
import { fmt, titleCaseFeature } from '../../features/urbanHeat/utils/format'

export default function ExplainableAI() {
  const { selectedGridId, forecastYear } = useUrbanHeat()
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!selectedGridId) { setData(null); return }
    const controller = new AbortController()
    setLoading(true); setError('')
    urbanHeatApi.xai(selectedGridId, forecastYear, controller.signal)
      .then(setData)
      .catch((e) => { if (e?.name !== 'AbortError') setError(e.message) })
      .finally(() => setLoading(false))
    return () => controller.abort()
  }, [selectedGridId, forecastYear])

  const chart = useMemo(() => {
    if (!data) return []
    const rows = []
    for (let i = 1; i <= 3; i += 1) {
      const hf = data[`top${i}_heating_feature`]
      const hv = Number(data[`top${i}_heating_shap_c`])
      if (hf && Number.isFinite(hv)) rows.push({ feature: titleCaseFeature(hf), shap: hv, direction: 'Heating' })
      const cf = data[`top${i}_cooling_feature`]
      const cv = Number(data[`top${i}_cooling_shap_c`])
      if (cf && Number.isFinite(cv)) rows.push({ feature: titleCaseFeature(cf), shap: cv, direction: 'Cooling' })
    }
    return rows.sort((a, b) => a.shap - b.shap)
  }, [data])

  return <div>
    <PageHeader eyebrow="Model Explainability" title="Explainable AI" subtitle="SHAP-based explanation of the selected grid's future LST prediction and feature contributions." badges={['Top heating contributors', 'Top cooling contributors', 'Signed SHAP values (°C)']}/>
    <div className="card-pad mb-5"><GridPicker showYear/></div>

    {loading ? <LoadingState message="Loading SHAP explanation…"/> : error ? <ErrorState error={error}/> : !selectedGridId ? <EmptyState title="Select a grid" message="Choose a real grid and forecast year to load its SHAP explanation."/> : !data ? <EmptyState/> : <>
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard title="Grid" value={data.grid_id} subtitle={data.area_name} icon={Brain} tone="teal"/>
        <MetricCard title={`Predicted LST ${data.year}`} value={`${fmt(data.predicted_lst_c, 2)}°C`} subtitle="Production model prediction" icon={Thermometer} tone="orange"/>
        <MetricCard title="Strongest Heating Contribution" value={`${fmt(Math.max(...chart.map((x) => x.shap)), 3)}°C`} subtitle="Largest positive SHAP contribution" icon={TrendingUp} tone="red"/>
        <MetricCard title="Strongest Cooling Contribution" value={`${fmt(Math.min(...chart.map((x) => x.shap)), 3)}°C`} subtitle="Most negative SHAP contribution" icon={TrendingDown} tone="blue"/>
      </div>

      <div className="mt-5 grid gap-5 xl:grid-cols-[1.25fr_.75fr]">
        <div className="card-pad">
          <h2 className="section-title">Feature contributions</h2>
          <p className="mt-1 text-[11px] text-slate-500">Positive values push the model prediction higher; negative values push it lower.</p>
          <div className="mt-5 h-[390px]"><ResponsiveContainer width="100%" height="100%"><BarChart data={chart} layout="vertical" margin={{ left: 45, right: 18 }}><CartesianGrid strokeDasharray="4 5"/><XAxis type="number" unit="°C"/><YAxis type="category" dataKey="feature" width={145} tick={{ fontSize: 10 }}/><Tooltip formatter={(v) => [`${fmt(v, 4)}°C`, 'SHAP contribution']}/><ReferenceLine x={0} stroke="#475569"/><Bar dataKey="shap" fill="#0f766e" radius={[5, 5, 5, 5]}/></BarChart></ResponsiveContainer></div>
        </div>
        <div className="space-y-5">
          <div className="card-pad">
            <div className="flex items-center gap-2"><TrendingUp size={17} className="text-red-600"/><h2 className="section-title">Heating contributors</h2></div>
            <div className="mt-4 space-y-3">{[1,2,3].map((i) => <div key={i} className="rounded-2xl border border-red-100 bg-red-50 p-4"><div className="text-xs font-semibold text-red-900">{titleCaseFeature(data[`top${i}_heating_feature`])}</div><div className="mt-1 text-lg font-bold text-red-700">+{fmt(data[`top${i}_heating_shap_c`], 4)}°C</div><div className="mt-1 text-[10px] text-red-700/70">Model contribution to a higher prediction</div></div>)}</div>
          </div>
          <div className="card-pad">
            <div className="flex items-center gap-2"><Snowflake size={17} className="text-sky-600"/><h2 className="section-title">Cooling contributors</h2></div>
            <div className="mt-4 space-y-3">{[1,2,3].map((i) => <div key={i} className="rounded-2xl border border-sky-100 bg-sky-50 p-4"><div className="text-xs font-semibold text-sky-900">{titleCaseFeature(data[`top${i}_cooling_feature`])}</div><div className="mt-1 text-lg font-bold text-sky-700">{fmt(data[`top${i}_cooling_shap_c`], 4)}°C</div><div className="mt-1 text-[10px] text-sky-700/70">Model contribution to a lower prediction</div></div>)}</div>
          </div>
        </div>
      </div>
    </>}
  </div>
}
