import { useEffect, useMemo, useState } from 'react'
import { Activity, Award, Brain, MapPin, ShieldCheck, Thermometer } from 'lucide-react'
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import PageHeader from '../../components/shared/PageHeader'
import MetricCard from '../../components/shared/MetricCard'
import Badge from '../../components/ui/Badge'
import GridPicker from '../../features/urbanHeat/components/GridPicker'
import { EmptyState, ErrorState, LoadingState } from '../../features/urbanHeat/components/AsyncState'
import { useUrbanHeat } from '../../features/urbanHeat/context/UrbanHeatContext'
import { urbanHeatApi } from '../../features/urbanHeat/services/urbanHeatApi'
import { confidenceTone, fmt, fmtInt, fmtSigned, titleCaseFeature, trajectoryTone } from '../../features/urbanHeat/utils/format'

export default function ZoneDetails() {
  const { selectedGridId, forecastYear } = useUrbanHeat()
  const [state, setState] = useState({ loading: false, error: '', metadata: null, history: [], future: [], trajectory: null, priority: null, xai: null, uncertainty: null })

  useEffect(() => {
    if (!selectedGridId) { setState((s) => ({ ...s, loading: false, error: '' })); return }
    const controller = new AbortController()
    setState((s) => ({ ...s, loading: true, error: '' }))
    Promise.allSettled([
      urbanHeatApi.gridMetadata(selectedGridId, controller.signal),
      urbanHeatApi.gridHistory(selectedGridId, controller.signal),
      urbanHeatApi.gridForecastAll(selectedGridId, controller.signal),
      urbanHeatApi.gridTrajectory(selectedGridId, controller.signal),
      urbanHeatApi.gridPriority(selectedGridId, controller.signal),
      urbanHeatApi.xai(selectedGridId, forecastYear, controller.signal),
      urbanHeatApi.uncertainty(selectedGridId, forecastYear, controller.signal),
    ]).then((items) => {
      const failures = items.filter((x) => x.status === 'rejected')
      const [m, h, f, t, p, x, u] = items
      setState({
        loading: false,
        error: failures.length === items.length ? failures[0]?.reason?.message || 'Could not load grid details.' : '',
        metadata: m.status === 'fulfilled' ? m.value : null,
        history: h.status === 'fulfilled' ? h.value?.data || [] : [],
        future: f.status === 'fulfilled' ? f.value?.data || [] : [],
        trajectory: t.status === 'fulfilled' ? t.value : null,
        priority: p.status === 'fulfilled' ? p.value : null,
        xai: x.status === 'fulfilled' ? x.value : null,
        uncertainty: u.status === 'fulfilled' ? u.value : null,
      })
    })
    return () => controller.abort()
  }, [selectedGridId, forecastYear])

  const selectedForecast = useMemo(() => state.future.find((row) => Number(row.year) === Number(forecastYear)), [state.future, forecastYear])
  const chart = useMemo(() => {
    const hist = state.history.map((row) => ({ year: Number(row.year), observed: Number(row.actual_lst_c), predicted: null }))
    const future = state.future.map((row) => ({ year: Number(row.year), observed: null, predicted: Number(row.predicted_lst_c) }))
    if (hist.length && future.length) future.unshift({ year: hist.at(-1).year, observed: hist.at(-1).observed, predicted: hist.at(-1).observed })
    return [...hist, ...future]
  }, [state.history, state.future])

  const shapRows = useMemo(() => {
    if (!state.xai) return []
    const rows = []
    for (let i = 1; i <= 3; i += 1) {
      if (state.xai[`top${i}_heating_feature`]) rows.push({ feature: titleCaseFeature(state.xai[`top${i}_heating_feature`]), value: Number(state.xai[`top${i}_heating_shap_c`]) })
      if (state.xai[`top${i}_cooling_feature`]) rows.push({ feature: titleCaseFeature(state.xai[`top${i}_cooling_feature`]), value: Number(state.xai[`top${i}_cooling_shap_c`]) })
    }
    return rows.sort((a, b) => Math.abs(b.value) - Math.abs(a.value)).slice(0, 4)
  }, [state.xai])

  return <div>
    <PageHeader eyebrow="Grid Intelligence" title="Grid Details" subtitle="Consolidated evidence for one real Kaduwela grid: metadata, observed LST, future forecast, relative trajectory, uncertainty, explainability and cooling priority." badges={[`Forecast context ${forecastYear}`, 'Grid ID is the stable join key']}/>
    <div className="card-pad mb-5"><GridPicker showYear/></div>

    {!selectedGridId ? <EmptyState title="Select a grid" message="Choose a grid from the selector or click a map on another Urban Heat page."/> : state.loading ? <LoadingState message={`Loading complete profile for grid ${selectedGridId}…`}/> : state.error ? <ErrorState error={state.error}/> : <>
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-5">
        <MetricCard title="Grid" value={selectedGridId} subtitle={state.metadata?.area_name || selectedForecast?.area_name || '—'} icon={MapPin} tone="teal"/>
        <MetricCard title={`Predicted LST ${forecastYear}`} value={selectedForecast ? `${fmt(selectedForecast.predicted_lst_c, 2)}°C` : '—'} subtitle={selectedForecast ? `Change from 2025: ${fmtSigned(selectedForecast.change_from_2025_grid_actual_c, 2, '°C')}` : 'Forecast unavailable'} icon={Thermometer} tone="orange"/>
        <MetricCard title="Trajectory" value={state.trajectory?.trajectory_class || '—'} subtitle="Relative spatial heat trajectory" icon={Activity} tone="amber"/>
        <MetricCard title="Cooling Rank" value={state.priority?.topsis_rank != null ? `#${fmtInt(state.priority.topsis_rank)}` : '—'} subtitle={state.priority ? `Score ${fmt(state.priority.topsis_priority_score, 4)}` : 'Priority unavailable'} icon={Award} tone="red"/>
        <MetricCard title="Confidence" value={state.uncertainty?.relative_confidence_label || '—'} subtitle={state.uncertainty ? `${fmt(state.uncertainty.forecast_pi95_lower_c, 2)}–${fmt(state.uncertainty.forecast_pi95_upper_c, 2)}°C PI95` : 'Uncertainty unavailable'} icon={ShieldCheck} tone="blue"/>
      </div>

      <div className="mt-5 grid gap-5 xl:grid-cols-[1.3fr_.7fr]">
        <div className="card-pad">
          <h2 className="section-title">Observed and forecast LST profile</h2>
          <div className="mt-4 h-[360px]"><ResponsiveContainer width="100%" height="100%"><LineChart data={chart}><CartesianGrid strokeDasharray="4 5"/><XAxis dataKey="year"/><YAxis domain={['auto', 'auto']} unit="°C"/><Tooltip formatter={(v) => [`${fmt(v, 2)}°C`]}/><Legend/><Line type="monotone" dataKey="observed" name="Observed LST" stroke="#0f766e" strokeWidth={2.5}/><Line type="monotone" dataKey="predicted" name="Predicted LST" stroke="#e0472b" strokeWidth={2.5} strokeDasharray="6 4"/></LineChart></ResponsiveContainer></div>
        </div>

        <div className="space-y-5">
          <div className="card-pad"><h2 className="section-title">Classification & uncertainty</h2><div className="mt-4 flex flex-wrap gap-2"><Badge variant={trajectoryTone(state.trajectory?.trajectory_class)}>{state.trajectory?.trajectory_class || 'No trajectory'}</Badge><Badge variant={confidenceTone(state.uncertainty?.relative_confidence_label)}>{state.uncertainty?.relative_confidence_label || 'No confidence label'}</Badge></div><div className="mt-4 grid grid-cols-2 gap-3 text-xs"><div className="soft-panel"><span className="text-slate-400">Sen slope</span><b className="mt-1 block">{fmt(state.trajectory?.sen_slope_c_per_year, 4)} °C/year</b></div><div className="soft-panel"><span className="text-slate-400">MK p-value</span><b className="mt-1 block">{fmt(state.trajectory?.mk_p_value, 4)}</b></div><div className="soft-panel"><span className="text-slate-400">Historical error tier</span><b className="mt-1 block">{state.uncertainty?.local_historical_error_tier || '—'}</b></div><div className="soft-panel"><span className="text-slate-400">PI95 width</span><b className="mt-1 block">{fmt(state.uncertainty?.forecast_pi95_width_c, 2)}°C</b></div></div></div>
          <div className="card-pad"><div className="flex items-center gap-2"><Brain size={17} className="text-teal-700"/><h2 className="section-title">Strongest model contributions</h2></div><div className="mt-4 space-y-2">{shapRows.length ? shapRows.map((row) => <div key={`${row.feature}-${row.value}`} className="flex items-center justify-between rounded-xl bg-slate-50 px-3 py-2 text-xs"><span className="pr-3 text-slate-600">{row.feature}</span><b className={row.value >= 0 ? 'text-red-700' : 'text-sky-700'}>{row.value > 0 ? '+' : ''}{fmt(row.value, 3)}°C</b></div>) : <p className="text-sm text-slate-500">XAI data unavailable.</p>}</div></div>
        </div>
      </div>

      <div className="mt-5">
        <div className="card-pad"><h2 className="section-title">Cooling-priority stability</h2><div className="mt-4 grid grid-cols-2 gap-3 text-xs"><div className="soft-panel"><span className="text-slate-400">Priority percentile</span><b className="mt-1 block">{fmt(state.priority?.priority_percentile, 1)}%</b></div><div className="soft-panel"><span className="text-slate-400">Mean sensitivity rank</span><b className="mt-1 block">{fmt(state.priority?.mean_sensitivity_rank, 1)}</b></div><div className="soft-panel"><span className="text-slate-400">Worst sensitivity rank</span><b className="mt-1 block">{fmtInt(state.priority?.worst_sensitivity_rank)}</b></div><div className="soft-panel"><span className="text-slate-400">Mean abs. rank shift</span><b className="mt-1 block">{fmt(state.priority?.mean_abs_rank_shift, 2)}</b></div></div></div>
      </div>
    </>}
  </div>
}
