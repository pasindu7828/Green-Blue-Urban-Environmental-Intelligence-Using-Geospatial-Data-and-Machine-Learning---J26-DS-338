import { useCallback, useEffect, useMemo, useState } from 'react'
import { Grid3X3, MapPinned, Thermometer, TrendingUp } from 'lucide-react'
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import MetricCard from '../../components/shared/MetricCard'
import PageHeader from '../../components/shared/PageHeader'
import Badge from '../../components/ui/Badge'
import GridPicker from '../../features/urbanHeat/components/GridPicker'
import KaduwelaMap from '../../features/urbanHeat/components/KaduwelaMap'
import { ErrorState, LoadingState } from '../../features/urbanHeat/components/AsyncState'
import { ThermalLegend, thermalColor } from '../../features/urbanHeat/components/MapLegends'
import { urbanHeatApi } from '../../features/urbanHeat/services/urbanHeatApi'
import { useUrbanHeat } from '../../features/urbanHeat/context/UrbanHeatContext'
import { confidenceTone, fmt, fmtInt, fmtSigned, trajectoryTone } from '../../features/urbanHeat/utils/format'

export default function Dashboard() {
  const { selectedGridId, setSelectedGridId, forecastYear } = useUrbanHeat()
  const [state, setState] = useState({ loading: true, error: '', map: null, health: null, priority: [], trajectory: [] })
  const [gridDetails, setGridDetails] = useState(null)

  const load = useCallback(async () => {
    const controller = new AbortController()
    setState((s) => ({ ...s, loading: true, error: '' }))
    try {
      const [map, health, priority, trajectory] = await Promise.all([
        urbanHeatApi.forecastMap(forecastYear, controller.signal),
        urbanHeatApi.dataHealth(controller.signal),
        urbanHeatApi.priority({ limit: 8 }, controller.signal),
        urbanHeatApi.trajectory('', controller.signal),
      ])
      setState({ loading: false, error: '', map, health, priority: priority?.data || [], trajectory: trajectory?.data || [] })
    } catch (error) {
      if (error?.name !== 'AbortError') setState((s) => ({ ...s, loading: false, error: error.message }))
    }
    return () => controller.abort()
  }, [forecastYear])

  useEffect(() => { load() }, [load])

  useEffect(() => {
    if (!selectedGridId) { setGridDetails(null); return }
    const controller = new AbortController()
    urbanHeatApi.gridDetails(selectedGridId, forecastYear, controller.signal)
      .then(setGridDetails)
      .catch(() => setGridDetails(null))
    return () => controller.abort()
  }, [selectedGridId, forecastYear])

  const summary = useMemo(() => {
    const features = state.map?.features || []
    const values = features.map((f) => Number(f?.properties?.predicted_lst_c)).filter(Number.isFinite)
    const mean = values.length ? values.reduce((a, b) => a + b, 0) / values.length : null
    const max = values.length ? Math.max(...values) : null
    const min = values.length ? Math.min(...values) : null
    return { count: features.length, mean, min, max }
  }, [state.map])

  const trajectoryChart = useMemo(() => {
    const counts = {}
    for (const row of state.trajectory) counts[row.trajectory_class || 'Unknown'] = (counts[row.trajectory_class || 'Unknown'] || 0) + 1
    return Object.entries(counts).map(([name, count]) => ({ name, count }))
  }, [state.trajectory])

  if (state.loading) return <LoadingState message={`Loading ${forecastYear} Urban Heat dashboard…`}/>
  if (state.error) return <ErrorState error={state.error} onRetry={load}/>

  const values = (state.map?.features || []).map((f) => Number(f?.properties?.predicted_lst_c)).filter(Number.isFinite)
  const minValue = values.length ? Math.min(...values) : null
  const maxValue = values.length ? Math.max(...values) : null
  const selectedForecast = gridDetails?.forecast
  const selectedPriority = gridDetails?.priority

  return <div>
    <PageHeader
      eyebrow="Urban Heat Intelligence"
      title="Kaduwela Heat Intelligence Dashboard"
      subtitle="Real backend-connected overview of grid-level heat forecasts, relative heat trajectories, uncertainty and cooling-priority evidence."
      badges={[`Forecast ${forecastYear}`, '2,194-grid study design', 'WGS84 map geometry']}
    />

    <div className="card-pad mb-5"><GridPicker showYear/></div>

    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
      <MetricCard title="Mapped Grids" value={fmtInt(summary.count)} subtitle="Forecast polygons returned by the API" icon={Grid3X3} tone="teal"/>
      <MetricCard title={`Mean Predicted LST ${forecastYear}`} value={`${fmt(summary.mean, 2)}°C`} subtitle="Mean across currently returned grids" icon={Thermometer} tone="orange"/>
      <MetricCard title="Highest Predicted LST" value={`${fmt(summary.max, 2)}°C`} subtitle="Maximum grid prediction on this map" icon={TrendingUp} tone="red"/>
      <MetricCard title="Data Pipeline" value={state.health?.status === 'ok' ? 'Ready' : 'Check'} subtitle={state.health?.status === 'ok' ? 'Required Step 34 files available' : 'One or more backend files are unavailable'} icon={MapPinned} tone={state.health?.status === 'ok' ? 'green' : 'amber'}/>
    </div>

    <div className="mt-5 grid gap-5 2xl:grid-cols-[1.6fr_.8fr]">
      <div className="card-pad">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3"><div><h2 className="section-title">Predicted land surface temperature</h2><p className="mt-1 text-[11px] text-slate-500">Click a real Kaduwela grid to load its detail record.</p></div><Badge variant="teal">{forecastYear}</Badge></div>
        <KaduwelaMap
          geojson={state.map}
          selectedGridId={selectedGridId}
          onSelect={(id) => setSelectedGridId(id)}
          fitKey={`dashboard-${forecastYear}`}
          colorForFeature={(feature) => thermalColor(feature?.properties?.predicted_lst_c, minValue, maxValue)}
          tooltipForFeature={(feature) => {
            const p = feature?.properties || {}
            return `<b>${p.area_name || 'Grid'}</b><br/>Grid: ${p.grid_id}<br/>Predicted LST: ${fmt(p.predicted_lst_c, 2)}°C<br/>Trajectory: ${p.trajectory_class || '—'}`
          }}
          legend={<ThermalLegend min={minValue} max={maxValue} title={`Predicted LST ${forecastYear} (°C)`}/>} 
        />
      </div>

      <div className="space-y-5">
        <div className="card-pad">
          <h2 className="section-title">Selected grid</h2>
          {!selectedGridId || !selectedForecast ? <p className="mt-4 text-sm text-slate-500">Click a map grid or choose a grid above.</p> : <div className="mt-4 space-y-3">
            <div className="rounded-2xl bg-slate-50 p-4"><div className="text-[10px] uppercase tracking-wide text-slate-400">Grid / area</div><div className="mt-1 font-bold text-slate-900">{selectedGridId}</div><div className="text-xs text-slate-500">{selectedForecast.area_name}</div></div>
            <div className="grid grid-cols-2 gap-3"><div className="soft-panel"><span className="text-[10px] text-slate-400">Predicted LST</span><b className="mt-1 block text-lg">{fmt(selectedForecast.predicted_lst_c, 2)}°C</b></div><div className="soft-panel"><span className="text-[10px] text-slate-400">Change from 2025</span><b className="mt-1 block text-lg">{fmtSigned(selectedForecast.change_from_2025_grid_actual_c, 2, '°C')}</b></div></div>
            <div className="flex flex-wrap gap-2"><Badge variant={trajectoryTone(selectedForecast.trajectory_class)}>{selectedForecast.trajectory_class || 'No trajectory'}</Badge><Badge variant={confidenceTone(selectedForecast.relative_confidence_label)}>{selectedForecast.relative_confidence_label || 'Confidence unavailable'}</Badge>{selectedPriority?.topsis_rank != null && <Badge variant="red">Priority rank #{fmtInt(selectedPriority.topsis_rank)}</Badge>}</div>
          </div>}
        </div>

        <div className="card-pad">
          <h2 className="section-title">Relative heat trajectories</h2>
          <div className="mt-4 h-[245px]"><ResponsiveContainer width="100%" height="100%"><BarChart data={trajectoryChart} layout="vertical" margin={{ left: 18, right: 10 }}><CartesianGrid strokeDasharray="4 5"/><XAxis type="number"/><YAxis type="category" dataKey="name" width={95} tick={{ fontSize: 9 }}/><Tooltip/><Bar dataKey="count" fill="#0f766e" radius={[0, 6, 6, 0]}/></BarChart></ResponsiveContainer></div>
        </div>
      </div>
    </div>

    <div className="mt-5 table-shell">
      <div className="border-b border-slate-100 px-5 py-4"><h2 className="section-title">Highest cooling-priority grids</h2><p className="mt-1 text-[11px] text-slate-500">CRITIC-weighted TOPSIS ranking returned by the backend.</p></div>
      <div className="overflow-x-auto"><table className="table-base"><thead><tr><th>Rank</th><th>Grid</th><th>Area</th><th>TOPSIS Score</th><th>Percentile</th><th>Trajectory</th></tr></thead><tbody>{state.priority.map((row) => <tr key={row.grid_id} className="cursor-pointer hover:bg-slate-50" onClick={() => setSelectedGridId(String(row.grid_id))}><td className="font-bold">#{fmtInt(row.topsis_rank)}</td><td>{row.grid_id}</td><td>{row.area_name}</td><td>{fmt(row.topsis_priority_score, 4)}</td><td>{fmt(row.priority_percentile, 1)}%</td><td><Badge variant={trajectoryTone(row.trajectory_class)}>{row.trajectory_class}</Badge></td></tr>)}</tbody></table></div>
    </div>
  </div>
}
