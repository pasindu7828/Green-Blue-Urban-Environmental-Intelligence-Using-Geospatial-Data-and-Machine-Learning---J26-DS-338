import { useCallback, useEffect, useMemo, useState } from 'react'
import { Activity, ArrowDownRight, ArrowUpRight, Grid3X3 } from 'lucide-react'
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import PageHeader from '../../components/shared/PageHeader'
import MetricCard from '../../components/shared/MetricCard'
import Badge from '../../components/ui/Badge'
import GridPicker from '../../features/urbanHeat/components/GridPicker'
import KaduwelaMap from '../../features/urbanHeat/components/KaduwelaMap'
import { ErrorState, LoadingState } from '../../features/urbanHeat/components/AsyncState'
import { TRAJECTORY_COLORS, TrajectoryLegend } from '../../features/urbanHeat/components/MapLegends'
import { useUrbanHeat } from '../../features/urbanHeat/context/UrbanHeatContext'
import { urbanHeatApi } from '../../features/urbanHeat/services/urbanHeatApi'
import { fmt, fmtInt, trajectoryTone } from '../../features/urbanHeat/utils/format'

const ORDER = ['Persistent Hot', 'Intensifying', 'Emerging', 'Diminishing', 'Persistent Cool']

const relativePercentile = (value) => {
  const n = Number(value)
  return Number.isFinite(n) ? `${(n * 100).toFixed(1)} percentile rank` : '—'
}

const relativeSlope = (value) => {
  const n = Number(value)
  return Number.isFinite(n) ? `${(n * 100).toFixed(2)} percentile points/year` : '—'
}

export default function TrendAnalysis() {
  const { selectedGridId, setSelectedGridId } = useUrbanHeat()
  const [map, setMap] = useState(null)
  const [detail, setDetail] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)

  const load = useCallback(async () => {
    const controller = new AbortController()
    setLoading(true); setError('')
    try { setMap(await urbanHeatApi.trajectoryMap(controller.signal)) }
    catch (e) { if (e?.name !== 'AbortError') setError(e.message) }
    finally { setLoading(false) }
    return () => controller.abort()
  }, [])

  useEffect(() => { load() }, [load])
  useEffect(() => {
    if (!selectedGridId) { setDetail(null); return }
    const controller = new AbortController()
    urbanHeatApi.gridTrajectory(selectedGridId, controller.signal).then(setDetail).catch(() => setDetail(null))
    return () => controller.abort()
  }, [selectedGridId])

  const counts = useMemo(() => {
    const result = Object.fromEntries(ORDER.map((x) => [x, 0]))
    for (const feature of map?.features || []) {
      const value = feature?.properties?.trajectory_class
      result[value] = (result[value] || 0) + 1
    }
    return result
  }, [map])
  const chart = ORDER.map((name) => ({ name, count: counts[name] || 0 }))

  if (loading) return <LoadingState message="Loading relative heat trajectory map…"/>
  if (error) return <ErrorState error={error} onRetry={load}/>

  return <div>
    <PageHeader eyebrow="Spatial-Temporal Intelligence" title="Trend & Heat Trajectory Analysis" subtitle="Heat trajectory tracks each grid's yearly heat rank within Kaduwela: lower percentiles are relatively cooler and higher percentiles are relatively hotter." badges={['Relative spatial trajectory', 'Mann–Kendall / Sen slope evidence']}/>
    <div className="card-pad mb-5"><GridPicker/></div>

    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-5">
      {ORDER.map((name, index) => <MetricCard key={name} title={name} value={fmtInt(counts[name])} subtitle="Grid count" icon={index < 3 ? ArrowUpRight : ArrowDownRight} tone={name === 'Persistent Hot' ? 'red' : name === 'Intensifying' ? 'orange' : name === 'Emerging' ? 'amber' : name === 'Diminishing' ? 'green' : 'blue'}/>) }
    </div>

    <div className="mt-5 grid gap-5 2xl:grid-cols-[1.45fr_.85fr]">
      <div className="card-pad">
        <div className="mb-4"><h2 className="section-title">Kaduwela relative heat trajectories</h2><p className="mt-1 text-[11px] text-slate-500">Click any grid to inspect its classification evidence. Past = 2015–2019 median rank, Recent = 2021–2025 median rank. Relative slope is in percentile points/year; absolute slope is in °C/year.</p></div>
        <KaduwelaMap
          geojson={map}
          selectedGridId={selectedGridId}
          onSelect={(id) => setSelectedGridId(id)}
          fitKey="trajectory"
          colorForFeature={(feature) => TRAJECTORY_COLORS[feature?.properties?.trajectory_class] || '#94a3b8'}
          tooltipForFeature={(feature) => {
            const p = feature?.properties || {}
            return `<b>${p.area_name || 'Grid'}</b><br/>
              Grid: ${p.grid_id}<br/>
              Trajectory: <b>${p.trajectory_class || '—'}</b><br/>
              Past heat position: ${relativePercentile(p.baseline_relative_heat_median)}<br/>
              Recent heat position: ${relativePercentile(p.recent_relative_heat_median)}<br/>
              Relative Sen slope: ${relativeSlope(p.relative_sen_slope_percentile_per_year)}<br/>
              Absolute LST Sen slope: ${fmt(p.sen_slope_c_per_year, 3)} °C/year`
          }}
          legend={<TrajectoryLegend/>}
        />
      </div>

      <div className="space-y-5">
        <div className="card-pad">
          <h2 className="section-title">Class distribution</h2>
          <div className="mt-4 h-[255px]"><ResponsiveContainer width="100%" height="100%"><BarChart data={chart} layout="vertical" margin={{ left: 18 }}><CartesianGrid strokeDasharray="4 5"/><XAxis type="number"/><YAxis type="category" dataKey="name" width={95} tick={{ fontSize: 9 }}/><Tooltip/><Bar dataKey="count" fill="#0f766e" radius={[0, 6, 6, 0]}/></BarChart></ResponsiveContainer></div>
        </div>

        <div className="card-pad">
          <div className="flex items-center justify-between"><h2 className="section-title">Selected grid evidence</h2>{detail?.trajectory_class && <Badge variant={trajectoryTone(detail.trajectory_class)}>{detail.trajectory_class}</Badge>}</div>
          {!detail ? <p className="mt-4 text-sm text-slate-500">Select a grid to inspect its trajectory record.</p> : <div className="mt-4 grid grid-cols-2 gap-3 text-xs">
            <div className="soft-panel"><span className="text-slate-400">Grid</span><b className="mt-1 block">{detail.grid_id}</b></div>
            <div className="soft-panel"><span className="text-slate-400">Area</span><b className="mt-1 block">{detail.area_name}</b></div>
            <div className="soft-panel"><span className="text-slate-400">Past relative heat position</span><b className="mt-1 block">{relativePercentile(detail.baseline_relative_heat_median)}</b></div>
            <div className="soft-panel"><span className="text-slate-400">Recent relative heat position</span><b className="mt-1 block">{relativePercentile(detail.recent_relative_heat_median)}</b></div>
            <div className="soft-panel"><span className="text-slate-400">Long-term relative heat position</span><b className="mt-1 block">{relativePercentile(detail.long_term_relative_heat_median)}</b></div>
            <div className="soft-panel"><span className="text-slate-400">Relative Sen slope</span><b className="mt-1 block">{relativeSlope(detail.relative_sen_slope_percentile_per_year)}</b></div>
            <div className="soft-panel"><span className="text-slate-400">Absolute LST Sen slope</span><b className="mt-1 block">{fmt(detail.sen_slope_c_per_year, 4)} °C/year</b></div>
            <div className="soft-panel"><span className="text-slate-400">Relative direction</span><b className="mt-1 block">{detail.relative_direction_operational || '—'}</b></div>
            <div className="soft-panel"><span className="text-slate-400">MK p-value</span><b className="mt-1 block">{fmt(detail.mk_p_value, 4)}</b></div>
            <div className="soft-panel"><span className="text-slate-400">FDR q-value</span><b className="mt-1 block">{fmt(detail.mk_fdr_q_value, 4)}</b></div>
          </div>}
        </div>
      </div>
    </div>
  </div>
}
