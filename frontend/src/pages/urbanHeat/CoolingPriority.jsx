import { useCallback, useEffect, useMemo, useState } from 'react'
import { Award, MapPin, ShieldCheck, Target } from 'lucide-react'
import PageHeader from '../../components/shared/PageHeader'
import MetricCard from '../../components/shared/MetricCard'
import Badge from '../../components/ui/Badge'
import GridPicker from '../../features/urbanHeat/components/GridPicker'
import KaduwelaMap from '../../features/urbanHeat/components/KaduwelaMap'
import { ErrorState, LoadingState } from '../../features/urbanHeat/components/AsyncState'
import { PriorityLegend, priorityColor } from '../../features/urbanHeat/components/MapLegends'
import { useUrbanHeat } from '../../features/urbanHeat/context/UrbanHeatContext'
import { urbanHeatApi } from '../../features/urbanHeat/services/urbanHeatApi'
import { fmt, fmtInt, trajectoryTone } from '../../features/urbanHeat/utils/format'

export default function CoolingPriority() {
  const { selectedGridId, setSelectedGridId, selectedArea } = useUrbanHeat()
  const [map, setMap] = useState(null)
  const [rows, setRows] = useState([])
  const [detail, setDetail] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const load = useCallback(async () => {
    const controller = new AbortController()
    setLoading(true); setError('')
    try {
      const [mapData, ranking] = await Promise.all([
        urbanHeatApi.priorityMap(controller.signal),
        urbanHeatApi.priority({ limit: 100, areaName: selectedArea || undefined }, controller.signal),
      ])
      setMap(mapData); setRows(ranking?.data || [])
    } catch (e) { if (e?.name !== 'AbortError') setError(e.message) }
    finally { setLoading(false) }
    return () => controller.abort()
  }, [selectedArea])

  useEffect(() => { load() }, [load])
  useEffect(() => {
    if (!selectedGridId) { setDetail(null); return }
    const controller = new AbortController()
    urbanHeatApi.gridPriority(selectedGridId, controller.signal).then(setDetail).catch(() => setDetail(null))
    return () => controller.abort()
  }, [selectedGridId])

  const mapForArea = useMemo(() => {
    if (!selectedArea || !map) return map
    return { ...map, features: (map.features || []).filter((f) => f?.properties?.area_name === selectedArea) }
  }, [map, selectedArea])

  if (loading) return <LoadingState message="Loading CRITIC–TOPSIS cooling priority…"/>
  if (error) return <ErrorState error={error} onRetry={load}/>

  return <div>
    <PageHeader eyebrow="Decision Support" title="Cooling Priority" subtitle="Grid-level cooling-priority ranking produced using objective CRITIC weighting and TOPSIS, with sensitivity/stability evidence from the finalized research workflow." badges={['CRITIC objective weighting', 'TOPSIS ranking', 'No invented priority classes']}/>
    <div className="card-pad mb-5"><GridPicker/></div>

    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
      <MetricCard title="Ranked Grids Shown" value={fmtInt(rows.length)} subtitle={selectedArea ? `Top records in ${selectedArea}` : 'Top 100 study-area records'} icon={Target} tone="teal"/>
      <MetricCard title="Selected Rank" value={detail?.topsis_rank != null ? `#${fmtInt(detail.topsis_rank)}` : '—'} subtitle={detail?.area_name || 'Select a grid'} icon={Award} tone="red"/>
      <MetricCard title="Priority Score" value={detail ? fmt(detail.topsis_priority_score, 4) : '—'} subtitle="TOPSIS closeness score" icon={ShieldCheck} tone="orange"/>
      <MetricCard title="Percentile" value={detail ? `${fmt(detail.priority_percentile, 1)}%` : '—'} subtitle="Higher percentile = higher relative priority" icon={MapPin} tone="blue"/>
    </div>

    <div className="mt-5 grid gap-5 2xl:grid-cols-[1.4fr_.85fr]">
      <div className="card-pad">
        <h2 className="section-title">Cooling-priority map</h2><p className="mt-1 mb-4 text-[11px] text-slate-500">Click a grid to inspect its rank, score and sensitivity stability.</p>
        <KaduwelaMap
          geojson={mapForArea}
          selectedGridId={selectedGridId}
          onSelect={(id) => setSelectedGridId(id)}
          fitKey={`priority-${selectedArea}`}
          colorForFeature={(feature) => priorityColor(feature?.properties?.priority_percentile)}
          tooltipForFeature={(feature) => { const p = feature?.properties || {}; return `<b>${p.area_name || 'Grid'}</b><br/>Grid: ${p.grid_id}<br/>Rank: #${fmtInt(p.topsis_rank)}<br/>Score: ${fmt(p.topsis_priority_score, 4)}<br/>Percentile: ${fmt(p.priority_percentile, 1)}%` }}
          legend={<PriorityLegend/>}
        />
      </div>
      <div className="card-pad">
        <h2 className="section-title">Selected grid priority</h2>
        {!detail ? <p className="mt-4 text-sm text-slate-500">Select a grid from the map or controls.</p> : <div className="mt-4 space-y-4">
          <div className="rounded-2xl bg-slate-50 p-4"><div className="text-[10px] uppercase text-slate-400">{detail.area_name}</div><div className="mt-1 font-bold">{detail.grid_id}</div></div>
          <div className="grid grid-cols-2 gap-3 text-xs">
            <div className="soft-panel"><span className="text-slate-400">TOPSIS rank</span><b className="mt-1 block text-lg">#{fmtInt(detail.topsis_rank)}</b></div>
            <div className="soft-panel"><span className="text-slate-400">Score</span><b className="mt-1 block text-lg">{fmt(detail.topsis_priority_score, 4)}</b></div>
            <div className="soft-panel"><span className="text-slate-400">Mean sensitivity rank</span><b className="mt-1 block">{fmt(detail.mean_sensitivity_rank, 1)}</b></div>
            <div className="soft-panel"><span className="text-slate-400">Worst sensitivity rank</span><b className="mt-1 block">{fmtInt(detail.worst_sensitivity_rank)}</b></div>
            <div className="soft-panel"><span className="text-slate-400">Mean abs. rank shift</span><b className="mt-1 block">{fmt(detail.mean_abs_rank_shift, 2)}</b></div>
            <div className="soft-panel"><span className="text-slate-400">Scenarios in top 100</span><b className="mt-1 block">{fmt(detail.pct_scenarios_in_top100, 1)}%</b></div>
          </div>
          <Badge variant={trajectoryTone(detail.trajectory_class)}>{detail.trajectory_class || 'Trajectory unavailable'}</Badge>
        </div>}
      </div>
    </div>

    <div className="mt-5 table-shell"><div className="border-b border-slate-100 px-5 py-4"><h2 className="section-title">Priority ranking table</h2><p className="mt-1 text-[11px] text-slate-500">Top 100 results, or up to 100 within the selected area.</p></div><div className="max-h-[520px] overflow-auto"><table className="table-base"><thead className="sticky top-0"><tr><th>Rank</th><th>Grid</th><th>Area</th><th>Score</th><th>Percentile</th><th>Trajectory</th><th>Mean Rank Shift</th></tr></thead><tbody>{rows.map((row) => <tr key={row.grid_id} onClick={() => setSelectedGridId(String(row.grid_id))} className="cursor-pointer hover:bg-slate-50"><td className="font-bold">#{fmtInt(row.topsis_rank)}</td><td>{row.grid_id}</td><td>{row.area_name}</td><td>{fmt(row.topsis_priority_score, 4)}</td><td>{fmt(row.priority_percentile, 1)}%</td><td><Badge variant={trajectoryTone(row.trajectory_class)}>{row.trajectory_class}</Badge></td><td>{fmt(row.mean_abs_rank_shift, 2)}</td></tr>)}</tbody></table></div></div>
  </div>
}
