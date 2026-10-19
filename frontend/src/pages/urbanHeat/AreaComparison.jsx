import { useCallback, useEffect, useMemo, useState } from 'react'
import { BarChart3, MapPinned, Target, Thermometer } from 'lucide-react'
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import PageHeader from '../../components/shared/PageHeader'
import MetricCard from '../../components/shared/MetricCard'
import { ErrorState, LoadingState } from '../../features/urbanHeat/components/AsyncState'
import { urbanHeatApi } from '../../features/urbanHeat/services/urbanHeatApi'
import { fmt, fmtInt } from '../../features/urbanHeat/utils/format'

export default function AreaComparison() {
  const [year, setYear] = useState(2030)
  const [heatRows, setHeatRows] = useState([])
  const [priorityRows, setPriorityRows] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const load = useCallback(async () => {
    const controller = new AbortController()
    setLoading(true); setError('')
    try {
      const [heat, priority] = await Promise.all([
        urbanHeatApi.areaHeat({ year }, controller.signal),
        urbanHeatApi.areaPriority(undefined, controller.signal),
      ])
      setHeatRows(heat?.data || [])
      setPriorityRows(priority?.data || [])
    } catch (e) { if (e?.name !== 'AbortError') setError(e.message) }
    finally { setLoading(false) }
    return () => controller.abort()
  }, [year])

  useEffect(() => { load() }, [load])

  const hottest = useMemo(() => [...heatRows].sort((a, b) => Number(b.mean_lst_c) - Number(a.mean_lst_c)).slice(0, 12), [heatRows])
  const priority = useMemo(() => [...priorityRows].sort((a, b) => Number(b.mean_priority_score) - Number(a.mean_priority_score)).slice(0, 12), [priorityRows])
  const mean = heatRows.length ? heatRows.reduce((s, r) => s + Number(r.mean_lst_c || 0), 0) / heatRows.length : null

  if (loading) return <LoadingState message={`Loading area summaries for ${year}…`}/>
  if (error) return <ErrorState error={error} onRetry={load}/>

  return <div>
    <PageHeader eyebrow="Area-Level Summary" title="Area Comparison" subtitle="Compare Kaduwela area-level heat summaries and descriptive cooling-priority aggregates without replacing the underlying grid-level decision evidence." badges={[`Year ${year}`, 'Area summaries', 'Grid ranking remains authoritative for priority']}/>
    <div className="card-pad mb-5"><label className="block max-w-xs text-[10px] font-bold uppercase tracking-wide text-slate-500">Comparison year<select className="control mt-1.5" value={year} onChange={(e) => setYear(Number(e.target.value))}>{Array.from({ length: 16 }, (_, i) => 2015 + i).map((y) => <option key={y} value={y}>{y} {y <= 2025 ? '· actual' : '· forecast'}</option>)}</select></label></div>

    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
      <MetricCard title="Areas Returned" value={fmtInt(heatRows.length)} subtitle="Area heat summaries" icon={MapPinned} tone="teal"/>
      <MetricCard title="Mean of Area Means" value={`${fmt(mean, 2)}°C`} subtitle={year <= 2025 ? 'Observed area summaries' : 'Forecast area summaries'} icon={Thermometer} tone="orange"/>
      <MetricCard title="Highest Area Mean" value={hottest[0] ? `${fmt(hottest[0].mean_lst_c, 2)}°C` : '—'} subtitle={hottest[0]?.area_name || '—'} icon={BarChart3} tone="red"/>
      <MetricCard title="Highest Mean Priority Score" value={priority[0] ? fmt(priority[0].mean_priority_score, 4) : '—'} subtitle={priority[0]?.area_name || '—'} icon={Target} tone="blue"/>
    </div>

    <div className="mt-5 grid gap-5 xl:grid-cols-2">
      <div className="card-pad"><h2 className="section-title">Highest area mean LST · {year}</h2><p className="mt-1 text-[11px] text-slate-500">Top 12 returned area summaries.</p><div className="mt-4 h-[390px]"><ResponsiveContainer width="100%" height="100%"><BarChart data={hottest} layout="vertical" margin={{ left: 42 }}><CartesianGrid strokeDasharray="4 5"/><XAxis type="number" unit="°C"/><YAxis type="category" dataKey="area_name" width={120} tick={{ fontSize: 9 }}/><Tooltip formatter={(v) => [`${fmt(v, 2)}°C`, 'Mean LST']}/><Bar dataKey="mean_lst_c" fill="#e0472b" radius={[0, 6, 6, 0]}/></BarChart></ResponsiveContainer></div></div>
      <div className="card-pad"><h2 className="section-title">Highest area mean priority score</h2><p className="mt-1 text-[11px] text-slate-500">Descriptive aggregation of grid-level TOPSIS scores.</p><div className="mt-4 h-[390px]"><ResponsiveContainer width="100%" height="100%"><BarChart data={priority} layout="vertical" margin={{ left: 42 }}><CartesianGrid strokeDasharray="4 5"/><XAxis type="number"/><YAxis type="category" dataKey="area_name" width={120} tick={{ fontSize: 9 }}/><Tooltip formatter={(v) => [fmt(v, 4), 'Mean priority score']}/><Bar dataKey="mean_priority_score" fill="#0f766e" radius={[0, 6, 6, 0]}/></BarChart></ResponsiveContainer></div></div>
    </div>

    <div className="mt-5 table-shell"><div className="border-b border-slate-100 px-5 py-4"><h2 className="section-title">Area heat summary · {year}</h2></div><div className="max-h-[500px] overflow-auto"><table className="table-base"><thead className="sticky top-0"><tr><th>Area</th><th>Grid Count</th><th>Mean LST</th><th>Median LST</th><th>Min</th><th>Max</th><th>Type</th></tr></thead><tbody>{[...heatRows].sort((a,b)=>Number(b.mean_lst_c)-Number(a.mean_lst_c)).map((row)=><tr key={row.area_name}><td className="font-semibold">{row.area_name}</td><td>{fmtInt(row.grid_count)}</td><td>{fmt(row.mean_lst_c,2)}°C</td><td>{fmt(row.median_lst_c,2)}°C</td><td>{fmt(row.min_lst_c,2)}°C</td><td>{fmt(row.max_lst_c,2)}°C</td><td>{row.value_type}</td></tr>)}</tbody></table></div></div>
  </div>
}
