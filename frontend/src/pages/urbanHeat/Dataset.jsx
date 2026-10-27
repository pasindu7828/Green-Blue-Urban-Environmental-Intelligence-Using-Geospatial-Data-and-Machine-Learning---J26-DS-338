import { useCallback, useEffect, useState } from 'react'
import { CheckCircle2, Database, FileCheck2, Grid3X3, XCircle } from 'lucide-react'
import PageHeader from '../../components/shared/PageHeader'
import MetricCard from '../../components/shared/MetricCard'
import Badge from '../../components/ui/Badge'
import { ErrorState, LoadingState } from '../../features/urbanHeat/components/AsyncState'
import { urbanHeatApi } from '../../features/urbanHeat/services/urbanHeatApi'
import { fmtInt } from '../../features/urbanHeat/utils/format'

export default function Dataset() {
  const [health, setHealth] = useState(null)
  const [manifest, setManifest] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const load = useCallback(async () => {
    const controller = new AbortController()
    setLoading(true); setError('')
    try {
      const [h, m] = await Promise.all([urbanHeatApi.dataHealth(controller.signal), urbanHeatApi.manifest(controller.signal)])
      setHealth(h); setManifest(m)
    } catch (e) { if (e?.name !== 'AbortError') setError(e.message) }
    finally { setLoading(false) }
    return () => controller.abort()
  }, [])
  useEffect(() => { load() }, [load])

  if (loading) return <LoadingState message="Checking dashboard data contract…"/>
  if (error) return <ErrorState error={error} onRetry={load}/>

  const fileEntries = Object.entries(health?.files || {})
  const available = fileEntries.filter(([, info]) => info?.exists).length

  return <div>
    <PageHeader eyebrow="Data Contract" title="Urban Heat Dataset Status" subtitle="Backend-visible Step 34 dashboard files, supported year ranges and dataset manifest. Raw research data are not copied into the browser." badges={['Backend health check', 'Manifest-driven', 'Read-only research outputs']}/>
    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
      <MetricCard title="Data Status" value={health?.status === 'ok' ? 'Ready' : 'Check'} subtitle="Backend dashboard-file availability" icon={Database} tone={health?.status === 'ok' ? 'green' : 'amber'}/>
      <MetricCard title="Expected Grids" value={fmtInt(health?.expected_grid_count || manifest?.grid_count)} subtitle="Stable grid count" icon={Grid3X3} tone="teal"/>
      <MetricCard title="Historical Range" value={(health?.historical_year_range || manifest?.historical_years)?.length ? `${(health?.historical_year_range || manifest?.historical_years)[0]}–${(health?.historical_year_range || manifest?.historical_years).at(-1)}` : '—'} subtitle="Observed dashboard coverage" icon={FileCheck2} tone="blue"/>
      <MetricCard title="Forecast Range" value={(health?.forecast_year_range || manifest?.future_years)?.length ? `${(health?.forecast_year_range || manifest?.future_years)[0]}–${(health?.forecast_year_range || manifest?.future_years).at(-1)}` : '—'} subtitle="Future dashboard coverage" icon={FileCheck2} tone="orange"/>
    </div>

    <div className="mt-5 grid gap-5 xl:grid-cols-[.9fr_1.1fr]">
      <div className="card-pad"><h2 className="section-title">Backend file availability</h2><p className="mt-1 text-[11px] text-slate-500">{available} of {fileEntries.length} required/important resources reported as available.</p><div className="mt-4 space-y-2">{fileEntries.map(([key, info]) => <div key={key} className="flex items-center justify-between rounded-xl border border-slate-100 bg-slate-50 px-3 py-2.5"><div className="flex items-center gap-2 text-xs text-slate-700">{info?.exists ? <CheckCircle2 size={15} className="text-emerald-600"/> : <XCircle size={15} className="text-red-500"/>}<span>{key.replace(/_/g, ' ')}</span></div><Badge variant={info?.exists ? 'green' : 'red'}>{info?.exists ? 'Available' : 'Missing'}</Badge></div>)}</div></div>

      <div className="card-pad"><h2 className="section-title">Dashboard manifest</h2><div className="mt-4 grid gap-3 sm:grid-cols-2 text-xs"><div className="soft-panel"><span className="text-slate-400">Export step</span><b className="mt-1 block">{manifest?.step || '—'}</b></div><div className="soft-panel"><span className="text-slate-400">Grid count</span><b className="mt-1 block">{fmtInt(manifest?.grid_count)}</b></div><div className="soft-panel"><span className="text-slate-400">Priority classes frozen?</span><b className="mt-1 block">{String(manifest?.priority_classes_frozen ?? '—')}</b></div></div>
        <div className="mt-5 overflow-x-auto rounded-2xl border border-slate-200"><table className="table-base"><thead><tr><th>Role</th><th>Rows</th><th>Notes</th></tr></thead><tbody>{(manifest?.files || []).map((file) => <tr key={file.file}><td className="font-semibold">{file.role}</td><td>{file.rows != null ? fmtInt(file.rows) : '—'}</td><td className="min-w-[260px] text-slate-500">{file.notes}</td></tr>)}</tbody></table></div>
      </div>
    </div>
  </div>
}
