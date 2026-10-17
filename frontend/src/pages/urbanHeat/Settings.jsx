import { useEffect, useState } from 'react'
import { Map, RotateCcw, Settings as SettingsIcon } from 'lucide-react'
import PageHeader from '../../components/shared/PageHeader'
import Button from '../../components/ui/Button'
import { API_BASE_URL } from '../../features/urbanHeat/services/apiClient'
import { useUrbanHeat } from '../../features/urbanHeat/context/UrbanHeatContext'

export default function Settings() {
  const { forecastYear, setForecastYear, historicalYear, setHistoricalYear } = useUrbanHeat()
  const [showTiles, setShowTiles] = useState(() => localStorage.getItem('urbanHeat.showTiles') !== '0')
  const [tableSize, setTableSize] = useState(() => Number(localStorage.getItem('urbanHeat.tableSize')) || 25)

  useEffect(() => localStorage.setItem('urbanHeat.showTiles', showTiles ? '1' : '0'), [showTiles])
  useEffect(() => localStorage.setItem('urbanHeat.tableSize', String(tableSize)), [tableSize])

  const reset = () => {
    setForecastYear(2030); setHistoricalYear(2025); setShowTiles(true); setTableSize(25)
    localStorage.removeItem('urbanHeat.selectedGridId')
  }

  return <div>
    <PageHeader eyebrow="Frontend Preferences" title="Urban Heat Settings" subtitle="Local display preferences only. This page does not mutate the research pipeline, model or backend data." badges={['Frontend-only', 'Stored in this browser']}/>
    <div className="grid gap-5 xl:grid-cols-2">
      <div className="card-pad"><div className="flex items-center gap-2"><SettingsIcon size={18} className="text-teal-700"/><h2 className="section-title">Defaults</h2></div><div className="mt-5 grid gap-4 sm:grid-cols-2"><label className="text-xs font-semibold text-slate-600">Default forecast year<select className="control mt-2" value={forecastYear} onChange={(e) => setForecastYear(Number(e.target.value))}>{[2026,2027,2028,2029,2030].map((y)=><option key={y}>{y}</option>)}</select></label><label className="text-xs font-semibold text-slate-600">Default historical year<select className="control mt-2" value={historicalYear} onChange={(e) => setHistoricalYear(Number(e.target.value))}>{Array.from({length:11},(_,i)=>2015+i).map((y)=><option key={y}>{y}</option>)}</select></label><label className="text-xs font-semibold text-slate-600">Table page size<select className="control mt-2" value={tableSize} onChange={(e) => setTableSize(Number(e.target.value))}>{[10,25,50,100].map((n)=><option key={n}>{n}</option>)}</select></label></div><Button variant="secondary" className="mt-5" onClick={reset}><RotateCcw size={15}/>Reset local preferences</Button></div>
      <div className="card-pad"><div className="flex items-center gap-2"><Map size={18} className="text-teal-700"/><h2 className="section-title">Map & API</h2></div><label className="mt-5 flex items-center justify-between rounded-2xl border border-slate-200 p-4"><div><div className="text-sm font-semibold text-slate-800">Base map tiles</div><div className="mt-1 text-[11px] text-slate-500">Preference is stored locally. The analytical grid layer remains the primary map evidence.</div></div><input type="checkbox" checked={showTiles} onChange={(e)=>setShowTiles(e.target.checked)} className="h-5 w-5 accent-teal-700"/></label><div className="mt-4 rounded-2xl bg-slate-50 p-4"><div className="text-[10px] uppercase tracking-wide text-slate-400">Configured API base URL</div><code className="mt-2 block break-all text-xs text-slate-700">{API_BASE_URL}</code></div></div>
    </div>
  </div>
}
