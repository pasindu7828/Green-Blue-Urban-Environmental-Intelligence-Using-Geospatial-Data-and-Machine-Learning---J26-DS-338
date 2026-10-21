import { useCallback, useEffect, useMemo, useState } from 'react'
import { Grid3X3, MapPinned, Thermometer } from 'lucide-react'
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import PageHeader from '../../components/shared/PageHeader'
import MetricCard from '../../components/shared/MetricCard'
import Badge from '../../components/ui/Badge'
import GridPicker from '../../features/urbanHeat/components/GridPicker'
import KaduwelaMap from '../../features/urbanHeat/components/KaduwelaMap'
import { ErrorState, LoadingState } from '../../features/urbanHeat/components/AsyncState'
import { ThermalLegend, thermalColor } from '../../features/urbanHeat/components/MapLegends'
import { useUrbanHeat } from '../../features/urbanHeat/context/UrbanHeatContext'
import { urbanHeatApi } from '../../features/urbanHeat/services/urbanHeatApi'
import { fmt, fmtInt } from '../../features/urbanHeat/utils/format'

export default function HeatMap() {
  const { historicalYear, selectedGridId, setSelectedGridId } = useUrbanHeat()
  const [map, setMap] = useState(null)
  const [history, setHistory] = useState([])
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)

  const loadMap = useCallback(async () => {
    const controller = new AbortController()
    setLoading(true); setError('')
    try { setMap(await urbanHeatApi.historicalMap(historicalYear, controller.signal)) }
    catch (e) { if (e?.name !== 'AbortError') setError(e.message) }
    finally { setLoading(false) }
    return () => controller.abort()
  }, [historicalYear])

  useEffect(() => { loadMap() }, [loadMap])

  useEffect(() => {
    if (!selectedGridId) { setHistory([]); return }
    const controller = new AbortController()
    urbanHeatApi.gridHistory(selectedGridId, controller.signal)
      .then((response) => setHistory(response?.data || []))
      .catch(() => setHistory([]))
    return () => controller.abort()
  }, [selectedGridId])

  const values = useMemo(() => (map?.features || []).map((f) => Number(f?.properties?.actual_lst_c)).filter(Number.isFinite), [map])
  const summary = useMemo(() => ({
    count: map?.features?.length || 0,
    min: values.length ? Math.min(...values) : null,
    max: values.length ? Math.max(...values) : null,
    mean: values.length ? values.reduce((a, b) => a + b, 0) / values.length : null,
  }), [map, values])

  if (loading) return <LoadingState message={`Loading historical LST map for ${historicalYear}…`}/>
  if (error) return <ErrorState error={error} onRetry={loadMap}/>

  return <div>
    <PageHeader eyebrow="Observed Heat Evidence" title="Historical Heat Map" subtitle="Actual annual land surface temperature (LST) from the validated dashboard dataset, mapped on real WGS84 Kaduwela grid polygons." badges={['Historical range 2015–2025', `Selected ${historicalYear}`, 'Click any grid']}/>
    <div className="card-pad mb-5"><GridPicker showYear yearType="historical"/></div>

    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
      <MetricCard title="Mapped Grids" value={fmtInt(summary.count)} subtitle={`Actual LST records in ${historicalYear}`} icon={Grid3X3} tone="teal"/>
      <MetricCard title="Mean LST" value={`${fmt(summary.mean, 2)}°C`} subtitle="Study-area grid mean" icon={Thermometer} tone="orange"/>
      <MetricCard title="Minimum LST" value={`${fmt(summary.min, 2)}°C`} subtitle="Coolest returned grid" icon={Thermometer} tone="blue"/>
      <MetricCard title="Maximum LST" value={`${fmt(summary.max, 2)}°C`} subtitle="Hottest returned grid" icon={MapPinned} tone="red"/>
    </div>

    <div className="mt-5 grid gap-5 2xl:grid-cols-[1.45fr_.8fr]">
      <div className="card-pad">
        <div className="mb-4 flex items-center justify-between"><div><h2 className="section-title">Actual LST · {historicalYear}</h2><p className="mt-1 text-[11px] text-slate-500">Real grid geometry; thermal color scale adapts to the selected year.</p></div><Badge variant="blue">Observed</Badge></div>
        <KaduwelaMap
          geojson={map}
          selectedGridId={selectedGridId}
          onSelect={(id) => setSelectedGridId(id)}
          fitKey={`historical-${historicalYear}`}
          colorForFeature={(feature) => thermalColor(feature?.properties?.actual_lst_c, summary.min, summary.max)}
          tooltipForFeature={(feature) => { const p = feature?.properties || {}; return `<b>${p.area_name || 'Grid'}</b><br/>Grid: ${p.grid_id}<br/>Actual LST (${historicalYear}): ${fmt(p.actual_lst_c, 2)}°C` }}
          legend={<ThermalLegend min={summary.min} max={summary.max} title={`Actual LST ${historicalYear} (°C)`}/>} 
        />
      </div>

      <div className="card-pad">
        <h2 className="section-title">Selected grid history</h2>
        <p className="mt-1 text-[11px] text-slate-500">2015–2025 observed LST for the currently selected grid.</p>
        {selectedGridId ? <><div className="mt-4 rounded-2xl bg-slate-50 p-4"><span className="text-[10px] uppercase text-slate-400">Grid ID</span><div className="mt-1 font-bold">{selectedGridId}</div><div className="text-xs text-slate-500">{history[0]?.area_name || 'Area loading / unavailable'}</div></div>
          <div className="mt-4 h-[300px]"><ResponsiveContainer width="100%" height="100%"><LineChart data={history}><CartesianGrid strokeDasharray="4 5"/><XAxis dataKey="year"/><YAxis domain={['auto', 'auto']} unit="°C"/><Tooltip formatter={(v) => [`${fmt(v, 2)}°C`, 'Actual LST']}/><Line type="monotone" dataKey="actual_lst_c" stroke="#0f766e" strokeWidth={2.5} dot={{ r: 3 }}/></LineChart></ResponsiveContainer></div>
        </> : <p className="mt-5 text-sm text-slate-500">Select a grid from the map or controls to view its history.</p>}
      </div>
    </div>
  </div>
}
