import { useCallback, useEffect, useMemo, useState } from 'react'
import { CalendarRange, MapPin, ShieldCheck, Thermometer, TrendingUp } from 'lucide-react'
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import PageHeader from '../../components/shared/PageHeader'
import MetricCard from '../../components/shared/MetricCard'
import Badge from '../../components/ui/Badge'
import GridPicker from '../../features/urbanHeat/components/GridPicker'
import KaduwelaMap from '../../features/urbanHeat/components/KaduwelaMap'
import { ErrorState, LoadingState } from '../../features/urbanHeat/components/AsyncState'
import { ThermalLegend, thermalColor } from '../../features/urbanHeat/components/MapLegends'
import { useUrbanHeat } from '../../features/urbanHeat/context/UrbanHeatContext'
import { urbanHeatApi } from '../../features/urbanHeat/services/urbanHeatApi'
import { confidenceTone, fmt, fmtSigned, trajectoryTone } from '../../features/urbanHeat/utils/format'

export default function Forecast() {
  const { forecastYear, selectedGridId, setSelectedGridId } = useUrbanHeat()
  const [map, setMap] = useState(null)
  const [detail, setDetail] = useState(null)
  const [uncertainty, setUncertainty] = useState(null)
  const [history, setHistory] = useState([])
  const [future, setFuture] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const loadMap = useCallback(async () => {
    const controller = new AbortController()
    setLoading(true); setError('')
    try { setMap(await urbanHeatApi.forecastMap(forecastYear, controller.signal)) }
    catch (e) { if (e?.name !== 'AbortError') setError(e.message) }
    finally { setLoading(false) }
    return () => controller.abort()
  }, [forecastYear])

  useEffect(() => { loadMap() }, [loadMap])

  useEffect(() => {
    if (!selectedGridId) { setDetail(null); setUncertainty(null); setHistory([]); setFuture([]); return }
    const controller = new AbortController()
    Promise.allSettled([
      urbanHeatApi.gridDetails(selectedGridId, forecastYear, controller.signal),
      urbanHeatApi.uncertainty(selectedGridId, forecastYear, controller.signal),
      urbanHeatApi.gridHistory(selectedGridId, controller.signal),
      urbanHeatApi.gridForecastAll(selectedGridId, controller.signal),
    ]).then(([d, u, h, f]) => {
      setDetail(d.status === 'fulfilled' ? d.value : null)
      setUncertainty(u.status === 'fulfilled' ? u.value : null)
      setHistory(h.status === 'fulfilled' ? h.value?.data || [] : [])
      setFuture(f.status === 'fulfilled' ? f.value?.data || [] : [])
    })
    return () => controller.abort()
  }, [selectedGridId, forecastYear])

  const values = useMemo(() => (map?.features || []).map((f) => Number(f?.properties?.predicted_lst_c)).filter(Number.isFinite), [map])
  const minValue = values.length ? Math.min(...values) : null
  const maxValue = values.length ? Math.max(...values) : null
  const meanValue = values.length ? values.reduce((a, b) => a + b, 0) / values.length : null
  const chart = useMemo(() => {
    const hist = history.map((row) => ({ year: Number(row.year), actual: Number(row.actual_lst_c), forecast: null, lower: null, upper: null }))
    const fc = future.map((row) => ({ year: Number(row.year), actual: null, forecast: Number(row.predicted_lst_c), lower: Number(row.forecast_pi95_lower_c), upper: Number(row.forecast_pi95_upper_c) }))
    if (hist.length && fc.length) {
      const last = hist[hist.length - 1]
      fc.unshift({ year: last.year, actual: last.actual, forecast: last.actual, lower: null, upper: null })
    }
    return [...hist, ...fc]
  }, [history, future])

  if (loading) return <LoadingState message={`Loading ${forecastYear} forecast map…`}/>
  if (error) return <ErrorState error={error} onRetry={loadMap}/>

  const forecast = detail?.forecast
  const priority = detail?.priority

  return <div>
    <PageHeader eyebrow="Future Heat Intelligence" title="Future Heat Forecast" subtitle="Grid-level land surface temperature forecasts for 2026–2030, with 2025 comparison, uncertainty and cooling-priority context." badges={['Forecast range 2026–2030', 'Frozen production model', 'Uncertainty visible']}/>
    <div className="card-pad mb-5"><GridPicker showYear/></div>

    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
      <MetricCard title={`Mean Forecast ${forecastYear}`} value={`${fmt(meanValue, 2)}°C`} subtitle="Across returned Kaduwela grids" icon={Thermometer} tone="orange"/>
      <MetricCard title="Selected Grid" value={selectedGridId || '—'} subtitle={forecast?.area_name || 'Choose a grid'} icon={MapPin} tone="teal"/>
      <MetricCard title={`Selected Prediction ${forecastYear}`} value={forecast ? `${fmt(forecast.predicted_lst_c, 2)}°C` : '—'} subtitle={forecast ? `2025 actual: ${fmt(forecast.actual_2025_lst_c, 2)}°C` : 'Select a grid'} icon={TrendingUp} tone="red"/>
      <MetricCard title="95% Reference Band" value={uncertainty ? `${fmt(uncertainty.forecast_pi95_width_c, 2)}°C` : '—'} subtitle={uncertainty ? `${fmt(uncertainty.forecast_pi95_lower_c, 2)} to ${fmt(uncertainty.forecast_pi95_upper_c, 2)}°C` : 'Select a grid'} icon={ShieldCheck} tone="blue"/>
    </div>

    <div className="mt-5 grid gap-5 2xl:grid-cols-[1.5fr_.85fr]">
      <div className="card-pad">
        <div className="mb-4 flex items-center justify-between"><div><h2 className="section-title">Predicted LST map · {forecastYear}</h2><p className="mt-1 text-[11px] text-slate-500">Click any real grid polygon for forecast detail.</p></div><Badge variant="orange">Model forecast</Badge></div>
        <KaduwelaMap
          geojson={map}
          selectedGridId={selectedGridId}
          onSelect={(id) => setSelectedGridId(id)}
          fitKey={`forecast-${forecastYear}`}
          colorForFeature={(feature) => thermalColor(feature?.properties?.predicted_lst_c, minValue, maxValue)}
          tooltipForFeature={(feature) => { const p = feature?.properties || {}; return `<b>${p.area_name || 'Grid'}</b><br/>Grid: ${p.grid_id}<br/>Predicted LST: ${fmt(p.predicted_lst_c, 2)}°C<br/>Change from 2025: ${fmtSigned(p.change_from_2025_grid_actual_c, 2, '°C')}<br/>Confidence: ${p.relative_confidence_label || '—'}` }}
          legend={<ThermalLegend min={minValue} max={maxValue} title={`Predicted LST ${forecastYear} (°C)`}/>} 
        />
      </div>

      <div className="card-pad">
        <h2 className="section-title">Selected grid forecast</h2>
        {!forecast ? <p className="mt-4 text-sm text-slate-500">Select a grid to load forecast, uncertainty and priority evidence.</p> : <div className="mt-4 space-y-4">
          <div className="rounded-2xl bg-slate-50 p-4"><div className="text-[10px] uppercase tracking-wide text-slate-400">{forecast.area_name}</div><div className="mt-1 text-lg font-bold">{selectedGridId}</div></div>
          <div className="grid grid-cols-2 gap-3 text-xs">
            <div className="soft-panel"><span className="text-slate-400">Predicted LST</span><b className="mt-1 block text-lg">{fmt(forecast.predicted_lst_c, 2)}°C</b></div>
            <div className="soft-panel"><span className="text-slate-400">2025 → {forecastYear}</span><b className="mt-1 block text-lg">{fmtSigned(forecast.change_from_2025_grid_actual_c, 2, '°C')}</b></div>
            <div className="soft-panel"><span className="text-slate-400">PI95 lower</span><b className="mt-1 block">{fmt(forecast.forecast_pi95_lower_c, 2)}°C</b></div>
            <div className="soft-panel"><span className="text-slate-400">PI95 upper</span><b className="mt-1 block">{fmt(forecast.forecast_pi95_upper_c, 2)}°C</b></div>
          </div>
          <div className="flex flex-wrap gap-2"><Badge variant={trajectoryTone(forecast.trajectory_class)}>{forecast.trajectory_class || 'Trajectory unavailable'}</Badge><Badge variant={confidenceTone(forecast.relative_confidence_label)}>{forecast.relative_confidence_label || 'Confidence unavailable'}</Badge>{priority?.topsis_rank != null && <Badge variant="red">Cooling rank #{Math.round(Number(priority.topsis_rank))}</Badge>}</div>
        </div>}
      </div>
    </div>

    <div className="mt-5 card-pad">
      <div className="flex flex-wrap items-center justify-between gap-3"><div><h2 className="section-title">Observed history + future forecast</h2><p className="mt-1 text-[11px] text-slate-500">Selected grid · observed 2015–2025 · model forecast 2026–2030.</p></div><div className="flex items-center gap-2 text-[10px] text-slate-500"><CalendarRange size={14}/> Forecast horizon limited to validated dashboard coverage.</div></div>
      {selectedGridId ? <div className="mt-4 h-[340px]"><ResponsiveContainer width="100%" height="100%"><LineChart data={chart}><CartesianGrid strokeDasharray="4 5"/><XAxis dataKey="year"/><YAxis domain={['auto', 'auto']} unit="°C"/><Tooltip formatter={(v, name) => [Number.isFinite(Number(v)) ? `${fmt(v, 2)}°C` : '—', name]}/><Legend/><Line type="monotone" dataKey="actual" name="Observed LST" stroke="#0f766e" strokeWidth={2.5} connectNulls={false}/><Line type="monotone" dataKey="forecast" name="Predicted LST" stroke="#e0472b" strokeWidth={2.5} strokeDasharray="6 4" connectNulls={false}/></LineChart></ResponsiveContainer></div> : <p className="mt-5 text-sm text-slate-500">Select a grid to view its time series.</p>}
    </div>
  </div>
}
