import { Search } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { useUrbanHeat } from '../context/UrbanHeatContext'
import { urbanHeatApi } from '../services/urbanHeatApi'
import { fmt, fmtInt } from '../utils/format'

export default function GridPicker({ showYear = false, yearType = 'forecast', className = '' }) {
  const {
    selectedGridId, setSelectedGridId,
    selectedArea, setSelectedArea,
    areas, gridOptions,
    forecastYear, setForecastYear,
    historicalYear, setHistoricalYear,
    bootstrapLoading,
  } = useUrbanHeat()
  const [query, setQuery] = useState('')
  const [gnSummary, setGnSummary] = useState(null)
  const [gnSummaryError, setGnSummaryError] = useState('')


  useEffect(() => {
    if (!selectedArea) {
      setGnSummary(null)
      setGnSummaryError('')
      return
    }
    const controller = new AbortController()
    const summaryYear = yearType === 'historical' ? historicalYear : forecastYear
    setGnSummary(null)
    setGnSummaryError('')
    urbanHeatApi.gnSummary(selectedArea, summaryYear, controller.signal)
      .then(setGnSummary)
      .catch((error) => {
        if (error?.name !== 'AbortError') setGnSummaryError(error.message || 'GN summary unavailable')
      })
    return () => controller.abort()
  }, [selectedArea, yearType, historicalYear, forecastYear])

  const options = useMemo(() => {
    const q = query.trim().toLowerCase()
    const filtered = gridOptions.filter((item) => {
      if (selectedArea && item.area_name !== selectedArea) return false
      if (!q) return true
      return item.grid_id.toLowerCase().includes(q) || item.area_name.toLowerCase().includes(q)
    })
    const limited = filtered.slice(0, 500)
    const selected = gridOptions.find((item) => item.grid_id === selectedGridId)
    if (selected && !limited.some((item) => item.grid_id === selectedGridId) && (!selectedArea || selected.area_name === selectedArea)) limited.unshift(selected)
    return limited
  }, [gridOptions, selectedArea, query, selectedGridId])

  const handleAreaChange = (value) => {
    setSelectedArea(value)
    const first = gridOptions.find((item) => !value || item.area_name === value)
    if (first) setSelectedGridId(first.grid_id)
  }

  const year = yearType === 'historical' ? historicalYear : forecastYear
  const setYear = yearType === 'historical' ? setHistoricalYear : setForecastYear
  const years = yearType === 'historical'
    ? Array.from({ length: 11 }, (_, i) => 2015 + i)
    : Array.from({ length: 5 }, (_, i) => 2026 + i)

  return <>
  <div className={`grid gap-3 md:grid-cols-2 ${showYear ? 'xl:grid-cols-4' : 'xl:grid-cols-3'} ${className}`}>
    <label className="text-[10px] font-bold uppercase tracking-wide text-slate-500">GN Division
      <select className="control mt-1.5" value={selectedArea} onChange={(e) => handleAreaChange(e.target.value)}>
        <option value="">All GN divisions</option>
        {areas.map((area) => <option key={area} value={area}>{area}</option>)}
      </select>
    </label>
    <label className="text-[10px] font-bold uppercase tracking-wide text-slate-500">Find grid / GN division
      <div className="relative mt-1.5"><Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400"/><input className="control pl-9" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="e.g. 1832,3615 or Malabe East"/></div>
    </label>
    <label className="text-[10px] font-bold uppercase tracking-wide text-slate-500">Grid
      <select className="control mt-1.5" value={selectedGridId} onChange={(e) => setSelectedGridId(e.target.value)} disabled={bootstrapLoading}>
        {!selectedGridId && <option value="">Select a grid</option>}
        {options.map((item) => <option key={item.grid_id} value={item.grid_id}>{item.area_name} · {item.grid_id}</option>)}
      </select>
      {gridOptions.length > 500 && !query && !selectedArea && <span className="mt-1 block text-[9px] normal-case tracking-normal text-slate-400">Showing the first 500 grids. Search or select a GN Division to narrow the list.</span>}
    </label>
    {showYear && <label className="text-[10px] font-bold uppercase tracking-wide text-slate-500">{yearType === 'historical' ? 'Historical year' : 'Forecast year'}
      <select className="control mt-1.5" value={year} onChange={(e) => setYear(Number(e.target.value))}>
        {years.map((item) => <option key={item} value={item}>{item}</option>)}
      </select>
    </label>}
  </div>
  {selectedArea && <div className="mt-3 rounded-2xl border border-teal-100 bg-teal-50/55 p-3">
    <div className="flex flex-wrap items-center justify-between gap-2">
      <div><span className="text-[9px] font-bold uppercase tracking-wide text-teal-700">Selected GN Division</span><div className="text-sm font-bold text-slate-900">{selectedArea}</div></div>
      {gnSummary && <span className="rounded-full bg-white px-2.5 py-1 text-[9px] font-semibold text-slate-500">{gnSummary.value_type === 'ACTUAL' ? 'Observed' : 'Forecast'} · {gnSummary.year}</span>}
    </div>
    {gnSummaryError ? <p className="mt-2 text-[10px] text-rose-600">{gnSummaryError}</p> : gnSummary ? <div className="mt-3 grid gap-2 sm:grid-cols-2 xl:grid-cols-6">
      <div className="rounded-xl bg-white p-2.5"><span className="text-[9px] text-slate-400">Heat grids</span><b className="mt-0.5 block text-xs">{fmtInt(gnSummary.grid_count)}</b></div>
      <div className="rounded-xl bg-white p-2.5"><span className="text-[9px] text-slate-400">Mean LST</span><b className="mt-0.5 block text-xs">{fmt(gnSummary.mean_lst_c, 2)}°C</b></div>
      <div className="rounded-xl bg-white p-2.5"><span className="text-[9px] text-slate-400">Max LST</span><b className="mt-0.5 block text-xs">{fmt(gnSummary.max_lst_c, 2)}°C</b></div>
      <div className="rounded-xl bg-white p-2.5"><span className="text-[9px] text-slate-400">Top 10% priority grids</span><b className="mt-0.5 block text-xs">{fmtInt(gnSummary.top10pct_priority_grid_count)}</b></div>
      <div className="rounded-xl bg-white p-2.5"><span className="text-[9px] text-slate-400">Dominant trajectory</span><b className="mt-0.5 block text-xs">{gnSummary.dominant_trajectory || '—'}</b></div>
      <div className="rounded-xl bg-white p-2.5"><span className="text-[9px] text-slate-400">Relative confidence</span><b className="mt-0.5 block text-xs">{gnSummary.dominant_relative_confidence || 'Not applicable'}</b></div>
    </div> : <p className="mt-2 text-[10px] text-slate-500">Loading GN Division summary…</p>}
  </div>}
  </>
}
