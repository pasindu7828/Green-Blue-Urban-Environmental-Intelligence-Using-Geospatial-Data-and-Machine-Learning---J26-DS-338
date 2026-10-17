import { Search } from 'lucide-react'
import { useMemo, useState } from 'react'
import { useUrbanHeat } from '../context/UrbanHeatContext'

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

  return <div className={`grid gap-3 md:grid-cols-2 ${showYear ? 'xl:grid-cols-4' : 'xl:grid-cols-3'} ${className}`}>
    <label className="text-[10px] font-bold uppercase tracking-wide text-slate-500">Area
      <select className="control mt-1.5" value={selectedArea} onChange={(e) => handleAreaChange(e.target.value)}>
        <option value="">All areas</option>
        {areas.map((area) => <option key={area} value={area}>{area}</option>)}
      </select>
    </label>
    <label className="text-[10px] font-bold uppercase tracking-wide text-slate-500">Find grid / area
      <div className="relative mt-1.5"><Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400"/><input className="control pl-9" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="e.g. 1832,3615 or Hokandara"/></div>
    </label>
    <label className="text-[10px] font-bold uppercase tracking-wide text-slate-500">Grid
      <select className="control mt-1.5" value={selectedGridId} onChange={(e) => setSelectedGridId(e.target.value)} disabled={bootstrapLoading}>
        {!selectedGridId && <option value="">Select a grid</option>}
        {options.map((item) => <option key={item.grid_id} value={item.grid_id}>{item.area_name} · {item.grid_id}</option>)}
      </select>
      {gridOptions.length > 500 && !query && !selectedArea && <span className="mt-1 block text-[9px] normal-case tracking-normal text-slate-400">Showing the first 500 grids. Search or select an area to narrow the list.</span>}
    </label>
    {showYear && <label className="text-[10px] font-bold uppercase tracking-wide text-slate-500">{yearType === 'historical' ? 'Historical year' : 'Forecast year'}
      <select className="control mt-1.5" value={year} onChange={(e) => setYear(Number(e.target.value))}>
        {years.map((item) => <option key={item} value={item}>{item}</option>)}
      </select>
    </label>}
  </div>
}
