import { createContext, useContext, useEffect, useMemo, useState } from 'react'
import { urbanHeatApi } from '../services/urbanHeatApi'

const UrbanHeatContext = createContext(null)

export function UrbanHeatProvider({ children }) {
  const [selectedGridId, setSelectedGridId] = useState(() => localStorage.getItem('urbanHeat.selectedGridId') || '')
  const [forecastYear, setForecastYear] = useState(() => Number(localStorage.getItem('urbanHeat.forecastYear')) || 2030)
  const [historicalYear, setHistoricalYear] = useState(() => Number(localStorage.getItem('urbanHeat.historicalYear')) || 2025)
  const [selectedArea, setSelectedArea] = useState('')
  const [areas, setAreas] = useState([])
  const [gridOptions, setGridOptions] = useState([])
  const [bootstrapLoading, setBootstrapLoading] = useState(true)
  const [bootstrapError, setBootstrapError] = useState('')

  useEffect(() => {
    const controller = new AbortController()
    async function load() {
      setBootstrapLoading(true)
      setBootstrapError('')
      try {
        const [areaResponse, gridGeojson] = await Promise.all([
          urbanHeatApi.areas(controller.signal),
          urbanHeatApi.gridMap(controller.signal),
        ])
        setAreas(areaResponse?.areas || [])
        const options = (gridGeojson?.features || [])
          .map((feature) => feature?.properties || {})
          .filter((item) => item.grid_id)
          .map((item) => ({ grid_id: String(item.grid_id), area_name: item.area_name || '' }))
          .sort((a, b) => (a.area_name || '').localeCompare(b.area_name || '') || a.grid_id.localeCompare(b.grid_id))
        setGridOptions(options)
      } catch (error) {
        if (error?.name !== 'AbortError') setBootstrapError(error.message || 'Failed to load Urban Heat lookup data.')
      } finally {
        setBootstrapLoading(false)
      }
    }
    load()
    return () => controller.abort()
  }, [])

  useEffect(() => {
    if (selectedGridId) localStorage.setItem('urbanHeat.selectedGridId', selectedGridId)
  }, [selectedGridId])
  useEffect(() => localStorage.setItem('urbanHeat.forecastYear', String(forecastYear)), [forecastYear])
  useEffect(() => localStorage.setItem('urbanHeat.historicalYear', String(historicalYear)), [historicalYear])

  const filteredGridOptions = useMemo(() => {
    if (!selectedArea) return gridOptions
    return gridOptions.filter((item) => item.area_name === selectedArea)
  }, [gridOptions, selectedArea])

  const value = useMemo(() => ({
    selectedGridId, setSelectedGridId,
    forecastYear, setForecastYear,
    historicalYear, setHistoricalYear,
    selectedArea, setSelectedArea,
    areas, gridOptions, filteredGridOptions,
    bootstrapLoading, bootstrapError,
  }), [selectedGridId, forecastYear, historicalYear, selectedArea, areas, gridOptions, filteredGridOptions, bootstrapLoading, bootstrapError])

  return <UrbanHeatContext.Provider value={value}>{children}</UrbanHeatContext.Provider>
}

export function useUrbanHeat() {
  const value = useContext(UrbanHeatContext)
  if (!value) throw new Error('useUrbanHeat must be used within UrbanHeatProvider')
  return value
}
