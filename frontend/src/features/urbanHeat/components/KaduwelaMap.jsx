import { useEffect, useMemo, useState } from 'react'
import L from 'leaflet'
import { GeoJSON, MapContainer, TileLayer, useMap } from 'react-leaflet'

const DEFAULT_CENTER = [6.902, 79.97]

function FitBounds({ geojson }) {
  const map = useMap()
  useEffect(() => {
    if (!geojson?.features?.length) return
    try {
      const bounds = L.geoJSON(geojson).getBounds()
      if (bounds.isValid()) map.fitBounds(bounds, { padding: [18, 18], maxZoom: 15 })
    } catch { /* keep default view */ }
  }, [geojson, map])
  return null
}

function SelectedPan({ geojson, selectedGridId }) {
  const map = useMap()
  useEffect(() => {
    if (!selectedGridId || !geojson?.features?.length) return
    const feature = geojson.features.find((f) => String(f?.properties?.grid_id) === String(selectedGridId))
    if (!feature) return
    try {
      const bounds = L.geoJSON(feature).getBounds()
      if (bounds.isValid() && !map.getBounds().contains(bounds)) map.panTo(bounds.getCenter())
    } catch { /* no-op */ }
  }, [geojson, selectedGridId, map])
  return null
}

export default function KaduwelaMap({
  geojson,
  selectedGridId,
  onSelect,
  colorForFeature,
  tooltipForFeature,
  height = 560,
  legend,
  showTiles,
  fitKey = '',
}) {
  const [boundary, setBoundary] = useState(null)
  const tilesEnabled = showTiles ?? (localStorage.getItem('urbanHeat.showTiles') !== '0')

  useEffect(() => {
    const controller = new AbortController()
    const boundaryUrl = `${import.meta.env.BASE_URL || '/'}data/kaduwela-boundary.geojson`
    fetch(boundaryUrl, { signal: controller.signal })
      .then((r) => r.ok ? r.json() : null)
      .then(setBoundary)
      .catch(() => {})
    return () => controller.abort()
  }, [])

  const dataKey = useMemo(() => `${fitKey}:${selectedGridId || ''}:${geojson?.features?.length || 0}`, [fitKey, selectedGridId, geojson])
  const style = (feature) => {
    const id = String(feature?.properties?.grid_id || '')
    const selected = id === String(selectedGridId || '')
    return {
      color: selected ? '#0f172a' : '#ffffff',
      weight: selected ? 2.4 : 0.45,
      opacity: selected ? 1 : 0.72,
      fillOpacity: selected ? 0.92 : 0.78,
      fillColor: colorForFeature?.(feature) || '#5aa7a0',
    }
  }

  const onEachFeature = (feature, layer) => {
    const props = feature?.properties || {}
    const tooltip = tooltipForFeature?.(feature)
    if (tooltip) layer.bindTooltip(tooltip, { sticky: true, direction: 'top', opacity: 0.96 })
    layer.on({
      click: () => onSelect?.(String(props.grid_id), feature),
      mouseover: (event) => event.target.setStyle({ weight: 1.8, color: '#0f172a', opacity: 1 }),
      mouseout: (event) => {
        const id = String(props.grid_id || '')
        event.target.setStyle(style(feature))
        if (id === String(selectedGridId || '')) event.target.bringToFront()
      },
    })
  }

  return <div className="relative overflow-hidden rounded-2xl border border-slate-200 bg-[#e9f0ef]" style={{ height }}>
    <MapContainer center={DEFAULT_CENTER} zoom={12} preferCanvas className="h-full w-full" zoomControl>
      {tilesEnabled && <TileLayer
        attribution="&copy; OpenStreetMap contributors"
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />}
      {geojson?.features?.length > 0 && <GeoJSON key={dataKey} data={geojson} style={style} onEachFeature={onEachFeature}/>} 
      {boundary?.features?.length > 0 && <GeoJSON
        key={`boundary-${boundary.features.length}`}
        data={boundary}
        style={{ color: '#0F2E28', weight: 3, opacity: 0.95, fillOpacity: 0, interactive: false }}
      />}
      <FitBounds geojson={boundary?.features?.length ? boundary : geojson}/>
      <SelectedPan geojson={geojson} selectedGridId={selectedGridId}/>
    </MapContainer>
    {boundary?.features?.length > 0 && <div className="pointer-events-none absolute right-3 top-3 z-[500] rounded-full border border-slate-200 bg-white/95 px-3 py-1.5 text-[9px] font-semibold text-slate-600 shadow-sm backdrop-blur">{boundary.features.some((feature) => feature?.properties?.authoritative_source) ? 'Official KMC boundary' : 'Real-grid study-area outline'}</div>}
    {legend && <div className="pointer-events-none absolute bottom-4 left-4 z-[500] max-w-[260px] rounded-2xl border border-slate-200 bg-white/95 p-3 shadow-lg backdrop-blur">{legend}</div>}
  </div>
}
