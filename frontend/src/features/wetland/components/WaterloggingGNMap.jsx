
import { useEffect, useState } from 'react'
import {
  MapContainer,
  TileLayer,
  ImageOverlay,
  GeoJSON,
  useMap,
} from 'react-leaflet'
import L from 'leaflet'

const API_BASE = 'http://127.0.0.1:8000/api/wetlands'

function FocusGN({ feature }) {
  const map = useMap()

  useEffect(() => {
    if (!feature?.geometry) return

    const bounds = L.geoJSON(feature).getBounds()

    if (bounds.isValid()) {
      map.fitBounds(bounds, {
        padding: [25, 25],
        maxZoom: 14,
        animate: false,
      })
    }
  }, [map, feature])

  return null
}

export default function WaterloggingGNMap({ feature }) {
  const [rasterBounds, setRasterBounds] = useState(null)
  const [rasterError, setRasterError] = useState('')

  useEffect(() => {
    const controller = new AbortController()

    async function loadRasterBounds() {
      try {
        const response = await fetch(
          `${API_BASE}/part4/raster/main/metadata`,
          { signal: controller.signal }
        )

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`)
        }

        const data = await response.json()
        const b = data.bounds_wgs84

        if (!b) {
          throw new Error('Raster bounds unavailable')
        }

        setRasterBounds([
          [b.south, b.west],
          [b.north, b.east],
        ])
      } catch (err) {
        if (err.name !== 'AbortError') {
          setRasterError(err.message)
        }
      }
    }

    loadRasterBounds()

    return () => controller.abort()
  }, [])

  if (!feature?.geometry) {
    return (
      <p className="text-sm text-slate-500">
        GN boundary geometry is unavailable.
      </p>
    )
  }

  const properties = feature.properties || {}

  return (
    <div className="overflow-hidden rounded-xl border border-slate-200">
      <MapContainer
        center={[6.9, 79.97]}
        zoom={11}
        scrollWheelZoom={false}
        style={{ height: '300px', width: '100%' }}
      >
        <TileLayer
          attribution="&copy; OpenStreetMap contributors"
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        <FocusGN feature={feature} />

        {rasterBounds && (
          <ImageOverlay
            url={`${API_BASE}/part4/raster/main/png`}
            bounds={rasterBounds}
            opacity={0.85}
          />
        )}

        <GeoJSON
          key={properties.admin_code}
          data={feature}
          style={{
            color: '#0F766E',
            weight: 3,
            fillColor: '#14B8A6',
            fillOpacity: 0.06,
          }}
          onEachFeature={(_, layer) => {
            layer.bindPopup(
              `${properties.gn_name || 'GN Division'} (${properties.gn_number || 'N/A'})`
            )
          }}
        />
      </MapContainer>

      <div className="border-t border-slate-200 bg-white p-3">
        <p className="text-xs font-semibold text-slate-700">
          Main Waterlogging Susceptibility
        </p>

        <div className="mt-2 flex flex-wrap gap-3 text-xs text-slate-600">
          {[
            ['Low', '#2EA043'],
            ['Moderate', '#FFD23C'],
            ['High', '#F58223'],
            ['Very High', '#D72D2D'],
          ].map(([label, color]) => (
            <div key={label} className="flex items-center gap-1.5">
              <span
                className="h-3 w-3 rounded-sm"
                style={{ backgroundColor: color }}
              />
              <span>{label}</span>
            </div>
          ))}
        </div>

        <p className="mt-3 text-xs leading-5 text-slate-500">
          Teal outline: selected GN administrative boundary.
          Coloured raster: relative susceptibility, including
          areas outside the selected GN. The raster is not
          clipped to the GN boundary.
        </p>

        {rasterError && (
          <p className="mt-2 text-xs text-red-600">
            Susceptibility raster unavailable: {rasterError}
          </p>
        )}
      </div>
    </div>
  )
}
