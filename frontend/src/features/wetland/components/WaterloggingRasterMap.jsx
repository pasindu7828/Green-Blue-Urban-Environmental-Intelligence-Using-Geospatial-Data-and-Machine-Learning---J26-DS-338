
import { useEffect, useState } from 'react'
import {
  MapContainer,
  TileLayer,
  ImageOverlay,
  useMap,
} from 'react-leaflet'

const API_BASE = 'http://127.0.0.1:8000/api/wetlands'


const legendColors = {
  main: {
    1: '#2EA043', // Low - Green
    2: '#FFD23C', // Moderate - Yellow
    3: '#F58223', // High - Orange
    4: '#D72D2D', // Very High - Red
  },
  terrain: {
    1: '#2EA043',
    2: '#FFD23C',
    3: '#F58223',
    4: '#D72D2D',
  },
  span: {
    0: '#5AAF69', // Same class
    1: '#FFD746', // One-class difference
    2: '#F58223', // Two-class difference
    3: '#C82832', // Three-class difference
  },
}

function FitRasterBounds({ bounds }) {
  const map = useMap()

  useEffect(() => {
    if (bounds) {
      map.fitBounds(bounds, {
        padding: [20, 20],
        animate: false,
      })
    }
  }, [map, bounds])

  return null
}

export default function WaterloggingRasterMap({ mapType = 'main' }) {
  const [metadata, setMetadata] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    const controller = new AbortController()

    async function loadMetadata() {
      setLoading(true)
      setError('')
      setMetadata(null)

      try {
        const response = await fetch(
          `${API_BASE}/part4/raster/${mapType}/metadata`,
          { signal: controller.signal }
        )

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`)
        }

        const data = await response.json()

        if (!data?.bounds_wgs84 || !data?.legend) {
          throw new Error('Invalid raster metadata')
        }

        setMetadata(data)
      } catch (err) {
        if (err.name !== 'AbortError') {
          setError(err.message)
        }
      } finally {
        if (!controller.signal.aborted) {
          setLoading(false)
        }
      }
    }

    loadMetadata()

    return () => controller.abort()
  }, [mapType])

  if (loading) {
    return (
      <div className="rounded-xl bg-slate-50 p-8 text-sm text-slate-600">
        Loading waterlogging raster map...
      </div>
    )
  }

  if (error || !metadata) {
    return (
      <div className="rounded-xl border border-red-200 bg-red-50 p-5 text-sm text-red-700">
        Unable to load raster metadata: {error}
      </div>
    )
  }

  const { west, south, east, north } = metadata.bounds_wgs84

  const bounds = [
    [south, west],
    [north, east],
  ]

  const imageUrl =
    `${API_BASE}/part4/raster/${mapType}/png`

  return (
    <div className="overflow-hidden rounded-xl border border-slate-200">
      <MapContainer
        center={[6.90, 79.97]}
        zoom={11}
        scrollWheelZoom={true}
        style={{ height: '500px', width: '100%' }}
      >
        <TileLayer
          attribution="&copy; OpenStreetMap contributors"
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        <FitRasterBounds bounds={bounds} />

        <ImageOverlay
          key={mapType}
          url={imageUrl}
          bounds={bounds}
          opacity={0.85}
        />
      </MapContainer>

      <div className="border-t border-slate-200 bg-white p-4">
        <p className="text-sm font-semibold text-slate-800">
          {mapType === 'span'
            ? 'Scenario Classification Disagreement'
            : 'Relative Waterlogging Susceptibility'}
        </p>

        <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {Object.entries(metadata.legend).map(([value, label]) => (
            <div
            key={value}
            className="flex items-center gap-3 rounded-lg border border-slate-200 bg-slate-50 p-3"
            >
            <span
                className="h-5 w-5 shrink-0 rounded border border-slate-300"
                style={{
                backgroundColor:
                    legendColors[mapType]?.[Number(value)] ?? '#94A3B8',
                }}
            />

            <div>
                <p className="text-sm font-semibold text-slate-800">
                {label}
                </p>

                <p className="text-xs text-slate-500">
                {mapType === 'span'
                    ? `Disagreement class ${value}`
                    : `Susceptibility class ${value}`}
                </p>
            </div>
            </div>
        ))}
        </div>

        <p className="mt-3 text-xs leading-5 text-slate-500">
            Map coverage is limited to the 56 GN divisions included
            in this study. Transparent areas outside these boundaries
            are intentionally excluded from the susceptibility analysis.
            They do not indicate low susceptibility or missing
            predictions within the study area.
        </p>

        <p className="mt-3 text-xs leading-5 text-slate-500">
          Research raster overlay. Relative susceptibility
          and scenario disagreement are not verified flood
          events or independently validated forecasts.
        </p>
      </div>
    </div>
  )
}
