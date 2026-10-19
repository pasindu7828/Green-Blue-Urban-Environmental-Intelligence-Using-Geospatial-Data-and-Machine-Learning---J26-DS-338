
import { useEffect } from 'react'
import {
  MapContainer,
  TileLayer,
  GeoJSON,
  useMap,
} from 'react-leaflet'
import L from 'leaflet'

// Use the historical wetland only to locate the map.
// Do not display it as a 2030 predicted boundary.
function FocusLocalArea({ historical }) {
  const map = useMap()

  useEffect(() => {
    if (!historical?.features?.length) return

    const layer = L.geoJSON(historical)
    const bounds = layer.getBounds()

    if (bounds.isValid()) {
      map.fitBounds(bounds.pad(1.5), {
        maxZoom: 16,
        animate: false,
      })
    }
  }, [map, historical])

  return null
}

export default function IndividualFutureScenarioMap({
  comparisonData,
}) {
  const historical = comparisonData?.historical
  const scenario = comparisonData?.scenario_2030

  const patches = scenario?.features || []
  const wetlandId = comparisonData?.wetland_id || 'Unknown'
  const bufferDistance =
    comparisonData?.metadata?.buffer_distance_m ?? 50

  return (
    <div className="overflow-hidden rounded-xl border border-slate-200">

      <MapContainer
        center={[6.9, 79.95]}
        zoom={12}
        scrollWheelZoom={false}
        style={{ height: '430px', width: '100%' }}
      >
        <TileLayer
          attribution='&copy; OpenStreetMap contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        <FocusLocalArea historical={historical} />

        {/* Show only locally clipped 2030 scenario polygons */}
        {patches.length > 0 && (
          <GeoJSON
            key={`future-${wetlandId}`}
            data={scenario}
            style={() => ({
              color: '#7C3AED',
              weight: 2,
              fillColor: '#A78BFA',
              fillOpacity: 0.5,
            })}
            onEachFeature={(feature, layer) => {
              const props = feature.properties || {}

              layer.bindPopup(`
                <strong>Exploratory 2030 Scenario</strong>
                <div>Patch ID: ${String(props.patch_id ?? 'N/A')}</div>
                <div>Display: Clipped local portion</div>
                <div>Not a validated wetland prediction</div>
              `)
            }}
          />
        )}
      </MapContainer>

      {/* Legend */}
      <div className="flex items-center gap-2 border-t border-slate-200 bg-white px-4 py-3">
        <span className="h-3 w-3 rounded-sm bg-violet-500" />

        <span className="text-xs font-medium text-slate-700">
          Local 2030 Model Scenario
        </span>
      </div>

      {/* Scenario Information */}
      <div className="border-t border-slate-200 bg-slate-50 px-4 py-4">

        {patches.length > 0 ? (
          <p className="text-xs leading-5 text-slate-600">
            Displaying {patches.length} model-generated 2030
            scenario patch portion(s) within a provisional{' '}
            {bufferDistance} m surrounding area of {wetlandId}.
            Only local clipped portions are displayed.
            These are exploratory scenario results, not
            predicted future boundaries of this wetland.
          </p>
        ) : (
          <p className="text-xs leading-5 text-amber-700">
            No display-eligible 2030 scenario patches were
            matched within the provisional {bufferDistance} m
            area of {wetlandId}. This does not mean the
            historical wetland will disappear.
          </p>
        )}

      </div>

    </div>
  )
}
