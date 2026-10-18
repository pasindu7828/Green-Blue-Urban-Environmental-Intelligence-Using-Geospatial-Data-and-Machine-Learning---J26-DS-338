
import { useEffect, useRef } from 'react'
import { GeoJSON, MapContainer, useMap } from 'react-leaflet'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

// Focus the map when a different wetland is selected.
// Do not reset zoom when changing dates for the same wetland.
function FocusWetland({ feature, wetlandId, comparisonFeatures }) {
  const map = useMap()
  const lastFocusedWetland = useRef(null)

  useEffect(() => {
    if (!feature?.geometry || !wetlandId) return

    if (lastFocusedWetland.current === wetlandId) return

    const geographicData =
      comparisonFeatures?.length === 2
        ? {
            type: 'FeatureCollection',
            features: comparisonFeatures,
          }
        : feature

    const bounds = L.geoJSON(geographicData).getBounds()

    if (bounds.isValid()) {
      map.invalidateSize()

      map.fitBounds(bounds, {
        padding: [40, 40],
        maxZoom: 17,
        animate: false,
      })

      lastFocusedWetland.current = wetlandId
    }
  }, [map, feature, wetlandId, comparisonFeatures])

  return null
}

export default function IndividualWetlandMap({
  feature,
  wetlandId,
  observationDate,
  comparisonFeatures,
}) {
  if (!feature?.geometry) {
    return (
      <div className="flex h-[400px] items-center justify-center rounded-xl bg-slate-100 text-sm text-slate-500">
        No polygon is available for this observation.
      </div>
    )
  }

  function onEachWetland(feature, layer) {
    const properties = feature.properties || {}

    const popup = document.createElement('div')

    const title = document.createElement('strong')
    title.textContent = `Wetland ${properties.wetland_id || 'Unknown'}`

    const date = document.createElement('p')
    date.textContent = `Observation date: ${
      properties.observation_date || 'N/A'
    }`

    const area = document.createElement('p')
    const areaValue = Number(properties.analysis_total_area_ha)

    area.textContent = `Mapped area: ${
      Number.isFinite(areaValue)
        ? `${areaValue.toFixed(4)} ha`
        : 'N/A'
    }`

    popup.append(title, date, area)
    layer.bindPopup(popup)
  }

  return (
    <div className="mt-5 overflow-hidden rounded-2xl border border-slate-200">
      <MapContainer
        center={[6.90, 79.97]}
        zoom={11}
        zoomSnap={0.1}
        scrollWheelZoom={false}
        style={{
          height: '400px',
          width: '100%',
          backgroundColor: '#EAF2EF',
        }}
      >
        <GeoJSON
          key={`${wetlandId}-${observationDate}`}
          data={feature}
          style={{
            color: '#0F766E',
            weight: 2,
            fillColor: '#2DD4BF',
            fillOpacity: 0.6,
          }}
          onEachFeature={onEachWetland}
        />

        <FocusWetland
          feature={feature}
          wetlandId={wetlandId}
          comparisonFeatures={comparisonFeatures}
        />
      </MapContainer>
    </div>
  )
}
