
import { useEffect } from 'react'
import {
  GeoJSON,
  MapContainer,
  LayersControl,
  useMap,
} from 'react-leaflet'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

function FitMapBounds({ gnData }) {
  const map = useMap()

  useEffect(() => {
    if (!gnData?.features?.length) return

    const bounds = L.geoJSON(gnData).getBounds()

    if (bounds.isValid()) {
      map.invalidateSize()

      map.fitBounds(bounds, {
        padding: [30, 30],
        animate: false,
      })
    }
  }, [map, gnData])

  return null
}

export default function HistoricalWetlandMap({ data, gnData, yearKey }) {
  if (!data?.features?.length) {
    return (
      <div className="flex h-[430px] items-center justify-center rounded-2xl bg-slate-100 text-sm text-slate-500">
        No wetland map data available.
      </div>
    )
  }

  // GN division popup
  function onEachGn(feature, layer) {
    const properties = feature.properties || {}

    const popup = document.createElement('div')

    const title = document.createElement('strong')
    title.textContent = properties.gn_name || 'Unknown GN Division'

    const description = document.createElement('p')
    description.textContent = 'GN administrative division'

    popup.append(title, description)
    layer.bindPopup(popup)
  }

  // Monitored wetland popup and hover highlighting
  function onEachWetland(feature, layer) {
    const properties = feature.properties || {}

    const popup = document.createElement('div')

    const title = document.createElement('strong')
    title.textContent = `Wetland ${properties.wetland_id || 'Unknown'}`

    const observation = document.createElement('p')
    observation.textContent = `Latest mapped observation: ${
      properties.observation_date || 'Not available'
    }`

    popup.append(title, observation)
    layer.bindPopup(popup)

    layer.on({
      mouseover: (event) => {
        event.target.setStyle({
          weight: 3,
          fillOpacity: 0.85,
        })
      },
      mouseout: (event) => {
        event.target.setStyle({
          weight: 2,
          fillOpacity: 0.55,
        })
      },
    })
  }

  return (
    <div className="overflow-hidden rounded-2xl border border-slate-200">
      <MapContainer
        center={[6.90, 79.97]}
        zoom={11}
        zoomSnap={0.1}
        scrollWheelZoom={false}
        style={{
          height: '430px',
          width: '100%',
          backgroundColor: '#EAF2EF',
        }}
      >

    {/* Interactive Map Layer Controls */}
    <LayersControl position="topright">

    {/* GN Division Boundaries */}
    {gnData?.features?.length > 0 && (
        <LayersControl.Overlay
        checked
        name="GN Division Boundaries"
        >
        <GeoJSON
            data={gnData}
            style={{
            color: '#64748B',
            weight: 1,
            opacity: 0.75,
            fill: true,
            fillColor: '#EAF2EF',
            fillOpacity: 0.03,
            }}
            onEachFeature={onEachGn}
        />
        </LayersControl.Overlay>
    )}

    {/* Monitored Wetland Polygons */}
    <LayersControl.Overlay
        checked
        name="Monitored Wetlands"
    >
        <GeoJSON
          key={yearKey}
          data={data}
          style={{
              color: '#0F766E',
              weight: 2,
              fillColor: '#2DD4BF',
              fillOpacity: 0.55,
          }}
          onEachFeature={onEachWetland}
        />
    </LayersControl.Overlay>

</LayersControl>

      <FitMapBounds gnData={gnData} />
      </MapContainer>
    </div>
  )
}
