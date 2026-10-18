
import { useEffect, useMemo } from 'react'
import {
  MapContainer,
  GeoJSON,
  LayersControl,
  useMap,
} from 'react-leaflet'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

// Fit map to the Kaduwela study area
function FitFutureMapBounds({ gnData, futureMapData }) {
  const map = useMap()

  useEffect(() => {
    const geographicData = gnData?.features?.length
      ? gnData
      : futureMapData

    if (!geographicData?.features?.length) return

    const bounds = L.geoJSON(geographicData).getBounds()

    if (bounds.isValid()) {
      map.invalidateSize()

      map.fitBounds(bounds, {
        padding: [30, 30],
        animate: false,
      })
    }
  }, [map, gnData, futureMapData])

  return null
}


function FocusFutureGn({ gnData, selectedGnId }) {
  const map = useMap()

  useEffect(() => {
    if (selectedGnId == null || !gnData?.features) {
      return
    }

    const selectedGn = gnData.features.find(
      (feature) =>
        String(
          feature.properties?.gn_source_id ??
          feature.properties?.gn_zone
        ) === String(selectedGnId)
    )

    if (!selectedGn?.geometry) return

    const bounds = L.geoJSON(selectedGn).getBounds()

    if (bounds.isValid()) {
      map.invalidateSize()

      map.fitBounds(bounds, {
        padding: [40, 40],
        maxZoom: 15,
        animate: true,
      })
    }
  }, [map, gnData, selectedGnId])

  return null
}

export default function FutureWetlandMap({
  futureMapData,
  gnData,
  selectedGnId,
  showAllPatches = false,
}) {


// Display all GN-specific patches when requested.
// Otherwise retain the five-pixel display rule.
const displayedPatches = useMemo(() => ({
  type: 'FeatureCollection',
  features: (futureMapData?.features || []).filter(
    (feature) =>
      showAllPatches ||
      Number(feature.properties?.display_ge5) === 1
  ),
}), [futureMapData, showAllPatches])


  if (!futureMapData?.features?.length) {
    return (
      <div className="flex h-[430px] items-center justify-center rounded-xl bg-slate-100 text-sm text-slate-500">
        No 2030 wetland scenario polygons available.
      </div>
    )
  }

  // Find the GN division selected from the scenario table
  const selectedGnFeature =
    selectedGnId == null
      ? null
      : gnData?.features?.find(
          (feature) =>
            String(
              feature.properties?.gn_source_id ??
              feature.properties?.gn_zone
            ) === String(selectedGnId)
        )

  // GN division popup
  function onEachGn(feature, layer) {
    const gn = feature.properties || {}

    const popup = document.createElement('div')

    const title = document.createElement('strong')
    title.textContent = gn.gn_name || 'Unknown GN Division'

    const note = document.createElement('p')
    note.textContent = 'Kaduwela GN division boundary'

    popup.append(title, note)
    layer.bindPopup(popup)
  }

  // Projected wetland patch popup
  function onEachFuturePatch(feature, layer) {
    const patch = feature.properties || {}

    const popup = document.createElement('div')

    const title = document.createElement('strong')
    title.textContent =
      `2030 Scenario Patch: ${patch.patch_id || 'Unknown'}`

    const area = document.createElement('p')
    const areaValue = Number(patch.area_ha)

    area.textContent =
      `Scenario patch area: ${
        patch.area_ha == null || !Number.isFinite(areaValue)
          ? 'N/A'
          : areaValue.toFixed(2) + ' ha'
      }`

    const pixels = document.createElement('p')
    pixels.textContent =
      `Pixel count: ${patch.pixel_count ?? 'N/A'}`

    const note = document.createElement('p')
    note.textContent =
      'Exploratory model scenario — not a verified future wetland.'

    popup.append(title, area, pixels, note)

    layer.bindPopup(popup)
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
                  fillColor: '#EAF2EF',
                  fillOpacity: 0.03,
                }}
                onEachFeature={onEachGn}
              />
            </LayersControl.Overlay>
          )}

          {/* 2030 Model-Generated Wetland Patches */}
          <LayersControl.Overlay
            checked
            name="2030 Scenario Patches (5+ Pixels)"
          >
            <GeoJSON
              data={displayedPatches}
              style={{
                color: '#0F766E',
                weight: 1.5,
                fillColor: '#2DD4BF',
                fillOpacity: 0.65,
              }}
              onEachFeature={onEachFuturePatch}
            />
          </LayersControl.Overlay>

        </LayersControl>

        {/* Highlight Selected GN Division */}
        {selectedGnFeature && (
          <GeoJSON
            key={`selected-future-gn-${selectedGnId}`}
            data={selectedGnFeature}
            interactive={false}
            style={{
              color: '#D97706',
              weight: 4,
              opacity: 1,
              fillColor: '#FBBF24',
              fillOpacity: 0.12,
            }}
          />
        )}

        <FitFutureMapBounds
          gnData={gnData}
          futureMapData={futureMapData}
        />

        <FocusFutureGn
          gnData={gnData}
          selectedGnId={selectedGnId}
        />
      </MapContainer>

    </div>
  )
}
