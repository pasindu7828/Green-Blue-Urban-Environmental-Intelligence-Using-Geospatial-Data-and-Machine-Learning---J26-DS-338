
import { useEffect } from 'react'
import {
  MapContainer,
  GeoJSON,
  LayersControl,
  useMap,
} from 'react-leaflet'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

// Fit the map to the Kaduwela GN boundaries
function FitCandidateMapBounds({ gnData, candidateData }) {
  const map = useMap()

  useEffect(() => {
    const geographicData =
      gnData?.features?.length
        ? gnData
        : candidateData

    if (!geographicData?.features?.length) return

    const bounds = L.geoJSON(geographicData).getBounds()

    if (bounds.isValid()) {
      map.invalidateSize()

      map.fitBounds(bounds, {
        padding: [30, 30],
        animate: false,
      })
    }
  }, [map, gnData, candidateData])

  return null
}

function FocusSelectedCandidate({
  candidateData,
  selectedCandidateId,
}) {
  const map = useMap()

  useEffect(() => {
    if (!selectedCandidateId || !candidateData?.features) {
      return
    }

    const selectedFeature = candidateData.features.find(
      (feature) =>
        feature.properties?.candidate_id === selectedCandidateId
    )

    if (!selectedFeature?.geometry) return

    const bounds = L.geoJSON(selectedFeature).getBounds()

    if (bounds.isValid()) {
      map.invalidateSize()

      map.fitBounds(bounds, {
        padding: [50, 50],
        maxZoom: 16,
        animate: true,
      })
    }
  }, [map, candidateData, selectedCandidateId])

  return null
}


function FocusSelectedGn({
  gnData,
  selectedGnId,
}) {
  const map = useMap()

  useEffect(() => {
    if (selectedGnId == null || !gnData?.features) {
      return
    }

    const selectedGn = gnData.features.find(
      (feature) =>
        String(feature.properties?.gn_id) === String(selectedGnId)
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

export default function CandidateWetlandMap({
  candidateData,
  gnData,
  selectedCandidateId,
  selectedGnId,
}) {
  if (!candidateData?.features?.length) {
    return (
      <div className="flex h-[430px] items-center justify-center rounded-xl bg-slate-100 text-sm text-slate-500">
        No candidate wetland polygons available.
      </div>
    )
  }

  // GN division popup
  function onEachGn(feature, layer) {
    const properties = feature.properties || {}

    const popup = document.createElement('div')

    const title = document.createElement('strong')
    title.textContent =
      properties.gn_name || 'Unknown GN Division'

    const count = document.createElement('p')
    count.textContent =
      `Intersecting primary candidates: ${
        properties.intersecting_primary_candidate_count ?? 'N/A'
      }`

    const area = document.createElement('p')
    const areaValue = Number(
      properties.primary_candidate_area_ha
    )

    area.textContent =
      `Candidate area within GN: ${
        Number.isFinite(areaValue)
          ? areaValue.toFixed(2) + ' ha'
          : 'N/A'
      }`

    popup.append(title, count, area)
    layer.bindPopup(popup)
  }

  // Primary candidate popup
  function onEachCandidate(feature, layer) {
    const properties = feature.properties || {}

    const popup = document.createElement('div')

    const title = document.createElement('strong')
    title.textContent =
      `Candidate ${properties.candidate_id || 'Unknown'}`

    const area = document.createElement('p')
    const areaValue = Number(properties.area_ha)

    area.textContent =
      `Screened area: ${
        Number.isFinite(areaValue)
          ? areaValue.toFixed(2) + ' ha'
          : 'N/A'
      }`

    const priority = document.createElement('p')
    priority.textContent =
      `Screening priority: ${
        properties.screening_priority || 'N/A'
      }`

    const review = document.createElement('p')
    review.textContent =
      `Manual review action: ${
        properties.qa_final_action || 'N/A'
      }`

    const note = document.createElement('p')
    note.textContent =
      'Screened candidate — not a confirmed wetland.'

    popup.append(title, area, priority, review, note)

    layer.bindPopup(popup)

    layer.on({
      mouseover: (event) => {
        event.target.setStyle({
          weight: 3,
          fillOpacity: 0.8,
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

        <LayersControl position="topright">

        {/* GN Division Boundaries */}
        {gnData?.features?.length > 0 && (
          <LayersControl.Overlay
            checked
            name="GN Division Boundaries"
          >
            <GeoJSON
              data={gnData}
              style={(feature) => {
                const isSelected =
                  selectedGnId != null &&
                  String(feature.properties?.gn_id) ===
                    String(selectedGnId)

                return {
                  color: isSelected ? '#D97706' : '#64748B',
                  weight: isSelected ? 4 : 1,
                  opacity: isSelected ? 1 : 0.75,
                  fillColor: isSelected ? '#FBBF24' : '#EAF2EF',
                  fillOpacity: isSelected ? 0.2 : 0.03,
                }
              }}
              onEachFeature={onEachGn}
            />
          </LayersControl.Overlay>
        )}


          {/* Primary Screened Candidates */}
          <LayersControl.Overlay
            checked
            name="Primary Screened Candidates"
          >
            <GeoJSON
              data={candidateData}
              style={{
                color: '#0F766E',
                weight: 2,
                fillColor: '#2DD4BF',
                fillOpacity: 0.55,
              }}
              onEachFeature={onEachCandidate}
            />
          </LayersControl.Overlay>

        </LayersControl>

        <FitCandidateMapBounds
          gnData={gnData}
          candidateData={candidateData}
        />

        <FocusSelectedCandidate
        candidateData={candidateData}
        selectedCandidateId={selectedCandidateId}
        />

        <FocusSelectedGn
        gnData={gnData}
        selectedGnId={selectedGnId}
        />

      </MapContainer>
    </div>
  )
}
