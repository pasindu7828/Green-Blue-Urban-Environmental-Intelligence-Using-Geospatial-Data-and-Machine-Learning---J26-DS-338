
import { useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import L from 'leaflet'
import {
  GeoJSON,
  MapContainer,
  TileLayer,
  ZoomControl,
  useMap,
} from 'react-leaflet'
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import {
  MapPinned,
  Layers,
  RefreshCcw,
  LocateFixed,
  ShieldAlert,
} from 'lucide-react'

import PageHeader from '../../components/shared/PageHeader'
import MetricCard from '../../components/shared/MetricCard'
import {
  urbanGreenApi,
} from '../../features/urbanGreen/services/urbanGreenApi'
import {
  useUrbanGreen,
} from '../../features/urbanGreen/context/UrbanGreenContext'

// =====================================================
// 1. FROZEN RESEARCH CONFIGURATION
// =====================================================

const STREAMS = {
  DEVELOPMENT_CONFLICT: {
    label: 'Development Conflict (DCI)',
    color: '#d75b48',
    field: 'dci_cluster_id',
    expectedGroups: 36,
    expectedCells: 135,
  },
  CONSERVATION_PRIORITY: {
    label: 'Conservation Priority (CP)',
    color: '#087f65',
    field: 'cp_cluster_id',
    expectedGroups: 22,
    expectedCells: 81,
  },
}

const GN_METRICS = [
  {
    key: 'dci_area_weighted_mean',
    label: 'Development Conflict Index',
  },
  {
    key: 'conservation_area_weighted_mean',
    label: 'Conservation Priority',
  },
  {
    key: 'pressure_area_weighted_mean',
    label: 'Urbanization Pressure',
  },
  {
    key: 'suitability_area_weighted_mean',
    label: 'Development Suitability',
  },
]

function valid(value) {
  return typeof value === 'number' &&
    Number.isFinite(value)
}

function fmt(value, digits = 3) {
  return valid(value)
    ? value.toLocaleString('en-US', {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
      })
    : 'Unavailable'
}

function whole(value) {
  return valid(value)
    ? value.toLocaleString('en-US', {
        maximumFractionDigits: 0,
      })
    : 'Unavailable'
}

function DetailRow({ label, value }) {
  return (
    <div className="flex items-start justify-between gap-3 border-b border-slate-100 py-2.5 last:border-0">
      <dt className="text-xs leading-5 text-slate-500">
        {label}
      </dt>

      <dd className="max-w-[62%] break-words text-right text-xs font-semibold leading-5 text-slate-800">
        {value ?? 'Unavailable'}
      </dd>
    </div>
  )
}

function Heading({ title, description }) {
  return (
    <div className="mb-4">
      <h2 className="section-title">{title}</h2>
      <p className="mt-1 text-xs leading-5 text-slate-500">
        {description}
      </p>
    </div>
  )
}

function asFeatureCollection(features) {
  return {
    type: 'FeatureCollection',
    features,
  }
}

// =====================================================
// 2. MAP CAMERA CONTROLLER
// =====================================================

function MapFocus({
  gnGeojson,
  candidateFeatures,
  selectedClusterId,
  resetKey,
}) {
  const map = useMap()
  const fittedInitially = useRef(false)

  useEffect(() => {
    if (fittedInitially.current) return

    const layer = L.geoJSON(gnGeojson)
    const bounds = layer.getBounds()

    if (bounds.isValid()) {
      map.fitBounds(bounds, {
        padding: [20, 20],
        animate: false,
      })
      fittedInitially.current = true
    }
  }, [gnGeojson, map])

  // Only selecting a hotspot moves the camera.
  // Changing the GN dropdown does not force zoom.
  useEffect(() => {
    if (!selectedClusterId) return

    const selected = candidateFeatures.filter(
      (feature) =>
        feature.properties?.cluster_id ===
        selectedClusterId
    )

    if (!selected.length) return

    const bounds = L.geoJSON(
      asFeatureCollection(selected)
    ).getBounds()

    if (bounds.isValid()) {
      map.fitBounds(bounds, {
        padding: [55, 55],
        maxZoom: 16,
      })
    }
  }, [
    candidateFeatures,
    map,
    selectedClusterId,
  ])

  useEffect(() => {
    if (resetKey === 0) return

    const bounds = L.geoJSON(gnGeojson).getBounds()

    if (bounds.isValid()) {
      map.fitBounds(bounds, {
        padding: [20, 20],
      })
    }
  }, [resetKey, gnGeojson, map])

  return null
}

// =====================================================
// 3. INTERACTIVE GN AND HOTSPOT MAP
// =====================================================

function GnHotspotMap({
  gnGeojson,
  gnDivisions,
  candidateFeatures,
  selectedGnId,
  selectGn,
  selectedClusterId,
  onSelectCluster,
  stream,
  resetKey,
}) {
  const config = STREAMS[stream]

  const gnByName = useMemo(
    () => new Map(
      gnDivisions.map((gn) => [
        String(gn.gn_name).toLowerCase(),
        gn.gn_id,
      ])
    ),
    [gnDivisions]
  )

  return (
    <div className="relative isolate h-[490px] overflow-hidden rounded-xl border border-slate-200 bg-slate-100 md:h-[600px]">
      <MapContainer
        center={[6.902, 79.97]}
        zoom={12}
        zoomControl={false}
        scrollWheelZoom
        className="h-full w-full"
        style={{ zIndex: 0 }}
      >
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          maxZoom={19}
        />

        {/* Real backend GN polygon boundaries */}
        <GeoJSON
          key={`gn-${selectedGnId || 'all'}`}
          data={gnGeojson}
          style={(feature) => {
            const props = feature?.properties || {}

            const gnId =
              props.gn_id ||
              gnByName.get(
                String(props.gn_name || '').toLowerCase()
              )

            const selected =
              selectedGnId &&
              String(gnId) === String(selectedGnId)

            return {
              color: selected ? '#102f2a' : '#64748b',
              weight: selected ? 3 : 1,
              opacity: selected ? 1 : 0.7,
              fillColor: '#0f766e',
              fillOpacity: selected ? 0.11 : 0.015,
            }
          }}
          onEachFeature={(feature, layer) => {
            const props = feature.properties || {}

            const gnId =
              props.gn_id ||
              gnByName.get(
                String(props.gn_name || '').toLowerCase()
              )

            if (props.gn_name) {
              layer.bindTooltip(
                String(props.gn_name),
                { sticky: true }
              )
            }

            if (gnId) {
              layer.on('click', () => {
                selectGn(String(gnId))
              })
            }
          }}
        />

        {/* Actual candidate-member 100 m polygons */}
        <GeoJSON
          key={`${stream}-${selectedClusterId || 'all'}`}
          data={asFeatureCollection(candidateFeatures)}
          style={(feature) => {
            const chosen =
              selectedClusterId &&
              feature?.properties?.cluster_id ===
              selectedClusterId

            return {
              color: chosen ? '#111827' : config.color,
              weight: chosen ? 2.7 : 1.2,
              opacity: 1,
              fillColor: config.color,
              fillOpacity: chosen ? 0.85 : 0.48,
            }
          }}
          onEachFeature={(feature, layer) => {
            const props = feature.properties || {}

            layer.bindTooltip(
              `${props.cluster_id || 'Unknown cluster'} | ${
                props.grid_id || 'Grid unavailable'
              }`,
              { sticky: true }
            )

            if (props.cluster_id) {
              layer.on('click', () => {
                onSelectCluster(props.cluster_id)
              })
            }
          }}
        />

        <ZoomControl position="topleft" />

        <MapFocus
          gnGeojson={gnGeojson}
          candidateFeatures={candidateFeatures}
          selectedClusterId={selectedClusterId}
          resetKey={resetKey}
        />
      </MapContainer>

      <div className="pointer-events-none absolute right-3 top-3 z-[450] max-w-[220px] rounded-lg border border-slate-200 bg-white/95 px-3 py-2 text-[11px] shadow-sm">
        <p className="font-bold text-slate-800">
          Spatial candidate screening
        </p>

        <p className="mt-1 text-slate-600">
          {config.label}
        </p>
      </div>
    </div>
  )
}

// =====================================================
// 4. MAIN GN AND HOTSPOT PAGE
// =====================================================

export default function GnHotspotExplorer() {
  const {
    selectedGnId,
    selectGn,
  } = useUrbanGreen()

  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [refresh, setRefresh] = useState(0)

  const [stream, setStream] = useState(
    'DEVELOPMENT_CONFLICT'
  )

  const [gnMetric, setGnMetric] = useState(
    'dci_area_weighted_mean'
  )

  const [selectedClusterId, setSelectedClusterId] =
    useState(null)

  const [resetKey, setResetKey] = useState(0)
  const [showAll, setShowAll] = useState(false)

  // ---------------------------------------------------
  // 4A. LOAD FROZEN GN AND HOTSPOT DATA
  // ---------------------------------------------------

  useEffect(() => {
    const controller = new AbortController()

    setLoading(true)
    setError('')

    Promise.all([
      urbanGreenApi.planning.gnDivisions(
        controller.signal
      ),

      urbanGreenApi.planning.gnMap(
        controller.signal
      ),

      urbanGreenApi.planning.hotspots(
        {},
        controller.signal
      ),

      urbanGreenApi.planning.map(
        controller.signal
      ),
    ])
      .then(([
        gnDivisions,
        gnMap,
        hotspots,
        planningMap,
      ]) => {
        if (controller.signal.aborted) return

        if (
          !Array.isArray(gnDivisions?.data) ||
          !Array.isArray(gnMap?.features) ||
          !Array.isArray(hotspots?.data) ||
          !Array.isArray(planningMap?.features)
        ) {
          throw new Error(
            'One or more GN or hotspot API responses have an unexpected format.'
          )
        }

        if (
          gnDivisions.total !== 57 ||
          gnDivisions.data.length !== 57 ||
          hotspots.total !== 58 ||
          hotspots.data.length !== 58 ||
          planningMap.features.length !== 9240
        ) {
          throw new Error(
            'The frozen 57 GN, 58 candidate-group or 9,240 grid research release is incomplete.'
          )
        }

        const ids = new Set(
          hotspots.data.map(
            (record) => record.cluster_id
          )
        )

        if (ids.size !== 58) {
          throw new Error(
            'The selected hotspot IDs must be unique.'
          )
        }

        const dciCount = hotspots.data.filter(
          (record) =>
            record.index_type === 'DEVELOPMENT_CONFLICT'
        ).length

        const cpCount = hotspots.data.filter(
          (record) =>
            record.index_type === 'CONSERVATION_PRIORITY'
        ).length

        if (dciCount !== 36 || cpCount !== 22) {
          throw new Error(
            'The DCI/CP candidate-group counts do not match the frozen Section 05 release.'
          )
        }

        setData({
          gnDivisions: gnDivisions.data,
          gnGeojson: gnMap,
          hotspots: hotspots.data,
          planningMap,
          statisticalTested:
            hotspots.statistical_significance_tested,
        })
      })
      .catch((reason) => {
        if (!controller.signal.aborted) {
          setError(
            reason?.message ||
            'GN and hotspot data loading failed.'
          )
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setLoading(false)
        }
      })

    return () => controller.abort()
  }, [refresh])

  // ---------------------------------------------------
  // 4B. SELECTED GN DIVISION
  // ---------------------------------------------------

  const selectedGn = useMemo(
    () => (
      data?.gnDivisions || []
    ).find(
      (gn) =>
        String(gn.gn_id) ===
        String(selectedGnId)
    ) || null,
    [data, selectedGnId]
  )

  // ---------------------------------------------------
  // 4C. REAL CANDIDATE GROUPS
  // ---------------------------------------------------

  const candidateGroups = useMemo(
    () => (
      data?.hotspots || []
    ).filter(
      (record) =>
        record.index_type === stream
    ),
    [data, stream]
  )

  const candidateIds = useMemo(
    () => new Set(
      candidateGroups.map(
        (record) => String(record.cluster_id)
      )
    ),
    [candidateGroups]
  )

  // ---------------------------------------------------
  // 4D. MAP CANDIDATE-MEMBER GRIDS
  // ---------------------------------------------------

  // Original Section 05 planning map has both
  // dci_cluster_id and cp_cluster_id.
  // Match only IDs in the 58 selected candidates.
  // Do NOT create new clusters or apply new thresholds.

  const candidateFeatures = useMemo(() => {
    const field = STREAMS[stream].field

    return (
      data?.planningMap?.features || []
    )
      .filter(
        (feature) =>
          candidateIds.has(
            String(feature.properties?.[field])
          )
      )
      .map((feature) => ({
        ...feature,
        properties: {
          ...feature.properties,
          cluster_id:
            feature.properties[field],
        },
      }))
  }, [data, stream, candidateIds])

  const selectedCluster = useMemo(
    () => (
      data?.hotspots || []
    ).find(
      (record) =>
        String(record.cluster_id) ===
        String(selectedClusterId)
    ) || null,
    [data, selectedClusterId]
  )

  // ---------------------------------------------------
  // 4E. GN FILTER OF BACKEND CANDIDATE GROUPS
  // ---------------------------------------------------

  // The backend lists intersected GN names for each
  // candidate. Preserve this recorded attribution;
  // do not assign all cluster area to one GN.

  const filteredGroups = useMemo(() => {
    let records = candidateGroups

    if (selectedGn) {
      records = records.filter((record) => {
        if (
          String(record.dominant_intersected_gn_id) ===
          String(selectedGn.gn_id)
        ) {
          return true
        }

        const names = String(
          record.intersected_gn_names || ''
        )
          .split(/[|;,]/)
          .map((name) =>
            name.trim().toLowerCase()
          )

        return names.includes(
          String(selectedGn.gn_name).toLowerCase()
        )
      })
    }

    return [...records].sort(
      (a, b) =>
        (
          b.mean_screening_score_area_weighted ?? 0
        ) -
        (
          a.mean_screening_score_area_weighted ?? 0
        )
    )
  }, [candidateGroups, selectedGn])

  // ---------------------------------------------------
  // 4F. GN COMPARISON CHART
  // ---------------------------------------------------

  const gnChartData = useMemo(
    () => (
      data?.gnDivisions || []
    )
      .filter(
        (gn) => valid(gn[gnMetric])
      )
      .sort(
        (a, b) =>
          b[gnMetric] - a[gnMetric]
      )
      .slice(0, 8)
      .map((gn) => ({
        name: gn.gn_name,
        value: gn[gnMetric],
        id: gn.gn_id,
      })),
    [data, gnMetric]
  )

  // ---------------------------------------------------
  // 4G. SELECT STREAM
  // ---------------------------------------------------

  function changeStream(next) {
    setStream(next)
    setSelectedClusterId(null)
    setShowAll(false)
  }

  function resetMap() {
    setSelectedClusterId(null)
    setResetKey((value) => value + 1)
  }

  // ===================================================
  // 5. LOADING AND ERROR STATES
  // ===================================================

  if (loading) {
    return (
      <section className="card-pad">
        <h2 className="section-title">
          Loading GN and hotspot research
        </h2>

        <p className="mt-3 text-sm text-slate-500">
          Reading 57 area-weighted GN summaries,
          selected spatial candidate groups and
          original research polygons...
        </p>
      </section>
    )
  }

  if (error || !data) {
    return (
      <section className="card-pad">
        <h2 className="section-title">
          GN and hotspot data unavailable
        </h2>

        <p className="mt-3 text-sm text-red-700">
          {error || 'No validated data returned.'}
        </p>

        <button
          type="button"
          className="btn-secondary mt-4"
          onClick={() =>
            setRefresh((value) => value + 1)
          }
        >
          <RefreshCcw size={15} />
          Retry
        </button>
      </section>
    )
  }

  const visibleGroups = showAll
    ? filteredGroups
    : filteredGroups.slice(0, 12)

  const mappedCellTotal = candidateFeatures.length

  const expectedCells =
    STREAMS[stream].expectedCells

  // ===================================================
  // 6. PAGE CONTENT
  // ===================================================

  return (
    <div>
      <PageHeader
        eyebrow="Urban Green Intelligence"
        title="GN Division and Hotspot Explorer"
        subtitle="Explore GN-level area-weighted planning indicators and separate exploratory Development Conflict and Conservation Priority spatial candidate groups."
        badges={[
          '57 GN divisions',
          '58 selected spatial candidates',
          'Exploratory screening · not statistical hotspots',
        ]}
        actions={
          <button
            type="button"
            className="btn-secondary"
            onClick={() =>
              setRefresh((value) => value + 1)
            }
          >
            <RefreshCcw size={15} />
            Refresh
          </button>
        }
      />

      {/* =========================================== */}
      {/* 6A. RESEARCH SUMMARY CARDS                 */}
      {/* =========================================== */}

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          title="GN divisions"
          value={whole(data.gnDivisions.length)}
          subtitle="Research-backed GN spatial summaries"
          icon={MapPinned}
          tone="teal"
        />

        <MetricCard
          title="Candidate spatial groups"
          value={whole(data.hotspots.length)}
          subtitle="Two independent screening streams"
          icon={Layers}
          tone="blue"
        />

        <MetricCard
          title="DCI groups"
          value="36"
          subtitle="135 selected original grid cells"
          icon={ShieldAlert}
          tone="orange"
        />

        <MetricCard
          title="Conservation groups"
          value="22"
          subtitle="81 selected original grid cells"
          icon={MapPinned}
          tone="green"
        />
      </div>

      {/* =========================================== */}
      {/* 6B. GIS MAP + SELECTED GN / HOTSPOT        */}
      {/* =========================================== */}

      <div className="mt-5 grid gap-5 2xl:grid-cols-[1.55fr_.85fr]">
        <section className="card-pad min-w-0">
          <Heading
            title="GN divisions and spatial candidate map"
            description="Real GN boundaries and selected candidate-group member cells from the original 100 m Section 05 GIS output."
          />

          <div className="mb-3 grid gap-3 md:grid-cols-2">
            <label className="text-xs font-semibold text-slate-600">
              GN division

              <select
                className="control mt-1"
                value={selectedGnId || ''}
                onChange={(event) =>
                  selectGn(
                    event.target.value || null
                  )
                }
              >
                <option value="">
                  All GN divisions
                </option>

                {data.gnDivisions.map((gn) => (
                  <option
                    key={gn.gn_id}
                    value={gn.gn_id}
                  >
                    {gn.gn_name}
                  </option>
                ))}
              </select>
            </label>

            <label className="text-xs font-semibold text-slate-600">
              Spatial screening stream

              <select
                className="control mt-1"
                value={stream}
                onChange={(event) =>
                  changeStream(event.target.value)
                }
              >
                {Object.entries(STREAMS).map(
                  ([key, value]) => (
                    <option key={key} value={key}>
                      {value.label}
                    </option>
                  )
                )}
              </select>
            </label>
          </div>

          <div className="mb-3 flex flex-wrap items-center justify-between gap-3 text-xs">
            <div className="flex flex-wrap gap-3 text-slate-600">
              <span className="flex items-center gap-1.5">
                <span
                  className="h-3 w-3 rounded-sm"
                  style={{
                    backgroundColor:
                      STREAMS[stream].color,
                  }}
                />

                Candidate 100 m cells
              </span>

              <span className="flex items-center gap-1.5">
                <span className="h-3 w-3 rounded-sm border-2 border-slate-700" />
                GN boundaries
              </span>
            </div>

            <button
              type="button"
              onClick={resetMap}
              className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 px-3 py-1.5 font-semibold text-slate-700 hover:bg-slate-50"
            >
              <LocateFixed size={14} />
              Fit entire study area
            </button>
          </div>

          <GnHotspotMap
            gnGeojson={data.gnGeojson}
            gnDivisions={data.gnDivisions}
            candidateFeatures={candidateFeatures}
            selectedGnId={selectedGnId}
            selectGn={selectGn}
            selectedClusterId={selectedClusterId}
            onSelectCluster={setSelectedClusterId}
            stream={stream}
            resetKey={resetKey}
          />

          <p className="mt-3 text-xs leading-5 text-slate-500">
            Original candidate-member grid polygons
            shown: <strong>{whole(mappedCellTotal)}</strong>.
            Expected research membership:{' '}
            <strong>{whole(expectedCells)}</strong>.

            {' '}Click a candidate cell to select its
            spatial group. Selecting a hotspot will
            zoom to that group; selecting a GN division
            only changes its outline and the tables,
            without forcing a map zoom.
          </p>

          {mappedCellTotal !== expectedCells && (
            <p className="mt-2 rounded-lg border border-amber-200 bg-amber-50 p-3 text-xs leading-5 text-amber-900">
              The candidate grid membership count
              does not match the frozen research
              summary. Verify the planning map's
              cluster IDs before interpreting
              the spatial layer.
            </p>
          )}
        </section>

        <div className="space-y-5">
          {/* SELECTED GN SUMMARY */}

          <section className="card-pad">
            <Heading
              title="Selected GN Division"
              description="Area-weighted research indicators, not simple averages of assigned grids."
            />

            {!selectedGn ? (
              <p className="text-sm leading-6 text-slate-500">
                Select a GN division using the
                dropdown or its polygon on the
                map to see the official research
                spatial summaries.
              </p>
            ) : (
              <>
                <p className="mb-3 rounded-lg bg-emerald-50 px-3 py-2 text-sm font-semibold text-emerald-900">
                  {selectedGn.gn_name}
                </p>

                <dl>
                  <DetailRow
                    label="GN identifier"
                    value={selectedGn.gn_id}
                  />

                  <DetailRow
                    label="GN area inside KMC"
                    value={`${fmt(
                      selectedGn.gn_area_inside_kmc_km2
                    )} km²`}
                  />

                  <DetailRow
                    label="Suitability mean"
                    value={fmt(
                      selectedGn.suitability_area_weighted_mean,
                      4
                    )}
                  />

                  <DetailRow
                    label="Urbanization Pressure mean"
                    value={fmt(
                      selectedGn.pressure_area_weighted_mean,
                      4
                    )}
                  />

                  <DetailRow
                    label="Development Conflict mean"
                    value={fmt(
                      selectedGn.dci_area_weighted_mean,
                      4
                    )}
                  />

                  <DetailRow
                    label="Conservation Priority mean"
                    value={fmt(
                      selectedGn.conservation_area_weighted_mean,
                      4
                    )}
                  />

                  <DetailRow
                    label="Pressure valid area"
                    value={`${fmt(
                      selectedGn.pressure_valid_area_km2
                    )} km²`}
                  />

                  <DetailRow
                    label="DCI valid area"
                    value={`${fmt(
                      selectedGn.dci_valid_area_km2
                    )} km²`}
                  />
                </dl>

                <p className="mt-3 text-[11px] leading-5 text-slate-500">
                  Indicator means apply only to
                  areas with available evidence.
                  An unavailable mean must not
                  be interpreted as zero.
                </p>
              </>
            )}
          </section>

          {/* SELECTED HOTSPOT DETAILS */}

          <section className="card-pad">
            <Heading
              title="Selected spatial candidate"
              description="Frozen group details for the selected DCI or CP cluster."
            />

            {!selectedCluster ? (
              <p className="text-sm leading-6 text-slate-500">
                Select a candidate polygon on
                the map or a group in the
                table below.
              </p>
            ) : (
              <dl>
                <DetailRow
                  label="Cluster ID"
                  value={selectedCluster.cluster_id}
                />

                <DetailRow
                  label="Screening stream"
                  value={
                    STREAMS[
                      selectedCluster.index_type
                    ]?.label || 'Unavailable'
                  }
                />

                <DetailRow
                  label="Original 100 m cells"
                  value={whole(
                    selectedCluster.grid_count
                  )}
                />

                <DetailRow
                  label="Candidate group area"
                  value={`${fmt(
                    selectedCluster.area_km2,
                    6
                  )} km²`}
                />

                <DetailRow
                  label="Mean screening score"
                  value={fmt(
                    selectedCluster.mean_screening_score_area_weighted,
                    6
                  )}
                />

                <DetailRow
                  label="Minimum screening score"
                  value={fmt(
                    selectedCluster.minimum_screening_score,
                    6
                  )}
                />

                <DetailRow
                  label="Maximum screening score"
                  value={fmt(
                    selectedCluster.maximum_screening_score,
                    6
                  )}
                />

                <DetailRow
                  label="Dominant intersected GN"
                  value={
                    selectedCluster.dominant_intersected_gn_name ||
                    'Unavailable'
                  }
                />

                <DetailRow
                  label="Intersected GN divisions"
                  value={
                    selectedCluster.intersected_gn_names ||
                    'Unavailable'
                  }
                />

                <DetailRow
                  label="GN-uncovered group area"
                  value={`${fmt(
                    selectedCluster.gn_uncovered_area_m2,
                    3
                  )} m²`}
                />
              </dl>
            )}

            <p className="mt-3 rounded-lg border border-amber-200 bg-amber-50 p-3 text-[11px] leading-5 text-amber-900">
              Candidate status does not indicate
              statistical significance, legal
              development conflict, an observed
              ecological loss, or an approved
              planning intervention.
            </p>
          </section>
        </div>
      </div>

      {/* =========================================== */}
      {/* 6C. GN COMPARISON CHART                    */}
      {/* =========================================== */}

      <section className="card-pad mt-5">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <Heading
            title="GN Division indicator comparison"
            description="Descriptive comparison of the eight highest available GN means for the selected continuous research indicator."
          />

          <select
            className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs"
            value={gnMetric}
            onChange={(event) =>
              setGnMetric(event.target.value)
            }
          >
            {GN_METRICS.map((metric) => (
              <option
                key={metric.key}
                value={metric.key}
              >
                {metric.label}
              </option>
            ))}
          </select>
        </div>

        <ResponsiveContainer
          width="100%"
          height={355}
        >
          <BarChart
            data={gnChartData}
            layout="vertical"
            margin={{
              top: 6,
              right: 20,
              bottom: 8,
              left: 10,
            }}
          >
            <CartesianGrid
              strokeDasharray="4 5"
            />

            <XAxis
              type="number"
              domain={[0, 1]}
              tick={{ fontSize: 11 }}
            />

            <YAxis
              type="category"
              dataKey="name"
              width={155}
              tick={{ fontSize: 11 }}
            />

            <Tooltip
              formatter={(value) => [
                fmt(value, 4),
                'Area-weighted mean',
              ]}
            />

            <Bar
              dataKey="value"
              fill="#0f766e"
              radius={[0, 4, 4, 0]}
              isAnimationActive={false}
            />
          </BarChart>
        </ResponsiveContainer>

        <p className="mt-3 text-[11px] leading-5 text-slate-500">
          These are descriptive comparisons,
          not official GN risk rankings.
          Different indicators have different
          valid-area coverage. Only GN divisions
          with a valid mean for the selected
          indicator appear in this chart.
        </p>
      </section>

      {/* =========================================== */}
      {/* 6D. CANDIDATE GROUP TABLE                  */}
      {/* =========================================== */}

      <section className="card-pad mt-5">
        <Heading
          title="Explore spatial candidate groups"
          description="Select a recorded group to highlight its original grid members and zoom the GIS map to its location."
        />

        <div className="mb-3 flex flex-wrap items-center justify-between gap-2 text-xs">
          <p className="text-slate-600">
            Showing <strong>{whole(visibleGroups.length)}</strong>
            {' '}of <strong>{whole(filteredGroups.length)}</strong>
            {' '}groups matching the current stream
            and GN filters.
          </p>

          <button
            type="button"
            onClick={() =>
              setShowAll((value) => !value)
            }
            className="rounded-lg border border-slate-200 px-3 py-2 font-semibold text-emerald-800 hover:bg-emerald-50"
          >
            {showAll
              ? 'Show first 12 groups'
              : 'Show all matching groups'}
          </button>
        </div>

        <div className="overflow-x-auto">
          <table className="min-w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50 text-slate-600">
                <th className="px-3 py-3">
                  Group ID
                </th>

                <th className="px-3 py-3">
                  Dominant GN
                </th>

                <th className="px-3 py-3">
                  Grid cells
                </th>

                <th className="px-3 py-3">
                  Area (km²)
                </th>

                <th className="px-3 py-3">
                  Mean screening score
                </th>

                <th className="px-3 py-3">
                  Map
                </th>
              </tr>
            </thead>

            <tbody>
              {visibleGroups.map((row) => (
                <tr
                  key={row.cluster_id}
                  className={
                    selectedClusterId === row.cluster_id
                      ? 'border-b border-emerald-200 bg-emerald-50'
                      : 'border-b border-slate-100'
                  }
                >
                  <td className="px-3 py-3 font-mono font-semibold text-slate-800">
                    {row.cluster_id}
                  </td>

                  <td className="px-3 py-3">
                    {row.dominant_intersected_gn_name ||
                      'Unassigned'}
                  </td>

                  <td className="px-3 py-3 tabular-nums">
                    {whole(row.grid_count)}
                  </td>

                  <td className="px-3 py-3 tabular-nums">
                    {fmt(row.area_km2, 5)}
                  </td>

                  <td className="px-3 py-3 tabular-nums">
                    {fmt(
                      row.mean_screening_score_area_weighted,
                      4
                    )}
                  </td>

                  <td className="px-3 py-3">
                    <button
                      type="button"
                      onClick={() =>
                        setSelectedClusterId(row.cluster_id)
                      }
                      className="font-semibold text-emerald-800 hover:underline"
                    >
                      Locate
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {filteredGroups.length === 0 && (
          <p className="mt-3 rounded-lg bg-slate-50 p-3 text-xs text-slate-600">
            No selected candidate group matches
            this GN and screening stream.
            That does not prove the GN has no
            environmental or planning concerns.
          </p>
        )}

        <p className="mt-3 text-[11px] leading-5 text-slate-500">
          GN filtering uses the cluster's recorded
          intersected-GN names and dominant GN
          identifier. A cluster may intersect
          multiple GN divisions. Its full area
          is not attributed to just the selected
          division.
        </p>
      </section>

      {/* =========================================== */}
      {/* 6E. RESEARCH LIMITATIONS                   */}
      {/* =========================================== */}

      {/* <section className="card-pad mt-5">
        <Heading
          title="Spatial screening method and limitations"
          description="How to interpret the completed Section 05 GN and hotspot research."
        />

        <div className="grid gap-3 md:grid-cols-3">
          <div className="rounded-xl bg-slate-50 p-4 text-xs leading-6 text-slate-700">
            <strong>GN summaries</strong>

            <p className="mt-2">
              Calculated using area-weighted
              intersections of original planning
              grid polygons with GN boundaries.
              Unavailable scores are excluded
              rather than replaced with zero.
            </p>
          </div>

          <div className="rounded-xl bg-slate-50 p-4 text-xs leading-6 text-slate-700">
            <strong>Candidate screening</strong>

            <p className="mt-2">
              DCI and CP use separate top-decile
              spatial screening streams. Connected
              candidate groups contain at least
              three original 100 m grid cells.
            </p>
          </div>

          <div className="rounded-xl bg-slate-50 p-4 text-xs leading-6 text-slate-700">
            <strong>Not statistical hotspots</strong>

            <p className="mt-2">
              The exploratory groups were not
              validated with a statistical
              hotspot-significance test. They
              are spatial review candidates only.
            </p>
          </div>
        </div>

        <p className="mt-4 rounded-xl border border-amber-200 bg-amber-50 p-4 text-xs leading-6 text-amber-900">
          <strong>Research caution:</strong>{' '}
          DCI reflects a relative planning-conflict
          screening index; CP reflects relative
          conservation priority where both required
          inputs are available.

          {' '}These outputs must remain separate
          and must not be interpreted as official
          zoning restrictions, legal permissions,
          statistically significant hotspots or
          field-confirmed environmental damage.

          {' '}GN boundary gaps and model-domain
          exclusions remain documented limitations.
        </p>

        <Link
          to="/green/methodology"
          className="mt-4 inline-flex items-center gap-2 text-xs font-semibold text-emerald-800 hover:underline"
        >
          Next: Research Methodology and Data QA →
        </Link>
      </section> */}
    </div>
  )
}
