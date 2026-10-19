
import { useEffect, useMemo, useRef, useState } from 'react'
import {
  GeoJSON,
  ImageOverlay,
  MapContainer,
  TileLayer,
  ZoomControl,
  useMap,
} from 'react-leaflet'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { Info, RefreshCcw } from 'lucide-react'

import PageHeader from '../../components/shared/PageHeader'
import { urbanGreenApi } from '../../features/urbanGreen/services/urbanGreenApi'

// =====================================================
// 1. RESEARCH CONFIGURATION
// =====================================================

const MAP_OPTIONS = [
  {
    id: 'GREEN_TRANSITION_ROBUST',
    label: 'Robust transitions',
    help: 'Primary 2-of-3 consensus transition map',
  },
  {
    id: 'GREEN_TRANSITION_STRICT',
    label: 'Strict transitions',
    help: 'Conservative 3-of-3 consensus transition map',
  },
  {
    id: 'GREEN_TO_BUILT_CONFIDENCE',
    label: 'Green-to-Built agreement',
    help: 'Robust-only versus strict-confirmed Green-to-Built pixels',
  },
  {
    id: 'TRANSITION_CONFIDENCE',
    label: 'All-transition agreement',
    help: 'Robust-only versus strict-confirmed transition pixels',
  },
]

const ROBUST = 'ROBUST_2_OF_3'
const STRICT = 'STRICT_3_OF_3'

function number(value, decimals = 2) {
  if (value == null || !Number.isFinite(Number(value))) return 'N/A'
  return Number(value).toFixed(decimals)
}

function signed(value, decimals = 2) {
  if (value == null || !Number.isFinite(Number(value))) return 'N/A'
  const n = Number(value)
  return `${n > 0 ? '+' : ''}${n.toFixed(decimals)}`
}

function integer(value) {
  return value == null || !Number.isFinite(Number(value))
    ? 'N/A'
    : Number(value).toLocaleString()
}

function isValidBounds(bounds) {
  return (
    Array.isArray(bounds) &&
    bounds.length === 2 &&
    bounds.every(
      (corner) =>
        Array.isArray(corner) &&
        corner.length === 2 &&
        corner.every(
          (coordinate) =>
            typeof coordinate === 'number' &&
            Number.isFinite(coordinate)
        )
    ) &&
    bounds[0][0] < bounds[1][0] &&
    bounds[0][1] < bounds[1][1]
  )
}

function rgbaCss(rgba) {
  if (!Array.isArray(rgba) || rgba.length !== 4) return '#94a3b8'
  return `rgba(${rgba[0]}, ${rgba[1]}, ${rgba[2]}, ${rgba[3] / 255})`
}

function SummaryCard({ title, value, description }) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
      <p className="text-xs font-semibold text-slate-500">{title}</p>
      <p className="mt-2 text-2xl font-bold tracking-tight text-slate-900">
        {value}
      </p>
      <p className="mt-1 text-xs leading-5 text-slate-500">{description}</p>
    </div>
  )
}

// =====================================================
// 2. LEAFLET MAP AND INITIAL FIT
// =====================================================

function FitMapOnFirstLoad({ bounds }) {
  const map = useMap()
  const hasFitted = useRef(false)

  useEffect(() => {
    if (!hasFitted.current && isValidBounds(bounds)) {
      map.fitBounds(bounds, {
        padding: [18, 18],
        animate: false,
      })
      hasFitted.current = true
    }
  }, [bounds, map])

  return null
}

function TransitionMap({
  record,
  boundary,
  showStreets,
  showBoundary,
  opacity,
  onImageError,
}) {
  return (
    <div className="relative isolate h-[470px] overflow-hidden rounded-xl border border-slate-200 bg-[#e9f0ef] md:h-[550px]">
      <MapContainer
        center={[6.902, 79.97]}
        zoom={12}
        zoomControl={false}
        scrollWheelZoom
        maxZoom={19}
        className="h-full w-full"
        style={{ zIndex: 0 }}
      >
        {showStreets && (
          <TileLayer
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            maxZoom={19}
          />
        )}

        <ImageOverlay
          key={record.id}
          url={urbanGreenApi.historical.mapImageUrl(record.id)}
          bounds={record.bounds}
          opacity={opacity}
          zIndex={10}
          eventHandlers={{ error: onImageError }}
        />

        {showBoundary && boundary?.features?.length > 0 && (
          <GeoJSON
            data={boundary}
            interactive={false}
            style={{
              color: '#102f2a',
              weight: 3,
              opacity: 0.95,
              fillOpacity: 0,
            }}
          />
        )}

        <ZoomControl position="topleft" />
        <FitMapOnFirstLoad bounds={record.bounds} />
      </MapContainer>

      <div className="pointer-events-none absolute right-3 top-3 z-[450] rounded-lg border border-slate-200 bg-white/95 px-3 py-2 text-[11px] font-semibold text-slate-700 shadow-sm">
        Historical transition research · 30 m LULC
      </div>
    </div>
  )
}

// =====================================================
// 3. MAIN PAGE
// =====================================================

export default function Transitions() {
  const [refreshToken, setRefreshToken] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [data, setData] = useState(null)

  const [mapId, setMapId] = useState('GREEN_TRANSITION_ROBUST')
  const [breakdownMethod, setBreakdownMethod] = useState(ROBUST)
  const [opacity, setOpacity] = useState(0.95)
  const [showStreets, setShowStreets] = useState(true)
  const [showBoundary, setShowBoundary] = useState(true)
  const [boundary, setBoundary] = useState(null)
  const [imageError, setImageError] = useState('')

  // ---------------------------------------------------
  // 3A. LOAD ALL FIVE RESEARCH ENDPOINTS AND MAPS
  // ---------------------------------------------------

  useEffect(() => {
    const controller = new AbortController()
    setLoading(true)
    setError('')

    Promise.all([
      urbanGreenApi.historical.transitions.analysis(controller.signal),
      urbanGreenApi.historical.transitions.breakdown(controller.signal),
      urbanGreenApi.historical.transitions.comparison(controller.signal),
      urbanGreenApi.historical.transitions.confidence(controller.signal),
      urbanGreenApi.historical.transitions.method(controller.signal),
      urbanGreenApi.historical.maps(controller.signal),
      urbanGreenApi.historical.mapsHealth(controller.signal),
    ])
      .then(([
        analysis,
        breakdown,
        comparison,
        confidence,
        method,
        catalogue,
        health,
      ]) => {
        if (controller.signal.aborted) return

        if (
          analysis?.overall_step10b_status !== 'PASS' ||
          !Array.isArray(breakdown?.data) ||
          !Array.isArray(comparison?.data) ||
          !Array.isArray(confidence?.data) ||
          !Array.isArray(catalogue?.maps)
        ) {
          throw new Error('Unexpected transition research API response or missing QA PASS.')
        }

        const transitionMaps = catalogue.maps.filter(
          (record) => MAP_OPTIONS.some((option) => option.id === record.id)
        )

        if (
          transitionMaps.length !== 4 ||
          !transitionMaps.every((record) => isValidBounds(record.bounds))
        ) {
          throw new Error('The four validated transition map overlays were not found.')
        }

        const methods = comparison.data.map((record) => record.consensus_method)
        if (!methods.includes(ROBUST) || !methods.includes(STRICT)) {
          throw new Error('Robust/strict comparison records are incomplete.')
        }

        setData({
          analysis,
          breakdown: breakdown.data,
          comparison: comparison.data,
          confidence: confidence.data,
          method,
          maps: transitionMaps,
          health,
        })
      })
      .catch((reason) => {
        if (!controller.signal.aborted) {
          setError(reason.message || String(reason))
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false)
      })

    return () => controller.abort()
  }, [refreshToken])

  // ---------------------------------------------------
  // 3B. KMC OUTLINE (SAME AS PREVIOUS TWO PAGES)
  // ---------------------------------------------------

  useEffect(() => {
    const controller = new AbortController()
    const base = import.meta.env.BASE_URL || '/'
    const url = `${base.endsWith('/') ? base : `${base}/`}data/kaduwela-boundary.geojson`

    fetch(url, { signal: controller.signal })
      .then((response) => response.ok ? response.json() : null)
      .then((geojson) => {
        if (!controller.signal.aborted) setBoundary(geojson)
      })
      .catch(() => {})

    return () => controller.abort()
  }, [])

  // ---------------------------------------------------
  // 3C. SELECT THE ACTUAL GIS OVERLAY
  // ---------------------------------------------------

  const activeMap = useMemo(
    () => data?.maps.find((record) => record.id === mapId) || null,
    [data, mapId]
  )

  const activeMapOption = MAP_OPTIONS.find((option) => option.id === mapId)

  useEffect(() => {
    setImageError('')
  }, [mapId])

  function chooseMap(nextId) {
    setMapId(nextId)
    if (nextId === 'GREEN_TRANSITION_ROBUST') {
      setBreakdownMethod(ROBUST)
    } else if (nextId === 'GREEN_TRANSITION_STRICT') {
      setBreakdownMethod(STRICT)
    }
  }

  // ---------------------------------------------------
  // 3D. REAL COMPARISON, BREAKDOWN, AGREEMENT RESULTS
  // ---------------------------------------------------

  const robust = data?.comparison.find(
    (row) => row.consensus_method === ROBUST
  )

  const strict = data?.comparison.find(
    (row) => row.consensus_method === STRICT
  )

  const selectedComparison = breakdownMethod === ROBUST ? robust : strict

  const breakdownRows = useMemo(
    () => (data?.breakdown || [])
      .filter((row) => row.consensus_method === breakdownMethod)
      .sort((a, b) => Number(a.green_transition_code) - Number(b.green_transition_code)),
    [data, breakdownMethod]
  )

  const greenBuiltAgreement = data?.confidence.find(
    (row) => String(row.metric).toLowerCase().includes('built')
  )

  const comparisonChart = useMemo(() => {
    if (!robust || !strict) return []

    return [
      {
        category: 'Green retained',
        robust: robust.green_retained_area_km2,
        strict: strict.green_retained_area_km2,
      },
      {
        category: 'Green → Built-up',
        robust: robust.green_to_built_area_km2,
        strict: strict.green_to_built_area_km2,
      },
      {
        category: 'Total green loss',
        robust: robust.total_green_loss_area_km2,
        strict: strict.total_green_loss_area_km2,
      },
      {
        category: 'Total green gain',
        robust: robust.total_green_gain_area_km2,
        strict: strict.total_green_gain_area_km2,
      },
    ]
  }, [robust, strict])

  const retry = () => setRefreshToken((n) => n + 1)

  // ===================================================
  // 4. RENDER
  // ===================================================

  return (
    <div>
      <PageHeader
        eyebrow="Urban Green Intelligence"
        title="Green–Built Transition Analysis"
        subtitle="Explore historically modelled land-cover transitions between the 2012–2014 and 2021–2023 consensus periods."
        badges={[
          'Retrospective LULC transitions',
          '30 m classified raster',
          'Robust vs strict consensus',
        ]}
      />

      <div className="mt-4 flex justify-end">
        <button
          type="button"
          onClick={retry}
          className="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-50"
        >
          <RefreshCcw size={15} /> Refresh data
        </button>
      </div>

      {loading && (
        <section className="card-pad mt-4 text-sm text-slate-600">
          Loading historical transition analysis and GIS maps...
        </section>
      )}

      {!loading && error && (
        <section className="card-pad mt-4">
          <h2 className="section-title">Transition research data unavailable</h2>
          <p className="mt-2 text-sm text-red-700">{error}</p>
          <p className="mt-2 text-xs text-slate-500">
            Check that the Urban Green FastAPI backend is running, then retry.
          </p>
        </section>
      )}

      {!loading && !error && data && (
        <>
          {/* ================================================= */}
          {/* 4A. SUMMARY CARDS                                */}
          {/* ================================================= */}

          <div className="mt-5 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <SummaryCard
              title="Green → Built-up · robust"
              value={`${number(data.analysis.robust_green_to_built_area_km2, 4)} km²`}
              description="Primary 2-of-3 historical consensus"
            />
            <SummaryCard
              title="Green → Built-up · strict"
              value={`${number(data.analysis.strict_green_to_built_area_km2, 4)} km²`}
              description="Conservative 3-of-3 confirmation subset"
            />
            <SummaryCard
              title="Robust-only Green → Built-up"
              value={`${number(greenBuiltAgreement?.robust_only_area_km2, 4)} km²`}
              description="Difference in consensus agreement, not an error margin"
            />
            <SummaryCard
              title="Strict share of robust conversion"
              value={`${number(greenBuiltAgreement?.strict_confirmation_percent_of_robust, 2)}%`}
              description="Overlap of strict and robust detected conversion pixels"
            />
          </div>

          {/* ================================================= */}
          {/* 4B. MAIN GIS MAP + BREAKDOWN                     */}
          {/* ================================================= */}

          <div className="mt-5 grid gap-5 2xl:grid-cols-[1.6fr_.9fr]">
            <section className="card-pad min-w-0">
              <div>
                <h2 className="section-title">Historical transition GIS map</h2>
                <p className="mt-1 text-xs leading-5 text-slate-500">
                  Official research export · Early period {data.analysis.early_window} ·
                  Late period {data.analysis.late_window}
                </p>
              </div>

              <div className="mt-4 grid gap-2 sm:grid-cols-2">
                {MAP_OPTIONS.map((option) => (
                  <button
                    key={option.id}
                    type="button"
                    onClick={() => chooseMap(option.id)}
                    aria-pressed={mapId === option.id}
                    className={`rounded-xl border px-3 py-3 text-left transition ${
                      mapId === option.id
                        ? 'border-emerald-700 bg-emerald-50 text-emerald-900'
                        : 'border-slate-200 bg-white text-slate-700 hover:bg-slate-50'
                    }`}
                  >
                    <span className="block text-xs font-bold">{option.label}</span>
                    <span className="mt-1 block text-[11px] leading-4 opacity-80">
                      {option.help}
                    </span>
                  </button>
                ))}
              </div>

              <div className="mt-3 flex flex-wrap items-center justify-between gap-3 text-xs text-slate-600">
                <div className="flex flex-wrap gap-3">
                  <label className="inline-flex items-center gap-1.5">
                    <input
                      type="checkbox"
                      checked={showStreets}
                      onChange={(event) => setShowStreets(event.target.checked)}
                    />
                    Streets
                  </label>
                  <label className="inline-flex items-center gap-1.5">
                    <input
                      type="checkbox"
                      checked={showBoundary}
                      onChange={(event) => setShowBoundary(event.target.checked)}
                    />
                    KMC boundary
                  </label>
                </div>

                <label className="inline-flex items-center gap-2">
                  Overlay opacity
                  <input
                    type="range"
                    min="0.5"
                    max="1"
                    step="0.05"
                    value={opacity}
                    onChange={(event) => setOpacity(Number(event.target.value))}
                  />
                  {Math.round(opacity * 100)}%
                </label>
              </div>

              {activeMap && isValidBounds(activeMap.bounds) ? (
                <TransitionMap
                  record={activeMap}
                  boundary={boundary}
                  showStreets={showStreets}
                  showBoundary={showBoundary}
                  opacity={opacity}
                  onImageError={() => setImageError(
                    'The selected transition image could not be loaded. Check the FastAPI image endpoint.'
                  )}
                />
              ) : (
                <div className="mt-3 rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">
                  This transition map is unavailable. No demonstration image is substituted.
                </div>
              )}

              {imageError && (
                <p className="mt-2 text-xs text-red-700">{imageError}</p>
              )}

              {/* Legend belongs to the current map, not hardcoded colors. */}
              <div className="mt-4">
                <p className="text-xs font-semibold text-slate-800">
                  {activeMapOption?.label} · original map legend
                </p>
                <div className="mt-2 grid gap-2 sm:grid-cols-2">
                  {(activeMap?.legend || []).map((entry) => (
                    <div
                      key={`${activeMap.id}-${entry.code}`}
                      className="flex items-center gap-2 text-[11px] text-slate-700"
                    >
                      <span
                        className="h-3.5 w-3.5 shrink-0 rounded-sm border border-slate-300"
                        style={{ backgroundColor: rgbaCss(entry.rgba) }}
                      />
                      <span>{entry.label}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* <p className="mt-3 text-[11px] leading-5 text-slate-500">
                The image represents classified 30 m historical transition pixels,
                not 100 m future-pressure grid predictions. Transparent areas represent
                no-data or unsupported classes. Switching map views preserves the
                current center and zoom; use + / − to adjust manually.
              </p> */}
            </section>

            {/* Transition-class breakdown, controlled separately from map. */}
            <div className="space-y-5">
              <section className="card-pad">
                <h2 className="section-title">Transition class breakdown</h2>
                <p className="mt-1 text-xs leading-5 text-slate-500">
                  Areas calculated from resolved historical consensus pixels.
                </p>

                <label className="mt-4 block text-xs font-semibold text-slate-600">
                  Compare consensus method
                  <select
                    className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900"
                    value={breakdownMethod}
                    onChange={(event) => setBreakdownMethod(event.target.value)}
                  >
                    <option value={ROBUST}>Robust · 2 of 3 years</option>
                    <option value={STRICT}>Strict · 3 of 3 years</option>
                  </select>
                </label>

                <div className="mt-3 max-h-[450px] space-y-2 overflow-y-auto pr-1">
                  {breakdownRows.map((row) => (
                    <div
                      key={`${row.consensus_method}-${row.green_transition_code}`}
                      className="rounded-lg border border-slate-100 bg-slate-50 px-3 py-2"
                    >
                      <div className="flex items-start justify-between gap-2">
                        <p className="text-xs font-medium text-slate-700">
                          {row.green_transition_name}
                        </p>
                        <p className="shrink-0 text-xs font-bold tabular-nums text-slate-900">
                          {number(row.area_km2, 4)} km²
                        </p>
                      </div>
                      <p className="mt-1 text-[11px] text-slate-500">
                        {number(row.percent_of_resolved_transition_area, 2)}%
                        {' '}of resolved area · {integer(row.pixels)} pixels
                      </p>
                    </div>
                  ))}

                  {!breakdownRows.length && (
                    <p className="text-xs text-amber-800">
                      No breakdown records returned for this consensus method.
                    </p>
                  )}
                </div>
              </section>

              <section className="card-pad">
                <h2 className="section-title">Selected consensus summary</h2>
                <p className="mt-1 text-xs text-slate-500">
                  {breakdownMethod === ROBUST ? 'Primary robust' : 'Strict sensitivity'} method
                </p>
                <dl className="mt-4 space-y-3 text-xs">
                  {[
                    ['Resolved transition pixels', integer(selectedComparison?.resolved_transition_pixels)],
                    ['Green retained', `${number(selectedComparison?.green_retained_area_km2, 4)} km²`],
                    ['Total green loss', `${number(selectedComparison?.total_green_loss_area_km2, 4)} km²`],
                    ['Total green gain', `${number(selectedComparison?.total_green_gain_area_km2, 4)} km²`],
                    ['Net consensus green change', `${signed(selectedComparison?.net_consensus_green_change_area_km2, 4)} km²`],
                  ].map(([label, value]) => (
                    <div key={label} className="flex justify-between gap-4 border-b border-slate-100 pb-2">
                      <dt className="text-slate-600">{label}</dt>
                      <dd className="shrink-0 font-semibold tabular-nums text-slate-900">{value}</dd>
                    </div>
                  ))}
                </dl>
                <p className="mt-3 text-[11px] leading-5 text-slate-500">
                  Net consensus change compares two multi-year periods; it is not
                  the same statistic as the 2011–2024 annual trend on Green-Cover Analysis.
                </p>
              </section>
            </div>
          </div>

          {/* ================================================= */}
          {/* 4C. ROBUST VS STRICT COMPARISON CHART            */}
          {/* ================================================= */}

          <section className="card-pad mt-5">
            <h2 className="section-title">Robust versus strict transition areas</h2>
            <p className="mt-1 text-xs text-slate-500">
              Comparison of modelled historical consensus transition areas (km²).
            </p>

            <div className="mt-4">
              <ResponsiveContainer width="100%" height={335}>
                <BarChart
                  data={comparisonChart}
                  layout="vertical"
                  margin={{ top: 8, right: 22, bottom: 4, left: 12 }}
                >
                  <CartesianGrid strokeDasharray="4 5" />
                  <XAxis
                    type="number"
                    tick={{ fontSize: 11 }}
                    tickFormatter={(value) => `${number(value, 0)}`}
                    label={{ value: 'Area (km²)', position: 'insideBottomRight', offset: -3 }}
                  />
                  <YAxis
                    dataKey="category"
                    type="category"
                    width={125}
                    tick={{ fontSize: 11 }}
                  />
                  <Tooltip
                    formatter={(value, name) => [`${number(value, 4)} km²`, name]}
                  />
                  <Legend />
                  <Bar
                    dataKey="robust"
                    name="Robust · 2-of-3"
                    fill="#15803d"
                    isAnimationActive={false}
                  />
                  <Bar
                    dataKey="strict"
                    name="Strict · 3-of-3"
                    fill="#2563eb"
                    isAnimationActive={false}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <p className="mt-2 text-[11px] leading-5 text-slate-500">
              Strict consensus requires stronger repeat-year agreement and resolves
              fewer pixels. The two methods have different resolved-area coverage.
              Their difference must not be reported as a statistical confidence interval.
            </p>
          </section>

          {/* ================================================= */}
          {/* 4D. METHODOLOGY AND QA                            */}
          {/* ================================================= */}

          <section className="card-pad mt-5">
            <div className="flex items-center gap-2">
              <Info size={18} className="text-emerald-700" />
              <h2 className="section-title">Interpretation, method and QA</h2>
            </div>

            <div className="mt-4 grid gap-3 md:grid-cols-3">
              <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
                <p className="text-xs font-bold text-slate-800">Robust · primary</p>
                <p className="mt-2 text-xs leading-5 text-slate-600">
                  {data.method.primary_rule_reason}
                </p>
              </div>
              <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
                <p className="text-xs font-bold text-slate-800">Strict · sensitivity</p>
                <p className="mt-2 text-xs leading-5 text-slate-600">
                  {data.method.strict_rule_role}
                  {' '}This is a stricter multi-year agreement subset.
                </p>
              </div>
              <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
                <p className="text-xs font-bold text-slate-800">Research data health</p>
                <p className="mt-2 text-xs leading-5 text-slate-600">
                  Historical transition QA: {data.analysis.overall_step10b_status}
                  <br />
                  Available historical GIS overlays: {integer(data.health?.overlay_count)}
                  <br />
                  Original research transition method locked: {data.method.transition_method_locked ? 'Yes' : 'Not confirmed'}
                </p>
              </div>
            </div>

            {/* <div className="mt-4 rounded-xl border border-amber-200 bg-amber-50 p-4 text-xs leading-6 text-amber-900">
              <strong>Research limitations:</strong> The maps show retrospective
              land-cover changes derived from modelled LULC, not field-verified
              conversion for every pixel. The early and late periods are
              2012–2014 and 2021–2023, not annual forecasts. The robust-only
              and strict-confirmed maps express agreement between the two
              consensus rules, not a quantified probability, statistical confidence
              interval, or new model accuracy estimate. No future-pressure model
              was retrained for this page.
            </div> */}
          </section>
        </>
      )}
    </div>
  )
}
