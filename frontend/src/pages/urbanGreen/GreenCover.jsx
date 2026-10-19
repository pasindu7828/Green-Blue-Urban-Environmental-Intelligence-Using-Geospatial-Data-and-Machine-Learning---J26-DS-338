
import { useEffect, useMemo, useRef, useState } from 'react'
import {
  ImageOverlay,
  GeoJSON,
  MapContainer,
  TileLayer,
  ZoomControl,
  useMap,
} from 'react-leaflet'
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import {
  Building2,
  Leaf,
  RefreshCcw,
  Info,
} from 'lucide-react'

import PageHeader from '../../components/shared/PageHeader'
import MetricCard from '../../components/shared/MetricCard'
import { urbanGreenApi } from '../../features/urbanGreen/services/urbanGreenApi'

// =====================================================
// 1. RESEARCH CONFIGURATION
// =====================================================

const CORE_START = 2011
const CORE_END = 2024

const YEARS = Array.from(
  { length: 17 },
  (_, i) => 2010 + i
)

const CLASSES = [
  {
    id: 1,
    label: 'Green / Vegetation',
    color: '#238b45',
  },
  {
    id: 2,
    label: 'Built-up',
    color: '#d65b52',
  },
  {
    id: 3,
    label: 'Open / Bare',
    color: '#d9a35d',
  },
  {
    id: 4,
    label: 'Water',
    color: '#347fc2',
  },
]

// =====================================================
// 2. FORMATTING AND VALIDATION
// =====================================================

function format(value, places = 2) {
  const n = Number(value)

  return value == null || !Number.isFinite(n)
    ? 'N/A'
    : n.toFixed(places)
}

function signed(value, places = 2) {
  if (
    value == null ||
    !Number.isFinite(Number(value))
  ) {
    return 'N/A'
  }

  const n = Number(value)

  return `${n > 0 ? '+' : ''}${n.toFixed(places)}`
}

function isCoreYear(year) {
  return year >= CORE_START && year <= CORE_END
}

function isStabilized(productType) {
  return String(productType ?? '')
    .toUpperCase()
    .includes('STABIL')
}

function matchesProduct(productType, selectedProduct) {
  return selectedProduct === 'stabilized'
    ? isStabilized(productType)
    : String(productType ?? '').toUpperCase() === 'RAW_RF'
}

function validBounds(bounds) {
  return (
    Array.isArray(bounds) &&
    bounds.length === 2 &&
    bounds.every(
      (corner) =>
        Array.isArray(corner) &&
        corner.length === 2 &&
        corner.every(
          (coord) =>
            typeof coord === 'number' &&
            Number.isFinite(coord)
        )
    )
  )
}

// =====================================================
// 3. FIT MAP ONLY WHEN INITIALLY LOADED
// =====================================================

function InitialMapFit({ bounds }) {
  const map = useMap()
  const fitted = useRef(false)

  useEffect(() => {
    if (!fitted.current && validBounds(bounds)) {
      map.fitBounds(bounds, {
        padding: [18, 18],
        animate: false,
      })

      fitted.current = true
    }
  }, [bounds, map])

  return null
}

// =====================================================
// 4. HISTORICAL LEAFLET MAP
// =====================================================

function HistoricalMap({
  mapRecord,
  boundary,
  showStreets,
  showBoundary,
  opacity,
  onImageError,
}) {
  const imageUrl =
    urbanGreenApi.historical.mapImageUrl(mapRecord.id)

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

        {/* Actual classified LULC image overlay */}
        <ImageOverlay
          key={mapRecord.id}
          url={imageUrl}
          bounds={mapRecord.bounds}
          opacity={opacity}
          zIndex={10}
          eventHandlers={{
            error: onImageError,
          }}
        />

        {/* Official study-area boundary */}
        {showBoundary &&
          boundary?.features?.length > 0 && (
            <GeoJSON
              data={boundary}
              style={{
                color: '#102f2a',
                weight: 3,
                opacity: 0.95,
                fillOpacity: 0,
              }}
              interactive={false}
            />
          )}

        <ZoomControl position="topleft" />

        <InitialMapFit bounds={mapRecord.bounds} />
      </MapContainer>

      <div className="pointer-events-none absolute right-3 top-3 z-[450] rounded-lg border border-slate-200 bg-white/95 px-3 py-2 text-[11px] font-semibold text-slate-700 shadow-sm">
        {mapRecord.year} ·{' '}
        {isStabilized(mapRecord.product_type)
          ? 'Stabilized LULC'
          : 'Raw RF LULC'}
      </div>
    </div>
  )
}

// =====================================================
// 5. ANNUAL CLASS STATISTICS COMPONENT
// =====================================================

function ClassRow({ item, data }) {
  const percent = data?.class_percent

  return (
    <div className="rounded-xl border border-slate-100 bg-slate-50 px-3 py-3">
      <div className="flex items-center justify-between gap-3 text-xs">
        <span className="flex items-center gap-2 font-semibold text-slate-700">
          <span
            className="h-3 w-3 rounded-sm"
            style={{
              backgroundColor: item.color,
            }}
          />

          {item.label}
        </span>

        <span className="font-bold tabular-nums text-slate-900">
          {format(percent)}%
        </span>
      </div>

      <div className="mt-2 h-2 overflow-hidden rounded-full bg-slate-200">
        <div
          className="h-full rounded-full"
          style={{
            width: `${
              Math.max(
                0,
                Math.min(100, Number(percent) || 0)
              )
            }%`,
            backgroundColor: item.color,
          }}
        />
      </div>

      <p className="mt-1 text-[11px] text-slate-500">
        Classified area: {format(data?.class_area_km2, 3)} km²
      </p>
    </div>
  )
}

// =====================================================
// 6. MAIN GREEN-COVER ANALYSIS PAGE
// =====================================================

export default function GreenCover() {
  const [refreshToken, setRefreshToken] = useState(0)

  const [dataset, setDataset] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const [year, setYear] = useState(2024)

  const [preferredProduct, setPreferredProduct] =
    useState('stabilized')

  const [showStreets, setShowStreets] = useState(true)
  const [showBoundary, setShowBoundary] = useState(true)
  const [opacity, setOpacity] = useState(0.94)

  const [boundary, setBoundary] = useState(null)
  const [imageError, setImageError] = useState('')

  // ---------------------------------------------------
  // 6A. LOAD REAL HISTORICAL RESEARCH DATA
  // ---------------------------------------------------

  useEffect(() => {
    const controller = new AbortController()

    setLoading(true)
    setError('')

    Promise.all([
      urbanGreenApi.historical.summary(controller.signal),
      urbanGreenApi.historical.trajectory(controller.signal),
      urbanGreenApi.historical.classAreas(controller.signal),
      urbanGreenApi.historical.annualQa(controller.signal),
      urbanGreenApi.historical.maps(controller.signal),
    ])
      .then(([
        summary,
        trajectory,
        classAreas,
        annualQa,
        maps,
      ]) => {
        if (controller.signal.aborted) return

        if (
          !Array.isArray(trajectory?.data) ||
          trajectory.data.length !== 17
        ) {
          throw new Error(
            'Historical trajectory must contain the 17 research years.'
          )
        }

        if (
          !Array.isArray(classAreas?.data) ||
          !Array.isArray(maps?.maps)
        ) {
          throw new Error(
            'Historical data or map catalogue has an unexpected format.'
          )
        }

        const annual = maps.maps.filter(
          (record) => record.kind === 'annual_lulc'
        )

        if (annual.length !== 34) {
          throw new Error(
            `Expected 34 annual LULC maps, received ${annual.length}.`
          )
        }

        if (
          annualQa?.overall_step09_status !== 'PASS'
        ) {
          throw new Error(
            'Historical research map QA has not reported PASS.'
          )
        }

        setDataset({
          summary,
          trajectory,
          classAreas,
          annualQa,
          annual,
        })
      })
      .catch((reason) => {
        if (!controller.signal.aborted) {
          setError(reason.message || String(reason))
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setLoading(false)
        }
      })

    return () => controller.abort()
  }, [refreshToken])

  // ---------------------------------------------------
  // 6B. LOAD KMC BOUNDARY
  // ---------------------------------------------------

  useEffect(() => {
    const controller = new AbortController()

    const base = import.meta.env.BASE_URL || '/'

    const url =
      `${base.endsWith('/') ? base : `${base}/`}` +
      'data/kaduwela-boundary.geojson'

    fetch(url, {
      signal: controller.signal,
    })
      .then((response) =>
        response.ok ? response.json() : null
      )
      .then((data) => {
        if (!controller.signal.aborted) {
          setBoundary(data)
        }
      })
      .catch(() => {})

    return () => controller.abort()
  }, [])

  // ---------------------------------------------------
  // 6C. RESOLVE MAP PRODUCT FOR SELECTED YEAR
  // ---------------------------------------------------

  // Core 2011-2024: raw or stabilized selectable.
  // Edge/context years: raw only in this page.

  const activeProduct = isCoreYear(year)
    ? preferredProduct
    : 'raw'

  const activeMap = useMemo(
    () =>
      dataset?.annual.find(
        (record) =>
          Number(record.year) === year &&
          matchesProduct(
            record.product_type,
            activeProduct
          )
      ) || null,
    [dataset, year, activeProduct]
  )

  // ---------------------------------------------------
  // 6D. MATCH STATISTICS WITH MAP PRODUCT
  // ---------------------------------------------------

  const annualClasses = useMemo(() => {
    const matches = (
      dataset?.classAreas?.data || []
    ).filter(
      (row) =>
        Number(row.year) === year &&
        matchesProduct(
          row.product_type,
          activeProduct
        )
    )

    return CLASSES.map((item) => ({
      ...item,
      data:
        matches.find(
          (row) =>
            Number(row.class_id) === item.id
        ) || null,
    }))
  }, [dataset, year, activeProduct])

  // ---------------------------------------------------
  // 6E. HISTORICAL CORE TREND
  // ---------------------------------------------------

  const trend = useMemo(
    () =>
      (dataset?.trajectory?.data || [])
        .filter(
          (row) =>
            row.included_in_core_trend === true
        )
        .map((row) => ({
          year: Number(row.year),
          green: row.green_percent,
          built: row.built_percent,
        }))
        .sort((a, b) => a.year - b.year),
    [dataset]
  )

  const summary = dataset?.summary

  const readyClasses = annualClasses.every(
    (row) => row.data
  )

  const contextYear = !isCoreYear(year)

  useEffect(() => {
    setImageError('')
  }, [activeMap?.id])

  const retry = () =>
    setRefreshToken((n) => n + 1)

  // ===================================================
  // 7. PAGE RENDERING
  // ===================================================

  return (
    <div>
      <PageHeader
        eyebrow="Urban Green Intelligence"
        title="Historical Green-Cover Analysis"
        subtitle="Explore classified land-cover maps, annual green and built-up patterns, and the stabilized historical trend for Kaduwela."
        badges={[
          'Historical core: 2011–2024',
          '30 m classified LULC',
          'Real GIS research outputs',
        ]}
        actions={
          <button
            type="button"
            className="btn-secondary"
            onClick={retry}
          >
            <RefreshCcw size={15} />
            Refresh
          </button>
        }
      />

      {/* LOADING */}

      {loading && (
        <div className="card-pad mt-5 text-sm text-slate-600">
          Loading historical research maps and land-cover
          statistics...
        </div>
      )}

      {/* ERROR */}

      {error && !loading && (
        <section className="card-pad mt-5">
          <h2 className="section-title">
            Historical data unavailable
          </h2>

          <p className="mt-3 text-sm text-red-700">
            {error}
          </p>

          <p className="mt-2 text-xs text-slate-500">
            Make sure your Urban Green FastAPI backend
            is running.
          </p>

          <button
            type="button"
            onClick={retry}
            className="btn-primary mt-4"
          >
            Retry
          </button>
        </section>
      )}

      {/* MAIN PAGE */}

      {dataset && !loading && !error && (
        <>
          {/* HISTORICAL SUMMARY CARDS */}

          <div className="mt-5 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <MetricCard
              title="Green cover · 2011"
              value={`${format(summary?.green_start_percent)}%`}
              subtitle="Core stabilized starting year"
              icon={Leaf}
              tone="green"
            />

            <MetricCard
              title="Green cover · 2024"
              value={`${format(summary?.green_end_percent)}%`}
              subtitle="Core stabilized ending year"
              icon={Leaf}
              tone="green"
            />

            <MetricCard
              title="Net green-cover change"
              value={`${signed(
                summary?.green_net_change_percent_points
              )} pp`}
              subtitle={`${signed(
                summary?.green_net_area_change_km2,
                3
              )} km² · 2011–2024`}
              icon={Leaf}
              tone="teal"
            />

            <MetricCard
              title="Net built-up change"
              value={`${signed(
                summary?.built_net_change_percent_points
              )} pp`}
              subtitle={`${signed(
                summary?.built_net_area_change_km2,
                3
              )} km² · 2011–2024`}
              icon={Building2}
              tone="orange"
            />
          </div>

          {/* MAP AND ANNUAL STATISTICS */}

          <div className="mt-5 grid gap-5 2xl:grid-cols-[1.6fr_.9fr]">
            <section className="card-pad min-w-0">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <h2 className="section-title">
                    Annual land-cover map
                  </h2>

                  {/* <p className="mt-1 text-xs text-slate-500">
                    Actual Landsat-derived, Random Forest
                    classified LULC raster visualization.
                  </p> */}
                </div>

                <span className="pill">
                  {year} ·{' '}
                  {activeProduct === 'stabilized'
                    ? 'Stabilized'
                    : 'Raw RF'}
                </span>
              </div>

              {/* YEAR AND PRODUCT SELECTORS */}

              <div className="mt-4 grid gap-3 sm:grid-cols-2">
                <label className="text-xs font-semibold text-slate-600">
                  Select year

                  <select
                    className="control mt-1"
                    value={year}
                    onChange={(event) =>
                      setYear(Number(event.target.value))
                    }
                  >
                    {YEARS.map((value) => (
                      <option
                        key={value}
                        value={value}
                      >
                        {value}
                        {value === 2026
                          ? ' · provisional'
                          : !isCoreYear(value)
                            ? ' · raw context'
                            : ''}
                      </option>
                    ))}
                  </select>
                </label>

                <label className="text-xs font-semibold text-slate-600">
                  Map product

                  <select
                    className="control mt-1"
                    value={activeProduct}
                    disabled={contextYear}
                    onChange={(event) =>
                      setPreferredProduct(event.target.value)
                    }
                  >
                    <option value="stabilized">
                      Stabilized analytical LULC
                    </option>

                    <option value="raw">
                      Raw Random Forest LULC
                    </option>
                  </select>
                </label>
              </div>

              {/* RESEARCH WARNINGS */}

              {contextYear && (
                <div className="mt-3 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs leading-5 text-amber-900">
                  {year === 2026
                    ? '2026 is a partial/provisional observation year. It is excluded from the 2011–2024 core trend and must not be interpreted as a comparable full-year trend value.'
                    : `${year} is provided as raw historical context, outside the 2011–2024 stabilized trend. Do not compare its raw percentages directly against the stabilized core values.`}
                </div>
              )}

              {!contextYear &&
                activeProduct === 'raw' && (
                  <p className="mt-3 rounded-lg border border-blue-200 bg-blue-50 px-3 py-2 text-xs leading-5 text-blue-900">
                    Raw RF mode shows the original annual
                    classification. The headline trend
                    cards and historical trend chart remain
                    based on the separate stabilized
                    analytical series.
                  </p>
                )}

              {/* MAP OPTIONS */}

              <div className="my-3 flex flex-wrap items-center justify-between gap-3 text-xs text-slate-600">
                <div className="flex flex-wrap gap-3">
                  <label className="flex items-center gap-1.5">
                    <input
                      type="checkbox"
                      checked={showStreets}
                      onChange={(e) =>
                        setShowStreets(e.target.checked)
                      }
                    />
                    Streets
                  </label>

                  <label className="flex items-center gap-1.5">
                    <input
                      type="checkbox"
                      checked={showBoundary}
                      onChange={(e) =>
                        setShowBoundary(e.target.checked)
                      }
                    />
                    KMC boundary
                  </label>
                </div>

                <label className="flex items-center gap-2">
                  LULC opacity

                  <input
                    type="range"
                    min="0.5"
                    max="1"
                    step="0.05"
                    value={opacity}
                    onChange={(e) =>
                      setOpacity(Number(e.target.value))
                    }
                  />

                  {Math.round(opacity * 100)}%
                </label>
              </div>

              {/* REAL HISTORICAL GIS MAP */}

              {activeMap &&
              validBounds(activeMap.bounds) ? (
                <HistoricalMap
                  mapRecord={activeMap}
                  boundary={boundary}
                  showStreets={showStreets}
                  showBoundary={showBoundary}
                  opacity={opacity}
                  onImageError={() =>
                    setImageError(
                      'The historical map image could not be loaded. Check the FastAPI image endpoint.'
                    )
                  }
                />
              ) : (
                <div className="rounded-xl border border-amber-200 bg-amber-50 p-5 text-sm text-amber-900">
                  The selected historical map or its
                  geographic bounds are unavailable.
                  No substitute map is displayed.
                </div>
              )}

              {imageError && (
                <p className="mt-2 text-xs text-red-700">
                  {imageError}
                </p>
              )}

              {/* MAP LEGEND */}

              <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-2 text-[11px] text-slate-700">
                {CLASSES.map((item) => (
                  <span
                    key={item.id}
                    className="inline-flex items-center gap-1.5"
                  >
                    <span
                      className="h-3 w-3 rounded-sm"
                      style={{
                        backgroundColor: item.color,
                      }}
                    />

                    {item.label}
                  </span>
                ))}
              </div>

              {/* <p className="mt-3 text-[11px] leading-5 text-slate-500">
                The colored layer is an actual 30 m
                raster visualization, not a 100 m
                planning grid or a new prediction.
                Transparent pixels represent no-data
                areas; the map uses nearest-neighbor
                image resampling. Switching years
                preserves your current zoom and map
                position; use + / − to adjust manually.
              </p> */}
            </section>

            {/* ANNUAL COMPOSITION SIDEBAR */}

            <div className="space-y-5">
              <section className="card-pad">
                <h2 className="section-title">
                  Land-cover composition · {year}
                </h2>

                <p className="mt-1 text-xs text-slate-500">
                  {activeProduct === 'stabilized'
                    ? 'Stabilized retrospective classification'
                    : 'Raw RF classification'}
                  {' '}· share of valid classified pixels
                </p>

                {readyClasses ? (
                  <div className="mt-4 space-y-3">
                    {annualClasses.map((row) => (
                      <ClassRow
                        key={row.id}
                        item={row}
                        data={row.data}
                      />
                    ))}

                    <div className="rounded-lg bg-emerald-50 px-3 py-2 text-xs text-emerald-900">
                      Classified pixels:{' '}
                      {Number(
                        annualClasses[0]?.data
                          ?.valid_pixels || 0
                      ).toLocaleString()}
                    </div>
                  </div>
                ) : (
                  <p className="mt-4 text-sm text-amber-800">
                    Matching class-area statistics are
                    not available for this year and
                    map product.
                  </p>
                )}
              </section>

              {/* INTERPRETATION */}

              <section className="card-pad">
                <h2 className="section-title">
                  Map interpretation
                </h2>

                <div className="mt-3 space-y-2 text-xs leading-5 text-slate-600">
                  <p>
                    <strong>Green:</strong> vegetation
                    and green-cover classification.
                  </p>

                  <p>
                    <strong>Built-up:</strong> classified
                    developed/built surfaces.
                  </p>

                  <p>
                    <strong>Open / Bare:</strong> exposed
                    or sparsely vegetated surfaces.
                  </p>

                  <p>
                    <strong>Water:</strong> classified
                    water surfaces.
                  </p>

                  <p>
                    <strong>Core historical period:</strong>
                    {' '}2011–2024 stabilized retrospective
                    analysis.
                  </p>

                  <p>
                    <strong>Selected map:</strong>{' '}
                    {activeProduct === 'stabilized'
                      ? 'stabilized analytical'
                      : 'raw RF'}.
                  </p>
                </div>
              </section>
            </div>
          </div>

          {/* HISTORICAL TREND CHART */}

          <section className="card-pad mt-5">
            <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
              <div>
                <h2 className="section-title">
                  Historical green and built-up trends
                </h2>

                <p className="mt-1 text-xs text-slate-500">
                  Stabilized retrospective LULC ·
                  2011–2024 only
                </p>
              </div>

              <span className="pill">
                {trend.length} core years
              </span>
            </div>

            <ResponsiveContainer
              width="100%"
              height={315}
            >
              <LineChart
                data={trend}
                margin={{
                  top: 8,
                  right: 15,
                  left: 0,
                  bottom: 6,
                }}
              >
                <CartesianGrid strokeDasharray="4 5" />

                <XAxis
                  dataKey="year"
                  type="category"
                  tick={{ fontSize: 11 }}
                />

                <YAxis
                  domain={[0, 100]}
                  tickFormatter={(v) => `${v}%`}
                  tick={{ fontSize: 11 }}
                />

                <Tooltip
                  formatter={(value, name) => [
                    `${format(value)}%`,
                    name,
                  ]}
                />

                <Legend />

                <Line
                  type="monotone"
                  dataKey="green"
                  name="Green cover"
                  stroke="#238b45"
                  strokeWidth={3}
                  dot={{ r: 2 }}
                  isAnimationActive={false}
                />

                <Line
                  type="monotone"
                  dataKey="built"
                  name="Built-up"
                  stroke="#d65b52"
                  strokeWidth={3}
                  dot={{ r: 2 }}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>

            {/* <p className="mt-2 text-[11px] leading-5 text-slate-500">
              These percentages describe classified
              LULC pixels, not direct field observations
              or independently verified ground truth.
              Net changes are percentage-point
              differences, not relative percent growth.
              Raw edge years and provisional 2026 are
              deliberately excluded.
            </p> */}
          </section>

          {/* DATA QUALITY AND RESEARCH LIMITATIONS */}

          <section className="card-pad mt-5">
            <div className="flex items-center gap-2">
              <Info
                size={18}
                className="text-emerald-700"
              />

              <h2 className="section-title">
                Research quality and scope
              </h2>
            </div>

            <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <div className="rounded-xl bg-slate-50 p-3 text-sm">
                <p className="text-xs text-slate-500">
                  Raw annual maps
                </p>

                <p className="mt-1 font-bold">
                  {dataset.annualQa.raw_map_count}
                </p>
              </div>

              <div className="rounded-xl bg-slate-50 p-3 text-sm">
                <p className="text-xs text-slate-500">
                  Stabilized annual maps
                </p>

                <p className="mt-1 font-bold">
                  {dataset.annualQa.stabilized_map_count}
                </p>
              </div>

              <div className="rounded-xl bg-slate-50 p-3 text-sm">
                <p className="text-xs text-slate-500">
                  Research raster resolution
                </p>

                <p className="mt-1 font-bold">
                  30 m
                </p>
              </div>

              <div className="rounded-xl bg-slate-50 p-3 text-sm">
                <p className="text-xs text-slate-500">
                  Annual GIS QA
                </p>

                <p className="mt-1 font-bold">
                  {dataset.annualQa.overall_step09_status}
                </p>
              </div>
            </div>

            {/* <p className="mt-3 text-xs leading-6 text-slate-600">
              The stabilized LULC series is for
              retrospective historical analysis only;
              it uses surrounding years and must not
              be treated as a causal predictor for the
              future-pressure model. In particular,
              2025 remains available as raw context
              and a pressure-model input, while the
              partial 2026 classification remains
              provisional. This page does not establish
              new independent annual classification
              accuracy.
            </p> */}
          </section>
        </>
      )}
    </div>
  )
}
