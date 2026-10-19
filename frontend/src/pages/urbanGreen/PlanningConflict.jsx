
import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
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
  AlertTriangle,
  Grid3X3,
  MapPinned,
  RefreshCcw,
  Search,
  ShieldCheck,
} from 'lucide-react'

import PageHeader from '../../components/shared/PageHeader'
import MetricCard from '../../components/shared/MetricCard'
import GreenGridMap, {
  GREEN_MAP_LAYERS,
  validScore,
} from '../../features/urbanGreen/components/GreenGridMap'
import { urbanGreenApi } from '../../features/urbanGreen/services/urbanGreenApi'
import { useUrbanGreen } from '../../features/urbanGreen/context/UrbanGreenContext'

// =====================================================
// 1. RESEARCH CONFIGURATION
// =====================================================

const MAP_OPTIONS = [
  'conflict',
  'pressure',
  'suitability',
  'conservation',
]

const TOTAL_GRIDS = 9240

function number(value, decimals = 3) {
  if (!validScore(value)) return 'Unavailable'

  return value.toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  })
}

function integer(value) {
  if (!validScore(value)) return 'Unavailable'

  return value.toLocaleString('en-US', {
    maximumFractionDigits: 0,
  })
}

function readable(value) {
  if (value === null || value === undefined || value === '') {
    return 'Unavailable'
  }

  return String(value)
    .replaceAll('_', ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

function flag(value) {
  return value === true ||
    value === 1 ||
    value === '1' ||
    String(value).toLowerCase() === 'true'
}

function EvidenceRow({ label, value }) {
  return (
    <div className="flex items-start justify-between gap-4 border-b border-slate-100 py-2.5 last:border-0">
      <dt className="text-xs leading-5 text-slate-500">
        {label}
      </dt>

      <dd className="max-w-[62%] break-words text-right text-xs font-semibold leading-5 text-slate-800">
        {value ?? 'Unavailable'}
      </dd>
    </div>
  )
}

function SectionTitle({ title, description }) {
  return (
    <div className="mb-4">
      <h2 className="section-title">{title}</h2>

      <p className="mt-1 text-xs leading-5 text-slate-500">
        {description}
      </p>
    </div>
  )
}

// Find the real grid record in a possible API wrapper.
// Never calculate or invent planning recommendations.
function extractGrid(response) {
  const candidates = [
    response?.data?.grid,
    response?.data?.record,
    response?.grid,
    response?.record,
    response?.data,
    response,
  ]

  for (const candidate of candidates) {
    if (!candidate ||
        typeof candidate !== 'object' ||
        Array.isArray(candidate)) {
      continue
    }

    const record = candidate.properties ?? candidate

    if (
      record &&
      typeof record === 'object' &&
      !Array.isArray(record) &&
      (
        record.grid_id ||
        record.review_rule_id ||
        record.review_action
      )
    ) {
      return record
    }
  }

  return null
}

// Numeric intervals for describing the index distribution.
// These are not policy classes or intervention thresholds.
function histogram(features, field) {
  const bins = Array.from(
    { length: 10 },
    (_, i) => ({
      interval:
        `${(i / 10).toFixed(1)}–${((i + 1) / 10).toFixed(1)}`,
      count: 0,
    })
  )

  for (const feature of features) {
    const value = feature.properties?.[field]

    if (!validScore(value)) continue

    const index = Math.min(
      9,
      Math.max(0, Math.floor(value * 10))
    )

    bins[index].count += 1
  }

  return bins
}

// =====================================================
// 2. MAIN PAGE
// =====================================================

export default function PlanningConflict() {
  const {
    selectedGridId,
    setSelectedGridId,
    selectedGnId,
    selectGn,
  } = useUrbanGreen()

  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [refresh, setRefresh] = useState(0)

  const [mapMetric, setMapMetric] =
    useState('conflict')

  const [gridDetails, setGridDetails] = useState(null)
  const [gridError, setGridError] = useState('')
  const [gridLoading, setGridLoading] = useState(false)

  const [searchGrid, setSearchGrid] = useState('')
  const [searchError, setSearchError] = useState('')

  // ---------------------------------------------------
  // 2A. LOAD FROZEN RESEARCH RESULTS
  // ---------------------------------------------------

  useEffect(() => {
    const controller = new AbortController()

    setLoading(true)
    setError('')

    Promise.all([
      urbanGreenApi.planning.map(
        controller.signal
      ),

      urbanGreenApi.planning.gnMap(
        controller.signal
      ),

      urbanGreenApi.planning.gnDivisions(
        controller.signal
      ),

      urbanGreenApi.planning.hotspots(
        {},
        controller.signal
      ),

      urbanGreenApi.pressure.overview(
        controller.signal
      ),

      urbanGreenApi.conservation.overview(
        controller.signal
      ),
    ])
      .then(([
        planningMap,
        gnMap,
        gnDivisions,
        hotspots,
        pressure,
        conservation,
      ]) => {
        if (controller.signal.aborted) return

        if (
          !Array.isArray(planningMap?.features) ||
          planningMap.features.length !== TOTAL_GRIDS ||
          !Array.isArray(gnMap?.features) ||
          !Array.isArray(gnDivisions?.data) ||
          !Array.isArray(hotspots?.data) ||
          pressure?.total_grid_cells !== TOTAL_GRIDS ||
          conservation?.grid_count !== TOTAL_GRIDS
        ) {
          throw new Error(
            'Planning GIS or supporting research API returned an unexpected structure.'
          )
        }

        const features = planningMap.features
        const ids = new Set(
          features.map(
            (feature) =>
              String(feature.properties?.grid_id)
          )
        )

        if (ids.size !== TOTAL_GRIDS) {
          throw new Error(
            'Planning GIS must contain 9,240 unique original grid IDs.'
          )
        }

        const scoredPressure = features.filter(
          (feature) =>
            validScore(
              feature.properties?.urbanization_pressure_score
            )
        ).length

        const scoredDCI = features.filter(
          (feature) =>
            validScore(
              feature.properties?.development_conflict_index_0_1
            )
        ).length

        const scoredCP = features.filter(
          (feature) =>
            validScore(
              feature.properties?.frozen_conservation_priority_0_1
            )
        ).length

        if (
          scoredPressure !==
            pressure.eligible_scored_cells ||
          scoredDCI !==
            pressure.eligible_scored_cells ||
          scoredCP !==
            conservation.primary_priority_scored_count
        ) {
          throw new Error(
            'Frozen pressure, DCI or Conservation Priority coverage disagrees with the planning map.'
          )
        }

        if (hotspots.total !== hotspots.data.length) {
          throw new Error(
            'The hotspot API response is incomplete. Full candidate-group results are required.'
          )
        }

        setData({
          planningMap,
          gnMap,
          gnDivisions,
          hotspots,
          pressure,
          conservation,
          scoredDCI,
          scoredCP,
        })
      })
      .catch((reason) => {
        if (!controller.signal.aborted) {
          setError(
            reason?.message ||
            'Unable to load planning research results.'
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
  // 2B. EXACT PLANNING-REVIEW ACTION FOR SELECTED GRID
  // ---------------------------------------------------

  useEffect(() => {
    if (!selectedGridId || !data) {
      setGridDetails(null)
      setGridError('')
      setGridLoading(false)
      return undefined
    }

    const controller = new AbortController()

    setGridDetails(null)
    setGridError('')
    setGridLoading(true)

    urbanGreenApi.planning.grid(
      selectedGridId,
      controller.signal
    )
      .then((result) => {
        if (!controller.signal.aborted) {
          const record = extractGrid(result)

          if (
            record &&
            record.grid_id &&
            String(record.grid_id) !==
              String(selectedGridId)
          ) {
            throw new Error(
              'Grid-detail response does not match the selected grid.'
            )
          }

          setGridDetails(record)
        }
      })
      .catch((reason) => {
        if (!controller.signal.aborted) {
          setGridError(
            reason?.message ||
            'Planning review detail is unavailable.'
          )
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setGridLoading(false)
        }
      })

    return () => controller.abort()
  }, [selectedGridId, data])

  // ---------------------------------------------------
  // 2C. GRID LOOKUP
  // ---------------------------------------------------

  const gridById = useMemo(
    () => new Map(
      (
        data?.planningMap?.features || []
      ).map((feature) => [
        String(feature.properties?.grid_id),
        feature.properties,
      ])
    ),
    [data]
  )

  const baseGrid = selectedGridId
    ? gridById.get(String(selectedGridId))
    : null

  const grid = baseGrid
    ? {
        ...baseGrid,
        ...(gridDetails || {}),
      }
    : null

  const selectedGn = (
    data?.gnDivisions?.data || []
  ).find(
    (gn) =>
      String(gn.gn_id) ===
      String(selectedGnId)
  )

  // ---------------------------------------------------
  // 2D. DCI HISTOGRAM
  // ---------------------------------------------------

  const dciHistogram = useMemo(
    () => histogram(
      data?.planningMap?.features || [],
      'development_conflict_index_0_1'
    ),
    [data]
  )

  // ---------------------------------------------------
  // 2E. RESEARCH REVIEW STREAMS
  // ---------------------------------------------------

  const reviewStreams = useMemo(() => {
    const counts = new Map()

    for (
      const feature of
      data?.planningMap?.features || []
    ) {
      const stream =
        feature.properties?.review_stream ||
        'UNSPECIFIED'

      counts.set(
        stream,
        (counts.get(stream) || 0) + 1
      )
    }

    return [...counts.entries()]
      .map(([stream, count]) => ({
        stream,
        name: readable(stream),
        count,
      }))
      .sort((a, b) => b.count - a.count)
  }, [data])

  // ---------------------------------------------------
  // 2F. HIGHEST-SCORING DCI CELLS
  // ---------------------------------------------------

  const highestDCI = useMemo(
    () => (
      data?.planningMap?.features || []
    )
      .map((feature) => feature.properties)
      .filter(
        (properties) =>
          validScore(
            properties
              ?.development_conflict_index_0_1
          )
      )
      .sort(
        (a, b) =>
          b.development_conflict_index_0_1 -
          a.development_conflict_index_0_1
      )
      .slice(0, 8),
    [data]
  )

  // ---------------------------------------------------
  // 2G. CANDIDATE HOTSPOT GROUPS
  // ---------------------------------------------------

  const hotspotGroups = useMemo(() => {
    const records =
      data?.hotspots?.data || []

    const dci = records.filter(
      (row) =>
        row.index_type ===
        'DEVELOPMENT_CONFLICT'
    )

    const conservation = records.filter(
      (row) =>
        row.index_type ===
        'CONSERVATION_PRIORITY'
    )

    return {
      dci,
      conservation,
      dciCells: dci.reduce(
        (sum, row) =>
          sum + (row.grid_count || 0),
        0
      ),
      cpCells: conservation.reduce(
        (sum, row) =>
          sum + (row.grid_count || 0),
        0
      ),
    }
  }, [data])

  // ---------------------------------------------------
  // 2H. GRID SEARCH
  // ---------------------------------------------------

  function searchForGrid(event) {
    event.preventDefault()

    const id =
      searchGrid.trim().toUpperCase()

    if (!gridById.has(id)) {
      setSearchError(
        'This grid ID was not found in the original 9,240-cell planning map.'
      )
      return
    }

    setSearchError('')
    setSelectedGridId(id)
    setSearchGrid('')
  }

  // ===================================================
  // 3. LOADING AND ERROR HANDLING
  // ===================================================

  if (loading) {
    return (
      <section className="card-pad">
        <h2 className="section-title">
          Loading Planning Conflict Intelligence
        </h2>

        <p className="mt-3 text-sm text-slate-500">
          Reading frozen DCI, suitability,
          conservation and planning-review
          evidence...
        </p>
      </section>
    )
  }

  if (error || !data) {
    return (
      <section className="card-pad">
        <h2 className="section-title">
          Planning Conflict data unavailable
        </h2>

        <p className="mt-3 text-sm text-red-700">
          {error || 'No validated research data returned.'}
        </p>

        <p className="mt-2 text-xs text-slate-500">
          Check FastAPI. Planning scores and
          review recommendations will not be
          replaced with demonstration data.
        </p>

        <button
          type="button"
          className="btn-secondary mt-4"
          onClick={() =>
            setRefresh((n) => n + 1)
          }
        >
          <RefreshCcw size={15} />
          Retry
        </button>
      </section>
    )
  }

  const {
    planningMap,
    gnMap,
    gnDivisions,
    hotspots,
    pressure,
    scoredDCI,
    scoredCP,
  } = data

  const reviewAction =
    gridDetails?.review_action ||
    gridDetails?.recommended_action ||
    gridDetails?.planning_recommendation ||
    null

  // ===================================================
  // 4. PAGE CONTENT
  // ===================================================

  return (
    <div>
      <PageHeader
        eyebrow="Urban Green Intelligence"
        title="Planning Conflict and Intervention Intelligence"
        subtitle="Review model-derived urbanization pressure alongside independent expert-AHP suitability, conservation evidence and frozen rule-based planning recommendations."
        badges={[
          'Planning review · Section 05',
          '2025-origin pressure · 2026–2028',
          'Deterministic, explainable rule-based screening',
        ]}
        actions={
          <button
            type="button"
            className="btn-secondary"
            onClick={() =>
              setRefresh((n) => n + 1)
            }
          >
            <RefreshCcw size={15} />
            Refresh
          </button>
        }
      />

      {/* ============================================ */}
      {/* 4A. RESEARCH SUMMARY CARDS                   */}
      {/* ============================================ */}

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          title="Original planning grids"
          value={integer(TOTAL_GRIDS)}
          subtitle="Kaduwela 100 m spatial analysis"
          icon={Grid3X3}
          tone="teal"
        />

        <MetricCard
          title="DCI-scored grids"
          value={integer(scoredDCI)}
          subtitle="Where modelled pressure evidence exists"
          icon={AlertTriangle}
          tone="orange"
        />

        <MetricCard
          title="Conservation Priority available"
          value={integer(scoredCP)}
          subtitle="Separate CP review evidence"
          icon={ShieldCheck}
          tone="green"
        />

        <MetricCard
          title="Exploratory hotspot groups"
          value={integer(hotspots.total)}
          subtitle="Selected DCI and CP spatial groups"
          icon={MapPinned}
          tone="blue"
        />
      </div>

      {/* ============================================ */}
      {/* 4B. GIS MAP AND SELECTED GRID REVIEW         */}
      {/* ============================================ */}

      <div className="mt-5 grid gap-5 2xl:grid-cols-[1.55fr_.85fr]">
        <section className="card-pad min-w-0">
          <SectionTitle
            title="Planning research GIS map"
            description="Select a research layer and inspect original 100 m grids using the established interactive map."
          />

          <div className="mb-4 grid gap-3 sm:grid-cols-2">
            <label className="text-xs font-semibold text-slate-600">
              Research map layer

              <select
                className="control mt-1"
                value={mapMetric}
                onChange={(event) =>
                  setMapMetric(event.target.value)
                }
              >
                {MAP_OPTIONS.map((key) => (
                  <option key={key} value={key}>
                    {GREEN_MAP_LAYERS[key].label}
                  </option>
                ))}
              </select>
            </label>

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

                {gnDivisions.data.map((gn) => (
                  <option
                    key={gn.gn_id}
                    value={gn.gn_id}
                  >
                    {gn.gn_name}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <GreenGridMap
            geojson={planningMap}
            gnGeojson={gnMap}
            metric={mapMetric}
            selectedGridId={selectedGridId}
            selectedGnId={selectedGnId}
            onSelect={setSelectedGridId}
            onClearGn={() => selectGn(null)}
            height={580}
          />

          {/* <p className="mt-3 text-[11px] leading-5 text-slate-500">
            The ~200 m overview is a visualization
            aggregation of original 100 m cells,
            not a newly trained model.

            {' '}Click an overview group to enter
            the original 100 m detail, then
            click a grid for its exact planning
            evidence.

            {' '}Unscored DCI or CP locations
            are not automatically low-conflict
            or low-conservation areas.
          </p> */}
        </section>

        {/* ========================================== */}
        {/* SELECTED GRID PLANNING EVIDENCE            */}
        {/* ========================================== */}

        <div className="space-y-5">
          <section className="card-pad">
            <SectionTitle
              title="Selected grid planning evidence"
              description="Frozen research scores, review stream, rule ID and supporting decision context."
            />

            <form
              onSubmit={searchForGrid}
              className="flex gap-2"
            >
              <input
                type="text"
                className="control min-w-0 flex-1"
                value={searchGrid}
                onChange={(event) =>
                  setSearchGrid(event.target.value)
                }
                placeholder="Enter original grid ID"
                aria-label="Search grid ID"
              />

              <button
                type="submit"
                className="btn-secondary"
                aria-label="Find original grid"
              >
                <Search size={17} />
              </button>
            </form>

            {searchError && (
              <p className="mt-2 text-xs text-red-700">
                {searchError}
              </p>
            )}

            {!grid ? (
              <p className="mt-4 text-sm leading-6 text-slate-500">
                Select an original 100 m grid
                on the map, or enter a grid ID,
                to see its planning conflict
                evidence and review rule.
              </p>
            ) : (
              <>
                <dl className="mt-4">
                  <EvidenceRow
                    label="Grid ID"
                    value={selectedGridId}
                  />

                  <EvidenceRow
                    label="GN division"
                    value={
                      grid.gn_name ||
                      'GN assignment unavailable'
                    }
                  />

                  <EvidenceRow
                    label="Pressure inference domain"
                    value={readable(
                      grid.pressure_domain_status
                    )}
                  />

                  <EvidenceRow
                    label="Relative Urbanization Pressure"
                    value={
                      validScore(
                        grid.urbanization_pressure_score
                      )
                        ? number(
                            grid.urbanization_pressure_score,
                            6
                          )
                        : 'Not calculated'
                    }
                  />

                  <EvidenceRow
                    label="Expert-AHP Development Suitability"
                    value={number(
                      grid.development_suitability_score,
                      6
                    )}
                  />

                  <EvidenceRow
                    label="Development Conflict Index"
                    value={
                      validScore(
                        grid.development_conflict_index_0_1
                      )
                        ? number(
                            grid.development_conflict_index_0_1,
                            6
                          )
                        : 'Not calculated'
                    }
                  />

                  <EvidenceRow
                    label="Conservation Priority"
                    value={
                      validScore(
                        grid.frozen_conservation_priority_0_1
                      )
                        ? number(
                            grid.frozen_conservation_priority_0_1,
                            6
                          )
                        : 'Not calculated'
                    }
                  />

                  <EvidenceRow
                    label="Frozen pressure threshold flag"
                    value={
                      grid.pressure_threshold_flag_frozen == null
                        ? 'Not applicable'
                        : flag(
                            grid.pressure_threshold_flag_frozen
                          )
                          ? 'Flagged'
                          : 'Not flagged'
                    }
                  />

                  <EvidenceRow
                    label="Planning review stream"
                    value={readable(
                      grid.review_stream
                    )}
                  />

                  <EvidenceRow
                    label="Research review rule"
                    value={
                      grid.review_rule_id ||
                      'Not available'
                    }
                  />
                </dl>

                <div className="mt-4 grid grid-cols-2 gap-2 text-xs">
                  <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
                    <p className="text-slate-500">
                      DCI candidate hotspot
                    </p>

                    <p className="mt-1 font-semibold text-slate-800">
                      {grid.dci_candidate_hotspot == null
                        ? 'Not assessed'
                        : flag(grid.dci_candidate_hotspot)
                          ? 'Candidate'
                          : 'Not a selected candidate'}
                    </p>

                    {grid.dci_cluster_id && (
                      <p className="mt-1 break-all font-mono text-[10px] text-slate-500">
                        {grid.dci_cluster_id}
                      </p>
                    )}
                  </div>

                  <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
                    <p className="text-slate-500">
                      CP candidate hotspot
                    </p>

                    <p className="mt-1 font-semibold text-slate-800">
                      {grid.cp_candidate_hotspot == null
                        ? 'Not assessed'
                        : flag(grid.cp_candidate_hotspot)
                          ? 'Candidate'
                          : 'Not a selected candidate'}
                    </p>

                    {grid.cp_cluster_id && (
                      <p className="mt-1 break-all font-mono text-[10px] text-slate-500">
                        {grid.cp_cluster_id}
                      </p>
                    )}
                  </div>
                </div>

                {!validScore(
                  grid.urbanization_pressure_score
                ) && (
                  <p className="mt-3 rounded-lg border border-amber-200 bg-amber-50 p-3 text-xs leading-5 text-amber-900">
                    This grid is outside the
                    scored Urbanization Pressure
                    domain. It cannot be classified
                    as low conflict merely because
                    DCI or CP is unavailable.
                  </p>
                )}

                {gridLoading && (
                  <p className="mt-3 text-xs text-slate-500">
                    Loading exact review action...
                  </p>
                )}

                {gridError && (
                  <p className="mt-3 text-xs text-amber-800">
                    Review API: {gridError}
                  </p>
                )}
              </>
            )}
          </section>

          {/* ======================================== */}
          {/* INTERVENTION FROM EXISTING BACKEND      */}
          {/* ======================================== */}

          {grid && (
            <section className="card-pad">
              <SectionTitle
                title="Planning review recommendation"
                description="Original backend rule-based recommendation for the selected grid."
              />

              {reviewAction ? (
                <div className="rounded-xl border border-emerald-200 bg-emerald-50 p-4">
                  <p className="text-xs font-bold text-emerald-900">
                    Research-derived review action
                  </p>

                  <p className="mt-2 text-sm leading-6 text-emerald-900">
                    {reviewAction}
                  </p>
                </div>
              ) : (
                <p className="rounded-xl bg-slate-50 p-4 text-xs leading-6 text-slate-600">
                  {gridLoading
                    ? 'Retrieving the existing planning recommendation.'
                    : 'No written intervention recommendation was returned by the grid-detail API. The review rule and evidence above remain available; no substitute recommendation is invented.'}
                </p>
              )}

              <p className="mt-3 text-xs leading-6 text-slate-500">
                Review stream:{' '}
                <strong>
                  {readable(grid.review_stream)}
                </strong>

                {' '}· Rule:{' '}
                <strong>
                  {grid.review_rule_id ||
                    'Unavailable'}
                </strong>
              </p>

              <p className="mt-3 text-[11px] leading-5 text-slate-500">
                This is decision-support evidence
                for planning review, not an approved
                land-use action, construction permit,
                legal restriction or automatically
                executable intervention.
              </p>
            </section>
          )}

          {/* ======================================== */}
          {/* SELECTED GN SUMMARY                     */}
          {/* ======================================== */}

          {selectedGn && (
            <section className="card-pad">
              <SectionTitle
                title="Selected GN planning context"
                description={selectedGn.gn_name}
              />

              <dl>
                <EvidenceRow
                  label="GN area inside KMC"
                  value={`${number(
                    selectedGn.gn_area_inside_kmc_km2
                  )} km²`}
                />

                <EvidenceRow
                  label="Area-weighted DCI mean"
                  value={number(
                    selectedGn.dci_area_weighted_mean,
                    4
                  )}
                />

                <EvidenceRow
                  label="Area-weighted suitability mean"
                  value={number(
                    selectedGn.suitability_area_weighted_mean,
                    4
                  )}
                />

                <EvidenceRow
                  label="Area-weighted Conservation Priority mean"
                  value={number(
                    selectedGn.conservation_area_weighted_mean,
                    4
                  )}
                />
              </dl>

              <p className="mt-3 text-[11px] leading-5 text-slate-500">
                GN summaries are calculated from
                actual polygon-intersection areas
                and exclude unavailable scores.
                They are not simple averages
                of assigned-grid values.
              </p>
            </section>
          )}
        </div>
      </div>

      {/* ============================================ */}
      {/* 4C. REVIEW STREAM SUMMARY                   */}
      {/* ============================================ */}

      <div className="mt-5 grid gap-5 xl:grid-cols-2">
        <section className="card-pad min-w-0">
          <SectionTitle
            title="Planning review streams"
            description="Distribution of the frozen Section 05 review-stream labels across all original 9,240 grid records."
          />

          <div className="space-y-3">
            {reviewStreams.map((row) => {
              const percent =
                (row.count / TOTAL_GRIDS) * 100

              return (
                <div
                  key={row.stream}
                  className="rounded-lg border border-slate-100 bg-slate-50 px-3 py-3"
                >
                  <div className="flex items-start justify-between gap-3 text-xs">
                    <p className="max-w-[72%] font-semibold text-slate-700">
                      {row.name}
                    </p>

                    <span className="font-bold tabular-nums text-slate-900">
                      {integer(row.count)}
                    </span>
                  </div>

                  <div className="mt-2 h-2 overflow-hidden rounded-full bg-slate-200">
                    <div
                      className="h-full rounded-full bg-teal-700"
                      style={{
                        width: `${percent}%`,
                      }}
                    />
                  </div>

                  <p className="mt-1 text-[11px] text-slate-500">
                    {number(percent, 2)}%
                    {' '}of all original cells
                  </p>
                </div>
              )
            })}
          </div>

          <p className="mt-3 text-[11px] leading-5 text-slate-500">
            These streams are deterministic
            research review categories, not
            newly predicted classes or official
            government planning classifications.
          </p>
        </section>

        {/* ========================================== */}
        {/* DCI DISTRIBUTION                          */}
        {/* ========================================== */}

        <section className="card-pad min-w-0">
          <SectionTitle
            title="Development Conflict Index distribution"
            description="Continuous DCI score intervals for scored original grid cells."
          />

          <ResponsiveContainer
            width="100%"
            height={335}
          >
            <BarChart
              data={dciHistogram}
              margin={{
                top: 8,
                right: 10,
                left: 0,
                bottom: 20,
              }}
            >
              <CartesianGrid
                strokeDasharray="4 5"
              />

              <XAxis
                dataKey="interval"
                angle={-35}
                textAnchor="end"
                height={60}
                interval={0}
                tick={{ fontSize: 10 }}
              />

              <YAxis
                allowDecimals={false}
                tick={{ fontSize: 11 }}
              />

              <Tooltip
                formatter={(value) => [
                  integer(value),
                  'Scored 100 m cells',
                ]}
              />

              <Bar
                dataKey="count"
                fill="#0f766e"
                radius={[4, 4, 0, 0]}
                isAnimationActive={false}
              />
            </BarChart>
          </ResponsiveContainer>

          <p className="mt-3 text-[11px] leading-5 text-slate-500">
            Numeric intervals are descriptive
            histogram bins, not official
            DCI risk classes.

            {' '}The DCI and Conservation Priority
            are different indices and should
            not be added together into an
            undocumented planning score.
          </p>
        </section>
      </div>

      {/* ============================================ */}
      {/* 4D. HIGHEST DCI SCORE TABLE                  */}
      {/* ============================================ */}

      <section className="card-pad mt-5">
        <SectionTitle
          title="Highest-scoring Development Conflict Index grids"
          description="Top eight saved DCI grid scores for exploration. Sorting does not create an additional planning-policy classification."
        />

        <div className="overflow-x-auto">
          <table className="min-w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50 text-slate-600">
                <th className="px-3 py-3">
                  Grid ID
                </th>

                <th className="px-3 py-3">
                  GN division
                </th>

                <th className="px-3 py-3">
                  Pressure
                </th>

                <th className="px-3 py-3">
                  Suitability
                </th>

                <th className="px-3 py-3">
                  DCI
                </th>

                <th className="px-3 py-3">
                  Review stream
                </th>
              </tr>
            </thead>

            <tbody>
              {highestDCI.map((item) => (
                <tr
                  key={item.grid_id}
                  className="border-b border-slate-100 last:border-0"
                >
                  <td className="px-3 py-3">
                    <button
                      type="button"
                      className="font-mono font-semibold text-emerald-800 hover:underline"
                      onClick={() =>
                        setSelectedGridId(
                          item.grid_id
                        )
                      }
                    >
                      {item.grid_id}
                    </button>
                  </td>

                  <td className="px-3 py-3">
                    {item.gn_name ||
                      'Unassigned'}
                  </td>

                  <td className="px-3 py-3 tabular-nums">
                    {number(
                      item.urbanization_pressure_score,
                      4
                    )}
                  </td>

                  <td className="px-3 py-3 tabular-nums">
                    {number(
                      item.development_suitability_score,
                      4
                    )}
                  </td>

                  <td className="px-3 py-3 font-bold tabular-nums">
                    {number(
                      item.development_conflict_index_0_1,
                      4
                    )}
                  </td>

                  <td className="px-3 py-3">
                    {readable(
                      item.review_stream
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <p className="mt-3 text-[11px] leading-5 text-slate-500">
          Click a grid ID to load its review
          evidence in the selected-grid panel.
          The GIS map's overview/detail mode
          remains under its own controls.
        </p>
      </section>

      {/* ============================================ */}
      {/* 4E. EXPLORATORY HOTSPOT GROUPS              */}
      {/* ============================================ */}

      <section className="card-pad mt-5">
        <SectionTitle
          title="Exploratory spatial candidate groups"
          description="Top-decile spatial screening and contiguous grouping from the completed Section 05 research."
        />

        <div className="grid gap-3 md:grid-cols-3">
          <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
            <p className="text-xs text-slate-500">
              Selected Development Conflict groups
            </p>

            <p className="mt-2 text-2xl font-bold text-slate-900">
              {integer(
                hotspotGroups.dci.length
              )}
            </p>

            <p className="mt-1 text-xs text-slate-500">
              {integer(
                hotspotGroups.dciCells
              )} original grid memberships
            </p>
          </div>

          <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
            <p className="text-xs text-slate-500">
              Selected Conservation Priority groups
            </p>

            <p className="mt-2 text-2xl font-bold text-slate-900">
              {integer(
                hotspotGroups.conservation.length
              )}
            </p>

            <p className="mt-1 text-xs text-slate-500">
              {integer(
                hotspotGroups.cpCells
              )} original grid memberships
            </p>
          </div>

          <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
            <p className="text-xs text-slate-500">
              Total selected candidate groups
            </p>

            <p className="mt-2 text-2xl font-bold text-slate-900">
              {integer(hotspots.total)}
            </p>

            <p className="mt-1 text-xs text-slate-500">
              Separate DCI and CP screening streams
            </p>
          </div>
        </div>

        <p className="mt-4 text-xs leading-6 text-slate-600">
          Candidate groups are exploratory spatial
          screening outputs. They have not been
          established as statistically significant
          hotspots.

          {' '}DCI and CP groups come from separate
          screening streams and may overlap.
          A selected candidate is not an official
          land-use designation or intervention order.
        </p>

        <Link
          to="/green/gn-hotspots"
          className="mt-4 inline-flex items-center gap-2 text-xs font-semibold text-emerald-800 hover:underline"
        >
          Explore GN Divisions and spatial hotspot groups →
        </Link>
      </section>

      {/* ============================================ */}
      {/* 4F. SCIENTIFIC INTERPRETATION               */}
      {/* ============================================ */}

      <section className="card-pad mt-5">
        <SectionTitle
          title="Planning intelligence methodology"
          description="Research decision-support synthesis of independent modelled evidence and documented review rules."
        />

        <div className="grid gap-3 md:grid-cols-3">
          <div className="rounded-xl bg-slate-50 p-4 text-xs leading-6 text-slate-700">
            <p className="font-bold text-slate-800">
              Urbanization Pressure
            </p>

            <p className="mt-2">
              Frozen XGBoost model output
              using 2025-origin features.

              {' '}The result applies to
              the single combined
              2026–2028 future period.

              {' '}Scores are relative,
              not calibrated probabilities.
            </p>
          </div>

          <div className="rounded-xl bg-slate-50 p-4 text-xs leading-6 text-slate-700">
            <p className="font-bold text-slate-800">
              Development Suitability
            </p>

            <p className="mt-2">
              Independent expert-AHP
              weighted GIS-MCDA using
              road proximity, local road
              density and terrain slope.

              {' '}The continuous index
              is not legal development
              approval.
            </p>
          </div>

          <div className="rounded-xl bg-slate-50 p-4 text-xs leading-6 text-slate-700">
            <p className="font-bold text-slate-800">
              Conservation Priority
            </p>

            <p className="mt-2">
              Frozen Green Importance
              multiplied by relative
              Urbanization Pressure.

              {' '}Available only for
              cells where both inputs
              exist; not a field-validated
              ecological-loss probability.
            </p>
          </div>
        </div>

        {/* <div className="mt-4 rounded-xl border border-amber-200 bg-amber-50 p-4 text-xs leading-6 text-amber-900">
          <strong>
            Important planning limitations:
          </strong>

          {' '}The planning-review engine
          applies documented deterministic
          rules to existing research results.

          {' '}It does not train an additional ML
          model, generate legal zoning decisions,
          or prove actual future land conversion.

          {' '}An unavailable pressure, DCI or CP
          value must never be interpreted as zero
          or low risk.

          {' '}The provisional UDA zoning dataset
          is context only and cannot establish
          formal planning compliance.

          {' '}The hotspot groups are exploratory
          candidates, not statistically significant
          official hotspots.
        </div>

        <p className="mt-3 text-xs leading-6 text-slate-500">
          Research source: completed Section 05
          planning conflict and intervention
          outputs, using frozen inputs from
          Sections 02, 03 and 04.

          {' '}The original model predictions,
          expert weights and conservation
          formula remain unchanged.
        </p> */}
      </section>
    </div>
  )
}
