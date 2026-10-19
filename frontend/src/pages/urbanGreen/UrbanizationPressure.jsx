
import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  Bar, BarChart, CartesianGrid, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from 'recharts'
import {
  ArrowRight, Grid3X3, MapPinned, RefreshCcw,
  TrendingUp, AlertTriangle,
} from 'lucide-react'

import PageHeader from '../../components/shared/PageHeader'
import MetricCard from '../../components/shared/MetricCard'
import GreenGridMap, { validScore } from '../../features/urbanGreen/components/GreenGridMap'
import { urbanGreenApi } from '../../features/urbanGreen/services/urbanGreenApi'
import { useUrbanGreen } from '../../features/urbanGreen/context/UrbanGreenContext'

// =====================================================
// 1. RESEARCH CONFIGURATION
// =====================================================

// Research-defined categories, NOT probability bands.
const CATEGORY_LABELS = [
  ['LOW_RELATIVE', 'Low relative'],
  ['MODERATE_RELATIVE', 'Moderate'],
  ['ELEVATED_RELATIVE', 'Elevated'],
  ['HIGH_RELATIVE', 'High relative'],
  ['THRESHOLD_FLAGGED', 'Threshold flagged'],
]

function number(value, decimals = 3) {
  return validScore(value)
    ? value.toLocaleString('en-US', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals,
      })
    : 'Unavailable'
}

function integer(value) {
  return validScore(value)
    ? value.toLocaleString('en-US', {
        maximumFractionDigits: 0,
      })
    : 'Unavailable'
}

function labelCategory(value) {
  if (value === 'NOT_IN_INFERENCE_DOMAIN') {
    return 'Outside prediction domain'
  }

  return CATEGORY_LABELS.find(
    ([key]) => key === value
  )?.[1] || 'Unavailable'
}

function DetailRow({ label, value }) {
  return (
    <div className="flex items-start justify-between gap-4 border-b border-slate-100 py-3 last:border-0">
      <dt className="text-xs text-slate-500">
        {label}
      </dt>
      <dd className="max-w-[60%] break-words text-right text-xs font-semibold text-slate-800">
        {value ?? 'Unavailable'}
      </dd>
    </div>
  )
}

function SectionIntro({ title, description }) {
  return (
    <div className="mb-4">
      <h2 className="section-title">{title}</h2>
      <p className="mt-1 text-xs leading-5 text-slate-500">
        {description}
      </p>
    </div>
  )
}

// =====================================================
// 2. MAIN PRESSURE PAGE
// =====================================================

export default function UrbanizationPressure() {
  const {
    selectedGridId,
    setSelectedGridId,
    selectedGnId,
    selectGn,
  } = useUrbanGreen()

  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [refreshToken, setRefreshToken] = useState(0)

  // ---------------------------------------------------
  // 2A. LOAD FROZEN RESEARCH OUTPUTS
  // ---------------------------------------------------

  useEffect(() => {
    const controller = new AbortController()

    setLoading(true)
    setError('')

    Promise.all([
      urbanGreenApi.pressure.overview(controller.signal),
      urbanGreenApi.pressure.map(controller.signal),
      urbanGreenApi.planning.map(controller.signal),
      urbanGreenApi.planning.gnMap(controller.signal),
      urbanGreenApi.planning.gnDivisions(controller.signal),
    ])
      .then(([
        overview,
        pressureMap,
        planningMap,
        gnMap,
        gnDivisions,
      ]) => {
        if (controller.signal.aborted) return

        if (
          overview?.status !== 'ok' ||
          !Array.isArray(pressureMap?.features) ||
          !Array.isArray(planningMap?.features) ||
          !Array.isArray(gnMap?.features) ||
          !Array.isArray(gnDivisions?.data)
        ) {
          throw new Error(
            'Pressure and planning API responses have unexpected structures.'
          )
        }

        if (
          overview.total_grid_cells !== 9240 ||
          pressureMap.features.length !== 9240 ||
          planningMap.features.length !== 9240
        ) {
          throw new Error(
            'Expected the original 9,240-cell study-area grid.'
          )
        }

        // Independently cross-check the saved scores.
        const scored = pressureMap.features.filter(
          (feature) =>
            validScore(
              feature.properties?.urbanization_pressure_score
            )
        )

        const flags = scored.filter(
          (feature) =>
            feature.properties?.threshold_flag === 1 ||
            feature.properties?.threshold_flag === true
        )

        const categoriesTotal = CATEGORY_LABELS.reduce(
          (sum, [key]) =>
            sum + (overview.category_counts?.[key] || 0),
          0
        )

        if (
          scored.length !== overview.eligible_scored_cells ||
          flags.length !== overview.threshold_flagged_cells ||
          categoriesTotal !== scored.length
        ) {
          throw new Error(
            'Frozen pressure summary and GIS scores disagree. No values have been substituted.'
          )
        }

        setData({
          overview,
          pressureMap,
          planningMap,
          gnMap,
          gnDivisions,
        })
      })
      .catch((reason) => {
        if (!controller.signal.aborted) {
          setError(
            reason?.message ||
            'Unable to load Urbanization Pressure data.'
          )
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
  // 2B. INDEX THE ORIGINAL 100 M GRID ATTRIBUTES
  // ---------------------------------------------------

  const pressureById = useMemo(
    () => new Map(
      (data?.pressureMap?.features || []).map(
        (feature) => [
          String(feature.properties?.grid_id),
          feature.properties,
        ]
      )
    ),
    [data]
  )

  const planningById = useMemo(
    () => new Map(
      (data?.planningMap?.features || []).map(
        (feature) => [
          String(feature.properties?.grid_id),
          feature.properties,
        ]
      )
    ),
    [data]
  )

  const selectedPressure = selectedGridId
    ? pressureById.get(String(selectedGridId))
    : null

  const selectedPlanning = selectedGridId
    ? planningById.get(String(selectedGridId))
    : null

  const selectedGn = useMemo(
    () => (data?.gnDivisions?.data || []).find(
      (gn) =>
        String(gn.gn_id) === String(selectedGnId)
    ) || null,
    [data, selectedGnId]
  )

  const selectedGridValid = Boolean(
    selectedPlanning && selectedPressure
  )

  const selectedScore =
    selectedPressure?.urbanization_pressure_score

  const selectedScored = validScore(selectedScore)

  // ---------------------------------------------------
  // 2C. CATEGORY CHART
  // ---------------------------------------------------

  const categoryData = useMemo(
    () => CATEGORY_LABELS.map(([key, label]) => ({
      category: label,
      count: data?.overview?.category_counts?.[key] ?? 0,
    })),
    [data]
  )

  // ---------------------------------------------------
  // 2D. SCORE DISTRIBUTION CHART
  // ---------------------------------------------------

  const histogramData = useMemo(() => {
    const bins = Array.from(
      { length: 10 },
      (_, index) => ({
        range: `${(index / 10).toFixed(1)}–${((index + 1) / 10).toFixed(1)}`,
        cells: 0,
      })
    )

    for (
      const feature of data?.pressureMap?.features || []
    ) {
      const value =
        feature.properties?.urbanization_pressure_score

      if (!validScore(value)) continue

      const bin = Math.min(
        9,
        Math.max(0, Math.floor(value * 10))
      )

      bins[bin].cells += 1
    }

    return bins
  }, [data])

  // ---------------------------------------------------
  // 2E. HIGHEST-SCORING ORIGINAL CELLS
  // ---------------------------------------------------

  const topCells = useMemo(
    () => (data?.pressureMap?.features || [])
      .map((feature) => feature.properties)
      .filter(
        (properties) =>
          validScore(
            properties?.urbanization_pressure_score
          )
      )
      .sort(
        (a, b) =>
          b.urbanization_pressure_score -
          a.urbanization_pressure_score
      )
      .slice(0, 8),
    [data]
  )

  // ---------------------------------------------------
  // 2F. SELECTED GN COUNTS
  // ---------------------------------------------------

  // Uses the assigned GN of each 100 m cell.
  // This is NOT fractional GN-intersection statistics.

  const gnStats = useMemo(() => {
    if (!data || !selectedGnId) return null

    const assigned = data.planningMap.features.filter(
      (feature) =>
        String(feature.properties?.gn_id) ===
        String(selectedGnId)
    )

    let scored = 0
    let flagged = 0

    for (const feature of assigned) {
      const result = pressureById.get(
        String(feature.properties?.grid_id)
      )

      if (
        validScore(result?.urbanization_pressure_score)
      ) {
        scored += 1
      }

      if (
        result?.threshold_flag === 1 ||
        result?.threshold_flag === true
      ) {
        flagged += 1
      }
    }

    return {
      assigned: assigned.length,
      scored,
      flagged,
    }
  }, [data, pressureById, selectedGnId])

  function refresh() {
    setRefreshToken((current) => current + 1)
  }

  // ===================================================
  // 3. LOADING AND ERROR STATES
  // ===================================================

  if (loading) {
    return (
      <div className="card-pad">
        <h2 className="section-title">
          Loading Urbanization Pressure Intelligence
        </h2>

        <p className="mt-3 text-sm text-slate-500">
          Reading the frozen XGBoost results,
          prediction-domain coverage, and original
          GIS cells...
        </p>
      </div>
    )
  }

  if (error || !data) {
    return (
      <div className="card-pad">
        <h2 className="section-title">
          Urbanization Pressure data unavailable
        </h2>

        <p className="mt-3 text-sm text-red-700">
          {error || 'No validated data returned.'}
        </p>

        <p className="mt-2 text-xs text-slate-500">
          Check that FastAPI is running.
          Research scores are not replaced with
          demonstration values.
        </p>

        <button
          type="button"
          className="btn-primary mt-4"
          onClick={refresh}
        >
          <RefreshCcw size={15} />
          Retry
        </button>
      </div>
    )
  }

  const {
    overview,
    planningMap,
    gnMap,
    gnDivisions,
  } = data

  const total = overview.total_grid_cells
  const eligible = overview.eligible_scored_cells
  const threshold = overview.locked_operating_threshold
  const flagged = overview.threshold_flagged_cells

  // ===================================================
  // 4. PAGE CONTENT
  // ===================================================

  return (
    <div>
      <PageHeader
        eyebrow="Urban Green Intelligence"
        title="Urbanization Pressure Intelligence"
        subtitle="Model-derived relative pressure for non-built-to-built conversion during the single combined 2026–2028 future assessment window."
        badges={[
          `Forecast origin: ${overview.origin_year}`,
          `Future window: ${overview.pressure_period}`,
          'Original 100 m GIS research cells',
        ]}
        actions={
          <button
            type="button"
            className="btn-secondary"
            onClick={refresh}
          >
            <RefreshCcw size={15} />
            Refresh
          </button>
        }
      />

      {/* ============================================== */}
      {/* 4A. SUMMARY CARDS                              */}
      {/* ============================================== */}

      <div className="mt-5 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          title="Eligible scored grids"
          value={integer(eligible)}
          subtitle="Original 100 m cells with an XGBoost score"
          icon={Grid3X3}
          tone="teal"
        />

        <MetricCard
          title="Outside prediction domain"
          value={integer(
            overview.outside_inference_domain_cells
          )}
          subtitle="No prediction; not automatically low pressure"
          icon={Grid3X3}
          tone="blue"
        />

        <MetricCard
          title="Threshold-flagged grids"
          value={integer(flagged)}
          subtitle={`${number(
            (flagged / eligible) * 100,
            2
          )}% of scored cells`}
          icon={AlertTriangle}
          tone="orange"
        />

        <MetricCard
          title="Locked operating threshold"
          value={number(threshold, 3)}
          subtitle="Frozen decision threshold; not a probability band"
          icon={TrendingUp}
          tone="green"
        />
      </div>

      {/* ============================================== */}
      {/* 4B. PRESSURE GIS MAP                           */}
      {/* ============================================== */}

      <div className="mt-5 grid gap-5 2xl:grid-cols-[1.55fr_.85fr]">
        <section className="card-pad min-w-0">
          <SectionIntro
            title="Future urbanization-pressure GIS map"
            description="2025-origin XGBoost pressure scores. Use the same validated 200 m overview and original 100 m detail controls as your Dashboard."
          />

          <div className="mb-4 grid gap-3 md:grid-cols-2">
            <label className="text-xs font-semibold text-slate-600">
              Prediction window

              <div className="control mt-1 bg-slate-50 text-slate-700">
                {overview.pressure_period}
                {' '}· combined period
              </div>
            </label>

            <label className="text-xs font-semibold text-slate-600">
              GN division

              <select
                className="control mt-1"
                value={selectedGnId || ''}
                onChange={(event) =>
                  selectGn(event.target.value || null)
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
            metric="pressure"
            selectedGridId={selectedGridId}
            selectedGnId={selectedGnId}
            onSelect={setSelectedGridId}
            onClearGn={() => selectGn(null)}
            height={580}
          />

          {/* <p className="mt-3 text-xs leading-5 text-slate-500">
            The ~200 m overview uses area-weighted
            display summaries; it is not a second
            prediction model. Click a 200 m group
            to highlight its original 100 m cells,
            then click one original grid for exact
            model evidence. GN selection does not
            automatically move or zoom the map.
          </p> */}
        </section>

        {/* ============================================ */}
        {/* 4C. SELECTED ORIGINAL GRID EVIDENCE           */}
        {/* ============================================ */}

        <div className="space-y-5">
          <section className="card-pad">
            <div className="flex items-center gap-2">
              <MapPinned
                size={18}
                className="text-emerald-700"
              />

              <h2 className="section-title">
                Selected original grid
              </h2>
            </div>

            {!selectedGridValid ? (
              <p className="mt-4 text-sm leading-6 text-slate-500">
                Switch to 100 m detail and click a
                cell to see its model-derived pressure
                score, category, percentile, rank,
                and domain eligibility.
              </p>
            ) : (
              <dl className="mt-3">
                <DetailRow
                  label="Grid ID"
                  value={selectedGridId}
                />

                <DetailRow
                  label="GN division"
                  value={
                    selectedPlanning.gn_name ||
                    'Not assigned'
                  }
                />

                <DetailRow
                  label="Inference domain"
                  value={
                    selectedScored
                      ? 'Eligible and scored'
                      : 'Outside prediction domain'
                  }
                />

                <DetailRow
                  label="Relative pressure score"
                  value={
                    selectedScored
                      ? number(selectedScore, 6)
                      : 'Not calculated'
                  }
                />

                <DetailRow
                  label="Research category"
                  value={labelCategory(
                    selectedPressure.pressure_category
                  )}
                />

                <DetailRow
                  label="Pressure percentile (scored cells)"
                  value={
                    selectedScored
                      ? `${number(
                          selectedPressure.pressure_percentile,
                          2
                        )}th`
                      : 'Not applicable'
                  }
                />

                <DetailRow
                  label="Pressure rank (scored cells)"
                  value={
                    selectedScored
                      ? integer(
                          selectedPressure.pressure_rank
                        )
                      : 'Not applicable'
                  }
                />

                <DetailRow
                  label="Threshold flag"
                  value={
                    !selectedScored
                      ? 'Not applicable'
                      : (
                          selectedPressure.threshold_flag === 1 ||
                          selectedPressure.threshold_flag === true
                        )
                        ? 'Flagged for model screening'
                        : 'Not threshold flagged'
                  }
                />

                <DetailRow
                  label="Forecast origin"
                  value={String(overview.origin_year)}
                />

                <DetailRow
                  label="Assessment window"
                  value={overview.pressure_period}
                />

                <p className="mt-3 rounded-lg bg-emerald-50 p-3 text-[11px] leading-5 text-emerald-900">
                  {selectedScored
                    ? 'This is a relative model score, not the probability that this grid will develop. Threshold flags are screening indicators, not confirmed conversions.'
                    : 'No future conversion-pressure prediction is available for this cell. Do not interpret an unscored grid as zero or low pressure.'}
                </p>
              </dl>
            )}
          </section>

          {/* ========================================== */}
          {/* 4D. SELECTED GN COVERAGE                    */}
          {/* ========================================== */}

          {selectedGn && gnStats && (
            <section className="card-pad">
              <SectionIntro
                title="Selected GN coverage"
                description={selectedGn.gn_name}
              />

              <dl>
                <DetailRow
                  label="Assigned 100 m grids"
                  value={integer(gnStats.assigned)}
                />

                <DetailRow
                  label="Pressure-scored assigned grids"
                  value={integer(gnStats.scored)}
                />

                <DetailRow
                  label="Threshold-flagged assigned grids"
                  value={integer(gnStats.flagged)}
                />
              </dl>

              <p className="mt-2 text-[11px] leading-5 text-slate-500">
                Counts use each grid's assigned GN.
                They are not fractional polygon-intersection
                statistics or GN-specific model forecasts.
              </p>
            </section>
          )}

          {/* ========================================== */}
          {/* 4E. MODEL AND DATA SCOPE                    */}
          {/* ========================================== */}

          <section className="card-pad">
            <SectionIntro
              title="Model and data scope"
              description="Research results already trained, evaluated and frozen."
            />

            <dl>
              <DetailRow
                label="Production model"
                value={overview.production_model}
              />

              <DetailRow
                label="Forecast origin"
                value={String(overview.origin_year)}
              />

              <DetailRow
                label="Future assessment"
                value={overview.pressure_period}
              />

              <DetailRow
                label="Total original grids"
                value={integer(total)}
              />

              <DetailRow
                label="QA Step 14C (scores)"
                value={
                  overview.research_qa?.['14c'] ||
                  'Unavailable'
                }
              />

              <DetailRow
                label="QA Step 14D (categories)"
                value={
                  overview.research_qa?.['14d'] ||
                  'Unavailable'
                }
              />
            </dl>
          </section>
        </div>
      </div>

      {/* ============================================== */}
      {/* 4F. CATEGORY DISTRIBUTION CHART                 */}
      {/* ============================================== */}

      <div className="mt-5 grid gap-5 xl:grid-cols-2">
        <section className="card-pad min-w-0">
          <SectionIntro
            title="Future pressure categories"
            description="Frozen category counts from the original 4,620 model-scored grids."
          />

          <ResponsiveContainer
            width="100%"
            height={310}
          >
            <BarChart
              data={categoryData}
              layout="vertical"
              margin={{
                left: 12,
                right: 22,
              }}
            >
              <CartesianGrid strokeDasharray="4 5" />

              <XAxis
                type="number"
                allowDecimals={false}
                tick={{ fontSize: 11 }}
              />

              <YAxis
                type="category"
                dataKey="category"
                width={124}
                tick={{ fontSize: 10 }}
              />

              <Tooltip
                formatter={(value) => [
                  integer(value),
                  'Original 100 m grids',
                ]}
              />

              <Bar
                dataKey="count"
                fill="#15803d"
                radius={[0, 5, 5, 0]}
                isAnimationActive={false}
              />
            </BarChart>
          </ResponsiveContainer>

          <p className="mt-2 text-[11px] leading-5 text-slate-500">
            Low, moderate, elevated and high are
            relative percentile-based visualization
            classes. Threshold-flagged cells use
            the locked{' '}
            {number(threshold, 3)} operating threshold.
            The 4,620 out-of-domain cells do not
            belong to the low category.
          </p>
        </section>

        {/* ============================================ */}
        {/* 4G. SCORE HISTOGRAM                           */}
        {/* ============================================ */}

        <section className="card-pad min-w-0">
          <SectionIntro
            title="Distribution of model-derived scores"
            description="Counts of the original scored 100 m cells in each numeric score interval."
          />

          <ResponsiveContainer
            width="100%"
            height={310}
          >
            <BarChart
              data={histogramData}
              margin={{
                left: 0,
                right: 12,
                bottom: 17,
              }}
            >
              <CartesianGrid strokeDasharray="4 5" />

              <XAxis
                dataKey="range"
                interval={0}
                angle={-35}
                textAnchor="end"
                height={59}
                tick={{ fontSize: 10 }}
              />

              <YAxis
                allowDecimals={false}
                tick={{ fontSize: 11 }}
              />

              <Tooltip
                formatter={(value) => [
                  integer(value),
                  'Original 100 m grids',
                ]}
              />

              <Bar
                dataKey="cells"
                fill="#0f766e"
                radius={[4, 4, 0, 0]}
                isAnimationActive={false}
              />
            </BarChart>
          </ResponsiveContainer>

          <p className="mt-2 text-[11px] leading-5 text-slate-500">
            Score ranges are descriptive numeric
            bins, not development probabilities
            or the research's separate pressure
            categories.
          </p>
        </section>
      </div>

      {/* ============================================== */}
      {/* 4H. HIGHEST SCORING GRIDS                       */}
      {/* ============================================== */}

      <section className="card-pad mt-5">
        <SectionIntro
          title="Highest-scoring original grids"
          description="The top eight cells ranked from the saved 2025-origin prediction map; not a list of confirmed future developments."
        />

        <div className="overflow-x-auto">
          <table className="min-w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50 text-slate-600">
                <th className="px-3 py-2">Rank</th>
                <th className="px-3 py-2">Grid ID</th>
                <th className="px-3 py-2">Relative score</th>
                <th className="px-3 py-2">Pressure category</th>
              </tr>
            </thead>

            <tbody>
              {topCells.map((row) => (
                <tr
                  key={row.grid_id}
                  className="border-b border-slate-100 last:border-0"
                >
                  <td className="px-3 py-2 tabular-nums">
                    {integer(row.pressure_rank)}
                  </td>

                  <td className="px-3 py-2 font-mono text-slate-800">
                    {row.grid_id}
                  </td>

                  <td className="px-3 py-2 font-semibold tabular-nums">
                    {number(
                      row.urbanization_pressure_score,
                      6
                    )}
                  </td>

                  <td className="px-3 py-2">
                    {labelCategory(
                      row.pressure_category
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* ============================================== */}
      {/* 4I. SCIENTIFIC INTERPRETATION                   */}
      {/* ============================================== */}

      <section className="card-pad mt-5">
        <SectionIntro
          title="Methodology and responsible interpretation"
          description="Interpretation rules for the final 2025-origin XGBoost production inference."
        />

        <div className="grid gap-3 md:grid-cols-3">
          <div className="rounded-xl bg-slate-50 p-4 text-xs leading-6 text-slate-700">
            <strong>Historical input:</strong> RAW
            LULC predictors through 2025, including
            the 2023–2025 three-year feature window
            and selected static terrain context.
            No provisional 2026 imagery is used
            for prediction.
          </div>

          <div className="rounded-xl bg-slate-50 p-4 text-xs leading-6 text-slate-700">
            <strong>Future output:</strong> A single
            three-year 2026–2028 non-built-to-built
            relative pressure score. It does not
            provide independent forecasts for
            2026, 2027 and 2028.
          </div>

          <div className="rounded-xl bg-slate-50 p-4 text-xs leading-6 text-slate-700">
            <strong>Model domain:</strong> Predictions
            are produced only for eligible cells.
            Areas already outside the defined
            conversion domain are left unscored,
            not assigned zero pressure.
          </div>
        </div>

        {/* <p className="mt-4 rounded-xl border border-amber-200 bg-amber-50 p-4 text-xs leading-6 text-amber-900">
          <strong>Research limitation:</strong> XGBoost
          relative scores are not calibrated probabilities.
          The conversion target was built from modelled
          historical LULC transitions, not complete
          independently field-verified land conversion.
          The locked threshold supports screening,
          not legal land-use decisions. This page
          displays saved results only; it does not
          retrain, retune or rescore the model.
        </p> */}

        <Link
          to="/green/xai"
          className="mt-4 inline-flex items-center gap-2 text-xs font-semibold text-emerald-800 hover:underline"
        >
          Explore model explanations (SHAP)
          <ArrowRight size={15} />
        </Link>
      </section>
    </div>
  )
}
