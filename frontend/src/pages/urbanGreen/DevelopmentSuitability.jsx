
import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  Bar, BarChart, CartesianGrid, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from 'recharts'
import {
  Calculator, CheckCircle2, MapPinned, RefreshCcw,
  Scale, ShieldAlert,
} from 'lucide-react'

import PageHeader from '../../components/shared/PageHeader'
import MetricCard from '../../components/shared/MetricCard'
import GreenGridMap, { validScore } from '../../features/urbanGreen/components/GreenGridMap'
import { urbanGreenApi } from '../../features/urbanGreen/services/urbanGreenApi'
import { useUrbanGreen } from '../../features/urbanGreen/context/UrbanGreenContext'

// =====================================================
// 1. VERIFIED RESEARCH CRITERIA
// =====================================================

const CRITERIA = [
  {
    id: 'C1',
    score: 'c1_road_access_score',
    raw: 'c1_major_road_distance_m',
    contribution: 'c1_weighted_contribution',
    rawLabel: 'Distance to major road',
    unit: 'm',
  },
  {
    id: 'C2',
    score: 'c2_road_context_score',
    raw: 'c2_road_density_500m_km_per_km2',
    contribution: 'c2_weighted_contribution',
    rawLabel: 'Road density in 500 m context',
    unit: 'km/km²',
  },
  {
    id: 'C4',
    score: 'c4_slope_score',
    raw: 'c4_slope_mean_deg',
    contribution: 'c4_weighted_contribution',
    rawLabel: 'Mean slope',
    unit: 'degrees',
  },
]

const display = (value, digits = 3) =>
  validScore(value)
    ? value.toLocaleString('en-US', {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
      })
    : 'Unavailable'

const count = (value) =>
  validScore(value)
    ? value.toLocaleString('en-US', {
        maximumFractionDigits: 0,
      })
    : 'Unavailable'

// =====================================================
// 2. REUSABLE PRESENTATION COMPONENTS
// =====================================================

function EvidenceRow({ label, value }) {
  return (
    <div className="flex items-start justify-between gap-3 border-b border-slate-100 py-3 last:border-0">
      <dt className="text-xs leading-5 text-slate-500">
        {label}
      </dt>

      <dd className="max-w-[62%] break-words text-right text-xs font-semibold leading-5 text-slate-800">
        {value ?? 'Unavailable'}
      </dd>
    </div>
  )
}

function SectionHeading({ title, description }) {
  return (
    <div className="mb-4">
      <h2 className="section-title">{title}</h2>

      <p className="mt-1 text-xs leading-5 text-slate-500">
        {description}
      </p>
    </div>
  )
}

function unwrapGrid(response) {
  const record =
    response?.data ?? response?.grid ?? response

  if (
    !record ||
    Array.isArray(record) ||
    typeof record !== 'object'
  ) {
    return null
  }

  const properties = record.properties ?? record

  return (
    properties &&
    typeof properties === 'object' &&
    !Array.isArray(properties)
  )
    ? properties
    : null
}

// =====================================================
// 3. MAIN DEVELOPMENT SUITABILITY PAGE
// =====================================================

export default function DevelopmentSuitability() {
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

  const [gridDetails, setGridDetails] = useState(null)
  const [gridError, setGridError] = useState('')

  // ---------------------------------------------------
  // 3A. LOAD FROZEN SUITABILITY RESEARCH DATA
  // ---------------------------------------------------

  useEffect(() => {
    const controller = new AbortController()

    setLoading(true)
    setError('')

    Promise.all([
      urbanGreenApi.suitability.overview(
        controller.signal
      ),

      urbanGreenApi.suitability.map(
        controller.signal
      ),

      urbanGreenApi.planning.map(
        controller.signal
      ),

      urbanGreenApi.planning.gnMap(
        controller.signal
      ),

      urbanGreenApi.planning.gnDivisions(
        controller.signal
      ),
    ])
      .then(([
        overview,
        suitabilityMap,
        planningMap,
        gnMap,
        gnDivisions,
      ]) => {
        if (controller.signal.aborted) return

        if (
          overview?.status !== 'ok' ||
          overview?.grid_count !== 9240 ||
          overview?.method !==
            'GIS_MCDA_EXPERT_AHP_WEIGHTED_LINEAR_COMBINATION' ||
          overview?.final_policy_classes_assigned !== false ||
          !Array.isArray(suitabilityMap?.features) ||
          !Array.isArray(planningMap?.features) ||
          !Array.isArray(gnMap?.features) ||
          !Array.isArray(gnDivisions?.data) ||
          suitabilityMap.features.length !== 9240 ||
          planningMap.features.length !== 9240
        ) {
          throw new Error(
            'Suitability contract or original 9,240-grid release does not match the verified research outputs.'
          )
        }

        const weights = CRITERIA.map(
          ({ id }) => overview.criteria?.[id]?.weight
        )

        if (
          weights.some(
            (weight) => !validScore(weight)
          ) ||
          Math.abs(
            weights.reduce((a, b) => a + b, 0) - 1
          ) > 1e-6
        ) {
          throw new Error(
            'Expert AHP criteria or weights are incomplete.'
          )
        }

        const actualScores =
          suitabilityMap.features.filter(
            (feature) =>
              validScore(
                feature.properties
                  ?.development_suitability_score
              )
          ).length

        if (
          actualScores !==
          overview.scored_grid_count
        ) {
          throw new Error(
            'Continuous suitability score coverage differs from the official GIS release.'
          )
        }

        setData({
          overview,
          suitabilityMap,
          planningMap,
          gnMap,
          gnDivisions,
        })
      })
      .catch((reason) => {
        if (!controller.signal.aborted) {
          setError(
            reason?.message || String(reason)
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
  // 3B. LOAD SELECTED ORIGINAL GRID CRITERIA
  // ---------------------------------------------------

  useEffect(() => {
    if (!selectedGridId || !data) {
      setGridDetails(null)
      setGridError('')
      return undefined
    }

    const controller = new AbortController()

    setGridDetails(null)
    setGridError('')

    urbanGreenApi.suitability.grid(
      selectedGridId,
      controller.signal
    )
      .then((response) => {
        if (!controller.signal.aborted) {
          setGridDetails(
            unwrapGrid(response)
          )
        }
      })
      .catch((reason) => {
        if (!controller.signal.aborted) {
          setGridError(
            reason?.message ||
            'Criterion detail not available'
          )
        }
      })

    return () => controller.abort()
  }, [selectedGridId, data])

  // ---------------------------------------------------
  // 3C. LOOK UP ORIGINAL GRID RECORDS
  // ---------------------------------------------------

  const suitabilityById = useMemo(
    () => new Map(
      (
        data?.suitabilityMap?.features || []
      ).map((feature) => [
        String(
          feature.properties?.grid_id
        ),
        feature.properties,
      ])
    ),
    [data]
  )

  const planningById = useMemo(
    () => new Map(
      (
        data?.planningMap?.features || []
      ).map((feature) => [
        String(
          feature.properties?.grid_id
        ),
        feature.properties,
      ])
    ),
    [data]
  )

  const selectedEvidence = selectedGridId
    ? {
        ...(
          suitabilityById.get(
            String(selectedGridId)
          ) || {}
        ),
        ...(gridDetails || {}),
      }
    : null

  const selectedPlanning = selectedGridId
    ? planningById.get(
        String(selectedGridId)
      )
    : null

  const selectedGn = (
    data?.gnDivisions?.data || []
  ).find(
    (gn) =>
      String(gn.gn_id) ===
      String(selectedGnId)
  )

  // ---------------------------------------------------
  // 3D. AHP WEIGHT CHART
  // ---------------------------------------------------

  const weightsChart = useMemo(
    () => CRITERIA.map(({ id }) => ({
      criterion: id,

      name:
        data?.overview?.criteria?.[id]?.name ||
        id,

      percentage:
        (
          data?.overview?.criteria?.[id]?.weight ??
          0
        ) * 100,
    })),
    [data]
  )

  // ---------------------------------------------------
  // 3E. CONTINUOUS SCORE DISTRIBUTION
  // ---------------------------------------------------

  // Numeric bins are only for visualization.
  // They are NOT official suitability classes.

  const distribution = useMemo(() => {
    const bins = Array.from(
      { length: 10 },
      (_, i) => ({
        interval:
          `${(i / 10).toFixed(1)}–` +
          `${((i + 1) / 10).toFixed(1)}`,

        grids: 0,
      })
    )

    for (
      const feature of
      data?.suitabilityMap?.features || []
    ) {
      const value =
        feature.properties
          ?.development_suitability_score

      if (!validScore(value)) continue

      const index = Math.min(
        9,
        Math.max(
          0,
          Math.floor(value * 10)
        )
      )

      bins[index].grids += 1
    }

    return bins
  }, [data])

  // ---------------------------------------------------
  // 3F. SELECTED GRID WEIGHTED CONTRIBUTIONS
  // ---------------------------------------------------

  const contributionRows = CRITERIA.map(
    (criterion) => ({
      ...criterion,

      config:
        data?.overview?.criteria
          ?.[criterion.id],

      normalized:
        selectedEvidence
          ?.[criterion.score],

      rawValue:
        selectedEvidence
          ?.[criterion.raw],

      weighted:
        selectedEvidence
          ?.[criterion.contribution],
    })
  )

  const canVerifyFormula =
    selectedEvidence &&
    validScore(
      selectedEvidence
        .development_suitability_score
    ) &&
    contributionRows.every(
      (item) => validScore(item.weighted)
    )

  const reconstructed = canVerifyFormula
    ? contributionRows.reduce(
        (sum, item) =>
          sum + item.weighted,
        0
      )
    : null

  // ===================================================
  // 4. LOADING AND ERROR STATES
  // ===================================================

  if (loading) {
    return (
      <section className="card-pad">
        <h2 className="section-title">
          Loading expert-AHP suitability results
        </h2>

        <p className="mt-3 text-sm text-slate-500">
          Reading the final continuous GIS-MCDA
          scores and criterion weights...
        </p>
      </section>
    )
  }

  if (error || !data) {
    return (
      <section className="card-pad">
        <h2 className="section-title">
          Development Suitability unavailable
        </h2>

        <p className="mt-3 text-sm text-red-700">
          {error || 'No research data returned.'}
        </p>

        <p className="mt-2 text-xs text-slate-500">
          Check that FastAPI is running.
          No mock suitability scores are substituted.
        </p>

        <button
          type="button"
          className="btn-secondary mt-4"
          onClick={() =>
            setRefresh((v) => v + 1)
          }
        >
          <RefreshCcw size={15} />
          Retry
        </button>
      </section>
    )
  }

  const {
    overview,
    planningMap,
    gnMap,
    gnDivisions,
  } = data

  const expert =
    overview.expert_consultation || {}

  const sensitivity =
    overview.sensitivity || {}

  const stats =
    overview.distribution || {}

  // ===================================================
  // 5. COMPLETE PAGE
  // ===================================================

  return (
    <div>
      <PageHeader
        eyebrow="Urban Green Intelligence"
        title="Expert-AHP Development Suitability"
        subtitle="Continuous GIS-MCDA development-suitability scores based on road proximity, local road-network density and terrain slope across Kaduwela."
        badges={[
          'Expert-AHP weighted GIS-MCDA',
          '9,240 original 100 m cells',
          'Continuous suitability scores — not permits',
        ]}
        actions={
          <button
            type="button"
            className="btn-secondary"
            onClick={() =>
              setRefresh((v) => v + 1)
            }
          >
            <RefreshCcw size={15} />
            Refresh
          </button>
        }
      />

      {/* =========================================== */}
      {/* 5A. SUMMARY CARDS                            */}
      {/* =========================================== */}

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          title="Scored GIS grids"
          value={count(
            overview.scored_grid_count
          )}
          subtitle="Original 100 m planning research cells"
          icon={MapPinned}
          tone="teal"
        />

        <MetricCard
          title="Average suitability"
          value={display(stats.mean)}
          subtitle="Continuous index between 0 and 1"
          icon={Scale}
          tone="green"
        />

        <MetricCard
          title="AHP consistency ratio"
          value={display(
            expert.consistency_ratio,
            3
          )}
          subtitle="Documented expert pairwise-weight consistency"
          icon={CheckCircle2}
          tone="blue"
        />

        <MetricCard
          title="Sensitivity scenarios"
          value={count(
            sensitivity.scenario_count
          )}
          subtitle={`${count(
            sensitivity.hypothetical_scenarios
          )} hypothetical scenarios plus baseline`}
          icon={Calculator}
          tone="orange"
        />
      </div>

      {/* =========================================== */}
      {/* 5B. REAL GIS MAP AND EVIDENCE                */}
      {/* =========================================== */}

      <div className="mt-5 grid gap-5 2xl:grid-cols-[1.55fr_.85fr]">
        <section className="card-pad min-w-0">
          <SectionHeading
            title="Continuous suitability GIS map"
            description="Research-derived relative suitability for the selected criteria. The map reuses your validated 200 m overview and original 100 m detail."
          />

          <label className="mb-3 block text-xs font-semibold text-slate-600">
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

          <GreenGridMap
            geojson={planningMap}
            gnGeojson={gnMap}
            metric="suitability"
            selectedGridId={selectedGridId}
            selectedGnId={selectedGnId}
            onSelect={setSelectedGridId}
            onClearGn={() => selectGn(null)}
            height={570}
          />

          {/* <p className="mt-3 text-[11px] leading-5 text-slate-500">
            Overview colors summarize existing 100 m
            grid values for display; they are not
            newly computed predictions. Suitability
            colors are continuous visual ranges,
            not officially assigned suitability classes.
            Select an original 100 m grid for its
            expert-weighted criterion evidence.
          </p> */}
        </section>

        <div className="space-y-5">

          {/* SELECTED GRID DETAILS */}

          <section className="card-pad">
            <SectionHeading
              title="Selected original grid evidence"
              description="The actual continuous score and underlying three criteria."
            />

            {!selectedGridId ? (
              <p className="text-sm leading-6 text-slate-500">
                Select a grid on the GIS map to
                inspect its suitability score,
                road context and slope evidence.
              </p>
            ) : (
              <>
                <dl>
                  <EvidenceRow
                    label="Grid ID"
                    value={selectedGridId}
                  />

                  <EvidenceRow
                    label="GN division"
                    value={
                      selectedPlanning?.gn_name ||
                      'Unavailable'
                    }
                  />

                  <EvidenceRow
                    label="Final continuous suitability score"
                    value={display(
                      selectedEvidence
                        ?.development_suitability_score,
                      6
                    )}
                  />

                  <EvidenceRow
                    label="Policy suitability class"
                    value="Not assigned"
                  />
                </dl>

                {gridError && (
                  <p className="mt-3 text-xs text-amber-800">
                    Criterion detail: {gridError}
                  </p>
                )}

                <div className="mt-4 space-y-3">
                  {contributionRows.map(({
                    id,
                    config,
                    normalized,
                    rawValue,
                    weighted,
                    rawLabel,
                    unit,
                  }) => (
                    <div
                      key={id}
                      className="rounded-xl border border-slate-200 bg-slate-50 p-3"
                    >
                      <div className="flex items-start justify-between gap-3 text-xs">
                        <p className="font-semibold text-slate-800">
                          {id} · {config?.name || id}
                        </p>

                        <span className="font-bold text-emerald-800">
                          {display(
                            (config?.weight ?? 0) * 100,
                            0
                          )}%
                          {' '}weight
                        </span>
                      </div>

                      <p className="mt-1 text-[11px] text-slate-500">
                        {rawLabel}:{' '}
                        {display(rawValue, 3)}
                        {' '}{unit}
                      </p>

                      <div className="mt-2 flex flex-wrap justify-between gap-2 text-[11px] text-slate-600">
                        <span>
                          Standardized score:{' '}
                          {display(normalized, 4)}
                        </span>

                        <span>
                          Weighted contribution:{' '}
                          {display(weighted, 4)}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>

                {canVerifyFormula && (
                  <p className="mt-3 rounded-lg bg-emerald-50 p-3 text-xs text-emerald-900">
                    Sum of weighted contributions:{' '}
                    {display(reconstructed, 6)}.

                    {' '}Matches saved suitability score:{' '}

                    {Math.abs(
                      reconstructed -
                      selectedEvidence
                        .development_suitability_score
                    ) < 1e-5
                      ? 'Yes'
                      : 'Review needed'}.
                  </p>
                )}

                <p className="mt-3 text-[11px] leading-5 text-slate-500">
                  The standardized C1 and C4 scores
                  account for their cost direction:
                  greater road distance or steeper
                  slopes are less favourable under
                  these selected criteria. This
                  score does not establish legal
                  building permission or the
                  absence of hazards.
                </p>
              </>
            )}
          </section>

          {/* GN SUMMARY */}

          {selectedGn && (
            <section className="card-pad">
              <SectionHeading
                title="Selected GN context"
                description={selectedGn.gn_name}
              />

              <dl>
                <EvidenceRow
                  label="KMC-intersected GN area"
                  value={`${display(
                    selectedGn.gn_area_inside_kmc_km2
                  )} km²`}
                />

                <EvidenceRow
                  label="Area-weighted suitability mean"
                  value={display(
                    selectedGn.suitability_area_weighted_mean
                  )}
                />
              </dl>

              <p className="mt-2 text-[11px] leading-5 text-slate-500">
                GN summary values use the backend's
                area-weighted spatial intersection
                method; they are not simply an
                average of assigned-grid values.
              </p>
            </section>
          )}
        </div>
      </div>

      {/* =========================================== */}
      {/* 5C. AHP WEIGHTS AND SCORE DISTRIBUTION      */}
      {/* =========================================== */}

      <div className="mt-5 grid gap-5 xl:grid-cols-2">
        <section className="card-pad min-w-0">
          <SectionHeading
            title="Expert AHP criterion weights"
            description="Weights from the documented engineer consultation. These are fixed research weights, not editable planning sliders."
          />

          <ResponsiveContainer
            width="100%"
            height={265}
          >
            <BarChart
              data={weightsChart}
              layout="vertical"
              margin={{
                top: 6,
                right: 25,
                bottom: 6,
                left: 0,
              }}
            >
              <CartesianGrid strokeDasharray="4 5" />

              <XAxis
                type="number"
                domain={[0, 100]}
                tick={{ fontSize: 11 }}
                unit="%"
              />

              <YAxis
                type="category"
                dataKey="name"
                width={150}
                tick={{ fontSize: 11 }}
              />

              <Tooltip
                formatter={(v) => [
                  `${display(v, 1)}%`,
                  'AHP weight',
                ]}
              />

              <Bar
                dataKey="percentage"
                fill="#15803d"
                radius={[0, 4, 4, 0]}
                isAnimationActive={false}
              />
            </BarChart>
          </ResponsiveContainer>

          <p className="mt-3 rounded-xl bg-slate-50 p-3 text-xs leading-6 text-slate-700">
            <strong>Frozen formula:</strong>{' '}
            {overview.formula}.

            {' '}C1 and C4 are standardized
            cost criteria; C2 is a benefit criterion.

            {' '}Expert consultation:{' '}
            {expert.role || 'Role unavailable'},
            {' '}{expert.date || 'date unavailable'}.

            {' '}Responses were researcher-transcribed
            {expert.independently_signed === false
              ? ' and not independently signed'
              : ''}.
          </p>
        </section>

        {/* CONTINUOUS SCORE HISTOGRAM */}

        <section className="card-pad min-w-0">
          <SectionHeading
            title="Distribution of continuous suitability scores"
            description="Original grid counts by numeric index interval; these intervals are only for chart display."
          />

          <ResponsiveContainer
            width="100%"
            height={265}
          >
            <BarChart
              data={distribution}
              margin={{
                top: 6,
                right: 8,
                bottom: 22,
                left: 0,
              }}
            >
              <CartesianGrid strokeDasharray="4 5" />

              <XAxis
                dataKey="interval"
                interval={0}
                angle={-35}
                textAnchor="end"
                height={52}
                tick={{ fontSize: 10 }}
              />

              <YAxis
                allowDecimals={false}
                tick={{ fontSize: 11 }}
              />

              <Tooltip
                formatter={(v) => [
                  count(v),
                  'Original 100 m cells',
                ]}
              />

              <Bar
                dataKey="grids"
                fill="#0f766e"
                radius={[4, 4, 0, 0]}
                isAnimationActive={false}
              />
            </BarChart>
          </ResponsiveContainer>

          <div className="mt-2 grid grid-cols-3 gap-2 rounded-xl bg-slate-50 p-3 text-center text-xs">
            <div>
              Minimum
              <p className="font-bold">
                {display(stats.minimum)}
              </p>
            </div>

            <div>
              Median
              <p className="font-bold">
                {display(stats.median)}
              </p>
            </div>

            <div>
              Maximum
              <p className="font-bold">
                {display(stats.maximum)}
              </p>
            </div>
          </div>

          <p className="mt-3 text-[11px] leading-5 text-slate-500">
            No Low/Moderate/High policy suitability
            classes were assigned in the final release.
            A higher numeric score means more
            favourable conditions only within the
            three modeled criteria.
          </p>
        </section>
      </div>

      {/* =========================================== */}
      {/* 5D. SENSITIVITY AND ROBUSTNESS               */}
      {/* =========================================== */}

      <section className="card-pad mt-5">
        <SectionHeading
          title="Sensitivity and ranking robustness"
          description="Hypothetical AHP-weight perturbations tested the stability of the frozen expert-weighted suitability ranking."
        />

        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          <div className="rounded-xl bg-slate-50 p-4">
            <p className="text-xs text-slate-500">
              Total scenarios
            </p>

            <p className="mt-1 text-xl font-bold">
              {count(sensitivity.scenario_count)}
            </p>
          </div>

          <div className="rounded-xl bg-slate-50 p-4">
            <p className="text-xs text-slate-500">
              Worst Spearman rank correlation
            </p>

            <p className="mt-1 text-xl font-bold">
              {display(
                sensitivity.worst_spearman,
                4
              )}
            </p>
          </div>

          <div className="rounded-xl bg-slate-50 p-4">
            <p className="text-xs text-slate-500">
              Minimum top-10% retention
            </p>

            <p className="mt-1 text-xl font-bold">
              {validScore(
                sensitivity.minimum_top10_retention
              )
                ? `${display(
                    sensitivity.minimum_top10_retention * 100,
                    1
                  )}%`
                : 'Unavailable'}
            </p>
          </div>

          <div className="rounded-xl bg-slate-50 p-4">
            <p className="text-xs text-slate-500">
              Minimum top-20% retention
            </p>

            <p className="mt-1 text-xl font-bold">
              {validScore(
                sensitivity.minimum_top20_retention
              )
                ? `${display(
                    sensitivity.minimum_top20_retention * 100,
                    1
                  )}%`
                : 'Unavailable'}
            </p>
          </div>
        </div>

        <p className="mt-3 text-xs leading-6 text-slate-600">
          {sensitivity.robustness_interpretation ||
            'Ranking sensitivity interpretation unavailable.'}

          {' '}The hypothetical scenarios do not
          replace the engineer's final weights.
          These are ranking-stability diagnostics,
          not predictive accuracy or uncertainty
          probabilities.
        </p>
      </section>

      {/* =========================================== */}
      {/* 5E. LIMITATIONS AND RESEARCH INTERPRETATION */}
      {/* =========================================== */}

      {/* <section className="card-pad mt-5">
        <div className="flex items-center gap-2">
          <ShieldAlert
            size={18}
            className="text-amber-700"
          />

          <h2 className="section-title">
            Research scope and limitations
          </h2>
        </div>

        <div className="mt-4 grid gap-3 md:grid-cols-3">
          <div className="rounded-xl bg-slate-50 p-4 text-xs leading-6 text-slate-700">
            <strong>
              Independent suitability:
            </strong>{' '}
            GIS-MCDA uses the selected road-proximity,
            local-road-density and slope criteria.
            Urbanization Pressure, SHAP and Conservation
            Priority are not suitability inputs.
          </div>

          <div className="rounded-xl bg-slate-50 p-4 text-xs leading-6 text-slate-700">
            <strong>
              Continuous scores only:
            </strong>{' '}
            All 9,240 grids have a relative
            suitability index. Official
            developable/non-developable decisions
            and final policy classes have not
            been established.
          </div>

          <div className="rounded-xl bg-slate-50 p-4 text-xs leading-6 text-slate-700">
            <strong>
              Important exclusions:
            </strong>{' '}
            Explicit flood-hazard screening,
            legal UDA zoning compliance and
            comprehensive on-site engineering
            validation are outside this index.
          </div>
        </div>

        {Array.isArray(
          overview.limitations
        ) && overview.limitations.length > 0 && (
          <div className="mt-4 rounded-xl border border-slate-200 bg-white p-4">
            <p className="text-xs font-bold text-slate-800">
              Backend-documented limitations
            </p>

            <ul className="mt-2 list-disc space-y-1 pl-5 text-xs leading-5 text-slate-600">
              {overview.limitations.map(
                (item, index) => (
                  <li key={index}>
                    {item}
                  </li>
                )
              )}
            </ul>
          </div>
        )}

        <p className="mt-4 rounded-xl border border-amber-200 bg-amber-50 p-4 text-xs leading-6 text-amber-900">
          <strong>
            Planning caution:
          </strong>{' '}
          A higher suitability score is not planning
          permission, a legal zoning determination,
          an unconditional development recommendation,
          or proof of environmental safety.
          Future development pressure and conservation
          importance must be assessed separately before
          planning intervention.
        </p>

        <Link
          to="/green/conservation"
          className="mt-4 inline-flex items-center gap-2 text-xs font-semibold text-emerald-800 hover:underline"
        >
          Next: Conservation Intelligence →
        </Link>
      </section> */}
    </div>
  )
}
