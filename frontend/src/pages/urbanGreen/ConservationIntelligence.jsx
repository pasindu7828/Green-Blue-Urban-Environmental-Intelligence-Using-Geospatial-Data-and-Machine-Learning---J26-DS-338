
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
  Leaf,
  ShieldCheck,
  ShieldAlert,
  RefreshCcw,
  MapPinned,
} from 'lucide-react'

import PageHeader from '../../components/shared/PageHeader'
import MetricCard from '../../components/shared/MetricCard'
import GreenGridMap, {
  validScore,
} from '../../features/urbanGreen/components/GreenGridMap'
import { urbanGreenApi } from '../../features/urbanGreen/services/urbanGreenApi'
import { useUrbanGreen } from '../../features/urbanGreen/context/UrbanGreenContext'

// =====================================================
// 1. FROZEN RESEARCH CONFIGURATION
// =====================================================

const GI_CRITERIA = [
  {
    id: 'G1',
    label: 'Current green condition',
    field: 'G1_current_green_0_1',
    contribution: 'G1_weighted_contribution',
    weight: 0.2778141474499054,
  },
  {
    id: 'G2',
    label: 'Historical green persistence',
    field: 'G2_historical_persistence_0_1',
    contribution: 'G2_weighted_contribution',
    weight: 0.4237380285685617,
  },
  {
    id: 'G3',
    label: 'Local green context (250 m)',
    field: 'G3_local_green_context_0_1',
    contribution: 'G3_weighted_contribution',
    weight: 0.298447823981533,
  },
]

const MISSING_STATES = [
  {
    key: 'NOT_IN_PRESSURE_DOMAIN',
    label: 'GI available; pressure outside domain',
  },
  {
    key: 'GI_UNSCORED_PRESSURE_AVAILABLE',
    label: 'Pressure available; GI unavailable',
  },
  {
    key: 'GI_UNSCORED_AND_PRESSURE_NOT_IN_DOMAIN',
    label: 'Both inputs unavailable',
  },
]

function fmt(value, digits = 3) {
  if (!validScore(value)) return 'Unavailable'

  return value.toLocaleString('en-US', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })
}

function fmtCount(value) {
  if (!validScore(value)) return 'Unavailable'

  return value.toLocaleString('en-US', {
    maximumFractionDigits: 0,
  })
}

function displayStatus(value) {
  if (value == null || value === '') {
    return 'Unavailable'
  }

  return String(value)
    .replaceAll('_', ' ')
    .toLowerCase()
    .replace(/^./, (s) => s.toUpperCase())
}

function InfoRow({ label, value }) {
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

function extractDetails(response) {
  const item =
    response?.data ??
    response?.grid ??
    response?.properties ??
    response

  if (
    item == null ||
    typeof item !== 'object' ||
    Array.isArray(item)
  ) {
    return null
  }

  const result = item.properties ?? item

  if (
    !result ||
    typeof result !== 'object' ||
    Array.isArray(result)
  ) {
    return null
  }

  return result
}

function createHistogram(features, field) {
  const bins = Array.from(
    { length: 10 },
    (_, i) => ({
      interval:
        `${(i / 10).toFixed(1)}–` +
        `${((i + 1) / 10).toFixed(1)}`,
      cells: 0,
    })
  )

  for (const feature of features) {
    const score = feature.properties?.[field]

    if (!validScore(score)) continue

    const index = Math.min(
      9,
      Math.max(0, Math.floor(score * 10))
    )

    bins[index].cells += 1
  }

  return bins
}

function ScoreHistogram({
  title,
  description,
  data,
  color,
  note,
}) {
  return (
    <section className="card-pad min-w-0">
      <SectionTitle
        title={title}
        description={description}
      />

      <ResponsiveContainer width="100%" height={310}>
        <BarChart
          data={data}
          margin={{
            left: 0,
            right: 12,
            bottom: 18,
          }}
        >
          <CartesianGrid strokeDasharray="4 5" />

          <XAxis
            dataKey="interval"
            interval={0}
            angle={-35}
            textAnchor="end"
            height={62}
            tick={{ fontSize: 10 }}
          />

          <YAxis
            allowDecimals={false}
            tick={{ fontSize: 11 }}
          />

          <Tooltip
            formatter={(value) => [
              fmtCount(value),
              'Original 100 m cells',
            ]}
          />

          <Bar
            dataKey="cells"
            fill={color}
            radius={[4, 4, 0, 0]}
            isAnimationActive={false}
          />
        </BarChart>
      </ResponsiveContainer>

      <p className="mt-3 text-[11px] leading-5 text-slate-500">
        {note}
      </p>
    </section>
  )
}

// =====================================================
// 2. MAIN CONSERVATION PAGE
// =====================================================

export default function ConservationIntelligence() {
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
  // 2A. LOAD REAL CONSERVATION GIS AND PLANNING DATA
  // ---------------------------------------------------

  useEffect(() => {
    const controller = new AbortController()

    setLoading(true)
    setError('')

    Promise.all([
      urbanGreenApi.conservation.overview(
        controller.signal
      ),

      urbanGreenApi.conservation.map(
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
        conservationMap,
        planningMap,
        gnMap,
        gnDivisions,
      ]) => {
        if (controller.signal.aborted) return

        if (
          overview?.status !== 'ok' ||
          overview?.grid_count !== 9240 ||
          overview?.final_priority_classes_assigned !== false ||
          !Array.isArray(conservationMap?.features) ||
          !Array.isArray(planningMap?.features) ||
          !Array.isArray(gnMap?.features) ||
          !Array.isArray(gnDivisions?.data) ||
          conservationMap.features.length !== 9240 ||
          planningMap.features.length !== 9240
        ) {
          throw new Error(
            'Unexpected Conservation Intelligence API or GIS release format.'
          )
        }

        const records = conservationMap.features

        const giScored = records.filter(
          (feature) =>
            validScore(
              feature.properties
                ?.green_importance_index_0_1
            )
        ).length

        const cpScored = records.filter(
          (feature) =>
            validScore(
              feature.properties
                ?.primary_conservation_priority_0_1
            )
        ).length

        if (
          giScored !==
          overview.green_importance_scored_count ||
          cpScored !==
          overview.primary_priority_scored_count
        ) {
          throw new Error(
            'Frozen GI or CP GIS coverage does not agree with the conservation overview.'
          )
        }

        const missing = MISSING_STATES.reduce(
          (sum, item) =>
            sum +
            (overview.status_counts?.[item.key] || 0),
          0
        )

        if (
          missing !==
            overview.primary_priority_unscored_count ||
          cpScored + missing !== 9240
        ) {
          throw new Error(
            'Conservation missing-data status counts are inconsistent.'
          )
        }

        const planningById = new Map(
          planningMap.features.map((feature) => [
            String(feature.properties?.grid_id),
            feature.properties,
          ])
        )

        // Cross-check the saved Section 05 planning layer.
        let compared = 0
        let mismatches = 0

        for (const feature of records) {
          const props = feature.properties || {}
          const other = planningById.get(
            String(props.grid_id)
          )

          if (!other) {
            throw new Error(
              `Grid ${props.grid_id} missing from planning GIS.`
            )
          }

          const cp =
            props.primary_conservation_priority_0_1

          const planningCp =
            other.frozen_conservation_priority_0_1

          if (
            validScore(cp) &&
            validScore(planningCp)
          ) {
            compared += 1

            if (Math.abs(cp - planningCp) > 1e-5) {
              mismatches += 1
            }
          }
        }

        if (
          compared !== cpScored ||
          mismatches > 0
        ) {
          throw new Error(
            'Conservation scores do not match the frozen planning map.'
          )
        }

        setData({
          overview,
          conservationMap,
          planningMap,
          gnMap,
          gnDivisions,
        })
      })
      .catch((reason) => {
        if (!controller.signal.aborted) {
          setError(
            reason?.message ||
            'Unable to load conservation data.'
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
  // 2B. FETCH EXACT SELECTED GRID DETAILS
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

    urbanGreenApi.conservation.grid(
      selectedGridId,
      controller.signal
    )
      .then((result) => {
        if (!controller.signal.aborted) {
          setGridDetails(
            extractDetails(result)
          )
        }
      })
      .catch((reason) => {
        if (!controller.signal.aborted) {
          setGridError(
            reason?.message ||
            'Unable to retrieve grid details.'
          )
        }
      })

    return () => controller.abort()
  }, [selectedGridId, data])

  // ---------------------------------------------------
  // 2C. INDEX ORIGINAL RESEARCH CELLS
  // ---------------------------------------------------

  const conservationById = useMemo(
    () => new Map(
      (
        data?.conservationMap?.features || []
      ).map((feature) => [
        String(feature.properties?.grid_id),
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
        String(feature.properties?.grid_id),
        feature.properties,
      ])
    ),
    [data]
  )

  const selectedBase = selectedGridId
    ? conservationById.get(String(selectedGridId))
    : null

  const selectedEvidence = selectedBase
    ? {
        ...selectedBase,
        ...(gridDetails || {}),
      }
    : null

  const selectedPlanning = selectedGridId
    ? planningById.get(String(selectedGridId))
    : null

  const selectedGn = (
    data?.gnDivisions?.data || []
  ).find(
    (gn) =>
      String(gn.gn_id) ===
      String(selectedGnId)
  )

  // ---------------------------------------------------
  // 2D. PREPARE REAL DISTRIBUTION CHARTS
  // ---------------------------------------------------

  const giHistogram = useMemo(
    () => createHistogram(
      data?.conservationMap?.features || [],
      'green_importance_index_0_1'
    ),
    [data]
  )

  const cpHistogram = useMemo(
    () => createHistogram(
      data?.conservationMap?.features || [],
      'primary_conservation_priority_0_1'
    ),
    [data]
  )

  // ---------------------------------------------------
  // 2E. HIGHEST RELATIVE CP SCORES
  // ---------------------------------------------------

  const topGrids = useMemo(
    () => (
      data?.conservationMap?.features || []
    )
      .map((feature) => feature.properties)
      .filter(
        (props) =>
          validScore(
            props?.primary_conservation_priority_0_1
          )
      )
      .sort(
        (a, b) =>
          b.primary_conservation_priority_0_1 -
          a.primary_conservation_priority_0_1
      )
      .slice(0, 8),
    [data]
  )

  // ---------------------------------------------------
  // 2F. SELECTED GRID GI AND CP CALCULATIONS
  // ---------------------------------------------------

  const giScore =
    selectedEvidence?.green_importance_index_0_1

  const futurePressure =
    selectedEvidence?.future_urbanization_pressure_score

  const priorityScore =
    selectedEvidence?.primary_conservation_priority_0_1

  const giContributionRows = GI_CRITERIA.map(
    (criterion) => ({
      ...criterion,
      score: selectedEvidence?.[criterion.field],
      weighted:
        selectedEvidence?.[criterion.contribution],
    })
  )

  const canCheckGI =
    validScore(giScore) &&
    giContributionRows.every(
      (item) => validScore(item.weighted)
    )

  const calculatedGI = canCheckGI
    ? giContributionRows.reduce(
        (total, item) => total + item.weighted,
        0
      )
    : null

  const canCheckCP =
    validScore(giScore) &&
    validScore(futurePressure) &&
    validScore(priorityScore)

  const calculatedCP = canCheckCP
    ? giScore * futurePressure
    : null

  // ===================================================
  // 3. LOADING AND ERROR STATES
  // ===================================================

  if (loading) {
    return (
      <section className="card-pad">
        <h2 className="section-title">
          Loading Green Conservation Intelligence
        </h2>

        <p className="mt-3 text-sm text-slate-500">
          Reading frozen Green Importance,
          Conservation Priority, model-domain
          coverage and GIS layers...
        </p>
      </section>
    )
  }

  if (error || !data) {
    return (
      <section className="card-pad">
        <h2 className="section-title">
          Conservation data unavailable
        </h2>

        <p className="mt-3 text-sm text-red-700">
          {error || 'No research data returned.'}
        </p>

        <p className="mt-2 text-xs text-slate-500">
          Check FastAPI and the conservation
          research exports. No missing values
          have been replaced with zero.
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

  const counts = overview.status_counts || {}
  const stats = overview.distribution || {}
  const sensitivity = overview.sensitivity || {}
  const methodology = overview.methodology || {}

  // ===================================================
  // 4. PAGE RENDERING
  // ===================================================

  return (
    <div>
      <PageHeader
        eyebrow="Urban Green Intelligence"
        title="Green Conservation Intelligence"
        subtitle="Explore frozen Green Importance and relative Conservation Priority, including grid-level evidence and explicit model-domain limitations."
        badges={[
          'Green Importance · CRITIC-weighted',
          'Conservation Priority · GI × Pressure',
          'Original 100 m research grids',
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

      {/* ============================================ */}
      {/* 4A. SUMMARY CARDS                            */}
      {/* ============================================ */}

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          title="Green Importance available"
          value={fmtCount(
            overview.green_importance_scored_count
          )}
          subtitle="Current green evidence and historical context available"
          icon={Leaf}
          tone="green"
        />

        <MetricCard
          title="Conservation Priority available"
          value={fmtCount(
            overview.primary_priority_scored_count
          )}
          subtitle="Cells with both GI and future pressure"
          icon={ShieldCheck}
          tone="teal"
        />

        <MetricCard
          title="Priority unavailable"
          value={fmtCount(
            overview.primary_priority_unscored_count
          )}
          subtitle="Missing joint evidence is not low priority"
          icon={ShieldAlert}
          tone="orange"
        />

        <MetricCard
          title="Median priority score"
          value={fmt(stats.median, 4)}
          subtitle="Within the 4,609 scored cells only"
          icon={MapPinned}
          tone="blue"
        />
      </div>

      {/* ============================================ */}
      {/* 4B. MAP AND GRID-LEVEL EVIDENCE              */}
      {/* ============================================ */}

      <div className="mt-5 grid gap-5 2xl:grid-cols-[1.55fr_.85fr]">
        <section className="card-pad min-w-0">
          <SectionTitle
            title="Conservation Priority GIS map"
            description="Frozen relative CP scores. The map uses the original 100 m cells and your existing validated 200 m overview for display."
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
            metric="conservation"
            selectedGridId={selectedGridId}
            selectedGnId={selectedGnId}
            onSelect={setSelectedGridId}
            onClearGn={() => selectGn(null)}
            height={580}
          />

          {/* <p className="mt-3 text-[11px] leading-5 text-slate-500">
            The map shows the frozen Conservation
            Priority layer, not the separate Green
            Importance index. GI is displayed in the
            grid evidence and distribution below.
            Gray/unscored areas have no calculated
            CP score; they are not classified as
            having low conservation value.
          </p> */}
        </section>

        {/* SELECTED GRID */}

        <div className="space-y-5">
          <section className="card-pad">
            <SectionTitle
              title="Selected original grid evidence"
              description="Inspect Green Importance, pressure, priority and input availability for one 100 m cell."
            />

            {!selectedGridId ? (
              <p className="text-sm leading-6 text-slate-500">
                Click a grid on the GIS map to
                inspect the original conservation
                evidence and its calculation.
              </p>
            ) : (
              <>
                <dl>
                  <InfoRow
                    label="Grid ID"
                    value={selectedGridId}
                  />

                  <InfoRow
                    label="GN division"
                    value={
                      selectedPlanning?.gn_name ||
                      'Unavailable'
                    }
                  />

                  <InfoRow
                    label="Current green condition"
                    value={displayStatus(
                      selectedEvidence
                        ?.current_green_condition_status
                    )}
                  />

                  <InfoRow
                    label="Green Importance availability"
                    value={displayStatus(
                      selectedEvidence
                        ?.green_importance_status
                    )}
                  />

                  <InfoRow
                    label="Green Importance (GI)"
                    value={
                      validScore(giScore)
                        ? fmt(giScore, 6)
                        : 'Not calculated'
                    }
                  />

                  <InfoRow
                    label="Future pressure domain"
                    value={displayStatus(
                      selectedEvidence
                        ?.future_pressure_domain_status
                    )}
                  />

                  <InfoRow
                    label="Future Urbanization Pressure"
                    value={
                      validScore(futurePressure)
                        ? fmt(futurePressure, 6)
                        : 'Outside domain / unavailable'
                    }
                  />

                  <InfoRow
                    label="Conservation Priority (CP)"
                    value={
                      validScore(priorityScore)
                        ? fmt(priorityScore, 6)
                        : 'Not calculated'
                    }
                  />

                  <InfoRow
                    label="CP assessment status"
                    value={displayStatus(
                      selectedEvidence
                        ?.primary_priority_status
                    )}
                  />

                  <InfoRow
                    label="Policy priority class"
                    value="Not assigned"
                  />
                </dl>

                {gridError && (
                  <p className="mt-3 text-xs text-amber-800">
                    Detail API: {gridError}
                  </p>
                )}

                {/* GI FORMULA INPUTS */}

                <div className="mt-4">
                  <p className="text-xs font-bold text-slate-800">
                    Frozen Green Importance criteria
                  </p>

                  <div className="mt-3 space-y-2">
                    {giContributionRows.map((item) => (
                      <div
                        key={item.id}
                        className="rounded-xl border border-slate-200 bg-slate-50 p-3"
                      >
                        <div className="flex items-center justify-between gap-2 text-xs">
                          <span className="font-semibold text-slate-800">
                            {item.id} · {item.label}
                          </span>

                          <span className="font-semibold text-emerald-800">
                            {fmt(item.weight * 100, 2)}%
                          </span>
                        </div>

                        <div className="mt-2 flex flex-wrap justify-between gap-2 text-[11px] text-slate-600">
                          <span>
                            Score: {fmt(item.score, 4)}
                          </span>

                          <span>
                            Contribution: {fmt(
                              item.weighted, 4
                            )}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {canCheckGI && (
                  <p className="mt-3 rounded-lg bg-emerald-50 p-3 text-xs leading-5 text-emerald-900">
                    GI contribution sum:{' '}
                    <strong>{fmt(calculatedGI, 6)}</strong>

                    {' '}· Matches saved GI:{' '}

                    <strong>
                      {Math.abs(
                        calculatedGI - giScore
                      ) < 1e-5
                        ? 'Yes'
                        : 'Review needed'}
                    </strong>
                  </p>
                )}

                {canCheckCP && (
                  <p className="mt-3 rounded-lg bg-sky-50 p-3 text-xs leading-5 text-sky-900">
                    Frozen CP calculation:{' '}
                    {fmt(giScore, 6)}
                    {' × '}
                    {fmt(futurePressure, 6)}
                    {' = '}
                    <strong>
                      {fmt(calculatedCP, 6)}
                    </strong>

                    {' '}· Matches saved CP:{' '}

                    <strong>
                      {Math.abs(
                        calculatedCP - priorityScore
                      ) < 1e-5
                        ? 'Yes'
                        : 'Review needed'}
                    </strong>
                  </p>
                )}

                {!validScore(priorityScore) && (
                  <p className="mt-3 rounded-lg border border-amber-200 bg-amber-50 p-3 text-xs leading-5 text-amber-900">
                    Conservation Priority is not
                    calculated because both required
                    scores are not jointly available.
                    Missing CP is not evidence of
                    low conservation importance.
                  </p>
                )}
              </>
            )}
          </section>

          {/* GN CONTEXT */}

          {selectedGn && (
            <section className="card-pad">
              <SectionTitle
                title="Selected GN context"
                description={selectedGn.gn_name}
              />

              <dl>
                <InfoRow
                  label="GN area inside KMC"
                  value={`${fmt(
                    selectedGn.gn_area_inside_kmc_km2,
                    3
                  )} km²`}
                />

                <InfoRow
                  label="Area-weighted Conservation Priority"
                  value={fmt(
                    selectedGn.conservation_area_weighted_mean,
                    4
                  )}
                />
              </dl>

              <p className="mt-2 text-[11px] leading-5 text-slate-500">
                The GN-level CP summary is area
                weighted and applies only where
                the necessary priority evidence
                is available. It is not the
                conservation value of every
                location in the GN division.
              </p>
            </section>
          )}
        </div>
      </div>

      {/* ============================================ */}
      {/* 4C. GI AND CP HISTOGRAMS                     */}
      {/* ============================================ */}

      <div className="mt-5 grid gap-5 xl:grid-cols-2">
        <ScoreHistogram
          title="Green Importance score distribution"
          description="Frozen GI scores for the 8,727 eligible green-evidence cells."
          data={giHistogram}
          color="#15803d"
          note="Green Importance represents current green condition, historical persistence and local green context. Numeric intervals are descriptive, not conservation classes."
        />

        <ScoreHistogram
          title="Conservation Priority score distribution"
          description="Relative CP scores for the 4,609 cells where both GI and pressure are available."
          data={cpHistogram}
          color="#0f766e"
          note="The 4,631 cells without CP are deliberately excluded from the histogram rather than counted as zero. No official high/medium/low priority thresholds were defined."
        />
      </div>

      {/* ============================================ */}
      {/* 4D. MISSING EVIDENCE AND COVERAGE            */}
      {/* ============================================ */}

      <section className="card-pad mt-5">
        <SectionTitle
          title="Conservation Priority evidence coverage"
          description="Reasons why a grid may have an available Green Importance score but no final Conservation Priority score."
        />

        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {MISSING_STATES.map((item) => (
            <div
              key={item.key}
              className="rounded-xl border border-slate-200 bg-slate-50 p-4"
            >
              <p className="text-xs leading-5 text-slate-600">
                {item.label}
              </p>

              <p className="mt-2 text-2xl font-bold text-slate-900">
                {fmtCount(
                  counts[item.key]
                )}
              </p>
            </div>
          ))}

          <div className="rounded-xl border border-emerald-200 bg-emerald-50 p-4">
            <p className="text-xs leading-5 text-emerald-800">
              Both GI and pressure available
            </p>

            <p className="mt-2 text-2xl font-bold text-emerald-900">
              {fmtCount(
                counts.FUTURE_CONSERVATION_PRIORITY_CALCULATED
              )}
            </p>
          </div>
        </div>

        <p className="mt-4 text-xs leading-6 text-slate-600">
          <strong>Why this matters:</strong>{' '}
          CP is available only in the intersection
          of the GI evidence domain and XGBoost
          future-pressure model domain.

          {' '}These availability states must not
          be converted into LOW priority categories.
        </p>
      </section>

      {/* ============================================ */}
      {/* 4E. HIGHEST SCORED CELLS                     */}
      {/* ============================================ */}

      <section className="card-pad mt-5">
        <SectionTitle
          title="Highest relative Conservation Priority scores"
          description="Eight original 100 m cells with the highest saved continuous CP values. These are descriptive model-screening ranks, not official conservation designations."
        />

        <div className="overflow-x-auto">
          <table className="min-w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50 text-slate-600">
                <th className="px-3 py-3">
                  Grid ID
                </th>

                <th className="px-3 py-3">
                  Green Importance
                </th>

                <th className="px-3 py-3">
                  Relative Pressure
                </th>

                <th className="px-3 py-3">
                  Conservation Priority
                </th>
              </tr>
            </thead>

            <tbody>
              {topGrids.map((row) => (
                <tr
                  key={row.grid_id}
                  className="border-b border-slate-100 last:border-0"
                >
                  <td className="px-3 py-3 font-mono font-semibold text-slate-800">
                    {row.grid_id}
                  </td>

                  <td className="px-3 py-3 tabular-nums">
                    {fmt(
                      row.green_importance_index_0_1,
                      4
                    )}
                  </td>

                  <td className="px-3 py-3 tabular-nums">
                    {fmt(
                      row.future_urbanization_pressure_score,
                      4
                    )}
                  </td>

                  <td className="px-3 py-3 font-bold tabular-nums text-emerald-800">
                    {fmt(
                      row.primary_conservation_priority_0_1,
                      4
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <p className="mt-3 text-[11px] leading-5 text-slate-500">
          A higher relative CP score identifies
          stronger joint Green Importance and
          modelled pressure under the frozen formula.
          It does not establish an observed future
          loss event or a statistically validated
          ecological risk.
        </p>
      </section>

      {/* ============================================ */}
      {/* 4F. METHODOLOGY AND SENSITIVITY              */}
      {/* ============================================ */}

      <section className="card-pad mt-5">
        <SectionTitle
          title="Research methodology and robustness"
          description="The Green Importance and Conservation Priority models remain separate, frozen research stages."
        />

        <div className="grid gap-3 md:grid-cols-3">
          <div className="rounded-xl bg-slate-50 p-4 text-xs leading-6 text-slate-700">
            <p className="font-bold text-slate-800">
              Green Importance
            </p>

            <p className="mt-2">
              {methodology.green_importance ||
                'CRITIC-weighted GI evidence'}

              {' '}The three selected criteria
              are combined using documented
              objective weights.
            </p>
          </div>

          <div className="rounded-xl bg-slate-50 p-4 text-xs leading-6 text-slate-700">
            <p className="font-bold text-slate-800">
              Conservation Priority
            </p>

            <p className="mt-2">
              Formula:{' '}
              <strong>
                {methodology.primary_priority_formula ||
                  'CP_i = GI_i × UP_i'}
              </strong>

              {' '}Uses frozen Green Importance
              and frozen 2025-origin pressure
              for the combined 2026–2028
              assessment.
            </p>
          </div>

          <div className="rounded-xl bg-slate-50 p-4 text-xs leading-6 text-slate-700">
            <p className="font-bold text-slate-800">
              Historical loss evidence
            </p>

            <p className="mt-2">
              {methodology.historical_loss_role ||
                'Supporting historical diagnostics only.'}

              {' '}It is not an additional
              weighted CP component.
            </p>
          </div>
        </div>

        <div className="mt-4 grid gap-3 sm:grid-cols-3">
          <div className="rounded-xl border border-slate-200 p-4">
            <p className="text-xs text-slate-500">
              Sensitivity scenarios including primary
            </p>

            <p className="mt-2 text-xl font-bold">
              {fmtCount(
                sensitivity.scenarios_including_primary
              )}
            </p>
          </div>

          <div className="rounded-xl border border-slate-200 p-4">
            <p className="text-xs text-slate-500">
              Minimum alternative top-10% retention
            </p>

            <p className="mt-2 text-xl font-bold">
              {validScore(
                sensitivity.lowest_alternative_top10_retention_pct
              )
                ? `${fmt(
                    sensitivity.lowest_alternative_top10_retention_pct,
                    2
                  )}%`
                : 'Unavailable'}
            </p>
          </div>

          <div className="rounded-xl border border-slate-200 p-4">
            <p className="text-xs text-slate-500">
              Lowest alternative Spearman correlation
            </p>

            <p className="mt-2 text-xl font-bold">
              {fmt(
                sensitivity.lowest_alternative_spearman,
                4
              )}
            </p>
          </div>
        </div>

        {/* <p className="mt-4 rounded-xl border border-amber-200 bg-amber-50 p-4 text-xs leading-6 text-amber-900">
          <strong>
            Scientific and planning limitations:
          </strong>{' '}

          Green Importance is a relative index of
          selected vegetation characteristics,
          not a field-validated biodiversity
          assessment. Urbanization Pressure is
          an XGBoost-derived relative score,
          not a calibrated development probability.

          {' '}Their multiplication does not
          produce an ecological-loss probability.

          {' '}Sensitivity agreement evaluates
          the stability of model rankings,
          not ground-truth accuracy.

          {' '}No HIGH/MEDIUM/LOW legal conservation
          categories have been approved.
          The provisional UDA zoning GIS is
          context only, not legal validation.
        </p> */}

        <Link
          to="/green/planning-conflict"
          className="mt-4 inline-flex items-center gap-2 text-xs font-semibold text-emerald-800 hover:underline"
        >
          Next: Planning Conflict Intelligence →
        </Link>
      </section>
    </div>
  )
}
