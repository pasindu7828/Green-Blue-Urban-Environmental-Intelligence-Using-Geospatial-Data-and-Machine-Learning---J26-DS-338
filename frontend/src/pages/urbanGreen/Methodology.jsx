
import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  BookOpen,
  Database,
  FileCheck2,
  GitBranch,
  MapPinned,
  RefreshCcw,
  ShieldCheck,
  ShieldAlert,
} from 'lucide-react'

import PageHeader from '../../components/shared/PageHeader'
import MetricCard from '../../components/shared/MetricCard'
import { urbanGreenApi } from '../../features/urbanGreen/services/urbanGreenApi'
import { apiRequest } from '../../features/urbanHeat/services/apiClient'

// =====================================================
// 1. RESEARCH REFERENCE NOTES
// Live numeric summaries and QA come from the backend.
// =====================================================

const WORKFLOW = [
  {
    number: '01',
    title: 'Historical Landsat land-cover analysis',
    method: 'Random Forest LULC classification',
    detail:
      'Annual Landsat surface-reflectance composites, four-class LULC classification, quality auditing and three-year centered retrospective stabilization.',
    result:
      '2011–2024 stabilized historical core; 2010 and 2025 are raw context; 2026 is provisional.',
    route: '/green/green-cover',
  },
  {
    number: '02',
    title: 'Historical Green-to-Built transitions',
    method: 'Multi-year consensus comparisons',
    detail:
      'Compare 2012–2014 against 2021–2023 using robust 2-of-3 consensus and conservative strict 3-of-3 consensus.',
    result:
      'Retrospective conversion estimates, not a statistical confidence interval or future forecast.',
    route: '/green/transitions',
  },
  {
    number: '03',
    title: 'Future Urbanization Pressure',
    method: 'Random Forest versus XGBoost; selected XGBoost',
    detail:
      'Historical three-year-forward non-built-to-built targets; leakage-aware temporal and spatial validation, tuning, held-out temporal evaluation, then production refit.',
    result:
      'One combined 2026–2028 relative pressure score from the 2025 predictor origin.',
    route: '/green/pressure',
  },
  {
    number: '04',
    title: 'Explainable AI',
    method: 'SHAP TreeExplainer',
    detail:
      'Explain frozen XGBoost behaviour for scored cells using 55 features, global importance, five interpretation domains and local feature contributions.',
    result:
      'Contributions are on the XGBoost raw-margin scale, not probabilities or causal effects.',
    route: '/green/xai',
  },
  {
    number: '05',
    title: 'Development Suitability',
    method: 'Independent expert-AHP GIS-MCDA',
    detail:
      'Standardize major-road proximity, local road density and slope; calculate expert pairwise weights, consistency and continuous weighted suitability.',
    result:
      'Relative suitability for 9,240 original 100 m cells; not building permission.',
    route: '/green/suitability',
  },
  {
    number: '06',
    title: 'Green Importance and Conservation Priority',
    method: 'CRITIC-weighted GI and frozen GI × future pressure',
    detail:
      'Combine current green condition, historical green persistence and local green context, then multiply GI by frozen pressure only where both values exist.',
    result:
      'Conservation Priority is a relative index, not an ecological-loss probability.',
    route: '/green/conservation',
  },
  {
    number: '07',
    title: 'Planning Conflict and GN screening',
    method: 'Deterministic review rules and spatial grouping',
    detail:
      'Retain Development Conflict Index and Conservation Priority as distinct evidence, apply documented planning-review rules and perform separate top-decile spatial screening.',
    result:
      'Exploratory intervention and hotspot candidates, not official planning classifications.',
    route: '/green/planning-conflict',
  },
]

// =====================================================
// 2. RESEARCH DATA SOURCES
// =====================================================

const SOURCES = [
  [
    'Satellite imagery',
    'Landsat annual 2010–2026 composites',
    '2026 imagery is partial/provisional.',
  ],
  [
    'External reference',
    'ESA WorldCover',
    'Reference benchmarking, not independently verified ground truth.',
  ],
  [
    'Administrative geography',
    'KMC study boundary and GN divisions',
    'GN polygon intersections support area-weighted reporting.',
  ],
  [
    'Present-day urban context',
    'OSM roads and buildings',
    'Current OSM context is not backdated into historical pressure training.',
  ],
  [
    'Terrain',
    'DEM-derived elevation and slope',
    'Terrain is used within the appropriate research stages.',
  ],
  [
    'Planning context',
    'Development Plan and UDA provisional zoning GIS',
    'Not used as proof of legal zoning compliance.',
  ],
]

// =====================================================
// 3. RESEARCH LIMITATIONS
// =====================================================

const LIMITATIONS = [
  'Annual LULC outputs are model classifications; complete independent annual ground truth is unavailable.',

  'The 2021 LULC spatial holdout is an internal evaluation, not a new accuracy estimate for every annual map.',

  'Centered three-year stabilized LULC uses surrounding years and is retrospective; the future-pressure predictor uses raw, time-appropriate features instead.',

  'The pressure conversion target is derived from classified historical land cover and contains relatively few positive events.',

  'The 2022 holdout evaluated the locked model configuration before the final 2012–2022 production refit; the refitted model has no separate untouched holdout.',

  'The frozen 2025-origin pressure score is relative, not a calibrated 2026–2028 development probability.',

  'SHAP importance measures model explanation magnitude, not causality or feature contributions to probability.',

  'AHP suitability assesses selected road and slope criteria, not flood safety, legal construction permission or complete engineering feasibility.',

  'Green Importance is not a field-validated biodiversity measure; Conservation Priority is unavailable outside the joint GI and pressure domain.',

  'Development Conflict and Conservation Priority must remain separate; no new combined official planning index was approved.',

  'Spatial candidate hotspot groups were not tested for statistical significance and may overlap across DCI and CP screening streams.',

  'The UDA zoning GIS is provisional planning context, not a validated legal parcel-level determination.',
]

// =====================================================
// 4. SAVED 2022 HOLDOUT EVALUATION
// Documented pre-refit temporal test results.
// These are NOT new frontend evaluations.
// =====================================================

const HELD_OUT_2022 = [
  ['PR-AUC', '0.063039'],
  ['PR-AUC lift over prevalence', '6.440×'],
  ['ROC-AUC', '0.804594'],
  ['Precision at locked 0.715 threshold', '0.057915'],
  ['Recall at locked 0.715 threshold', '0.333333'],
  ['F1 at locked 0.715 threshold', '0.098684'],
  ['Balanced accuracy', '0.639865'],
  ['2022 positive events', '45'],
]

// =====================================================
// 5. BACKEND QA / METADATA ENDPOINTS
// These optional endpoints must not be assumed healthy.
// =====================================================

const OPTIONAL_ENDPOINTS = [
  {
    key: 'metadata',
    label: 'Research metadata',
    route: '/metadata',
  },
  {
    key: 'manifest',
    label: 'Release manifest',
    route: '/manifest',
  },
  {
    key: 'dataHealth',
    label: 'Backend data health',
    route: '/data-health',
  },
]

// =====================================================
// 6. FORMATTING FUNCTIONS
// =====================================================

function finiteNumber(value) {
  return typeof value === 'number' &&
    Number.isFinite(value)
}

function fmt(value, digits = 3) {
  return finiteNumber(value)
    ? value.toLocaleString('en-US', {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
      })
    : 'Unavailable'
}

function count(value) {
  return finiteNumber(value)
    ? value.toLocaleString('en-US', {
        maximumFractionDigits: 0,
      })
    : 'Unavailable'
}

// =====================================================
// 7. REUSABLE UI COMPONENTS
// =====================================================

function Row({ label, value }) {
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

function Heading({ title, description }) {
  return (
    <div className="mb-4">
      <h2 className="section-title">
        {title}
      </h2>

      {description && (
        <p className="mt-1 text-xs leading-5 text-slate-500">
          {description}
        </p>
      )}
    </div>
  )
}

function QaRow({
  name,
  result,
  details,
  passed,
}) {
  return (
    <div className="grid gap-2 border-b border-slate-100 py-3 last:border-0 sm:grid-cols-[1.1fr_.45fr_1.6fr] sm:items-start">
      <p className="text-xs font-semibold text-slate-800">
        {name}
      </p>

      <span
        className={`text-xs font-bold ${
          passed === true
            ? 'text-emerald-700'
            : passed === false
              ? 'text-amber-800'
              : 'text-slate-600'
        }`}
      >
        {result}
      </span>

      <p className="text-xs leading-5 text-slate-500">
        {details}
      </p>
    </div>
  )
}

// =====================================================
// 8. BACKEND DATA HEALTH RECORD DISPLAY
// =====================================================

function AuditSource({ record }) {
  const details =
    record.data &&
    typeof record.data === 'object'
      ? Object.entries(record.data)
          .filter(([, value]) =>
            value === null ||
            ['number', 'string', 'boolean'].includes(
              typeof value
            )
          )
          .slice(0, 12)
      : []

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-sm font-semibold text-slate-800">
            {record.label}
          </p>

          <p className="mt-1 font-mono text-[11px] text-slate-500">
            {record.route}
          </p>
        </div>

        <span
          className={`text-xs font-semibold ${
            record.ok
              ? 'text-emerald-800'
              : 'text-amber-800'
          }`}
        >
          {record.ok ? 'Available' : 'Unavailable'}
        </span>
      </div>

      {record.ok ? (
        <>
          {details.length > 0 && (
            <dl className="mt-3">
              {details.map(([key, value]) => (
                <Row
                  key={key}
                  label={key.replaceAll('_', ' ')}
                  value={String(value ?? 'null')}
                />
              ))}
            </dl>
          )}

          <details className="mt-3">
            <summary className="cursor-pointer text-xs font-semibold text-emerald-800">
              View raw backend record
            </summary>

            <pre className="mt-2 max-h-72 overflow-auto rounded-lg bg-slate-900 p-3 text-[10px] leading-4 text-slate-100">
              {JSON.stringify(record.data, null, 2)?.slice(
                0,
                16000
              )}
            </pre>

            <p className="mt-1 text-[10px] text-slate-500">
              Preview is limited to 16,000 characters.
            </p>
          </details>
        </>
      ) : (
        <p className="mt-2 text-xs leading-5 text-amber-800">
          {record.error}. This source has not been
          treated as passing QA.
        </p>
      )}
    </div>
  )
}

// =====================================================
// 9. MAIN METHODOLOGY PAGE
// =====================================================

export default function Methodology() {
  const [data, setData] = useState(null)
  const [extras, setExtras] = useState([])

  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [refresh, setRefresh] = useState(0)

  // ---------------------------------------------------
  // 9A. LOAD RESEARCH AND QA ENDPOINTS
  // ---------------------------------------------------

  useEffect(() => {
    const controller = new AbortController()
    const signal = controller.signal

    setLoading(true)
    setError('')
    setExtras([])

    async function load() {
      try {
        const [
          historical,
          annualQa,
          transition,
          pressure,
          xai,
          suitability,
          conservation,
          gn,
          hotspots,
        ] = await Promise.all([
          urbanGreenApi.historical.summary(signal),

          urbanGreenApi.historical.annualQa(signal),

          urbanGreenApi.historical.transitions.analysis(
            signal
          ),

          urbanGreenApi.pressure.overview(signal),

          urbanGreenApi.pressure.xai.overview(signal),

          urbanGreenApi.suitability.overview(signal),

          urbanGreenApi.conservation.overview(signal),

          urbanGreenApi.planning.gnDivisions(signal),

          urbanGreenApi.planning.hotspots({}, signal),
        ])

        // Validate existing frozen research contracts.

        if (
          historical?.core_period !== '2011-2024' ||
          annualQa?.overall_step09_status !== 'PASS' ||
          transition?.overall_step10b_status !== 'PASS' ||
          pressure?.total_grid_cells !== 9240 ||
          pressure?.eligible_scored_cells !== 4620 ||
          xai?.research_qa?.['15a'] !== 'PASS' ||
          xai?.research_qa?.['15b'] !== 'PASS' ||
          suitability?.grid_count !== 9240 ||
          conservation?.green_importance_scored_count !==
            8727 ||
          conservation?.primary_priority_scored_count !==
            4609 ||
          gn?.total !== 57 ||
          hotspots?.total !== 58
        ) {
          throw new Error(
            'A required research endpoint or frozen QA/count contract does not match the documented final release.'
          )
        }

        if (signal.aborted) return

        setData({
          historical,
          annualQa,
          transition,
          pressure,
          xai,
          suitability,
          conservation,
          gn,
          hotspots,
        })

        // General metadata / manifest / health requests.
        // These are optional until verified.
        // Failed requests must remain unavailable.

        const optional = await Promise.allSettled(
          OPTIONAL_ENDPOINTS.map((item) =>
            apiRequest(
              `/api/urban-green${item.route}`,
              { signal }
            )
          )
        )

        if (signal.aborted) return

        setExtras(
          OPTIONAL_ENDPOINTS.map((item, index) => ({
            ...item,

            ok:
              optional[index].status === 'fulfilled',

            data:
              optional[index].status === 'fulfilled'
                ? optional[index].value
                : null,

            error:
              optional[index].status === 'rejected'
                ? String(
                    optional[index].reason?.message ||
                      'Request failed'
                  )
                : '',
          }))
        )
      } catch (reason) {
        if (!signal.aborted) {
          setError(
            reason?.message ||
              'Could not load research QA'
          )
        }
      } finally {
        if (!signal.aborted) {
          setLoading(false)
        }
      }
    }

    load()

    return () => controller.abort()
  }, [refresh])

  // ---------------------------------------------------
  // 9B. RESOLVE FINAL AHP CRITERIA
  // ---------------------------------------------------

  const ahpCriteria = useMemo(() => {
    if (!data?.suitability?.criteria) return []

    return ['C1', 'C2', 'C4'].map((id) => ({
      id,

      name:
        data.suitability.criteria[id]?.name ||
        id,

      weight:
        data.suitability.criteria[id]?.weight,
    }))
  }, [data])

  // ===================================================
  // 10. LOADING AND ERROR STATES
  // ===================================================

  if (loading) {
    return (
      <section className="card-pad">
        <h2 className="section-title">
          Loading methodology and QA evidence
        </h2>

        <p className="mt-3 text-sm text-slate-500">
          Reading existing historical, pressure,
          SHAP, AHP, conservation and planning
          records...
        </p>
      </section>
    )
  }

  if (error || !data) {
    return (
      <section className="card-pad">
        <h2 className="section-title">
          Research QA could not be verified
        </h2>

        <p className="mt-3 text-sm text-red-700">
          {error || 'No research results returned.'}
        </p>

        <p className="mt-2 text-xs text-slate-500">
          The page does not replace missing backend
          evidence with demonstration values.
          Confirm that the Urban Green FastAPI
          server is running.
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

  // ===================================================
  // 11. RESEARCH DATA AND QA SUMMARY
  // ===================================================

  const {
    historical,
    annualQa,
    transition,
    pressure,
    xai,
    suitability,
    conservation,
    gn,
    hotspots,
  } = data

  const expert =
    suitability.expert_consultation || {}

  const ahpCr =
    expert.consistency_ratio

  const scoreCount =
    suitability.scored_grid_count

  const criteriaWeightSum =
    ahpCriteria.reduce(
      (sum, row) =>
        sum +
        (
          finiteNumber(row.weight)
            ? row.weight
            : 0
        ),
      0
    )

  const missingCp =
    conservation.primary_priority_unscored_count

  // Live quality-assurance records.
  // Each row identifies exactly what was checked.

  const qaRows = [
    {
      name: 'Annual LULC maps and statistics',

      result:
        annualQa.overall_step09_status,

      passed:
        annualQa.overall_step09_status === 'PASS',

      details:
        `${count(annualQa.raw_map_count)} raw and ` +
        `${count(annualQa.stabilized_map_count)} ` +
        'stabilized annual maps; raster QA, class and temporal checks.',
    },
    {
      name: 'Historical multi-year transitions',

      result:
        transition.overall_step10b_status,

      passed:
        transition.overall_step10b_status === 'PASS',

      details:
        `Early ${transition.early_window}; late ` +
        `${transition.late_window}; robust and strict consensus outputs.`,
    },
    {
      name: 'Pressure inference coverage',

      result:
        pressure.eligible_scored_cells === 4620
          ? 'MATCH'
          : 'REVIEW',

      passed:
        pressure.eligible_scored_cells === 4620,

      details:
        `${count(pressure.eligible_scored_cells)} scored of ` +
        `${count(pressure.total_grid_cells)} cells; ` +
        `${count(pressure.outside_inference_domain_cells)} unscored.`,
    },
    {
      name: 'SHAP features and explanations',

      result:
        xai.research_qa['15a'],

      passed:
        xai.research_qa['15a'] === 'PASS',

      details:
        `${count(
          xai.groups?.all_scored?.feature_count
        )} features; explanations on raw XGBoost margin.`,
    },
    {
      name: 'SHAP domain explanations',

      result:
        xai.research_qa['15b'],

      passed:
        xai.research_qa['15b'] === 'PASS',

      details:
        `${count(
          xai.groups?.all_scored?.domain_count
        )} domains across ` +
        `${count(
          xai.groups?.all_scored?.explained_grid_count
        )} scored cells.`,
    },
    {
      name: 'AHP pairwise consistency',

      result:
        finiteNumber(ahpCr)
          ? `CR ${fmt(ahpCr, 3)}`
          : 'UNAVAILABLE',

      passed:
        finiteNumber(ahpCr)
          ? ahpCr < 0.10
          : null,

      details:
        'Consistency measures the expert pairwise judgments; it is not predictive model accuracy.',
    },
    {
      name: 'AHP final weights',

      result:
        Math.abs(criteriaWeightSum - 1) < 1e-6
          ? 'SUM = 1'
          : 'REVIEW',

      passed:
        Math.abs(criteriaWeightSum - 1) < 1e-6,

      details:
        'Three final independent criteria: C1, C2 and C4.',
    },
    {
      name: 'Suitability grid coverage',

      result:
        scoreCount === 9240
          ? 'MATCH'
          : 'REVIEW',

      passed:
        scoreCount === 9240,

      details:
        'Continuous GIS-MCDA scores for all 9,240 original 100 m cells.',
    },
    {
      name: 'Conservation joint-domain coverage',

      result:
        conservation.primary_priority_scored_count +
          missingCp ===
        9240
          ? 'MATCH'
          : 'REVIEW',

      passed:
        conservation.primary_priority_scored_count +
          missingCp ===
        9240,

      details:
        `${count(
          conservation.primary_priority_scored_count
        )} CP scored; ` +
        `${count(missingCp)} unavailable, not zero priority.`,
    },
    {
      name: 'GN and candidate-group records',

      result:
        gn.total === 57 &&
        hotspots.total === 58
          ? 'MATCH'
          : 'REVIEW',

      passed:
        gn.total === 57 &&
        hotspots.total === 58,

      details:
        '57 GN division summaries and 58 selected exploratory hotspot groups.',
    },
  ]

  // ===================================================
  // 12. COMPLETE PAGE CONTENT
  // ===================================================

  return (
    <div>
      <PageHeader
        eyebrow="Urban Green Intelligence"
        title="Research Methodology and Data Quality"
        subtitle="Trace the complete research workflow, evaluate saved validation evidence and inspect live Urban Green research QA records."
        badges={[
          'Research documentation',
          'Frozen model results',
          'Read-only QA inspection',
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

      {/* =========================================== */}
      {/* 12A. RESEARCH OVERVIEW                     */}
      {/* =========================================== */}

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          title="Research study area"
          value="Kaduwela"
          subtitle="Municipal Council study boundary"
          icon={MapPinned}
          tone="teal"
        />

        <MetricCard
          title="Primary research grids"
          value={count(suitability.grid_count)}
          subtitle="Original 100 m GIS cells"
          icon={Database}
          tone="green"
        />

        <MetricCard
          title="Historical core"
          value={historical.core_period}
          subtitle="Stabilized retrospective LULC"
          icon={BookOpen}
          tone="blue"
        />

        <MetricCard
          title="Future assessment"
          value={pressure.pressure_period}
          subtitle="One combined period · origin 2025"
          icon={GitBranch}
          tone="orange"
        />
      </div>

      {/* =========================================== */}
      {/* 12B. COMPLETE RESEARCH PIPELINE            */}
      {/* =========================================== */}

      <section className="card-pad mt-5">
        <Heading
          title="Complete research development workflow"
          description="The stages below describe the methods used to produce the saved results, not operations executed by opening this page."
        />

        <div className="grid gap-3 lg:grid-cols-2">
          {WORKFLOW.map((stage) => (
            <div
              key={stage.number}
              className="rounded-xl border border-slate-200 bg-slate-50 p-4"
            >
              <div className="flex items-start gap-3">
                <span className="grid h-9 w-9 shrink-0 place-items-center rounded-lg bg-emerald-100 text-xs font-bold text-emerald-900">
                  {stage.number}
                </span>

                <div>
                  <p className="text-sm font-bold text-slate-800">
                    {stage.title}
                  </p>

                  <p className="mt-1 text-xs font-semibold text-emerald-800">
                    {stage.method}
                  </p>
                </div>
              </div>

              <p className="mt-3 text-xs leading-6 text-slate-600">
                {stage.detail}
              </p>

              <p className="mt-2 text-xs leading-6 text-slate-700">
                <strong>Output:</strong>{' '}
                {stage.result}
              </p>

              <Link
                to={stage.route}
                className="mt-3 inline-block text-xs font-semibold text-emerald-800 hover:underline"
              >
                Open research page →
              </Link>
            </div>
          ))}
        </div>
      </section>

      {/* =========================================== */}
      {/* 12C. DATA SOURCES AND PROVENANCE           */}
      {/* =========================================== */}

      <section className="card-pad mt-5">
        <Heading
          title="Study data and provenance"
          description="Research source roles and cautions are shown separately from modelled results and independent ground truth."
        />

        <div className="overflow-x-auto">
          <table className="min-w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50 text-slate-600">
                <th className="px-3 py-3">
                  Data category
                </th>

                <th className="px-3 py-3">
                  Research source
                </th>

                <th className="px-3 py-3">
                  Interpretation
                </th>
              </tr>
            </thead>

            <tbody>
              {SOURCES.map(([
                type,
                source,
                limitation,
              ]) => (
                <tr
                  key={type}
                  className="border-b border-slate-100 last:border-0"
                >
                  <td className="px-3 py-3 font-semibold text-slate-800">
                    {type}
                  </td>

                  <td className="px-3 py-3 text-slate-700">
                    {source}
                  </td>

                  <td className="px-3 py-3 text-slate-600">
                    {limitation}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* =========================================== */}
      {/* 12D. ML EVALUATION AND AHP METHODS          */}
      {/* =========================================== */}

      <div className="mt-5 grid gap-5 xl:grid-cols-2">
        <section className="card-pad min-w-0">
          <Heading
            title="Pressure model validation and testing"
            description="Previously completed one-time 2022 temporal holdout evaluation of the locked XGBoost configuration."
          />

          <dl>
            {HELD_OUT_2022.map(([label, value]) => (
              <Row
                key={label}
                label={label}
                value={value}
              />
            ))}
          </dl>

          <p className="mt-3 rounded-lg bg-slate-50 p-3 text-xs leading-6 text-slate-700">
            <strong>Evaluation sequence:</strong>{' '}
            historical development and temporal/spatial
            validation → freeze model family,
            hyperparameters and threshold → evaluate
            the reserved 2022 origin once → production
            refit using labelled historical origins
            through 2022 → 2025-origin inference for
            2026–2028.
          </p>

          <p className="mt-3 text-[11px] leading-5 text-slate-500">
            These values are documented historical
            test results, not metrics newly calculated
            by this page. The pre-refit holdout does
            not independently evaluate the final
            refitted model.
          </p>
        </section>

        {/* AHP AND CONSERVATION METHODS */}

        <section className="card-pad min-w-0">
          <Heading
            title="Independent AHP and conservation methods"
            description="GIS-MCDA, conservation and ML pressure are independent stages, not one blended trained model."
          />

          <p className="text-xs font-bold text-slate-800">
            Final expert-AHP weights
          </p>

          <div className="mt-3 space-y-3">
            {ahpCriteria.map((criterion) => (
              <div key={criterion.id}>
                <div className="flex items-center justify-between gap-2 text-xs">
                  <span className="font-semibold text-slate-700">
                    {criterion.id}
                    {' · '}
                    {criterion.name}
                  </span>

                  <span className="font-bold text-slate-800">
                    {finiteNumber(
                      criterion.weight
                    )
                      ? `${fmt(
                          criterion.weight * 100,
                          0
                        )}%`
                      : 'Unavailable'}
                  </span>
                </div>

                <div className="mt-1 h-2 overflow-hidden rounded-full bg-slate-100">
                  <div
                    className="h-full rounded-full bg-emerald-700"
                    style={{
                      width:
                        `${
                          finiteNumber(
                            criterion.weight
                          )
                            ? criterion.weight * 100
                            : 0
                        }%`,
                    }}
                  />
                </div>
              </div>
            ))}
          </div>

          <dl className="mt-4">
            <Row
              label="Consistency ratio"
              value={fmt(ahpCr, 3)}
            />

            <Row
              label="Mean continuous suitability"
              value={fmt(
                suitability.distribution?.mean,
                3
              )}
            />

            <Row
              label="Green Importance scored cells"
              value={count(
                conservation.green_importance_scored_count
              )}
            />

            <Row
              label="Conservation Priority scored cells"
              value={count(
                conservation.primary_priority_scored_count
              )}
            />

            <Row
              label="Conservation Priority unavailable"
              value={count(missingCp)}
            />
          </dl>

          <p className="mt-3 rounded-lg bg-slate-50 p-3 text-xs leading-6 text-slate-700">
            <strong>AHP suitability:</strong>{' '}
            weighted sum of standardized road
            proximity, local road density and slope.

            {' '}<strong>Green Importance:</strong>{' '}
            CRITIC-weighted current green condition,
            historical persistence and local green
            context.

            {' '}<strong>Conservation Priority:</strong>{' '}
            GI × frozen 2025-origin future pressure,
            only within the common domain.
          </p>
        </section>
      </div>

      {/* =========================================== */}
      {/* 12E. LIVE QUALITY ASSURANCE                */}
      {/* =========================================== */}

      <section className="card-pad mt-5">
        <div className="mb-3 flex items-center gap-2">
          <FileCheck2
            size={18}
            className="text-emerald-700"
          />

          <h2 className="section-title">
            Live backend QA and research coverage
          </h2>
        </div>

        <p className="mb-4 text-xs leading-5 text-slate-500">
          PASS refers to the named technical QA
          stage, not proof that every research
          output has external field-validation
          accuracy.
        </p>

        <div className="rounded-xl border border-slate-200 px-4">
          {qaRows.map((row) => (
            <QaRow
              key={row.name}
              {...row}
            />
          ))}
        </div>
      </section>

      {/* =========================================== */}
      {/* 12F. METADATA AND DATA HEALTH              */}
      {/* =========================================== */}

      <section className="card-pad mt-5">
        <Heading
          title="Metadata, manifest and backend data-health records"
          description="Read-only live requests. If an endpoint is not available, the page shows that fact rather than inventing its contents."
        />

        <div className="grid gap-3 xl:grid-cols-3">
          {extras.map((record) => (
            <AuditSource
              key={record.key}
              record={record}
            />
          ))}
        </div>
      </section>

      {/* =========================================== */}
      {/* 12G. RESEARCH LIMITATIONS                  */}
      {/* =========================================== */}

      <section className="card-pad mt-5">
        <div className="mb-3 flex items-center gap-2">
          <ShieldAlert
            size={18}
            className="text-amber-700"
          />

          <h2 className="section-title">
            Research interpretation and documented limitations
          </h2>
        </div>

        <div className="grid gap-2 md:grid-cols-2">
          {LIMITATIONS.map((text, index) => (
            <div
              key={index}
              className="flex items-start gap-3 rounded-xl bg-slate-50 p-3"
            >
              <span className="mt-0.5 text-xs font-bold text-slate-500">
                {index + 1}.
              </span>

              <p className="text-xs leading-5 text-slate-700">
                {text}
              </p>
            </div>
          ))}
        </div>

        <div className="mt-4 rounded-xl border border-amber-200 bg-amber-50 p-4 text-xs leading-6 text-amber-900">
          <strong>Planning caution:</strong>{' '}
          This research provides relative GIS and
          ML decision-support evidence. It does
          not issue development permits, replace
          legal zoning review, or verify future
          conversion or ecological damage at
          field level. Missing model outputs
          remain unknown, not low or zero.
        </div>
      </section>

      {/* =========================================== */}
      {/* 12H. FRONTEND COMPLETION                   */}
      {/* =========================================== */}

      <section className="card-pad mt-5">
        <div className="flex items-center gap-2">
          <ShieldCheck
            size={18}
            className="text-emerald-700"
          />

          <h2 className="section-title">
            Urban Green frontend research release
          </h2>
        </div>

        <p className="mt-3 text-xs leading-6 text-slate-600">
          This documentation page reads saved results
          only. It neither retrains Random
          Forest/XGBoost, recalculates SHAP or
          AHP/CRITIC weights, nor reassigns
          planning classifications. All research
          pages stay scoped to the Urban Green
          component.
        </p>

        <div className="mt-4 flex flex-wrap gap-4 text-xs font-semibold text-emerald-800">
          <Link
            to="/green/dashboard"
            className="hover:underline"
          >
            Return to Dashboard →
          </Link>

          <Link
            to="/green/gn-hotspots"
            className="hover:underline"
          >
            GN and Hotspot Explorer →
          </Link>
        </div>
      </section>
    </div>
  )
}
