
import { useEffect, useMemo, useState } from 'react'
import {
  Bar, BarChart, CartesianGrid, Cell, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { BrainCircuit, RefreshCcw, Search, ShieldAlert, TrendingUp } from 'lucide-react'

import PageHeader from '../../components/shared/PageHeader'
import MetricCard from '../../components/shared/MetricCard'
import GreenGridMap, { validScore } from '../../features/urbanGreen/components/GreenGridMap'
import { urbanGreenApi } from '../../features/urbanGreen/services/urbanGreenApi'
import { useUrbanGreen } from '../../features/urbanGreen/context/UrbanGreenContext'
import { apiRequest } from '../../features/urbanHeat/services/apiClient'

const ALL = 'all_scored'
const FLAGGED = 'threshold_flagged'
const SAMPLE_GRID = 'G100_R017_C109' // Known top-ranked grid, not a synthetic example
const LOCAL_ASSET = `${import.meta.env.BASE_URL || '/'}data/urban-green/xai_grid_explanations.json`

const fmt = (value, digits = 2) =>
  validScore(value) ? value.toLocaleString('en-US', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }) : 'Unavailable'

const int = (value) => validScore(value)
  ? value.toLocaleString('en-US', { maximumFractionDigits: 0 }) : 'Unavailable'

const signed = (value, digits = 3) =>
  validScore(value) ? `${value > 0 ? '+' : ''}${fmt(value, digits)}` : 'Unavailable'

function Row({ label, value }) {
  return (
    <div className="flex items-start justify-between gap-3 border-b border-slate-100 py-2.5 last:border-0">
      <span className="text-xs text-slate-500">{label}</span>
      <span className="max-w-[65%] text-right text-xs font-semibold text-slate-800">{value ?? 'Unavailable'}</span>
    </div>
  )
}

function PanelTitle({ title, description }) {
  return <div className="mb-4">
    <h2 className="section-title">{title}</h2>
    <p className="mt-1 text-xs leading-5 text-slate-500">{description}</p>
  </div>
}

function ApiError({ error, retry }) {
  return <section className="card-pad">
    <h2 className="section-title">Explainable AI data unavailable</h2>
    <p className="mt-3 text-sm text-red-700">{error}</p>
    <button onClick={retry} className="btn-secondary mt-4" type="button">Retry</button>
  </section>
}

function LocalFeatureRow({ feature }) {
  const isPositive = validScore(feature.shap) && feature.shap > 0
  const isNegative = validScore(feature.shap) && feature.shap < 0
  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
      <div className="flex items-start justify-between gap-2">
        <div>
          <p className="text-xs font-semibold text-slate-800">{feature.name || 'Unnamed feature'}</p>
          <p className="mt-1 text-[11px] text-slate-500">{feature.domain || 'Domain unavailable'}</p>
        </div>
        <span className={`shrink-0 text-xs font-bold tabular-nums ${isPositive ? 'text-rose-700' : isNegative ? 'text-sky-700' : 'text-slate-700'}`}>
          {signed(feature.shap, 4)}
        </span>
      </div>
      <p className="mt-2 text-[11px] leading-5 text-slate-600">
        Feature value: {fmt(feature.value, 4)} · {feature.direction === 'INCREASES_PRESSURE'
          ? 'Pushes the raw model margin higher'
          : feature.direction === 'DECREASES_PRESSURE'
            ? 'Pushes the raw model margin lower'
            : 'Direction not established'}
      </p>
    </div>
  )
}

export default function ExplainableAI() {
  const { selectedGridId, setSelectedGridId, selectedGnId, selectGn } = useUrbanGreen()
  const [refresh, setRefresh] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [data, setData] = useState(null)
  const [group, setGroup] = useState(ALL)
  const [inputGrid, setInputGrid] = useState('')
  const [inputError, setInputError] = useState('')

  // Frozen responses: all-scored and threshold-flagged. Never recalculate SHAP.
  useEffect(() => {
    const controller = new AbortController()
    setLoading(true)
    setError('')
    const getXai = (path) => apiRequest(
      `/api/urban-green/pressure/xai/${path}`,
      { signal: controller.signal }
    )
    const getLocal = async () => {
      const response = await fetch(LOCAL_ASSET, { signal: controller.signal })
      if (!response.ok) throw new Error(
        `Local SHAP export not found (${response.status}). Run Step 17B before opening this page.`
      )
      return response.json()
    }

    Promise.all([
      urbanGreenApi.pressure.xai.overview(controller.signal),
      getXai('features?group=all_scored&limit=10'),
      getXai('features?group=threshold_flagged&limit=10'),
      getXai('domains?group=all_scored'),
      getXai('domains?group=threshold_flagged'),
      urbanGreenApi.pressure.overview(controller.signal),
      urbanGreenApi.planning.map(controller.signal),
      urbanGreenApi.planning.gnMap(controller.signal),
      urbanGreenApi.planning.gnDivisions(controller.signal),
      getLocal(),
    ]).then(([
      xai, featuresAll, featuresFlagged, domainsAll, domainsFlagged,
      pressure, planningMap, gnMap, gnDivisions, local,
    ]) => {
      if (controller.signal.aborted) return
      const valid = xai?.status === 'ok' &&
        xai?.shap_output_space === 'RAW_XGBOOST_MARGIN' &&
        xai?.shap_is_probability_contribution === false &&
        xai?.explanations_are_causal_claims === false &&
        xai?.research_qa?.['15a'] === 'PASS' && xai?.research_qa?.['15b'] === 'PASS' &&
        xai?.groups?.[ALL]?.explained_grid_count === 4620 &&
        xai?.groups?.[FLAGGED]?.explained_grid_count === 255 &&
        featuresAll?.group === ALL && featuresFlagged?.group === FLAGGED &&
        domainsAll?.group === ALL && domainsFlagged?.group === FLAGGED &&
        Array.isArray(featuresAll?.features) &&
        Array.isArray(featuresFlagged?.features) &&
        Array.isArray(domainsAll?.domains) &&
        Array.isArray(domainsFlagged?.domains) &&
        featuresAll.features.length === 10 && featuresFlagged.features.length === 10 &&
        domainsAll.domains.length === 5 && domainsFlagged.domains.length === 5 &&
        pressure?.eligible_scored_cells === 4620 &&
        Array.isArray(planningMap?.features) && planningMap.features.length === 9240 &&
        Array.isArray(gnMap?.features) && Array.isArray(gnDivisions?.data) &&
        local?.status === 'ok' && local?.shap_output_space === 'RAW_XGBOOST_MARGIN' &&
        local?.origin_year === 2025 && local?.pressure_period === '2026-2028' &&
        local?.total === 4620 && Array.isArray(local?.data) && local.data.length === 4620

      if (!valid) throw new Error('An XAI response or local export failed data-contract validation. No substitute values are shown.')
      setData({
        xai, pressure, planningMap, gnMap, gnDivisions,
        features: { [ALL]: featuresAll.features, [FLAGGED]: featuresFlagged.features },
        domains: { [ALL]: domainsAll.domains, [FLAGGED]: domainsFlagged.domains },
        local,
      })
    }).catch((reason) => {
      if (!controller.signal.aborted) setError(reason.message || String(reason))
    }).finally(() => {
      if (!controller.signal.aborted) setLoading(false)
    })

    return () => controller.abort()
  }, [refresh])

  const localById = useMemo(() => new Map(
    (data?.local?.data || []).map((row) => [row.grid_id, row])
  ), [data])
  const planningById = useMemo(() => new Map(
    (data?.planningMap?.features || []).map((f) => [f.properties?.grid_id, f.properties])
  ), [data])

  const activeGridId = selectedGridId || SAMPLE_GRID
  const local = localById.get(activeGridId)
  const planning = planningById.get(activeGridId)
  const selectedGn = data?.gnDivisions?.data?.find(
    (row) => String(row.gn_id) === String(selectedGnId)
  )
  const stats = data?.xai?.groups?.[group]
  const featureRows = useMemo(() => (data?.features?.[group] || []).map((item) => ({
    rank: item.rank,
    feature: item.human_readable_feature || item.feature_name,
    technical: item.feature_name,
    domain: item.explanation_domain,
    importance: item.importance_percent,
    absolute: item.mean_absolute_shap_raw_margin,
  })).sort((a, b) => a.rank - b.rank), [data, group])
  const domainRows = useMemo(() => (data?.domains?.[group] || []).map((item) => ({
    name: item.explanation_domain,
    importance: item.absolute_domain_importance_percent,
    count: item.feature_count,
    rank: item.rank,
  })).sort((a, b) => a.rank - b.rank), [data, group])
  const localFeatures = local?.features || []
  const localChart = localFeatures.map((feature) => ({
    name: feature.name || 'Unknown', value: feature.shap,
  })).filter((row) => validScore(row.value))

  function selectTypedGrid(event) {
    event.preventDefault()
    const id = inputGrid.trim().toUpperCase()
    if (!planningById.has(id)) {
      setInputError('Grid ID not found in the original 9,240-cell map.')
      return
    }
    setInputError('')
    setSelectedGridId(id)
    setInputGrid('')
  }

  if (loading) return <section className="card-pad">
    <h2 className="section-title">Loading frozen Explainable AI evidence</h2>
    <p className="mt-2 text-sm text-slate-500">Reading global and local SHAP explanations, GIS maps and QA...</p>
  </section>
  if (error || !data) return <ApiError error={error || 'No research data returned.'} retry={() => setRefresh(v => v + 1)} />

  return <div>
    <PageHeader
      eyebrow="Urban Green Intelligence"
      title="Urbanization Pressure Explainable AI"
      subtitle="Understand the frozen XGBoost model's global feature importance and local SHAP contributions for the single 2026–2028 future-pressure assessment."
      badges={[
        'Origin: 2025', 'Future pressure: 2026–2028',
        'SHAP scale: XGBoost raw margin',
      ]}
      actions={<button type="button" className="btn-secondary" onClick={() => setRefresh(v => v + 1)}><RefreshCcw size={15}/> Refresh</button>}
    />

    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
      <MetricCard title="Explained grids" value={int(data.xai.groups[ALL].explained_grid_count)} subtitle="Original pressure-scored 100 m cells" icon={BrainCircuit} tone="teal" />
      <MetricCard title="Model features" value={int(data.xai.groups[ALL].feature_count)} subtitle="Historical RAW LULC, neighbourhood context and terrain" icon={TrendingUp} tone="green" />
      <MetricCard title="Explanation domains" value={int(data.xai.groups[ALL].domain_count)} subtitle="Understandable feature groups" icon={BrainCircuit} tone="blue" />
      <MetricCard title="Flagged grid explanations" value={int(data.xai.groups[FLAGGED].explained_grid_count)} subtitle="Above the frozen pressure threshold (0.715)" icon={ShieldAlert} tone="orange" />
    </div>

    <section className="card-pad mt-5">
      <PanelTitle title="Global model explanations" description="Compare all scored grids with the saved threshold-flagged subset. Importance percentages are normalized mean absolute SHAP within each respective group, not conversion probabilities." />
      <div className="flex flex-wrap gap-2">
        <button type="button" onClick={() => setGroup(ALL)} className={`rounded-lg border px-4 py-2 text-xs font-semibold ${group === ALL ? 'border-emerald-800 bg-emerald-800 text-white' : 'border-slate-300 bg-white text-slate-700'}`}>
          All 4,620 scored grids
        </button>
        <button type="button" onClick={() => setGroup(FLAGGED)} className={`rounded-lg border px-4 py-2 text-xs font-semibold ${group === FLAGGED ? 'border-emerald-800 bg-emerald-800 text-white' : 'border-slate-300 bg-white text-slate-700'}`}>
          255 threshold-flagged grids
        </button>
      </div>
      <div className="mt-3 grid gap-3 rounded-xl bg-slate-50 p-4 text-xs text-slate-700 sm:grid-cols-3">
        <div>Explained cells: <strong>{int(stats.explained_grid_count)}</strong></div>
        <div>Top feature: <strong>{stats.top_feature}</strong></div>
        <div>Top domain: <strong>{stats.top_domain}</strong></div>
      </div>

      <div className="mt-5 grid gap-5 xl:grid-cols-2">
        <div className="min-w-0 rounded-xl border border-slate-200 p-3">
          <h3 className="text-sm font-bold text-slate-800">Top 10 SHAP features</h3>
          <p className="mt-1 text-[11px] text-slate-500">Percentage of all 55 features' summed mean absolute SHAP within this group</p>
          <ResponsiveContainer width="100%" height={390}>
            <BarChart layout="vertical" data={featureRows} margin={{ top: 10, right: 20, bottom: 10, left: 8 }}>
              <CartesianGrid strokeDasharray="4 5" />
              <XAxis type="number" tick={{ fontSize: 10 }} unit="%" />
              <YAxis type="category" dataKey="technical" width={154} tick={{ fontSize: 10 }} />
              <Tooltip formatter={(v) => [`${fmt(v, 2)}%`, 'Relative SHAP importance']} labelFormatter={(_, payload) => payload?.[0]?.payload?.feature || ''} />
              <Bar dataKey="importance" fill="#15803d" radius={[0, 4, 4, 0]} isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
          <p className="text-[11px] leading-5 text-slate-500">Ranks describe importance magnitude, not whether a feature always increases pressure.</p>
        </div>
        <div className="min-w-0 rounded-xl border border-slate-200 p-3">
          <h3 className="text-sm font-bold text-slate-800">Five explanatory domains</h3>
          <p className="mt-1 text-[11px] text-slate-500">Each bar is the share of overall mean absolute SHAP attributable to a domain</p>
          <ResponsiveContainer width="100%" height={390}>
            <BarChart layout="vertical" data={domainRows} margin={{ top: 10, right: 20, bottom: 10, left: 8 }}>
              <CartesianGrid strokeDasharray="4 5" />
              <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 10 }} unit="%" />
              <YAxis type="category" dataKey="name" width={177} tick={{ fontSize: 10 }} />
              <Tooltip formatter={(v) => [`${fmt(v, 2)}%`, 'Domain importance']} />
              <Bar dataKey="importance" fill="#2563eb" radius={[0, 4, 4, 0]} isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
          <p className="text-[11px] leading-5 text-slate-500">Domain importance is aggregated model explanation magnitude, not a causal effect.</p>
        </div>
      </div>
    </section>

    <div className="mt-5 grid gap-5 2xl:grid-cols-[1.5fr_.85fr]">
      <section className="card-pad min-w-0">
        <PanelTitle title="Select a 100 m grid for a local explanation" description="The GIS reuses your validated 200 m overview and 100 m detail workflow. Click a displayed block to drill down, then select the original grid to inspect saved SHAP reasons." />
        <label className="mb-3 block text-xs font-semibold text-slate-600">GN division
          <select className="control mt-1" value={selectedGnId || ''} onChange={e => selectGn(e.target.value || null)}>
            <option value="">All GN divisions</option>
            {data.gnDivisions.data.map(gn => <option key={gn.gn_id} value={gn.gn_id}>{gn.gn_name}</option>)}
          </select>
        </label>
        <GreenGridMap
          geojson={data.planningMap}
          gnGeojson={data.gnMap}
          metric="pressure"
          selectedGridId={selectedGridId}
          selectedGnId={selectedGnId}
          onSelect={setSelectedGridId}
          onClearGn={() => selectGn(null)}
          height={545}
        />
        {/* <p className="mt-3 text-[11px] leading-5 text-slate-500">
          Gray grids are outside the prediction domain or unscored, not low pressure.
          The 200 m view is a display summary only; local SHAP results always correspond to a single original 100 m grid.
        </p> */}
      </section>
      <div className="space-y-5">
        <section className="card-pad">
          <PanelTitle title="Local XGBoost explanation" description="Saved SHAP reasons for the selected original research grid; no live model inference or SHAP recomputation." />
          <form onSubmit={selectTypedGrid} className="flex gap-2">
            <input className="control min-w-0 flex-1" placeholder="G100_R017_C109" aria-label="Grid ID" value={inputGrid} onChange={e => setInputGrid(e.target.value)} />
            <button type="submit" className="btn-secondary" aria-label="Find grid"><Search size={17} /></button>
          </form>
          {inputError && <p className="mt-2 text-xs text-red-700">{inputError}</p>}
          {!selectedGridId && <p className="mt-3 rounded-lg bg-sky-50 p-3 text-xs leading-5 text-sky-900">
            Showing the real highest-ranked grid as a starting example. Click another grid or enter its ID to change the explanation.
          </p>}
          <dl className="mt-3">
            <Row label="Grid ID" value={activeGridId} />
            <Row label="GN division" value={planning?.gn_name || 'Unavailable'} />
            <Row label="Pressure inference domain" value={local ? 'Eligible and explained' : planning ? 'No local SHAP explanation (unscored)' : 'Unknown grid'} />
            <Row label="Relative pressure score" value={local ? fmt(local.urbanization_pressure_score, 6) : 'Not calculated'} />
            <Row label="Pressure category" value={local?.pressure_category || 'Unavailable'} />
            <Row label="Pressure rank" value={local ? int(local.pressure_rank) : 'Not applicable'} />
            <Row label="Dominant explanation domain" value={local?.dominant_domain || 'Not available'} />
          </dl>
          {!local && <div className="mt-3 rounded-lg border border-amber-200 bg-amber-50 p-3 text-xs leading-5 text-amber-900">
            No SHAP output is available for this cell. It must not be interpreted as a zero contribution or as low pressure.
          </div>}
          {local && <>
            <p className="mt-4 text-xs font-bold text-slate-700">Top three saved feature contributions</p>
            <div className="mt-2 space-y-2">{localFeatures.map((f, idx) => <LocalFeatureRow key={`${idx}-${f.name}`} feature={f} />)}</div>
          </>}
        </section>
        <section className="card-pad">
          <PanelTitle title="Selected GN context" description={selectedGn?.gn_name || 'Select a GN division using the map controls.'} />
          <p className="text-xs leading-5 text-slate-600">
            Grid-level SHAP explanations are individual model explanations. They are not automatically GN-wide explanations and should not be averaged without a separately validated aggregation method.
          </p>
        </section>
      </div>
    </div>

    {local && <section className="card-pad mt-5">
      <PanelTitle title="Why this selected grid received its model score" description="Signed SHAP contribution of the top three local features, measured on the raw XGBoost margin scale." />
      <ResponsiveContainer width="100%" height={230}>
        <BarChart data={localChart} layout="vertical" margin={{ left: 16, right: 30, top: 6, bottom: 6 }}>
          <CartesianGrid strokeDasharray="4 5" />
          <XAxis type="number" tick={{ fontSize: 10 }} />
          <YAxis type="category" dataKey="name" width={205} tick={{ fontSize: 10 }} />
          <Tooltip formatter={v => [signed(v, 4), 'Raw-margin SHAP contribution']} />
          <ReferenceLine x={0} stroke="#475569" strokeWidth={1.5} />
          <Bar dataKey="value" radius={[3, 3, 3, 3]} isAnimationActive={false}>
            {localChart.map((row, index) => <Cell key={index} fill={row.value > 0 ? '#c2413b' : '#2563eb'} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="mt-2 flex flex-wrap gap-4 text-[11px] text-slate-600">
        <span><span className="mr-1 inline-block h-2.5 w-2.5 bg-[#c2413b]" />Positive: pushes model margin higher</span>
        <span><span className="mr-1 inline-block h-2.5 w-2.5 bg-[#2563eb]" />Negative: pushes model margin lower</span>
      </div>
      <p className="mt-2 text-[11px] leading-5 text-slate-500">Only the three dominant saved feature contributions are plotted. They do not sum to the model's prediction; remaining features and the SHAP base value are omitted. They are not probability percentage-point contributions.</p>
    </section>}

    <section className="card-pad mt-5">
      <PanelTitle title="Research method, QA and limitations" description="Official frozen model explanation evidence; no re-training, threshold tuning or new scoring." />
      <div className="grid gap-3 md:grid-cols-3">
        <div className="rounded-xl bg-slate-50 p-4 text-xs leading-6 text-slate-700"><strong>SHAP method:</strong> TreeExplainer applied to the frozen 2025-origin production XGBoost model on the 4,620 scored cells.</div>
        <div className="rounded-xl bg-slate-50 p-4 text-xs leading-6 text-slate-700"><strong>Output space:</strong> RAW_XGBOOST_MARGIN. Positive SHAP values push the raw margin higher and negative values push it lower.</div>
        <div className="rounded-xl bg-slate-50 p-4 text-xs leading-6 text-slate-700"><strong>Validation:</strong> Stage 15A: {data.xai.research_qa['15a']}; Stage 15B: {data.xai.research_qa['15b']}. Saved pressure scores remain unchanged.</div>
      </div>
      {/* <p className="mt-4 rounded-xl border border-amber-200 bg-amber-50 p-4 text-xs leading-6 text-amber-900">
        <strong>Responsible interpretation:</strong> SHAP explains associations learned by the prediction model, not proven causes of urbanization. The feature and domain percentages represent normalized average absolute SHAP importance, not conversion probabilities. The raw margin contributions are not probability changes. The 2026–2028 forecast is one combined three-year pressure period, not three annual forecasts. Unscored locations have no model explanation.
      </p> */}
    </section>
  </div>
}
