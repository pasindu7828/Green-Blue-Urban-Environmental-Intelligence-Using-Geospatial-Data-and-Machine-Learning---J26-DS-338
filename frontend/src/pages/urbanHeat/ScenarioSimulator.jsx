import { useEffect, useMemo, useState } from 'react'
import { Beaker, Building2, Droplets, Leaf, Play, ShieldAlert, Thermometer } from 'lucide-react'
import PageHeader from '../../components/shared/PageHeader'
import MetricCard from '../../components/shared/MetricCard'
import Button from '../../components/ui/Button'
import Badge from '../../components/ui/Badge'
import GridPicker from '../../features/urbanHeat/components/GridPicker'
import { ErrorState, LoadingState } from '../../features/urbanHeat/components/AsyncState'
import { useUrbanHeat } from '../../features/urbanHeat/context/UrbanHeatContext'
import { urbanHeatApi } from '../../features/urbanHeat/services/urbanHeatApi'
import { fmt, fmtSigned } from '../../features/urbanHeat/utils/format'

const familyMeta = {
  GREEN: { label: 'Green Cover', icon: Leaf, description: 'Persistent green-cover increase in percentage points.' },
  BUILT: { label: 'Built-up Reduction', icon: Building2, description: 'Persistent NDBI reduction within research-supported limits.' },
  BLUE: { label: 'Blue-space / NDWI', icon: Droplets, description: 'Persistent NDWI increase within research-supported limits.' },
  COMBINED: { label: 'Combined Green + Built', icon: Beaker, description: 'Joint green-cover increase and NDBI reduction.' },
}

export default function ScenarioSimulator() {
  const { selectedGridId, forecastYear } = useUrbanHeat()
  const [options, setOptions] = useState(null)
  const [family, setFamily] = useState('GREEN')
  const [green, setGreen] = useState(10)
  const [built, setBuilt] = useState(0.03)
  const [blue, setBlue] = useState(0.05)
  const [result, setResult] = useState(null)
  const [loadingOptions, setLoadingOptions] = useState(true)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    const controller = new AbortController()
    setLoadingOptions(true)
    urbanHeatApi.scenarioOptions(controller.signal)
      .then((data) => {
        setOptions(data)
        if (data?.green?.precomputed_research_levels?.length) setGreen(data.green.precomputed_research_levels[1] ?? data.green.precomputed_research_levels[0])
        if (data?.built?.research_levels?.MODERATE != null) setBuilt(data.built.research_levels.MODERATE)
        if (data?.blue?.research_levels?.MODERATE != null) setBlue(data.blue.research_levels.MODERATE)
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoadingOptions(false))
    return () => controller.abort()
  }, [])

  const payload = useMemo(() => {
    const base = { grid_id: selectedGridId, year: forecastYear, scenario_family: family }
    if (family === 'GREEN') return { ...base, green_increase_pp: Number(green) }
    if (family === 'BUILT') return { ...base, ndbi_reduction: Number(built) }
    if (family === 'BLUE') return { ...base, ndwi_increase: Number(blue) }
    return { ...base, green_increase_pp: Number(green), ndbi_reduction: Number(built) }
  }, [selectedGridId, forecastYear, family, green, built, blue])

  async function runScenario() {
    if (!selectedGridId) { setError('Select a grid before running a scenario.'); return }
    setRunning(true); setError(''); setResult(null)
    try { setResult(await urbanHeatApi.simulateScenario(payload)) }
    catch (e) { setError(e.message) }
    finally { setRunning(false) }
  }

  if (loadingOptions) return <LoadingState message="Loading research-supported scenario bounds…"/>

  const scenarioPredicted = result?.scenario_predicted_lst_c ?? result?.combined_scenario_predicted_lst_c
  const scenarioDelta = result?.modeled_lst_change_c ?? result?.combined_modeled_lst_change_c
  const coolingBenefit = result?.cooling_benefit_c ?? result?.combined_cooling_benefit_c
  const effect = result?.scenario_effect_direction ?? result?.combined_effect_direction

  return <div>
    <PageHeader eyebrow="Planning Sensitivity" title="Scenario Simulator" subtitle="Run one-grid, one-year model-based planning scenarios using the frozen Urban Heat production model. Custom values are constrained to the research-supported magnitude envelope." badges={['GREEN', 'BUILT', 'BLUE', 'COMBINED = GREEN + BUILT']}/>
    <div className="card-pad mb-5"><GridPicker showYear/></div>

    {error && <div className="mb-5"><ErrorState error={error}/></div>}

    <div className="grid gap-5 xl:grid-cols-[.9fr_1.1fr]">
      <div className="card-pad">
        <h2 className="section-title">1. Select scenario family</h2>
        <div className="mt-4 grid gap-3 sm:grid-cols-2">
          {Object.entries(familyMeta).map(([key, meta]) => { const Icon = meta.icon; const active = family === key; return <button key={key} onClick={() => { setFamily(key); setResult(null) }} className={`rounded-2xl border p-4 text-left transition ${active ? 'border-teal-500 bg-teal-50 ring-2 ring-teal-500/10' : 'border-slate-200 bg-white hover:bg-slate-50'}`}><div className="flex items-center gap-2"><Icon size={18} className={active ? 'text-teal-700' : 'text-slate-500'}/><b className="text-sm">{meta.label}</b></div><p className="mt-2 text-[11px] leading-5 text-slate-500">{meta.description}</p></button> })}
        </div>

        <h2 className="section-title mt-6">2. Intervention magnitude</h2>
        <div className="mt-4 space-y-5">
          {(family === 'GREEN' || family === 'COMBINED') && <label className="block text-xs font-semibold text-slate-600">Green-cover increase: <b>{fmt(green, 1)} pp</b><input type="range" className="mt-3 w-full accent-teal-700" min={options?.green?.min ?? 0} max={options?.green?.max ?? 20} step="0.5" value={green} onChange={(e) => setGreen(Number(e.target.value))}/><div className="mt-1 flex justify-between text-[9px] text-slate-400"><span>{options?.green?.min ?? 0} pp</span><span>Research levels: {(options?.green?.precomputed_research_levels || []).join(', ')} pp</span><span>{options?.green?.max ?? 20} pp</span></div></label>}
          {(family === 'BUILT' || family === 'COMBINED') && <label className="block text-xs font-semibold text-slate-600">NDBI reduction: <b>{fmt(built, 4)}</b><input type="range" className="mt-3 w-full accent-teal-700" min={options?.built?.min ?? 0} max={options?.built?.max ?? 0.07} step="0.001" value={built} onChange={(e) => setBuilt(Number(e.target.value))}/><div className="mt-1 flex justify-between text-[9px] text-slate-400"><span>0</span><span>Max {fmt(options?.built?.max, 4)}</span></div></label>}
          {family === 'BLUE' && <label className="block text-xs font-semibold text-slate-600">NDWI increase: <b>{fmt(blue, 4)}</b><input type="range" className="mt-3 w-full accent-teal-700" min={options?.blue?.min ?? 0} max={options?.blue?.max ?? 0.18} step="0.001" value={blue} onChange={(e) => setBlue(Number(e.target.value))}/><div className="mt-1 flex justify-between text-[9px] text-slate-400"><span>0</span><span>Max {fmt(options?.blue?.max, 4)}</span></div></label>}
        </div>

        <Button className="mt-6 w-full" onClick={runScenario} disabled={running || !selectedGridId}><Play size={16}/>{running ? 'Running model scenario…' : 'Run scenario'}</Button>
      </div>

      <div className="card-pad">
        <h2 className="section-title">Scenario result</h2>
        {!result ? <div className="mt-5 grid min-h-[420px] place-items-center rounded-2xl border border-dashed border-slate-300 bg-slate-50 p-8 text-center"><div><Beaker size={34} className="mx-auto text-slate-400"/><h3 className="mt-3 font-bold text-slate-800">Ready to simulate</h3><p className="mt-2 max-w-md text-sm leading-6 text-slate-500">Choose a real grid, forecast year and intervention magnitude, then run the model-based scenario.</p></div></div> : <div className="mt-5 space-y-5">
          <div className="flex flex-wrap items-center gap-2"><Badge variant="teal">{result.scenario_family}</Badge><Badge variant="slate">{result.area_name}</Badge><Badge variant="slate">Grid {result.grid_id}</Badge><Badge variant="blue">{result.year}</Badge></div>
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
            <MetricCard compact title="Baseline predicted LST" value={`${fmt(result.baseline_predicted_lst_c, 2)}°C`} icon={Thermometer} tone="slate"/>
            <MetricCard compact title="Scenario predicted LST" value={`${fmt(scenarioPredicted, 2)}°C`} icon={Thermometer} tone={Number(scenarioDelta) <= 0 ? 'green' : 'red'}/>
            <MetricCard compact title="Modeled LST change" value={fmtSigned(scenarioDelta, 3, '°C')} icon={Number(scenarioDelta) <= 0 ? Leaf : ShieldAlert} tone={Number(scenarioDelta) <= 0 ? 'green' : 'red'}/>
          </div>
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="soft-panel"><div className="text-[10px] uppercase text-slate-400">Cooling benefit</div><b className="mt-1 block text-lg">{fmt(coolingBenefit, 3)}°C</b></div>
            <div className="soft-panel"><div className="text-[10px] uppercase text-slate-400">Effect direction</div><b className="mt-1 block text-lg capitalize">{effect || '—'}</b></div>
          </div>

          <div className="rounded-2xl border border-slate-200 p-4 text-xs">
            <div className="font-bold text-slate-800">Intervention audit</div>
            <div className="mt-3 grid gap-2 sm:grid-cols-2">
              {result.requested_green_increase_pp != null && <div><span className="text-slate-400">Requested green</span><b className="ml-2">{fmt(result.requested_green_increase_pp, 2)} pp</b></div>}
              {result.actual_green_increase_pp != null && <div><span className="text-slate-400">Actual green</span><b className="ml-2">{fmt(result.actual_green_increase_pp, 2)} pp</b></div>}
              {result.requested_ndbi_reduction != null && <div><span className="text-slate-400">Requested NDBI reduction</span><b className="ml-2">{fmt(result.requested_ndbi_reduction, 4)}</b></div>}
              {result.actual_ndbi_reduction != null && <div><span className="text-slate-400">Actual NDBI reduction</span><b className="ml-2">{fmt(result.actual_ndbi_reduction, 4)}</b></div>}
              {result.requested_ndwi_increase != null && <div><span className="text-slate-400">Requested NDWI increase</span><b className="ml-2">{fmt(result.requested_ndwi_increase, 4)}</b></div>}
              {result.actual_ndwi_increase != null && <div><span className="text-slate-400">Actual NDWI increase</span><b className="ml-2">{fmt(result.actual_ndwi_increase, 4)}</b></div>}
              {result.support_class && <div><span className="text-slate-400">Support class</span><b className="ml-2">{result.support_class}</b></div>}
              {result.matched_research_level && <div><span className="text-slate-400">Matched level</span><b className="ml-2">{String(result.matched_research_level)}</b></div>}
            </div>
          </div>

        </div>}
      </div>
    </div>
  </div>
}
