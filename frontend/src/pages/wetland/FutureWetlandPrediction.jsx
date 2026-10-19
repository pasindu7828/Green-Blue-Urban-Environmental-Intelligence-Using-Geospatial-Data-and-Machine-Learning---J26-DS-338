import { useEffect, useRef, useState } from 'react'
import { wetlandApi } from '../../features/wetland/services/wetlandApi'
import { TrendingUp, ShieldAlert } from 'lucide-react'
import FutureWetlandMap from '../../features/wetland/components/FutureWetlandMap'

export default function FutureWetlandPrediction() {

  // Part 3 — GN division future scenario data
  const [gnData, setGnData] = useState(null)
  const [gnLoading, setGnLoading] = useState(true)
  const [gnError, setGnError] = useState(null)
  const [gnSearch, setGnSearch] = useState('')
  const [gnChangeFilter, setGnChangeFilter] = useState('all')
  const [focusedGnId, setFocusedGnId] = useState(null)
  const futureMapRef = useRef(null)
  const [focusedMapData, setFocusedMapData] = useState(null)
  const [focusedMapLoading, setFocusedMapLoading] = useState(true)
  const [focusedMapError, setFocusedMapError] = useState(null)
  const [scenarioView, setScenarioView] = useState('focused')
  const [gnBoundaryData, setGnBoundaryData] = useState(null)

  useEffect(() => {
    const controller = new AbortController()

    async function loadFutureGnData() {
      try {
        const result = await wetlandApi.focusedFutureGnSummary(
          controller.signal
        )

        if (
          result.type !== 'FeatureCollection' ||
          !Array.isArray(result.features)
        ) {
          throw new Error('Unexpected future GN data format')
        }

        if (!controller.signal.aborted) {
          setGnData(result)
          setGnError(null)
        }
      } catch (err) {
        if (err.name !== 'AbortError' && !controller.signal.aborted) {
          setGnError(err.message)
        }
      } finally {
        if (!controller.signal.aborted) {
          setGnLoading(false)
        }
      }
    }

    loadFutureGnData()

    return () => controller.abort()
  }, [])

// Search and filter GN future scenario records
const filteredGnDivisions = [...(gnData?.features || [])]
  .filter((feature) => {
    const gn = feature.properties || {}
    const query = gnSearch.trim().toLowerCase()

    const matchesSearch = [
      gn.gn_name,
      gn.gn_source_id,
      gn.gn_zone,
    ].some((value) =>
      String(value ?? '').toLowerCase().includes(query)
    )

    const rawNetChange = gn.scenario_net_change_ha
    const netChange = Number(rawNetChange)

    const hasValidChange =
      rawNetChange !== null &&
      rawNetChange !== undefined &&
      rawNetChange !== '' &&
      Number.isFinite(netChange)

    const matchesChangeFilter =
      gnChangeFilter === 'all' ||
      (hasValidChange &&
        (
          (gnChangeFilter === 'decrease' && netChange < 0) ||
          (gnChangeFilter === 'increase' && netChange > 0) ||
          (gnChangeFilter === 'unchanged' && netChange === 0)
        ))

    return matchesSearch && matchesChangeFilter
  })
  .sort((a, b) =>
    String(a.properties?.gn_name || '').localeCompare(
      String(b.properties?.gn_name || '')
    )
  )

// Part 3 — Load focused 2030 scenario polygons
useEffect(() => {
  const controller = new AbortController()

  async function loadFocusedScenario() {
    try {
      const result = await wetlandApi.focusedFutureScenario(
        controller.signal
      )

      if (
        result.type !== 'FeatureCollection' ||
        !Array.isArray(result.features)
      ) {
        throw new Error('Unexpected focused scenario data format')
      }

      if (
        result.metadata?.selected_patches !==
        result.features.length
      ) {
        throw new Error('Focused scenario feature count mismatch')
      }

      if (!controller.signal.aborted) {
        setFocusedMapData(result)
        setFocusedMapError(null)
      }
    } catch (err) {
      if (err.name !== 'AbortError' && !controller.signal.aborted) {
        setFocusedMapError(err.message)
      }
    } finally {
      if (!controller.signal.aborted) {
        setFocusedMapLoading(false)
      }
    }
  }

  loadFocusedScenario()

  return () => controller.abort()
}, [])

 // Use only the focused 2030 scenario for the map
 const activeMapData = focusedMapData
 const activeMapLoading = focusedMapLoading
 const activeMapError = focusedMapError

// Step 145 — Load all original GN boundaries for the map
useEffect(() => {
  const controller = new AbortController()

  async function loadGnBoundaries() {
    try {
      const result = await wetlandApi.futureGnSummary(
        controller.signal
      )

      if (
        result.type !== 'FeatureCollection' ||
        !Array.isArray(result.features) ||
        result.features.length !== 56
      ) {
        throw new Error('Expected 56 original GN boundaries')
      }

      if (!controller.signal.aborted) {
        setGnBoundaryData(result)
      }
    } catch (error) {
      if (!controller.signal.aborted) {
        console.error('Unable to load original GN boundaries:', error)
      }
    }
  }

  loadGnBoundaries()

  return () => controller.abort()
}, [])

  return (
    <div className="space-y-6">

      {/* Page Header */}
      <div>
        <p className="text-xs font-bold uppercase tracking-[0.18em] text-teal-700">
          Wetland Research — Part 3
        </p>

        <h1 className="mt-2 text-3xl font-bold tracking-tight text-slate-900">
          Future Wetland Scenarios
        </h1>

        <p className="mt-3 max-w-4xl text-sm leading-6 text-slate-600">
          Explore model-generated wetland change scenarios
          for 2030, including projected wetland distribution,
          area changes and GN division-level analysis
          across the Kaduwela study area.
        </p>
      </div>

      {/* Future Scenario Summary */}
      <div className="grid gap-4 md:grid-cols-3">
        {[
          'Scenario Reference Year',
          'Projected Scenario Year',
          'Focused GN Divisions',
        ].map((title) => (
          <div
            key={title}
            className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm"
          >
            <p className="text-xs font-semibold uppercase text-slate-500">
              {title}
            </p>
            <p className="mt-4 text-4xl font-bold text-teal-700">
            {title === 'Scenario Reference Year'
                ? '2025'
                : title === 'Projected Scenario Year'
                ? '2030'
                : gnLoading
                    ? '...'
                    : gnError
                    ? 'Unavailable'
                    : gnData?.features?.length ?? 0}
            </p>
          </div>
        ))}
      </div>

        {/* Future Wetland Map */}
        <div
          ref={futureMapRef}
          className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm"
        >
        <div className="flex items-center gap-3">
          <TrendingUp size={22} className="text-teal-700" />

          <h2 className="text-lg font-bold text-slate-900">
            Future Wetland Scenario Map
          </h2>
        </div>

        <p className="mt-2 text-sm text-slate-500">
          Explore model-generated wetland polygons
          for the available scenario years.
        </p>

        
        {/* Interactive Future Wetland Scenario Map */}
        <div className="mt-5">

          {/* Focused / Full Scenario Switch */}
          <div className="mb-4 flex flex-wrap items-center justify-between gap-3">

            <p className="text-xs font-medium text-slate-500">          
              Selected historical wetlands and primary candidate areas
            </p>

          </div>

          {/* Selection Explanation */}
          <p className="mb-4 text-xs leading-5 text-slate-600"> 
            Focused 2030 scenario: Model-generated patches meeting the
            five-pixel display rule and intersecting the 25 historical
            wetland areas or 48 primary screened candidate polygons
            within a provisional 50 m buffer. This is an exploratory
            spatial selection, not a validated prediction improvement.
          </p>

          {/* Scenario Map */}
          {activeMapLoading || gnLoading ? (
            <div className="flex h-[430px] items-center justify-center rounded-xl bg-slate-50 text-sm text-slate-500">
              Loading 2030 wetland scenario map...
            </div>
          ) : activeMapError ? (
            <div className="rounded-xl border border-red-200 bg-red-50 p-5 text-sm text-red-700">
              Unable to load scenario map: {activeMapError}
            </div>
          ) : (
            <>
              {gnError && (
                <p className="mb-3 text-sm text-amber-700">
                  GN division boundaries could not be loaded: {gnError}
                </p>
              )}

              <FutureWetlandMap
                key={scenarioView}
                futureMapData={activeMapData}
                gnData={gnBoundaryData}
                selectedGnId={focusedGnId}
              />

              {scenarioView === 'focused' && focusedMapData && (
                <p className="mt-3 text-xs leading-5 text-slate-500">
                  Focused selection:{' '}
                  {focusedMapData.metadata?.selected_patches ?? focusedMapData.features.length}
                  {' '}of{' '}
                  {focusedMapData.metadata?.display_eligible_patches ?? 501}
                  {' '}display-eligible 2030 patches. Other areas are
                  excluded only from this view and remain available
                  through the Full Scenario option.
                </p>
              )}
            </>
          )}

        </div>

      </div>

      {/* GN Division Scenario Analysis */}
      <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <h2 className="text-lg font-bold text-slate-900">
          GN Division Future Scenario Analysis
        </h2>

        <p className="mt-2 text-sm text-slate-500">
          Explore scenario wetland loss, gain and net area
          changes for individual GN divisions.
        </p>

        
{/* GN Division Future Scenario Records */}
<div className="mt-5">

{/* GN Division Search */}
<div className="mb-4 flex flex-wrap items-center gap-3">
  <div className="relative min-w-[240px] flex-1">
    <input
      type="text"
      value={gnSearch}
      onChange={(event) => setGnSearch(event.target.value)}
      placeholder="Search GN division name or ID..."
      aria-label="Search future scenario GN divisions"
      className="w-full rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm outline-none transition focus:border-teal-500 focus:ring-2 focus:ring-teal-100"
    />
  </div>


{/* Future Net Change Filter */}
<select
  value={gnChangeFilter}
  onChange={(event) => setGnChangeFilter(event.target.value)}
  aria-label="Filter GN divisions by scenario net change"
  className="w-full rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-700 outline-none focus:border-teal-500 sm:w-auto"
>
  <option value="all">All Net Changes</option>
  <option value="decrease">Projected Net Decrease</option>
  <option value="increase">Projected Net Increase</option>
  <option value="unchanged">No Net Change</option>
</select>


  <p className="text-xs font-medium text-slate-500">
    {gnLoading
      ? 'Loading GN divisions...'
      : `Showing ${filteredGnDivisions.length} of ${gnData?.features?.length ?? 0} GN divisions`}
  </p>
</div>


  {gnLoading ? (
    <p className="text-sm text-slate-500">
      Loading GN division scenario results...
    </p>
  ) : gnError ? (
    <p className="text-sm text-red-600">
      Unable to load GN scenario data: {gnError}
    </p>
  ) : (
    <div className="overflow-x-auto rounded-xl border border-slate-200">

      <table className="w-full text-left text-sm">
        <thead className="bg-slate-50 text-xs uppercase text-slate-500">
          <tr>
            <th className="px-4 py-4">GN Division</th>
            <th className="px-4 py-4">2025 Wetland (ha)</th>
            <th className="px-4 py-4">2030 Scenario (ha)</th>
            <th className="px-4 py-4">Scenario Loss (ha)</th>
            <th className="px-4 py-4">Scenario Gain (ha)</th>
            <th className="px-4 py-4">Net Change (ha)</th>
          </tr>
        </thead>

        <tbody className="divide-y divide-slate-100">
          
            {filteredGnDivisions.map((feature, index) => {
              const gn = feature.properties || {}
              const formatArea = (value) => {
                if (value == null || value === '') return 'N/A'

                const number = Number(value)

                return Number.isFinite(number)
                  ? number.toFixed(2)
                  : 'N/A'
              }

              const netChange = Number(
                gn.scenario_net_change_ha
              )

              return (
                <tr
                  key={gn.gn_source_id ?? gn.gn_zone ?? index}
                  className="hover:bg-teal-50/40"
                >
                  <td className="px-4 py-4">
                    <button
                      type="button"
                      onClick={() => {
                        setFocusedGnId(gn.gn_source_id ?? gn.gn_zone)

                        futureMapRef.current?.scrollIntoView({
                          behavior: 'smooth',
                          block: 'start',
                        })
                      }}
                      className="font-semibold text-teal-700 transition hover:text-teal-900 hover:underline"
                      title={`View ${gn.gn_name || 'GN division'} on map`}
                    >
                      {gn.gn_name || 'N/A'}
                    </button>
                  </td>
                  <td className="px-4 py-4">
                    {formatArea(gn.wetland_2025_ha)}
                  </td>

                  <td className="px-4 py-4">
                    {formatArea(gn.wetland_2030_ha)}
                  </td>

                  <td className="px-4 py-4">
                    {formatArea(gn.scenario_loss_ha)}
                  </td>

                  <td className="px-4 py-4">
                    {formatArea(gn.scenario_gain_ha)}
                  </td>

                  <td
                    className={`px-4 py-4 font-semibold ${
                      !Number.isFinite(netChange) ||
                      gn.scenario_net_change_ha == null
                        ? 'text-slate-600'
                        : netChange < 0
                          ? 'text-red-600'
                          : netChange > 0
                            ? 'text-teal-700'
                            : 'text-slate-600'
                    }`}
                  >
                    {formatArea(gn.scenario_net_change_ha)}
                  </td>
                </tr>
              )
            })}
        </tbody>
      </table>

    </div>
  )}
  
  <p className="mt-4 text-xs leading-5 text-slate-500">
    This focused analysis includes GN divisions intersecting
    selected 2025 or 2030 model patches. Areas are calculated
    from selected polygons clipped to GN boundaries. Scenario
    gain and loss describe spatial differences between the
    filtered model outputs, while net change is their area
    difference. These are exploratory statistics, not
    independently validated wetland predictions.
  </p>

</div>

      </div>

      {/* Scientific Limitations */}
      <div className="flex gap-3 rounded-xl border border-amber-200 bg-amber-50 p-4">
        <ShieldAlert
          size={20}
          className="shrink-0 text-amber-700"
        />

        <p className="text-xs leading-5 text-amber-900">
          The 2030 results are exploratory CA-Markov
          model scenarios, not validated future predictions.
          Earlier hindcast evaluation did not outperform
          the persistence baseline. These results should
          not be treated as reliable forecasts or used
          alone for planning decisions.
        </p>
      </div>

    </div>
  )
}
