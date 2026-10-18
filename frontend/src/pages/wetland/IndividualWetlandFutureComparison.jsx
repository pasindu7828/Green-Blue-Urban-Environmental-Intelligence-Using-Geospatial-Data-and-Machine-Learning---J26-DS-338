import { useEffect, useState } from 'react'
import { GitCompareArrows } from 'lucide-react'
import { wetlandApi } from '../../features/wetland/services/wetlandApi'
import IndividualWetlandMap from '../../features/wetland/components/IndividualWetlandMap'
import IndividualFutureScenarioMap from '../../features/wetland/components/IndividualFutureScenarioMap'


export default function IndividualWetlandFutureComparison() {

  const [wetlands, setWetlands] = useState([])
  const [selectedWetland, setSelectedWetland] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [comparisonData, setComparisonData] = useState(null)
  const [comparisonLoading, setComparisonLoading] = useState(false)
  const [comparisonError, setComparisonError] = useState(null)

  // Load the 25 monitored historical wetlands
  useEffect(() => {
    const controller = new AbortController()

    async function loadWetlands() {
      try {
        const result = await wetlandApi.historicalSummary(
          controller.signal
        )

        if (!Array.isArray(result.data)) {
          throw new Error('Unexpected wetland data format')
        }

        const records = [...result.data].sort((a, b) =>
          a.wetland_id.localeCompare(b.wetland_id)
        )

        if (!controller.signal.aborted) {
          setWetlands(records)
          setSelectedWetland(records[0]?.wetland_id || '')
          setError(null)
        }
      } catch (err) {
        if (!controller.signal.aborted) {
          setError(err.message)
        }
      } finally {
        if (!controller.signal.aborted) {
          setLoading(false)
        }
      }
    }

    loadWetlands()

    return () => controller.abort()
  }, [])

// Step 171 — Load individual wetland future comparison
useEffect(() => {
  if (!selectedWetland) {
    setComparisonData(null)
    setComparisonLoading(false)
    setComparisonError(null)
    return
  }

  const controller = new AbortController()

  async function loadComparison() {
    setComparisonLoading(true)
    setComparisonError(null)
    setComparisonData(null)

    try {
      const result = await wetlandApi.individualFutureComparison(
        selectedWetland,
        controller.signal
      )

      if (
        result.wetland_id !== selectedWetland ||
        !Array.isArray(result.historical?.features) ||
        !Array.isArray(result.scenario_2030?.features)
      ) {
        throw new Error('Unexpected individual comparison data format')
      }

      if (!controller.signal.aborted) {
        setComparisonData(result)
        setComparisonError(null)
      }
    } catch (err) {
      if (!controller.signal.aborted) {
        setComparisonError(err.message)
      }
    } finally {
      if (!controller.signal.aborted) {
        setComparisonLoading(false)
      }
    }
  }

  loadComparison()

  return () => controller.abort()
}, [selectedWetland])

  return (
    <div className="space-y-6">

      {/* Page Header */}
      <div>
        <p className="text-xs font-bold uppercase tracking-[0.18em] text-teal-700">
          Wetland Research — Part 3
        </p>

        <h1 className="mt-2 text-3xl font-bold tracking-tight text-slate-900">
          Individual Wetland — Local 2030 Scenario
        </h1>        
        <p className="mt-3 max-w-4xl text-sm leading-6 text-slate-600">
            Explore the geographic surroundings of individual monitored
            wetlands using the exploratory 2030 CA-Markov scenario.
            Local scenario polygons are displayed within a provisional
            50 m area around each selected historical wetland.
            These results describe model-generated local context,
            not predicted future boundaries or validated wetland change.
        </p>
      </div>

      {/* Comparison Workspace */}
      <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">

        <div className="flex items-center gap-3">
          <GitCompareArrows size={22} className="text-teal-700" />

          <h2 className="text-lg font-bold text-slate-900">
            Local 2030 Scenario Explorer
          </h2>
        </div>

        <p className="mt-3 text-sm leading-6 text-slate-600">
          Select an individual historical wetland to explore
          its mapped boundary alongside nearby model-generated
          2030 scenario patches.
        </p>

        {/* Historical Wetland Selector */}
        <div className="mt-6 space-y-3">
        <label
            htmlFor="future-comparison-wetland"
            className="block text-sm font-semibold text-slate-700"
        >
            Select Historical Wetland
        </label>

        <select
            id="future-comparison-wetland"
            value={selectedWetland}
            onChange={(event) => setSelectedWetland(event.target.value)}
            disabled={loading || Boolean(error)}
            className="w-full max-w-lg rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-700 outline-none transition focus:border-teal-500 focus:ring-2 focus:ring-teal-100 disabled:opacity-60"
        >
            {loading && (
            <option value="">Loading wetlands...</option>
            )}

            {!loading && wetlands.length === 0 && (
            <option value="">No wetlands available</option>
            )}

            {wetlands.map((wetland) => (
            <option
                key={wetland.wetland_id}
                value={wetland.wetland_id}
            >
                {wetland.wetland_id} — {wetland.primary_gn || 'GN not available'}
            </option>
            ))}
        </select>

        {error && (
            <p className="text-sm text-red-600">
            Unable to load historical wetlands: {error}
            </p>
        )}

        {!loading && !error && selectedWetland && (
            <p className="text-sm text-teal-700">
            Selected wetland: <strong>{selectedWetland}</strong>
            </p>
        )}
        </div>
        
{/* Individual Wetland — Local 2030 Scenario Map */}
<div className="mt-6 rounded-xl border border-slate-200 bg-white p-4">

  <h3 className="text-base font-bold text-slate-900">
    Local 2030 Wetland Scenario — {selectedWetland}
  </h3>

  <p className="mt-2 text-xs leading-5 text-slate-500">
    Explore the model-generated 2030 scenario within the
    provisional 50 m surrounding area of the selected
    historical wetland. This is not an independently
    validated individual-wetland forecast.
  </p>

  {comparisonLoading ? (
    <div className="mt-4 flex h-[430px] items-center justify-center rounded-xl bg-slate-50 text-sm text-slate-500">
      Loading local 2030 scenario...
    </div>
  ) : comparisonError ? (
    <p className="mt-4 text-sm text-red-600">
      Unable to load local scenario: {comparisonError}
    </p>
  ) : comparisonData?.wetland_id === selectedWetland ? (
    <div className="mt-4">
      <IndividualFutureScenarioMap
        key={selectedWetland}
        comparisonData={comparisonData}
      />
    </div>
  ) : (
    <p className="mt-4 text-sm text-slate-500">
      Select a historical wetland to explore its local
      2030 scenario.
    </p>
  )}

</div>
      </div>

      {/* Scientific Disclaimer */}
      <div className="rounded-xl border border-amber-200 bg-amber-50 p-4">
        <p className="text-xs leading-5 text-amber-900">
          The 2030 results are exploratory CA-Markov scenario
          outputs, not independently validated future wetland
          boundaries. Spatial proximity does not establish
          that a model patch represents the future state of
          a particular historical wetland.
        </p>
      </div>

    </div>
  )
}
