import { useEffect, useState } from 'react'
import { MapPin, CalendarDays, ChartNoAxesCombined } from 'lucide-react'
import { wetlandApi } from '../../features/wetland/services/wetlandApi'
import IndividualWetlandMap from '../../features/wetland/components/IndividualWetlandMap'
import {
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from 'recharts'

function formatMeasurement(value, decimals = 4) {
  if (value === null || value === undefined || value === '') {
    return 'N/A'
  }

  const number = Number(value)

  return Number.isFinite(number)
    ? number.toFixed(decimals)
    : 'N/A'
}

export default function IndividualWetlandExplorer() {

  const [wetlands, setWetlands] = useState([])
  const [selectedWetland, setSelectedWetland] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [observations, setObservations] = useState(null)
  const [observationsLoading, setObservationsLoading] = useState(false)
  const [observationsError, setObservationsError] = useState(null)
  const [selectedYear, setSelectedYear] = useState('')
  const [selectedDate, setSelectedDate] = useState('')
  const [wetlandDetails, setWetlandDetails] = useState(null)
  const [detailsLoading, setDetailsLoading] = useState(false)
  const [detailsError, setDetailsError] = useState(null)
  const [beforeDate, setBeforeDate] = useState('')
  const [afterDate, setAfterDate] = useState('')
  const [comparisonEvidence, setComparisonEvidence] = useState(null)
  const [evidenceLoading, setEvidenceLoading] = useState(false)
  const [evidenceError, setEvidenceError] = useState(null)

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

        if (!controller.signal.aborted) {
          const records = [...result.data].sort((a, b) =>
            a.wetland_id.localeCompare(b.wetland_id)
          )

          setWetlands(records)

          if (records.length > 0) {
            setSelectedWetland(records[0].wetland_id)
          }
        }
      } catch (err) {
        if (err.name !== 'AbortError' && !controller.signal.aborted) {
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


  // Load historical observations for the selected wetland
  useEffect(() => {
    if (!selectedWetland) {
      setObservations(null)
      return
    }

    const controller = new AbortController()

    async function loadObservations() {
      setObservationsLoading(true)
      setObservationsError(null)
      setObservations(null)

      try {
        const result = await wetlandApi.individualWetlandObservations(
          selectedWetland,
          controller.signal
        )

        if (
          result.type !== 'FeatureCollection' ||
          !Array.isArray(result.features)
        ) {
          throw new Error('Unexpected historical observation format')
        }

        if (!controller.signal.aborted) {
          setObservations(result)
        }
      } catch (err) {
        if (err.name !== 'AbortError' && !controller.signal.aborted) {
          setObservationsError(err.message)
        }
      } finally {
        if (!controller.signal.aborted) {
          setObservationsLoading(false)
        }
      }
    }

    loadObservations()

    return () => controller.abort()
  }, [selectedWetland])


  // Select the latest available year for the chosen wetland
  useEffect(() => {
    const years = observations?.metadata?.available_years || []

    if (years.length === 0) {
      setSelectedYear('')
      return
    }

    const latestYear = Math.max(...years)

    setSelectedYear(String(latestYear))
  }, [observations])


  // Get actual observation dates for the selected year
  const availableDates = (
    observations?.features || []
  )
    .filter((feature) =>
      String(feature.properties?.year) === selectedYear
    )
    .map((feature) => feature.properties.observation_date)
    .filter(Boolean)
    .sort((a, b) => b.localeCompare(a))

  
  // Preserve a valid selected date, otherwise choose the latest
  useEffect(() => {
    setSelectedDate((currentDate) =>
      availableDates.includes(currentDate)
        ? currentDate
        : availableDates[0] || ''
    )
  }, [observations, selectedYear])

  // Get only the observation matching the selected year and date
  const selectedObservation =
    observations?.features?.find((feature) => {
      const properties = feature.properties || {}

      return (
        String(properties.year) === selectedYear &&
        properties.observation_date === selectedDate
      )
    }) || null


  // Load the overall research summary for the selected wetland
  useEffect(() => {
    if (!selectedWetland) {
      setWetlandDetails(null)
      return
    }

    const controller = new AbortController()

    async function loadWetlandDetails() {
      setDetailsLoading(true)
      setDetailsError(null)
      setWetlandDetails(null)

      try {
        const result = await wetlandApi.wetlandDetails(
          selectedWetland,
          controller.signal
        )

        if (!result || result.wetland_id !== selectedWetland) {
          throw new Error('Unexpected wetland details format')
        }

        if (!controller.signal.aborted) {
          setWetlandDetails(result)
        }
      } catch (err) {
        if (err.name !== 'AbortError' && !controller.signal.aborted) {
          setDetailsError(err.message)
        }
      } finally {
        if (!controller.signal.aborted) {
          setDetailsLoading(false)
        }
      }
    }

    loadWetlandDetails()

    return () => controller.abort()
  }, [selectedWetland])


  // Actual historical observations for the area chart
  const areaChartData = (observations?.features || [])
    .map((feature) => {
      const properties = feature.properties || {}
      const area = Number(properties.analysis_total_area_ha)
      const date = properties.observation_date
      const timestamp = date
        ? Date.parse(`${date}T00:00:00Z`)
        : NaN

      return {
        date,
        timestamp,
        area,
      }
    })
    .filter(
      (item) =>
        Number.isFinite(item.timestamp) &&
        Number.isFinite(item.area)
    )
    .sort((a, b) => a.timestamp - b.timestamp)

  // All actual historical observations for the timeline
  const timelineObservations = [
    ...(observations?.features || []),
  ].sort((a, b) =>
    String(a.properties?.observation_date || '').localeCompare(
      String(b.properties?.observation_date || '')
    )
  )


  // Available recorded dates for before-and-after comparison
  const comparisonDates = [
    ...new Set(
      timelineObservations
        .map((feature) => feature.properties?.observation_date)
        .filter(Boolean)
    ),
  ].sort()

  // Reset comparison dates when a different wetland is loaded
  useEffect(() => {
    setBeforeDate(comparisonDates[0] || '')
    setAfterDate(comparisonDates.at(-1) || '')
  }, [observations])

  const comparisonDatesValid =
    Boolean(beforeDate && afterDate) &&
    beforeDate < afterDate


  // Find actual historical polygons for the comparison dates
  const beforeObservation =
    observations?.features?.find(
      (feature) =>
        feature.properties?.observation_date === beforeDate
    ) || null

  const afterObservation =
    observations?.features?.find(
      (feature) =>
        feature.properties?.observation_date === afterDate
    ) || null

  // Both maps use the same geographic bounds
  const comparisonFeatures = [
    beforeObservation,
    afterObservation,
  ].filter(Boolean)


  // Raw mapped-area comparison between selected observations
  const beforeAreaValue =
    beforeObservation?.properties?.analysis_total_area_ha

  const afterAreaValue =
    afterObservation?.properties?.analysis_total_area_ha

  const hasValidComparisonAreas =
    beforeAreaValue != null &&
    afterAreaValue != null &&
    Number.isFinite(Number(beforeAreaValue)) &&
    Number.isFinite(Number(afterAreaValue))

  const beforeArea = hasValidComparisonAreas
    ? Number(beforeAreaValue)
    : null

  const afterArea = hasValidComparisonAreas
    ? Number(afterAreaValue)
    : null

  const areaDifference = hasValidComparisonAreas
    ? afterArea - beforeArea
    : null

  const areaChangePercent =
    hasValidComparisonAreas && beforeArea > 0
      ? (areaDifference / beforeArea) * 100
      : null
  

  // Load quality evidence for the selected historical comparison
  useEffect(() => {
    if (!selectedWetland || !comparisonDatesValid) {
      setComparisonEvidence(null)
      setEvidenceLoading(false)
      setEvidenceError(null)
      return
    }

    const controller = new AbortController()

    async function loadComparisonEvidence() {
      setEvidenceLoading(true)
      setEvidenceError(null)
      setComparisonEvidence(null)

      try {
        const result = await wetlandApi.comparisonEvidence(
          selectedWetland,
          beforeDate,
          afterDate,
          controller.signal
        )

        if (!controller.signal.aborted) {
          setComparisonEvidence(result)
        }
      } catch (err) {
        if (err.name !== 'AbortError' && !controller.signal.aborted) {
          setEvidenceError(err.message)
        }
      } finally {
        if (!controller.signal.aborted) {
          setEvidenceLoading(false)
        }
      }
    }

    loadComparisonEvidence()

    return () => controller.abort()
  }, [
    selectedWetland,
    beforeDate,
    afterDate,
    comparisonDatesValid,
  ])

  // Prevent displaying evidence for a previously selected pair
  const evidenceMatchesSelection =
    comparisonEvidence?.wetland_id === selectedWetland &&
    comparisonEvidence?.before_date === beforeDate &&
    comparisonEvidence?.after_date === afterDate

  const currentEvidence = evidenceMatchesSelection
    ? comparisonEvidence
    : null


  return (
    <div className="space-y-6">

      {/* Page Header */}
      <div>
        <p className="text-xs font-bold uppercase tracking-[0.18em] text-teal-700">
          Wetland Research — Part 1
        </p>

        <h1 className="mt-2 text-3xl font-bold tracking-tight text-slate-900">
          Individual Wetland Explorer
        </h1>

        <p className="mt-3 max-w-4xl text-sm leading-6 text-slate-600">
          Explore individual monitored wetlands, their historical
          observation dates, mapped boundaries, area measurements
          and available evidence of environmental change.
        </p>
      </div>

      {/* Wetland Selection Section */}
      <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="flex items-center gap-3">
          <MapPin size={22} className="text-teal-700" />

          <h2 className="text-lg font-bold text-slate-900">
            Select a Wetland
          </h2>
        </div>

        <p className="mt-2 text-sm text-slate-500">
          Choose one of the 25 monitored wetlands to explore
          its available historical observations.
        </p>
   
        <div className="mt-5 space-y-3">
        <label
            htmlFor="individual-wetland"
            className="block text-sm font-semibold text-slate-700"
        >
            Monitored Wetland
        </label>

        <select
            id="individual-wetland"
            value={selectedWetland}
            onChange={(event) => setSelectedWetland(event.target.value)}
            disabled={loading || Boolean(error)}
            className="w-full max-w-lg rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-700 outline-none transition focus:border-teal-500 focus:ring-2 focus:ring-teal-100 disabled:opacity-60"
        >
            {loading && <option value="">Loading wetlands...</option>}

            {!loading && wetlands.length === 0 && (
            <option value="">No wetlands available</option>
            )}

            {wetlands.map((wetland) => (
            <option key={wetland.wetland_id} value={wetland.wetland_id}>
                {wetland.wetland_id} — {wetland.primary_gn || 'GN not available'}
            </option>
            ))}
        </select>

        {error && (
            <p className="text-sm text-red-600">
            Unable to load wetlands: {error}
            </p>
        )}

        {!loading && !error && selectedWetland && (
            <p className="text-sm text-teal-700">
            Selected wetland: <strong>{selectedWetland}</strong>
            </p>
        )}
        </div>

      </div>

      {/* Historical Map Section */}
      <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="flex items-center gap-3">
          <CalendarDays size={22} className="text-teal-700" />

          <h2 className="text-lg font-bold text-slate-900">
            Historical Observation Map
          </h2>
        </div>

        <p className="mt-2 text-sm text-slate-500">
          Display the selected wetland's actual mapped boundary
          for an available observation date.
        </p>

        {/* Individual Wetland Observation Status */}
        <div className="mt-4 rounded-xl border border-teal-100 bg-teal-50 px-5 py-4">

        {observationsLoading ? (
            <p className="text-sm text-teal-700">
            Loading historical observations...
            </p>
        ) : observationsError ? (
            <p className="text-sm text-red-600">
            Unable to load observations: {observationsError}
            </p>
        ) : observations ? (
            <div className="space-y-2">
            <p className="text-sm font-semibold text-teal-800">
                {observations.metadata.wetland_id}
                {' — '}
                {observations.metadata.observation_count}
                {' historical observations available'}
            </p>

            <p className="text-xs text-slate-600">
                Available years:
                {' '}
                {observations.metadata.available_years.join(', ')}
            </p>

            <p className="text-xs text-slate-600">
                Observation period:
                {' '}
                {observations.metadata.observation_dates[0]}
                {' to '}
                {observations.metadata.observation_dates.at(-1)}
            </p>
            </div>
        ) : (
            <p className="text-sm text-slate-500">
            Select a wetland to load its historical observations.
            </p>
        )}

        </div>

        {/* Individual Wetland Year Selector */}
        <div className="mt-5 flex flex-wrap items-center gap-4">

        <div className="space-y-2">
            <label
            htmlFor="individual-observation-year"
            className="block text-sm font-semibold text-slate-700"
            >
            Observation Year
            </label>

            <select
            id="individual-observation-year"
            value={selectedYear}
            onChange={(event) => setSelectedYear(event.target.value)}
            disabled={observationsLoading || !observations}
            className="min-w-[200px] rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-700 outline-none transition focus:border-teal-500 focus:ring-2 focus:ring-teal-100 disabled:opacity-60"
            >
            {!selectedYear && (
                <option value="">Select a year</option>
            )}

            {observations?.metadata?.available_years
                ?.slice()
                .sort((a, b) => b - a)
                .map((year) => (
                <option key={year} value={String(year)}>
                    {year}
                </option>
                ))}
            </select>
        </div>

        {/* Exact Observation Date Selector */}
        <div className="space-y-2">
        <label
            htmlFor="individual-observation-date"
            className="block text-sm font-semibold text-slate-700"
        >
            Observation Date
        </label>

        <select
            id="individual-observation-date"
            value={selectedDate}
            onChange={(event) => setSelectedDate(event.target.value)}
            disabled={observationsLoading || availableDates.length === 0}
            className="min-w-[220px] rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-700 outline-none transition focus:border-teal-500 focus:ring-2 focus:ring-teal-100 disabled:opacity-60"
        >
            {!selectedDate && (
            <option value="">Select an observation date</option>
            )}

            {availableDates.map((date) => (
            <option key={date} value={date}>
                {date}
            </option>
            ))}
        </select>
        </div>

        </div>

            {/* Selected Individual Wetland Map */}
            {observationsLoading ? (
            <div className="mt-5 flex h-[400px] items-center justify-center rounded-xl bg-slate-50 text-sm text-slate-500">
                Loading wetland map...
            </div>
            ) : selectedObservation ? (
            <IndividualWetlandMap
                feature={selectedObservation}
                wetlandId={selectedWetland}
                observationDate={selectedDate}
            />
            ) : (
            <div className="mt-5 flex h-[400px] items-center justify-center rounded-xl bg-slate-50 text-sm text-slate-500">
                No mapped observation is available for the selected date.
            </div>
            )}

            {/* Historical Observation Details */}
            {selectedObservation && (
            <div className="mt-6">

                <h3 className="text-base font-bold text-slate-900">
                Historical Observation Details
                </h3>

                <p className="mt-1 text-sm text-slate-500">
                Measurements from the selected mapped observation.
                </p>

                <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">

                {[
                    {
                    label: 'Mapped Wetland Area',
                    value: `${formatMeasurement(
                        selectedObservation.properties.analysis_total_area_ha
                    )} ha`,
                    },
                    {
                    label: 'Analysis Patch Count',
                    value:
                        selectedObservation.properties.analysis_patch_count ?? 'N/A',
                    },
                    {
                    label: 'Largest Patch Area',
                    value: `${formatMeasurement(
                        selectedObservation.properties.largest_patch_ha
                    )} ha`,
                    },
                    {
                    label: 'Largest Patch Index',
                    value: `${formatMeasurement(
                        selectedObservation.properties.lpi_percent,
                        2
                    )}%`,
                    },
                    {
                    label: 'Observation Season',
                    value: selectedObservation.properties.season || 'N/A',
                    },
                    {
                    label: 'Observation Date',
                    value:
                        selectedObservation.properties.observation_date || 'N/A',
                    },
                ].map((item) => (
                    <div
                    key={item.label}
                    className="rounded-xl border border-slate-200 bg-slate-50 p-4"
                    >
                    <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                        {item.label}
                    </p>

                    <p className="mt-2 text-lg font-bold text-teal-700">
                        {item.value}
                    </p>
                    </div>
                ))}

                </div>

                <p className="mt-4 text-xs leading-5 text-slate-500">
                These measurements describe the selected historical
                observation. They are not independent evidence of
                confirmed wetland loss or gain.
                </p>

            </div>
            )}

      </div>

      {/* Research Analysis Section */}
      <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="flex items-center gap-3">
          <ChartNoAxesCombined size={22} className="text-teal-700" />

          <h2 className="text-lg font-bold text-slate-900">
            Historical Change Analysis
          </h2>
        </div>

        <p className="mt-2 text-sm text-slate-500">
          Explore observation timelines, mapped area changes,
          fragmentation indicators and comparison evidence.
        </p>

        <div className="mt-5 rounded-xl border border-dashed border-slate-200 bg-slate-50 p-5 text-sm text-slate-500">
          
            {/* Individual Wetland Research Summary */}
            <div className="mt-5">

            {detailsLoading ? (
                <p className="text-sm text-slate-500">
                Loading wetland research summary...
                </p>
            ) : detailsError ? (
                <p className="text-sm text-red-600">
                Unable to load wetland details: {detailsError}
                </p>
            ) : wetlandDetails ? (
                <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">

                {[
                    {
                    label: 'Historical Change Status',
                    value: wetlandDetails.change_status,
                    },
                    {
                    label: 'Fragmentation Status',
                    value: wetlandDetails.fragmentation_status,
                    },
                    {
                    label: 'Historical Observations',
                    value: wetlandDetails.observation_count,
                    },
                    {
                    label: 'Trusted Comparison Pairs',
                    value: wetlandDetails.trusted_pair_count,
                    },
                ].map((item) => (
                    <div
                    key={item.label}
                    className="rounded-xl border border-slate-200 bg-slate-50 p-4"
                    >
                    <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                        {item.label}
                    </p>

                    <p className="mt-3 text-lg font-bold text-teal-700">
                        {item.value ?? 'N/A'}
                    </p>
                    </div>
                ))}

                </div>
            ) : (
                <p className="text-sm text-slate-500">
                Select a wetland to view its research summary.
                </p>
            )}

            </div>

        </div>


{/* Historical Wetland Area Chart */}
{areaChartData.length > 0 && (
  <div className="mt-8 border-t border-slate-100 pt-6">

    <h3 className="text-lg font-bold text-slate-900">
      Historical Mapped Wetland Area
    </h3>

    <p className="mt-2 text-sm text-slate-500">
      Recorded wetland area measurements across available
      observation dates for {selectedWetland}.
    </p>

    <div className="mt-6 h-[340px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart
          margin={{ top: 15, right: 20, bottom: 20, left: 10 }}
        >
          <CartesianGrid strokeDasharray="3 3" />

          <XAxis
            type="number"
            dataKey="timestamp"
            domain={['dataMin', 'dataMax']}
            tickFormatter={(value) =>
              new Date(value).getUTCFullYear()
            }
            name="Observation date"
            tick={{ fontSize: 12 }}
            label={{
              value: 'Observation Year',
              position: 'insideBottom',
              offset: -10,
            }}
          />

          <YAxis
            type="number"
            dataKey="area"
            domain={[0, 'auto']}
            name="Mapped area"
            tick={{ fontSize: 12 }}
            width={60}
            label={{
              value: 'Area (ha)',
              angle: -90,
              position: 'insideLeft',
            }}
          />

          <Tooltip
            cursor={{ strokeDasharray: '3 3' }}
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null

              const point = payload[0]?.payload

              if (!point) return null

              return (
                <div className="rounded-xl border border-slate-200 bg-white p-3 shadow-lg">
                  <p className="text-sm font-semibold text-slate-900">
                    {point.date}
                  </p>

                  <p className="mt-1 text-sm text-teal-700">
                    Mapped area: {point.area.toFixed(4)} ha
                  </p>
                </div>
              )
            }}
          />

          <Scatter
            name="Observed wetland area"
            data={areaChartData}
            fill="#0F766E"
          />
        </ScatterChart>
      </ResponsiveContainer>
    </div>

    <p className="mt-3 text-xs leading-5 text-slate-500">
      Each point represents an actual mapped observation.
      Missing dates are not interpolated. Differences between
      observations may reflect seasonal conditions, imagery
      limitations or changes in wetland extent, and should not
      automatically be interpreted as validated loss or gain.
    </p>

  </div>
)}


{/* Historical Observation Timeline */}
{timelineObservations.length > 0 && (
  <div className="mt-8 border-t border-slate-100 pt-6">

    <h3 className="text-lg font-bold text-slate-900">
      Historical Observation Timeline
    </h3>

    <p className="mt-2 text-sm text-slate-500">
      Select an actual observation date to explore its mapped
      wetland boundary and measurements.
    </p>

    <div className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">

      {timelineObservations.map((feature) => {
        const properties = feature.properties || {}
        const date = properties.observation_date
        const year = String(properties.year)
        const area = Number(properties.analysis_total_area_ha)
        const isSelected =
          selectedYear === year && selectedDate === date

        return (
          <button
            key={`${properties.wetland_id}-${date}`}
            type="button"
            aria-pressed={isSelected}
            onClick={() => {
              setSelectedYear(year)
              setSelectedDate(date)
            }}
            className={`rounded-xl border p-4 text-left transition ${
              isSelected
                ? 'border-teal-600 bg-teal-50 ring-2 ring-teal-100'
                : 'border-slate-200 bg-slate-50 hover:border-teal-300 hover:bg-teal-50/50'
            }`}
          >
            <p className="text-xs font-semibold text-slate-500">
              {year}
            </p>

            <p className="mt-2 text-sm font-bold text-teal-700">
              {date}
            </p>

            <p className="mt-2 text-xs text-slate-600">
              Mapped area:{' '}
              {Number.isFinite(area)
                ? `${area.toFixed(4)} ha`
                : 'N/A'}
            </p>

            <p className="mt-1 text-xs text-slate-500">
              {properties.season || 'Season not available'}
            </p>

            {isSelected && (
              <p className="mt-3 text-xs font-semibold text-teal-700">
                Currently selected
              </p>
            )}
          </button>
        )
      })}

    </div>

    <p className="mt-4 text-xs leading-5 text-slate-500">
      The timeline contains recorded observations only.
      Missing years and dates are not interpolated.
      Observation intervals and seasonal conditions vary.
    </p>

  </div>
)}


{/* Before-and-After Wetland Comparison */}
<div className="mt-8 border-t border-slate-100 pt-6">

  <h3 className="text-lg font-bold text-slate-900">
    Before-and-After Wetland Comparison
  </h3>

  <p className="mt-2 text-sm text-slate-500">
    Select two recorded observations to compare the
    historical mapped boundaries of {selectedWetland}.
  </p>

  <div className="mt-5 grid gap-4 md:grid-cols-2">

    {/* Before Observation */}
    <div className="rounded-xl border border-slate-200 bg-slate-50 p-5">
      <label
        htmlFor="before-observation"
        className="block text-sm font-bold text-slate-800"
      >
        Before — Earlier Observation
      </label>

      <select
        id="before-observation"
        value={beforeDate}
        onChange={(event) => setBeforeDate(event.target.value)}
        disabled={observationsLoading || comparisonDates.length < 2}
        className="mt-3 w-full rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm outline-none focus:border-teal-500"
      >
        {comparisonDates.map((date) => (
          <option key={date} value={date}>
            {date}
          </option>
        ))}
      </select>
    </div>

    {/* After Observation */}
    <div className="rounded-xl border border-slate-200 bg-slate-50 p-5">
      <label
        htmlFor="after-observation"
        className="block text-sm font-bold text-slate-800"
      >
        After — Later Observation
      </label>

      <select
        id="after-observation"
        value={afterDate}
        onChange={(event) => setAfterDate(event.target.value)}
        disabled={observationsLoading || comparisonDates.length < 2}
        className="mt-3 w-full rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm outline-none focus:border-teal-500"
      >
        {comparisonDates.map((date) => (
          <option key={date} value={date}>
            {date}
          </option>
        ))}
      </select>
    </div>

  </div>

  <p
    className={`mt-4 text-sm ${
      comparisonDatesValid ? 'text-teal-700' : 'text-amber-700'
    }`}
  >
    {comparisonDatesValid
      ? `Comparison selected: ${beforeDate} → ${afterDate}`
      : 'Please select two different dates, with the earlier date first.'}
  </p>


{/* Before and After Historical Maps */}
{comparisonDatesValid &&
  beforeObservation &&
  afterObservation && (
    <div className="mt-6 grid gap-5 lg:grid-cols-2">

      {/* Before Map */}
      <div className="rounded-2xl border border-slate-200 bg-white p-4">
        <h4 className="text-base font-bold text-slate-900">
          Before — {beforeDate}
        </h4>

        <p className="mt-1 text-xs text-slate-500">
          Earlier recorded wetland boundary
        </p>

        <IndividualWetlandMap
          key={`before-${selectedWetland}-${beforeDate}-${afterDate}`}
          feature={beforeObservation}
          wetlandId={selectedWetland}
          observationDate={beforeDate}
          comparisonFeatures={comparisonFeatures}
        />
      </div>

      {/* After Map */}
      <div className="rounded-2xl border border-slate-200 bg-white p-4">
        <h4 className="text-base font-bold text-slate-900">
          After — {afterDate}
        </h4>

        <p className="mt-1 text-xs text-slate-500">
          Later recorded wetland boundary
        </p>

        <IndividualWetlandMap
          key={`after-${selectedWetland}-${beforeDate}-${afterDate}`}
          feature={afterObservation}
          wetlandId={selectedWetland}
          observationDate={afterDate}
          comparisonFeatures={comparisonFeatures}
        />
      </div>

    </div>
  )}


  <p className="mt-3 text-xs leading-5 text-slate-500">
    Selecting two observations does not establish a validated
    wetland change. Comparison reliability depends on the
    available evidence and observation quality.
  </p>


{/* Before-and-After Area Measurements */}
{comparisonDatesValid && hasValidComparisonAreas && (
  <div className="mt-6">

    <h4 className="text-base font-bold text-slate-900">
      Mapped Area Comparison
    </h4>

    <p className="mt-1 text-sm text-slate-500">
      Raw area measurements from the two selected observations.
    </p>

    <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">

      {[
        {
          label: 'Before Area',
          value: `${beforeArea.toFixed(4)} ha`,
        },
        {
          label: 'After Area',
          value: `${afterArea.toFixed(4)} ha`,
        },
        {
          label: 'Raw Area Difference',
          value: `${areaDifference >= 0 ? '+' : ''}${areaDifference.toFixed(4)} ha`,
        },
        {
          label: 'Raw Percentage Change',
          value:
            areaChangePercent === null
              ? 'N/A'
              : `${areaChangePercent >= 0 ? '+' : ''}${areaChangePercent.toFixed(2)}%`,
        },
      ].map((item) => (
        <div
          key={item.label}
          className="rounded-xl border border-slate-200 bg-slate-50 p-4"
        >
          <p className="text-xs font-semibold uppercase text-slate-500">
            {item.label}
          </p>

          <p className="mt-3 text-xl font-bold text-teal-700">
            {item.value}
          </p>
        </div>
      ))}

    </div>

    <p className="mt-4 text-xs leading-5 text-amber-800">
      These are raw differences between mapped observations.
      Positive and negative values indicate differences in
      recorded area, not independently validated wetland gain
      or loss. Comparison quality must be assessed separately.
    </p>

  </div>
)}


</div>

{/* Historical Comparison Evidence */}
{comparisonDatesValid && (
  <div className="mt-8 border-t border-slate-100 pt-6">

    <h4 className="text-lg font-bold text-slate-900">
      Evidence Quality & Reliability
    </h4>

    <p className="mt-2 text-sm text-slate-500">
      Quality assessment from the existing historical
      wetland comparison research.
    </p>

    {evidenceLoading || (!currentEvidence && !evidenceError) ? (
      <p className="mt-5 text-sm text-slate-500">
        Loading comparison evidence...
      </p>
    ) : evidenceError ? (
      <p className="mt-5 text-sm text-red-600">
        Unable to load comparison evidence: {evidenceError}
      </p>
    ) : currentEvidence.assessed ? (
      <div className="mt-5">

        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">

          {[
            {
              label: 'Comparison Quality',
              value: currentEvidence.comparison_quality,
            },
            {
              label: 'Same Observation Season',
              value: currentEvidence.same_season ? 'Yes' : 'No',
            },
            {
              label: 'Rainfall Similarity',
              value: currentEvidence.rainfall_similarity || 'N/A',
            },
            {
              label: 'Manual QA Flag',
              value: currentEvidence.manual_qa_required
                ? 'Required'
                : 'Not flagged',
            },
          ].map((item) => (
            <div
              key={item.label}
              className="rounded-xl border border-slate-200 bg-slate-50 p-4"
            >
              <p className="text-xs font-semibold uppercase text-slate-500">
                {item.label}
              </p>

              <p className="mt-3 text-lg font-bold text-teal-700">
                {item.value}
              </p>
            </div>
          ))}

        </div>

        <p className="mt-4 rounded-xl border border-amber-200 bg-amber-50 p-4 text-xs leading-5 text-amber-900">
          This exact observation pair has a recorded quality
          assessment. The classification describes comparison
          evidence quality, not independent validation of
          wetland loss or gain. A missing manual QA flag does
          not mean manual validation was completed.
        </p>

      </div>
    ) : (
      <div className="mt-5 rounded-xl border border-amber-200 bg-amber-50 p-5">

        <p className="text-sm font-bold text-amber-900">
          Comparison Not Assessed
        </p>

        <p className="mt-2 text-sm leading-6 text-amber-800">
          No exact quality-assessed comparison record exists
          for these two observation dates. The displayed
          polygons and raw area differences are exploratory
          and must not be treated as validated wetland change.
        </p>

      </div>
    )}

  </div>
)}
      </div>

      {/* Research Interpretation */}
      <div className="rounded-xl border border-amber-200 bg-amber-50 p-4 text-xs leading-5 text-amber-900">
        Historical wetland observations may differ in date,
        season and evidence quality. Mapped area differences
        should not automatically be interpreted as independently
        validated wetland loss or gain.
      </div>

    </div>
  )
}
