import { useEffect, useState } from 'react'
import { History, Database, MapPin, Search } from 'lucide-react'
import { wetlandApi } from '../../features/wetland/services/wetlandApi'
import HistoricalWetlandMap from '../../features/wetland/components/HistoricalWetlandMap'

function formatArea(value) {
  if (value === null || value === undefined || value === '') {
    return 'N/A'
  }

  const number = Number(value)
  return Number.isFinite(number) ? number.toFixed(2) : 'N/A'
}

export default function HistoricalChange() {
  const [wetlands, setWetlands] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [mapData, setMapData] = useState(null)
  const [mapLoading, setMapLoading] = useState(true)
  const [mapError, setMapError] = useState(null)
  const [gnBoundaryData, setGnBoundaryData] = useState(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedStatus, setSelectedStatus] = useState('all')
  const [availableYears, setAvailableYears] = useState([])
  const [yearsError, setYearsError] = useState(null)
  const [selectedYear, setSelectedYear] = useState('latest')

  useEffect(() => {
    const controller = new AbortController()

    async function loadWetlands() {
      try {
        const result = await wetlandApi.historicalSummary(
          controller.signal
        )

        if (!Array.isArray(result.data)) {
          throw new Error('Unexpected historical wetland data format')
        }

        setWetlands(result.data)
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

  // Load historical wetland polygons for the selected year
  useEffect(() => {
    const controller = new AbortController()

    async function loadWetlandMap() {
      setMapLoading(true)
      setMapError(null)

      try {
        const result =
          selectedYear === 'latest'
            ? await wetlandApi.historicalMap(controller.signal)
            : await wetlandApi.historicalMapByYear(
                selectedYear,
                controller.signal
              )

        if (
          result.type !== 'FeatureCollection' ||
          !Array.isArray(result.features)
        ) {
          throw new Error('Unexpected historical map data format')
        }

        if (!controller.signal.aborted) {
          setMapData(result)
        }
      } catch (err) {
        if (err.name !== 'AbortError' && !controller.signal.aborted) {
          setMapError(err.message)
          setMapData(null)
        }
      } finally {
        if (!controller.signal.aborted) {
          setMapLoading(false)
        }
      }
    }

    loadWetlandMap()

    return () => controller.abort()
  }, [selectedYear])

  // Load GN division boundaries for geographic context
  useEffect(() => {
    const controller = new AbortController()

    async function loadGnBoundaries() {
      try {
        const result = await wetlandApi.waterloggingGnSummary(
          controller.signal
        )

        if (
          result.type !== 'FeatureCollection' ||
          !Array.isArray(result.features)
        ) {
          throw new Error('Unexpected GN boundary data format')
        }

        if (!controller.signal.aborted) {
          setGnBoundaryData(result)
        }
      } catch (err) {
        if (err.name !== 'AbortError' && !controller.signal.aborted) {
          console.error('Unable to load GN boundaries:', err)
        }
      }
    }

    loadGnBoundaries()

    return () => controller.abort()
  }, [])

  // Load available historical observation years
  useEffect(() => {
    const controller = new AbortController()

    async function loadHistoricalYears() {
      try {
        const result = await wetlandApi.historicalYears(
          controller.signal
        )

        if (!Array.isArray(result.years)) {
          throw new Error('Unexpected historical years format')
        }

        if (!controller.signal.aborted) {
          setAvailableYears(result.years)
          setYearsError(null)
        }
      } catch (err) {
        if (err.name !== 'AbortError' && !controller.signal.aborted) {
          setYearsError(err.message)
        }
      }
    }

    loadHistoricalYears()

    return () => controller.abort()
  }, [])
  
    const changeStatuses = [
    ...new Set(
        wetlands
        .map((wetland) => wetland.change_status)
        .filter(Boolean)
    ),
    ].sort()

    const filteredWetlands = wetlands.filter((wetland) => {
    const query = searchQuery.trim().toLowerCase()

    const matchesSearch = [
        wetland.wetland_id,
        wetland.primary_gn,
        wetland.change_status,
    ].some((value) =>
        String(value || '').toLowerCase().includes(query)
    )

    const matchesStatus =
        selectedStatus === 'all' ||
        wetland.change_status === selectedStatus

    return matchesSearch && matchesStatus
    })


  return (
    <div className="space-y-6">

      {/* Page Header */}
      <div>
        <p className="text-xs font-bold uppercase tracking-[0.18em] text-teal-700">
          Wetland Research — Part 1
        </p>

        <h1 className="mt-2 text-3xl font-bold tracking-tight text-slate-900">
          Historical Wetland Change
        </h1>

        <p className="mt-3 max-w-4xl text-sm leading-6 text-slate-600">
          Explore monitored wetlands, historical observation records,
          wetland area measurements and change classifications across
          the Kaduwela study area.
        </p>
      </div>

      {/* Live Summary */}
      <div className="grid gap-4 md:grid-cols-2">
        <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
          <History className="text-teal-700" size={25} />

          <p className="mt-4 text-xs font-semibold uppercase text-slate-500">
            Monitored Wetlands
          </p>

          <p className="mt-2 text-4xl font-bold text-teal-700">
            {loading ? '...' : error ? '—' : wetlands.length}
          </p>

          <p className="mt-2 text-sm text-slate-500">
            Wetland monitoring records available from FastAPI.
          </p>
        </div>

        <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
          <MapPin className="text-teal-700" size={25} />

          <p className="mt-4 text-xs font-semibold uppercase text-slate-500">
            GN Divisions Represented
          </p>

          <p className="mt-2 text-4xl font-bold text-teal-700">
            {loading ? '...' : error ? '—' :
              new Set(
                wetlands.flatMap((wetland) =>
                  String(wetland.primary_gn || '')
                    .split(';')
                    .map((name) => name.trim())
                    .filter(Boolean)
                )
              ).size}
          </p>

          <p className="mt-2 text-sm text-slate-500">
            Based on primary GN assignments in the monitoring records.
          </p>
        </div>
      </div>

      {/* Historical Wetland Table */}
      
        {/* Historical Wetland Distribution Map */}
        <div className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">

        <div className="border-b border-slate-100 p-5">
            <h2 className="text-lg font-bold text-slate-900">
            Historical Wetland Distribution Map
            </h2>

            <p className="mt-1 text-sm text-slate-500">
              Explore the latest mapped observations of 25 monitored
              wetlands across the Kaduwela study area.
              Select a polygon to view its wetland ID and observation date.
            </p>

            <p className="mt-2 text-xs font-medium text-teal-700">
              {yearsError
                ? `Unable to load historical years: ${yearsError}`
                : availableYears.length > 0
                  ? `${availableYears.length} historical observation years available`
                  : 'Loading available historical years...'}
            </p>


{/* Historical Year Selector */}
<div className="mt-4 flex flex-wrap items-center gap-3">

  <label
    htmlFor="historical-year"
    className="text-sm font-semibold text-slate-700"
  >
    Observation Year:
  </label>

  <select
    id="historical-year"
    value={selectedYear}
    onChange={(event) => setSelectedYear(event.target.value)}
    className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-700 outline-none transition focus:border-teal-500 focus:ring-2 focus:ring-teal-100"
  >
    <option value="latest">Latest Available Observations</option>

    {availableYears.map((item) => (
      <option key={item.year} value={String(item.year)}>
        {item.year} — {item.wetland_count} wetlands
      </option>
    ))}
  </select>

</div>


        </div>

        <div className="p-4">

        {/* Historical Map Legend */}
        <div className="flex flex-wrap items-center gap-x-6 gap-y-3 border-b border-slate-100 bg-slate-50 px-5 py-3">

        <div className="flex items-center gap-2">
            <span className="h-4 w-4 rounded border-2 border-teal-700 bg-teal-300" />
            <span className="text-xs font-medium text-slate-700">
            Monitored Wetlands ({mapLoading ? '...' : mapData?.features?.length ?? 0})
            </span>
        </div>

        <div className="flex items-center gap-2">
            <span className="h-[3px] w-6 rounded bg-slate-500" />
            <span className="text-xs font-medium text-slate-700">
            GN Division Boundaries (56)
            </span>
        </div>

        <div className="ml-auto text-xs text-slate-500">
            Click a polygon to view details
        </div>

        </div>

        {!mapData ? (
          <div className="flex h-[430px] items-center justify-center rounded-2xl bg-slate-50 text-sm text-slate-500">
            {mapError
              ? `Unable to load wetland map: ${mapError}`
              : mapLoading
                ? 'Loading wetland geographic data...'
                : 'No wetland map data available.'}
          </div>
        ) : (
          <div className="relative">
            <HistoricalWetlandMap
              data={mapData}
              gnData={gnBoundaryData}
              yearKey={selectedYear}
            />

            {mapLoading && (
              <div className="absolute inset-0 z-[1000] flex items-center justify-center rounded-2xl bg-white/30">
                <div className="rounded-xl bg-white px-5 py-3 text-sm font-medium text-teal-700 shadow-lg">
                  Updating historical wetland polygons...
                </div>
              </div>
            )}
          </div>
        )}

        </div> 
        <div className="border-t border-slate-100 bg-slate-50 px-5 py-3 text-xs leading-5 text-slate-600">
          Map interpretation: {selectedYear === 'latest'
            ? 'Each polygon represents the latest available mapped observation of a monitored wetland.'
            : `Each polygon represents the latest recorded observation within ${selectedYear} for that wetland.`}
          {' '}These polygons do not represent every wetland in Kaduwela.
          Years without observations are not interpolated.
          The map does not include an external basemap.
        </div>
        </div>

      <div className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">

        <div className="flex items-center gap-3 border-b border-slate-100 p-5">
          <Database className="text-teal-700" size={21} />

          <div>
            <h2 className="text-lg font-bold text-slate-900">
              Historical Wetland Records
            </h2>
            <p className="text-xs text-slate-500">
              Research observations retrieved from the backend
            </p>
          </div>
        </div>

        
        {/* Historical Wetland Search and Filters */}
        <div className="flex flex-wrap items-center gap-4 border-b border-slate-100 px-5 py-4">

        {/* Search Input */}
        <div className="relative min-w-[240px] flex-1">
            <Search
            size={18}
            className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400"
            />

            <input
            type="text"
            value={searchQuery}
            onChange={(event) => setSearchQuery(event.target.value)}
            placeholder="Search wetland ID, GN division or change status..."
            className="w-full rounded-xl border border-slate-200 bg-slate-50 py-3 pl-10 pr-4 text-sm outline-none transition focus:border-teal-500 focus:ring-2 focus:ring-teal-100"
            />
        </div>

        {/* Change Status Dropdown */}
        <select
            value={selectedStatus}
            onChange={(event) => setSelectedStatus(event.target.value)}
            className="w-full rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-700 outline-none transition focus:border-teal-500 focus:ring-2 focus:ring-teal-100 sm:w-auto sm:max-w-[290px]"
        >
            <option value="all">All Change Statuses</option>

            {changeStatuses.map((status) => (
            <option key={status} value={status}>
                {status}
            </option>
            ))}
        </select>

        {/* Result Counter */}
        <p className="ml-auto text-xs font-medium text-slate-500">
            {loading
            ? 'Loading records...'
            : `Showing ${filteredWetlands.length} of ${wetlands.length} wetlands`}
        </p>

        </div>

        {loading ? (
          <p className="p-6 text-sm text-slate-500">
            Loading historical wetland records...
          </p>
        ) : error ? (
          <p className="p-6 text-sm text-red-600">
            Unable to load historical wetland data: {error}
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="bg-slate-50 text-xs uppercase text-slate-500">
                <tr>
                  <th className="px-5 py-4">Wetland ID</th>
                  <th className="px-5 py-4">Primary GN</th>
                  <th className="px-5 py-4">First Observation</th>
                  <th className="px-5 py-4">Latest Observation</th>
                  <th className="px-5 py-4">First Area (ha)</th>
                  <th className="px-5 py-4">Latest Area (ha)</th>
                  <th className="px-5 py-4">Change Status</th>
                </tr>
              </thead>

              <tbody className="divide-y divide-slate-100">
                {filteredWetlands.map((wetland) => (
                  <tr key={wetland.wetland_id} className="hover:bg-teal-50/40">
                    <td className="px-5 py-4 font-bold text-teal-700">
                      {wetland.wetland_id}
                    </td>
                    <td className="px-5 py-4">
                      {wetland.primary_gn || 'N/A'}
                    </td>
                    <td className="px-5 py-4">
                      {wetland.first_observation_date || 'N/A'}
                    </td>
                    <td className="px-5 py-4">
                      {wetland.latest_observation_date || 'N/A'}
                    </td>
                    <td className="px-5 py-4">
                      {formatArea(wetland.first_area_ha)}
                    </td>
                    <td className="px-5 py-4">
                      {formatArea(wetland.latest_area_ha)}
                    </td>
                    <td className="px-5 py-4">
                      {wetland.change_status || 'N/A'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Research Note */}
      <div className="rounded-xl border border-amber-200 bg-amber-50 p-4 text-xs leading-5 text-amber-900">
        Historical change classifications depend on the availability
        and quality of comparable observations. Raw differences between
        first and latest areas should not automatically be interpreted
        as independently validated wetland loss or gain.
      </div>

    </div>
  )
}
