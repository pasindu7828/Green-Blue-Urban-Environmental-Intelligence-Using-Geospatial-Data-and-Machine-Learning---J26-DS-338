import { useEffect, useState } from 'react'
import WaterloggingGNMap from './WaterloggingGNMap'

const API_URL =
  'http://127.0.0.1:8000/api/wetlands/part4/gn-summary'

export default function WaterloggingGNSummary() {
  const [features, setFeatures] = useState([])
  const [search, setSearch] = useState('')
  const [selectedGN, setSelectedGN] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    const controller = new AbortController()

    async function loadData() {
      try {
        const response = await fetch(API_URL, {
          signal: controller.signal,
        })

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`)
        }

        const data = await response.json()

        if (
          data?.type !== 'FeatureCollection' ||
          !Array.isArray(data.features)
        ) {
          throw new Error('Invalid GN summary response')
        }

        setFeatures(
          [...data.features].sort((a, b) =>
            String(a.properties?.gn_name ?? '').localeCompare(
              String(b.properties?.gn_name ?? '')
            )
          )
        )
      } catch (err) {
        if (err.name !== 'AbortError') {
          setError(err.message)
        }
      } finally {
        if (!controller.signal.aborted) {
          setLoading(false)
        }
      }
    }

    loadData()

    return () => controller.abort()
  }, [])

  const filtered = features.filter((feature) => {
    const name = feature.properties?.gn_name ?? ''
    const number = feature.properties?.gn_number ?? ''

    return `${name} ${number}`
      .toLowerCase()
      .includes(search.toLowerCase())
  })

  const format = (value) =>
    Number(value ?? 0).toLocaleString(undefined, {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    })

  return (
    <section className="rounded-xl border border-slate-200 bg-white p-5">
      <h2 className="text-lg font-bold text-slate-900">
        GN-Level Waterlogging Susceptibility
      </h2>

      <p className="mt-2 text-sm leading-6 text-slate-600">
        Explore relative susceptibility across the 56 GN
        divisions using the existing research summary.
      </p>

      {loading ? (
        <p className="mt-5 text-sm text-slate-500">
          Loading GN statistics...
        </p>
      ) : error ? (
        <p className="mt-5 text-sm text-red-600">
          Unable to load GN summary: {error}
        </p>
      ) : (
        <>
          <div className="mt-5 flex flex-wrap items-center justify-between gap-3">
            <p className="text-sm font-medium text-teal-700">
              {filtered.length} of {features.length} GN divisions
            </p>

            <input
              type="text"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search GN name or number..."
              className="w-full rounded-lg border border-slate-300 px-4 py-2 text-sm outline-none focus:border-teal-600 sm:w-72"
            />
          </div>

          <div className="mt-4 overflow-x-auto rounded-lg border border-slate-200">
            <table className="min-w-full divide-y divide-slate-200 text-sm">
              <thead className="bg-slate-50 text-left text-slate-700">
                <tr>
                  <th className="px-4 py-3 font-semibold">GN Division</th>
                  <th className="px-4 py-3 font-semibold">GN Number</th>
                  <th className="px-4 py-3 font-semibold">Low (ha)</th>
                  <th className="px-4 py-3 font-semibold">Moderate (ha)</th>
                  <th className="px-4 py-3 font-semibold">High (ha)</th>
                  <th className="px-4 py-3 font-semibold">Very High (ha)</th>
                  <th className="px-4 py-3 font-semibold">High + Very High (%)</th>
                </tr>
              </thead>

              <tbody className="divide-y divide-slate-100">
                {filtered.map((feature) => {
                  const p = feature.properties

                  return (
                    <tr
                        key={`${p.admin_code}-${p.gn_number}`}
                        onClick={() => setSelectedGN(feature)}
                        className={`cursor-pointer transition ${
                            selectedGN?.properties?.admin_code === p.admin_code
                            ? 'bg-teal-50'
                            : 'hover:bg-slate-50'
                        }`}
                    >
                      <td className="px-4 py-3 font-medium text-slate-900">
                        {p.gn_name}
                      </td>
                      <td className="px-4 py-3 text-slate-600">
                        {p.gn_number}
                      </td>
                      <td className="px-4 py-3">{format(p.low_area_ha)}</td>
                      <td className="px-4 py-3">{format(p.moderate_area_ha)}</td>
                      <td className="px-4 py-3">{format(p.high_area_ha)}</td>
                      <td className="px-4 py-3">{format(p.very_high_area_ha)}</td>
                      <td className="px-4 py-3 font-semibold text-orange-700">
                        {format(p.high_very_high_percent)}%
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>

{/* Selected GN Details Popup */}
{selectedGN && (
  <div
    className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/50 p-4"
    onClick={() => setSelectedGN(null)}
  >
    <div
      role="dialog"
      aria-modal="true"
      aria-label="GN Waterlogging Susceptibility Details"
      className="max-h-[90vh] w-full max-w-2xl overflow-y-auto rounded-2xl bg-white p-6 shadow-2xl"
      onClick={(event) => event.stopPropagation()}
    >
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-teal-700">
            GN Waterlogging Susceptibility
          </p>

          <h3 className="mt-2 text-xl font-bold text-slate-900">
            {selectedGN.properties.gn_name}
          </h3>

          <p className="mt-1 text-sm text-slate-500">
            GN Number: {selectedGN.properties.gn_number}
          </p>
        </div>

        <button
          type="button"
          onClick={() => setSelectedGN(null)}
          aria-label="Close popup"
          className="rounded-lg px-2 py-1 text-xl text-slate-500 hover:bg-slate-100"
        >
          ×
        </button>
      </div>

      <div className="mt-5 grid grid-cols-2 gap-3">
        {[
          {
            label: 'Low',
            value: selectedGN.properties.low_area_ha,
            color: 'bg-green-500',
          },
          {
            label: 'Moderate',
            value: selectedGN.properties.moderate_area_ha,
            color: 'bg-yellow-400',
          },
          {
            label: 'High',
            value: selectedGN.properties.high_area_ha,
            color: 'bg-orange-500',
          },
          {
            label: 'Very High',
            value: selectedGN.properties.very_high_area_ha,
            color: 'bg-red-600',
          },
        ].map((item) => (
          <div
            key={item.label}
            className="rounded-xl border border-slate-200 p-3"
          >
            <div className="flex items-center gap-2">
              <span className={`h-3 w-3 rounded ${item.color}`} />
              <span className="text-xs text-slate-600">
                {item.label}
              </span>
            </div>

            <p className="mt-2 text-lg font-bold text-slate-900">
              {format(item.value)} ha
            </p>
          </div>
        ))}
      </div>

      <div className="mt-4 rounded-xl bg-orange-50 p-4">
        <p className="text-sm text-orange-800">
          High + Very High Susceptibility
        </p>

        <p className="mt-1 text-2xl font-bold text-orange-700">
          {format(
            selectedGN.properties.high_very_high_percent
          )}%
        </p>

        <p className="mt-1 text-xs text-orange-800">
          Of classified general-land cells.
        </p>
      </div>

        {/* Selected GN Geographic Location */}
        <div className="mt-5">
        <h4 className="mb-3 text-sm font-bold text-slate-900">
            GN Division Geographic Location
        </h4>

        <WaterloggingGNMap
            key={selectedGN.properties.admin_code}
            feature={selectedGN}
        />
        </div>

      <p className="mt-4 text-xs leading-5 text-slate-500">
        These values indicate relative waterlogging
        susceptibility, not observed flooding or a
        validated flood prediction.
      </p>

      <button
        type="button"
        onClick={() => setSelectedGN(null)}
        className="mt-5 w-full rounded-lg bg-teal-700 px-4 py-2.5 text-sm font-semibold text-white hover:bg-teal-800"
      >
        Close
      </button>
    </div>
  </div>
)}


          <p className="mt-3 text-xs leading-5 text-slate-500">
            Susceptibility area statistics refer to classified
            general-land cells. The High + Very High percentage
            is relative to the GN's general-land classification,
            not its entire geographic area.
            These are relative susceptibility indicators,
            not observed flood frequencies.
          </p>
        </>
      )}
    </section>
  )
}
