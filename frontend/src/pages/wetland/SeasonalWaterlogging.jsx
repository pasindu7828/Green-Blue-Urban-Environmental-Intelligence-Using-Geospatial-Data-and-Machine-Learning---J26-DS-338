import { useState } from 'react'
import WaterloggingRasterMap from '../../features/wetland/components/WaterloggingRasterMap'
import WaterloggingGNSummary from '../../features/wetland/components/WaterloggingGNSummary'

const mapOptions = [
  {
    id: 'main',
    label: 'Main Susceptibility',
    description:
      'Relative seasonal waterlogging susceptibility across the study area.',
  },
  {
    id: 'terrain',
    label: 'Terrain Susceptibility',
    description:
      'Terrain-based relative waterlogging susceptibility classification.',
  },
  {
    id: 'span',
    label: 'Scenario Disagreement',
    description:
      'Differences in susceptibility classification across model scenarios.',
  },
]

export default function SeasonalWaterlogging() {
  const [selectedMap, setSelectedMap] = useState('main')

  const activeMap = mapOptions.find(
    (option) => option.id === selectedMap
  )

  return (
    <div className="space-y-6">
      <div>
        <p className="text-xs font-semibold uppercase tracking-wide text-teal-700">
          Part 4 — Wetland Research
        </p>

        <h1 className="mt-2 text-2xl font-bold text-slate-900">
          Seasonal Waterlogging Susceptibility
        </h1>

        <p className="mt-3 max-w-4xl text-sm leading-6 text-slate-600">
          Explore spatial patterns of relative waterlogging
          susceptibility across the Kaduwela study area,
          including terrain-based classifications and
          disagreement between modelling scenarios.
        </p>
      </div>

      <div className="rounded-xl border border-slate-200 bg-white p-5">
        <h2 className="text-lg font-bold text-slate-900">
          Waterlogging Susceptibility Map
        </h2>

        <p className="mt-2 text-sm text-slate-500">
          Select a map layer to explore the available
          research outputs.
        </p>

        <div className="mt-5 flex flex-wrap gap-2">
          {mapOptions.map((option) => (
            <button
              key={option.id}
              type="button"
              onClick={() => setSelectedMap(option.id)}
              className={`rounded-lg border px-4 py-2 text-sm font-medium transition ${
                selectedMap === option.id
                  ? 'border-teal-700 bg-teal-700 text-white'
                  : 'border-slate-200 bg-white text-slate-700 hover:bg-slate-50'
              }`}
            >
              {option.label}
            </button>
          ))}
        </div>

        
        <div className="mt-5">
        <h3 className="text-base font-semibold text-slate-800">
            {activeMap.label}
        </h3>

        <p className="mt-2 text-sm text-slate-600">
            {activeMap.description}
        </p>

        <div className="mt-4">
            <WaterloggingRasterMap
            key={selectedMap}
            mapType={selectedMap}
            />
        </div>
        </div>
      </div>
      {/* Part 4 — GN-Level Susceptibility Statistics */}
        <WaterloggingGNSummary />

      <div className="rounded-xl border border-amber-200 bg-amber-50 p-4">
        <p className="text-sm leading-6 text-amber-900">
          <strong>Research interpretation:</strong> These
          maps represent relative waterlogging susceptibility
          and scenario classification differences. They are
          not verified flood events, flood-depth predictions,
          or independently validated waterlogging forecasts.
        </p>
      </div>
    </div>
  )
}
