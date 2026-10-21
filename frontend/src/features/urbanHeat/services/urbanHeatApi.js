import { apiRequest, queryString } from './apiClient'

const P = '/api/urban-heat'
const enc = (value) => encodeURIComponent(String(value))

export const urbanHeatApi = {
  health: (signal) => apiRequest(`${P}/health`, { signal }),
  dataHealth: (signal) => apiRequest(`${P}/data-health`, { signal }),
  manifest: (signal) => apiRequest(`${P}/manifest`, { signal }),
  areas: (signal) => apiRequest(`${P}/areas`, { signal }),
  gridMetadata: (gridId, signal) => apiRequest(`${P}/grid-metadata/${enc(gridId)}`, { signal }),

  historical: (year, signal) => apiRequest(`${P}/historical${queryString({ year })}`, { signal }),
  gridHistory: (gridId, signal) => apiRequest(`${P}/history/${enc(gridId)}`, { signal }),
  historicalMap: (year, signal) => apiRequest(`${P}/map/historical${queryString({ year })}`, { signal }),

  forecast: (year, signal) => apiRequest(`${P}/forecast${queryString({ year })}`, { signal }),
  gridForecast: (gridId, year, signal) => apiRequest(`${P}/forecast/grid/${enc(gridId)}${queryString({ year })}`, { signal }),
  gridForecastAll: (gridId, signal) => apiRequest(`${P}/forecast/grid/${enc(gridId)}/all`, { signal }),
  gridDetails: (gridId, year, signal) => apiRequest(`${P}/grid/${enc(gridId)}${queryString({ year })}`, { signal }),
  forecastMap: (year, signal) => apiRequest(`${P}/map/forecast-lite${queryString({ year })}`, { signal }),

  currentVsForecast: (year, areaName, signal) => apiRequest(`${P}/current-vs-forecast${queryString({ year, area_name: areaName })}`, { signal }),
  gridCurrentVsForecast: (gridId, year, signal) => apiRequest(`${P}/current-vs-forecast/grid/${enc(gridId)}${queryString({ year })}`, { signal }),

  xai: (gridId, year, signal) => apiRequest(`${P}/xai/grid/${enc(gridId)}${queryString({ year })}`, { signal }),
  uncertainty: (gridId, year, signal) => apiRequest(`${P}/uncertainty/grid/${enc(gridId)}${queryString({ year })}`, { signal }),

  trajectory: (areaName, signal) => apiRequest(`${P}/trajectory${queryString({ area_name: areaName })}`, { signal }),
  gridTrajectory: (gridId, signal) => apiRequest(`${P}/trajectory/grid/${enc(gridId)}`, { signal }),
  trajectoryMap: (signal) => apiRequest(`${P}/map/trajectory`, { signal }),

  priority: ({ limit, areaName } = {}, signal) => apiRequest(`${P}/priority${queryString({ limit, area_name: areaName })}`, { signal }),
  gridPriority: (gridId, signal) => apiRequest(`${P}/priority/grid/${enc(gridId)}`, { signal }),
  priorityMap: (signal) => apiRequest(`${P}/map/priority-lite`, { signal }),

  areaHeat: ({ areaName, year } = {}, signal) => apiRequest(`${P}/areas/heat${queryString({ area_name: areaName, year })}`, { signal }),
  areaPriority: (areaName, signal) => apiRequest(`${P}/areas/priority${queryString({ area_name: areaName })}`, { signal }),

  gridMap: (signal) => apiRequest(`${P}/map/grids`, { signal }),

  primaryScenario: (gridId, year, signal) => apiRequest(`${P}/scenario/grid/${enc(gridId)}${queryString({ year })}`, { signal }),
  gridScenarios: (gridId, year, signal) => apiRequest(`${P}/scenarios/grid/${enc(gridId)}${queryString({ year })}`, { signal }),
  scenarioOptions: (signal) => apiRequest(`${P}/scenario/options`, { signal }),
  simulateScenario: (payload, signal) => apiRequest(`${P}/scenario/simulate`, { method: 'POST', body: payload, signal }),
}
