
import { apiRequest, API_BASE_URL } from '../../urbanHeat/services/apiClient'

const P = '/api/wetlands'

const enc = (value) => encodeURIComponent(String(value))

export const wetlandApi = {
  // Part 1 - Historical Wetland Change
  historicalSummary: (signal) =>
    apiRequest(`${P}/part1/summary`, { signal }),

  historicalMap: (signal) =>
    apiRequest(`${P}/part1/map`, { signal }),

  historicalYears: (signal) =>
    apiRequest('/api/wetlands/part1/historical/years', { signal }),

  historicalMapByYear: (year, signal) =>
    apiRequest(`/api/wetlands/part1/historical/map/${year}`, { signal }),


  individualWetlandObservations: (wetlandId, signal) =>
    apiRequest(
      `/api/wetlands/part1/historical/wetlands/${encodeURIComponent(wetlandId)}/observations`,
      { signal }
    ),

  comparisonEvidence: (wetlandId, beforeDate, afterDate, signal) =>
    apiRequest(
      `/api/wetlands/part1/historical/wetlands/${encodeURIComponent(wetlandId)}/comparison-evidence?${new URLSearchParams({
        before_date: beforeDate,
        after_date: afterDate,
      }).toString()}`,
      { signal }
    ),

  historicalValidation: (signal) =>
    apiRequest(`${P}/part1/validation`, { signal }),

  wetlandDetails: (wetlandId, signal) =>
    apiRequest(`${P}/part1/wetlands/${enc(wetlandId)}`, { signal }),

  // Part 2 - Candidate Wetland Detection
  primaryCandidates: (signal) =>
    apiRequest(`${P}/part2/primary-candidates`, { signal }),

  allCandidates: (signal) =>
    apiRequest(`${P}/part2/all-candidates`, { signal }),

  candidateGnSummary: (signal) =>
    apiRequest(`${P}/part2/gn-summary`, { signal }),

  // Part 3 - Exploratory Future Wetland Scenarios
  futureGnSummary: (signal) =>
    apiRequest(`${P}/part3/gn-summary`, { signal }),

  // Part 3 — GN summary for selected 2025 and 2030 model patches
  focusedFutureGnSummary: (signal) =>
    apiRequest(`${P}/part3/focused-gn-summary`, { signal }),

  // Part 3 — Historical wetland and nearby 2030 scenario patches
  individualFutureComparison: (wetlandId, signal) =>
    apiRequest(
      `${P}/part3/individual-future-comparison/${enc(wetlandId)}`,
      { signal }
    ),

  futureWetlandMap: (year, signal) =>
    apiRequest(`${P}/part3/wetland-map/${enc(year)}`, { signal }),

  futureScenario: (changeType, signal) =>
    apiRequest(`${P}/part3/scenario/${enc(changeType)}`, { signal }),

  // Focused 2030 scenario — provisional 50 m selection
  focusedFutureScenario: (signal) =>
    apiRequest(`${P}/part3/focused-scenario`, { signal }),

  // Part 4 - Seasonal Waterlogging Susceptibility
  waterloggingGnSummary: (signal) =>
    apiRequest(`${P}/part4/gn-summary`, { signal }),

  rasterMetadata: (mapType, signal) =>
    apiRequest(`${P}/part4/raster/${enc(mapType)}/metadata`, { signal }),

  rasterPngUrl: (mapType) =>
    `${API_BASE_URL}${P}/part4/raster/${enc(mapType)}/png`,
}
