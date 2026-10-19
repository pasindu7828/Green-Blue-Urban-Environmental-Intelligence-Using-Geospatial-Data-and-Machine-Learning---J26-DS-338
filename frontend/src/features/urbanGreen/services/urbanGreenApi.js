export const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ||
  'http://127.0.0.1:8000'
).replace(/\/+$/, '')

const PREFIX = '/api/urban-green'

export class UrbanGreenApiError extends Error {
  constructor(message, status = 0) {
    super(message)
    this.name = 'UrbanGreenApiError'
    this.status = status
  }
}

export function urbanGreenUrl(path) {
  return `${API_BASE_URL}${PREFIX}${path}`
}

function withQuery(path, params = {}) {
  const query = new URLSearchParams()

  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== '') {
      query.set(key, String(value))
    }
  }

  const suffix = query.toString()
  return suffix ? `${path}?${suffix}` : path
}

function encodeId(value) {
  return encodeURIComponent(String(value))
}

export async function urbanGreenRequest(
  path,
  { params = {}, signal } = {}
) {
  const url = urbanGreenUrl(withQuery(path, params))
  let response

  try {
    response = await fetch(url, {
      method: 'GET',
      signal,
      headers: {
        Accept: 'application/json, application/geo+json',
      },
    })
  } catch (error) {
    if (error?.name === 'AbortError') throw error

    throw new UrbanGreenApiError(
      'Could not connect to the Urban Green backend. Check that FastAPI is running.'
    )
  }

  let data = null

  try {
    data = await response.json()
  } catch {
    throw new UrbanGreenApiError(
      `Invalid JSON response from ${url}`,
      response.status
    )
  }

  if (!response.ok) {
    const detail = data?.detail
    const message =
      typeof detail === 'string'
        ? detail
        : `Urban Green API returned HTTP ${response.status}`

    throw new UrbanGreenApiError(message, response.status)
  }

  return data
}

const get = (path, signal, params) =>
  urbanGreenRequest(path, { signal, params })

export const urbanGreenApi = {
  health: (signal) => get('/health', signal),
  dataHealth: (signal) => get('/data-health', signal),
  metadata: (signal) => get('/metadata', signal),
  manifest: (signal) => get('/manifest', signal),

  historical: {
    summary: (signal) => get('/historical/summary', signal),
    classAreas: (signal) => get('/historical/class-areas', signal),
    trajectory: (signal) => get('/historical/trajectory', signal),
    annualQa: (signal) => get('/historical/annual-qa', signal),
    periodStatus: (signal) => get('/historical/period-status', signal),
    dataHealth: (signal) => get('/historical/data-health', signal),

    maps: (signal) => get('/historical/maps', signal),
    mapsHealth: (signal) => get('/historical/maps/health', signal),
    mapDetail: (mapId, signal) =>
      get(`/historical/maps/${encodeId(mapId)}`, signal),
    mapImageUrl: (mapId) =>
      urbanGreenUrl(`/historical/maps/${encodeId(mapId)}/image`),

    transitions: {
      analysis: (signal) =>
        get('/historical/transitions/analysis', signal),
      breakdown: (signal) =>
        get('/historical/transitions/breakdown', signal),
      comparison: (signal) =>
        get('/historical/transitions/comparison', signal),
      confidence: (signal) =>
        get('/historical/transitions/confidence', signal),
      method: (signal) =>
        get('/historical/transitions/method', signal),
    },
  },

  pressure: {
    overview: (signal) => get('/pressure/overview', signal),
    grids: (params = {}, signal) =>
      get('/pressure/grids', signal, params),
    grid: (gridId, signal) =>
      get(`/pressure/grids/${encodeId(gridId)}`, signal),
    map: (signal) => get('/pressure/map', signal),
    mapMetadata: (signal) =>
      get('/pressure/map/metadata', signal),

    xai: {
      overview: (signal) =>
        get('/pressure/xai/overview', signal),
      features: (signal) =>
        get('/pressure/xai/features', signal),
      domains: (signal) =>
        get('/pressure/xai/domains', signal),
    },
  },

  suitability: {
    overview: (signal) => get('/suitability/overview', signal),
    grids: (params = {}, signal) =>
      get('/suitability/grids', signal, params),
    grid: (gridId, signal) =>
      get(`/suitability/grids/${encodeId(gridId)}`, signal),
    map: (signal) => get('/suitability/map', signal),
    mapMetadata: (signal) =>
      get('/suitability/map/metadata', signal),
  },

  conservation: {
    overview: (signal) => get('/conservation/overview', signal),
    grids: (params = {}, signal) =>
      get('/conservation/grids', signal, params),
    grid: (gridId, signal) =>
      get(`/conservation/grids/${encodeId(gridId)}`, signal),
    map: (signal) => get('/conservation/map', signal),
    mapMetadata: (signal) =>
      get('/conservation/map/metadata', signal),
  },

  planning: {
    grids: (params = {}, signal) =>
      get('/planning/grids', signal, params),
    grid: (gridId, signal) =>
      get(`/planning/grids/${encodeId(gridId)}`, signal),
    map: (signal) => get('/planning/map', signal),

    gnDivisions: (signal) =>
      get('/planning/gn-divisions', signal),
    gnDivision: (gnId, signal) =>
      get(`/planning/gn-divisions/${encodeId(gnId)}`, signal),
    gnMap: (signal) => get('/planning/gn-map', signal),

    hotspots: (params = {}, signal) =>
      get('/planning/hotspots', signal, params),
    hotspotMap: (signal) =>
      get('/planning/hotspot-map', signal),
  },
}
