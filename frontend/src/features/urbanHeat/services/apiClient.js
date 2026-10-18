export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '')

export class ApiError extends Error {
  constructor(message, { status = 0, data = null, url = '' } = {}) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.data = data
    this.url = url
  }
}

export async function apiRequest(path, { method = 'GET', body, signal, headers = {} } = {}) {
  const url = `${API_BASE_URL}${path.startsWith('/') ? path : `/${path}`}`
  let response
  try {
    response = await fetch(url, {
      method,
      signal,
      headers: {
        ...(body !== undefined ? { 'Content-Type': 'application/json' } : {}),
        ...headers,
      },
      body: body !== undefined ? JSON.stringify(body) : undefined,
    })
  } catch (error) {
    if (error?.name === 'AbortError') throw error
    throw new ApiError('Could not connect to the backend API. Make sure FastAPI is running on the configured base URL.', { url })
  }

  const contentType = response.headers.get('content-type') || ''
  let data = null
  if (contentType.includes('application/json')) {
    data = await response.json().catch(() => null)
  } else {
    data = await response.text().catch(() => null)
  }

  if (!response.ok) {
    const detail = data && typeof data === 'object' ? data.detail : null
    const message = typeof detail === 'string'
      ? detail
      : Array.isArray(detail)
        ? detail.map((item) => item.msg || JSON.stringify(item)).join('; ')
        : `API request failed with HTTP ${response.status}.`
    throw new ApiError(message, { status: response.status, data, url })
  }

  return data
}

export function queryString(params = {}) {
  const qs = new URLSearchParams()
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== '') qs.set(key, String(value))
  })
  const value = qs.toString()
  return value ? `?${value}` : ''
}
