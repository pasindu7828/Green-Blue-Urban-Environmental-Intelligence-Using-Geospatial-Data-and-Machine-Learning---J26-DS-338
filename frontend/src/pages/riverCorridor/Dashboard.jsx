
import { useEffect, useMemo, useState } from 'react'
import {
  MapContainer,
  TileLayer,
  GeoJSON,
  useMap,
} from 'react-leaflet'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

import {
  ResponsiveContainer,
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from 'recharts'

import './Dashboard.css'

const API_BASE = (
  import.meta.env.VITE_API_BASE_URL ||
  'http://127.0.0.1:8000'
).replace(/\/$/, '')

async function getResearchData(endpoint, signal) {
  const response = await fetch(
    `${API_BASE}/api/river-corridor${endpoint}`,
    { signal }
  )

  if (!response.ok) {
    throw new Error(
      `${endpoint}: HTTP ${response.status}`
    )
  }

  return response.json()
}

function MapBounds({ data }) {
  const map = useMap()

  useEffect(() => {
    if (!data?.features?.length) return

    const bounds = L.geoJSON(data).getBounds()

    if (bounds.isValid()) {
      map.fitBounds(bounds, {
        padding: [25, 25],
      })
    }
  }, [data, map])

  return null
}

function displayWidth(value) {
  return typeof value === 'number' &&
    Number.isFinite(value)
    ? `${value.toFixed(1)} m`
    : 'Unavailable'
}

function MetricCard({ label, value, detail }) {
  return (
    <div className="rc-card rc-stat">
      <span className="rc-muted">{label}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </div>
  )
}

export default function RiverCorridorDashboard() {
  const [summary, setSummary] = useState(null)
  const [geojson, setGeojson] = useState(null)
  const [metrics, setMetrics] = useState([])
  const [annualQA, setAnnualQA] = useState([])

  const [selectedId, setSelectedId] = useState('T0001')
  const [selectedHistory, setSelectedHistory] = useState([])

  const [loading, setLoading] = useState(true)
  const [historyLoading, setHistoryLoading] = useState(false)
  const [error, setError] = useState('')

  // Load real research outputs from FastAPI.
  useEffect(() => {
    const controller = new AbortController()

    async function loadDashboard() {
      try {
        setLoading(true)
        setError('')

        const [
          summaryData,
          geojsonData,
          metricsData,
          annualData,
        ] = await Promise.all([
          getResearchData('/summary', controller.signal),
          getResearchData('/geojson', controller.signal),
          getResearchData('/metrics', controller.signal),
          getResearchData('/annual-qa', controller.signal),
        ])

        if (
          geojsonData.type !== 'FeatureCollection' ||
          !Array.isArray(geojsonData.features)
        ) {
          throw new Error('Invalid GeoJSON response')
        }

        setSummary(summaryData)
        setGeojson(geojsonData)
        setMetrics(metricsData.records ?? [])
        setAnnualQA(annualData.records ?? [])
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

    loadDashboard()

    return () => controller.abort()
  }, [])

  // Load individual historical observations when a
  // transect is selected on the map.
  useEffect(() => {
    const controller = new AbortController()

    async function loadHistory() {
      try {
        setHistoryLoading(true)
        setSelectedHistory([])

        const response = await getResearchData(
          `/history?transect_id=${encodeURIComponent(selectedId)}`,
          controller.signal
        )

        setSelectedHistory(response.records ?? [])
      } catch (err) {
        if (err.name !== 'AbortError') {
          console.error('Transect history:', err)
        }
      } finally {
        if (!controller.signal.aborted) {
          setHistoryLoading(false)
        }
      }
    }

    loadHistory()
    return () => controller.abort()
  }, [selectedId])

  const selectedFeature = useMemo(() => {
    return geojson?.features?.find(
      feature =>
        feature.properties?.transect_id === selectedId
    ) ?? null
  }, [geojson, selectedId])

  const selected = selectedFeature?.properties

  const benchmark = useMemo(() => {
    const labels = {
      Persistence: 'Persistence',
      TrainingMeanChange: 'Mean change',
      RandomForest: 'Random Forest',
    }

    return metrics
      .filter(row => row.split === 'test_2024_2025')
      .map(row => ({
        model: labels[row.method] ?? row.method,
        mae: Number(row.mae_m),
      }))
  }, [metrics])

  const annualChart = useMemo(() => {
    return annualQA.map(row => ({
      year: String(row.year),
      screened: Number(row.pilot_screened),
    }))
  }, [annualQA])

  const historyChart = useMemo(() => {
    return selectedHistory
      .map(row => ({
        year: String(row.year),
        width:
          row.qa_status === 'PILOT_UNVALIDATED' &&
          typeof row.pilot_width_m === 'number'
            ? row.pilot_width_m
            : null,
      }))
      .sort((a, b) => Number(a.year) - Number(b.year))
  }, [selectedHistory])

  const projectionCoverage = summary
    ? (
        (summary.projected_transects /
          summary.total_transects) * 100
      ).toFixed(1)
    : '0.0'

  function getTransectStyle(feature) {
    const props = feature.properties ?? {}

    if (props.transect_id === selectedId) {
      return {
        color: '#f59e0b',
        weight: 5,
        opacity: 1,
      }
    }

    return {
      color: props.screened ? '#0284c7' : '#94a3b8',
      weight: props.screened ? 3 : 2,
      opacity: props.screened ? 0.9 : 0.65,
    }
  }

  function attachTransectEvents(feature, layer) {
    const props = feature.properties ?? {}

    layer.bindTooltip(
      `${props.transect_id} · ${props.reach_id}`
    )

    layer.on('click', () => {
      setSelectedId(props.transect_id)
    })
  }

  if (loading) {
    return (
      <main className="rc-page">
        <div className="rc-card">
          Loading River Corridor research outputs...
        </div>
      </main>
    )
  }

  if (error) {
    return (
      <main className="rc-page">
        <div className="rc-card rc-error">
          <h2>Unable to load River Corridor dashboard</h2>
          <p>{error}</p>
          <p>
            Check that FastAPI is running on port 8000
            and that the research outputs are available.
          </p>
        </div>
      </main>
    )
  }

  return (
    <main className="rc-page">
      <header className="rc-header">
        <div>
          <span className="rc-eyebrow">
            J26-DS-338 · River Corridor Research
          </span>
          <h1>Kelani River Corridor Intelligence</h1>
          <p>
            Historical water-width proxies, experimental
            forecasting and measurement quality analysis.
          </p>
        </div>

        <span className="rc-stage">
          PP1 · Experimental
        </span>
      </header>

      <div className="rc-notice">
        <strong>Research limitation:</strong>{' '}
        Measurements are preliminary satellite-derived
        water-width proxies from 30 m Landsat imagery.
        They are not surveyed riverbank positions,
        confirmed erosion measurements or validated
        erosion-risk predictions.
      </div>

      <section className="rc-stats">
        <MetricCard
          label="Fixed river transects"
          value={summary.total_transects}
          detail="50 m spacing"
        />

        <MetricCard
          label="2026 baseline available"
          value={summary.projected_transects}
          detail={`${projectionCoverage}% of transects`}
        />

        <MetricCard
          label="Unavailable after QA"
          value={summary.unavailable_transects}
          detail="No projection generated"
        />

        <MetricCard
          label="Holdout MAE"
          value={`${summary.selected_test_mae_m.toFixed(2)} m`}
          detail="2024–2025 · screened pairs only"
        />
      </section>

      <section className="rc-card rc-map-card">
        <div className="rc-section-heading">
          <div>
            <h2>River Transect Explorer</h2>
            <p>
              Select a transect to explore its historical
              observations and experimental projection.
            </p>
          </div>

          <span className="rc-tag">
            2025 → 2026
          </span>
        </div>

        <div className="rc-map-layout">
          <div className="rc-map">
            {geojson && (
              <MapContainer
                center={[6.93, 80.02]}
                zoom={12}
                scrollWheelZoom
                style={{ height: '100%', width: '100%' }}
              >
                <TileLayer
                  attribution='&copy; OpenStreetMap contributors'
                  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                />

                <GeoJSON
                  key={selectedId}
                  data={geojson}
                  style={getTransectStyle}
                  onEachFeature={attachTransectEvents}
                />

                <MapBounds data={geojson} />
              </MapContainer>
            )}
          </div>

          <aside className="rc-selected">
            <span className="rc-eyebrow">
              Selected transect
            </span>
            <h2>{selected?.transect_id ?? 'None'}</h2>

            <dl className="rc-details">
              <div>
                <dt>Reach</dt>
                <dd>{selected?.reach_id ?? '—'}</dd>
              </div>
              <div>
                <dt>River chainage</dt>
                <dd>
                  {selected
                    ? `${(selected.chainage_m / 1000).toFixed(2)} km`
                    : '—'}
                </dd>
              </div>
              <div>
                <dt>2025 pilot width</dt>
                <dd>
                  {displayWidth(selected?.pilot_width_m)}
                </dd>
              </div>
              <div>
                <dt>2026 selected baseline</dt>
                <dd>
                  {displayWidth(
                    selected?.persistence_2026_m
                  )}
                </dd>
              </div>
              <div>
                <dt>2026 RF candidate</dt>
                <dd>
                  {displayWidth(
                    selected?.rf_candidate_2026_m
                  )}
                </dd>
              </div>
              <div>
                <dt>Quality flag</dt>
                <dd>{selected?.qa_status ?? '—'}</dd>
              </div>
            </dl>

            <p className="rc-small-note">
              Selected method: Persistence.
              Random Forest is shown only as an
              unselected experimental candidate.
            </p>
          </aside>
        </div>

        <div className="rc-legend">
          <span>
            <i className="rc-legend-dot rc-blue" />
            Pilot-screened
          </span>
          <span>
            <i className="rc-legend-dot rc-grey" />
            No 2026 projection
          </span>
          <span>
            <i className="rc-legend-dot rc-orange" />
            Selected transect
          </span>
        </div>
      </section>

      <section className="rc-chart-grid">
        <div className="rc-card">
          <h2>Annual Measurement Availability</h2>
          <p className="rc-muted">
            Transects passing preliminary screening
          </p>

          <div className="rc-chart">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={annualChart}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="year" />
                <YAxis domain={[0, 271]} />
                <Tooltip />
                <Line
                  type="linear"
                  dataKey="screened"
                  stroke="#0284c7"
                  strokeWidth={2}
                  dot={{ r: 3 }}
                  name="Screened transects"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="rc-card">
          <h2>Forecasting Model Comparison</h2>
          <p className="rc-muted">
            2024–2025 conditional test MAE
            — lower is better
          </p>

          <div className="rc-chart">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={benchmark}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="model"
                  interval={0}
                  tick={{ fontSize: 11 }}
                />
                <YAxis />
                <Tooltip
                  formatter={value => [
                    `${Number(value).toFixed(3)} m`,
                    'MAE',
                  ]}
                />
                <Bar
                  dataKey="mae"
                  name="MAE"
                  fill="#2563eb"
                  radius={[5, 5, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <p className="rc-small-note">
            Persistence was selected using 2023
            validation MAE, not the final test results.
          </p>
        </div>
      </section>

      <section className="rc-card">
        <h2>Selected Transect — Historical Pilot Widths</h2>
        <p className="rc-muted">
          {selectedId} · 2010–2025.
          Only values passing pilot screening are plotted.
        </p>

        <div className="rc-chart">
          {historyLoading ? (
            <p>Loading transect history...</p>
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={historyChart}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="year" />
                <YAxis
                  domain={['auto', 'auto']}
                  unit=" m"
                />
                <Tooltip />
                <Line
                  dataKey="width"
                  name="Pilot water width"
                  stroke="#0f766e"
                  strokeWidth={2}
                  connectNulls={false}
                  dot={{ r: 4 }}
                />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>
      </section>

      <footer className="rc-footer">
        <strong>Research status:</strong> Experimental
        baseline only · Not independently validated ·
        No confirmed riverbank erosion or physical-risk
        predictions.
      </footer>
    </main>
  )
}
