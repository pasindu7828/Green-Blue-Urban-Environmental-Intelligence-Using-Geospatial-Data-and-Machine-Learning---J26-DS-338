import { useEffect, useMemo, useState } from 'react'
import { MapContainer, TileLayer, GeoJSON, ImageOverlay, useMap } from 'react-leaflet'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import {
  LayoutDashboard, Waves, Leaf, Building2, BrainCircuit, ShieldCheck,
  Satellite, Map as MapIcon, Layers, Activity, Info, ChevronRight,
  Menu, ArrowUpRight, ChevronDown, RefreshCw, MapPin, AlertTriangle,
  BarChart3, Search, X, Eye, TrendingDown, ClipboardCheck, Database,
  Compass, ArrowRight, CheckCircle2, CircleHelp, ExternalLink,
} from 'lucide-react'
import {
  ResponsiveContainer, BarChart, Bar, LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend, ReferenceLine, Cell,
} from 'recharts'
import './Dashboard.css'

const API_BASE = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '')
const API = `${API_BASE}/api/river-corridor`
const NAV = [
  { id: 'overview', label: 'Overview', icon: LayoutDashboard, sub: 'Research command center' },
  { id: 'morphology', label: 'River Morphology', icon: Waves, sub: 'Water and width change' },
  { id: 'vegetation', label: 'Vegetation', icon: Leaf, sub: 'Riparian environment' },
  { id: 'urban', label: 'Urban Pressure', icon: Building2, sub: 'OSM spatial intelligence' },
  { id: 'forecast', label: 'ML Forecasting', icon: BrainCircuit, sub: 'Models and 2026 scenarios' },
  { id: 'planning', label: 'Planning Priority', icon: ShieldCheck, sub: 'RCPI decision support' },
]
const CLASS_COLORS = {
  LOW: '#35ad83', MEDIUM: '#edbe65', HIGH: '#f18a57',
  VERY_HIGH: '#ed5666', INSUFFICIENT_EVIDENCE: '#94a3b8',
}
const CLASS_LABELS = {
  LOW: 'Low', MEDIUM: 'Medium', HIGH: 'High',
  VERY_HIGH: 'Very High', INSUFFICIENT_EVIDENCE: 'Insufficient evidence',
}
const TARGETS = [
  { id: 'width_delta_m', label: 'Channel-width change', unit: 'm' },
  { id: 'start_edge_delta_m', label: 'Start water-edge displacement', unit: 'm' },
  { id: 'end_edge_delta_m', label: 'End water-edge displacement', unit: 'm' },
  { id: 'vegetation_loss_pct', label: 'Vegetation loss', unit: 'percentage points' },
  { id: 'median_ndvi_delta', label: 'Median NDVI change', unit: 'NDVI units' },
]
const PERIODS = [
  { key: '2020-2024', a: 2020, b: 2024, label: '2020 → 2024' },
  { key: '2018-2024', a: 2018, b: 2024, label: '2018 → 2024' },
  { key: '2018-2025', a: 2018, b: 2025, label: '2018 → 2025' },
]
const uiNumber = (x, places = 2, suffix = '') => (x == null || !Number.isFinite(Number(x)))
  ? 'Not available' : `${Number(x).toFixed(places)}${suffix}`
const shortNum = (x) => (x == null || !Number.isFinite(Number(x))) ? '—' : Number(x).toLocaleString()
const fmtSigned = (x, places = 2, suffix = '') => (x == null || !Number.isFinite(Number(x)))
  ? 'Not available' : `${Number(x) > 0 ? '+' : ''}${Number(x).toFixed(places)}${suffix}`
const targetName = id => TARGETS.find(t => t.id === id)?.label || id
const getPeriodRow = (rows, a, b) => (rows || []).find(r => Number(r.year_a) === a && Number(r.year_b) === b)

async function fetchData(url, signal) {
  const response = await fetch(`${API}${url}`, { signal })
  if (!response.ok) {
    let message = `${url} returned HTTP ${response.status}`
    try { const error = await response.json(); message = error.detail || message } catch { /* no JSON error body */ }
    throw new Error(message)
  }
  return response.json()
}

function PriorityChip({ value, compact = false }) {
  const label = CLASS_LABELS[value] || 'Not classified'
  return <span className={`rcv-chip ${compact ? 'rcv-chip-sm' : ''}`} style={{
    '--chip-color': CLASS_COLORS[value] || '#94a3b8',
  }}><span className="rcv-chip-circle" />{label}</span>
}

function IconHeading({ kicker, title, desc, icon: Icon, trailing }) {
  return <div className="rcv-section-head">
    <div className="rcv-section-identity">
      {Icon && <div className="rcv-section-icon"><Icon size={19} /></div>}
      <div><div className="rcv-kicker">{kicker}</div><h2>{title}</h2>{desc && <p>{desc}</p>}</div>
    </div>
    {trailing}
  </div>
}
function StatCard({ icon: Icon, label, number, detail, color = 'teal' }) {
  return <article className="rcv-stat"><span className={`rcv-stat-icon rcv-stat-${color}`}><Icon size={18}/></span>
    <div className="rcv-stat-label">{label}</div><div className="rcv-stat-value">{number}</div>
    <div className="rcv-stat-caption">{detail}</div></article>
}
function DataNote({ children, level = 'info' }) {
  return <div className={`rcv-note rcv-note-${level}`}><Info size={16}/><div>{children}</div></div>
}
function Field({ name, value, note }) {
  return <div className="rcv-field"><div><span>{name}</span>{note && <small>{note}</small>}</div><strong>{value}</strong></div>
}
function Plot({ title, subtitle, children, height = 265, legend }) {
  return <article className="rcv-panel rcv-plot"><div className="rcv-panel-head"><div><h3>{title}</h3>{subtitle && <p>{subtitle}</p>}</div>{legend}</div>
    <div className="rcv-plot-area" style={{ height }}>{children}</div></article>
}
function EmptyState({ title, detail }) {
  return <div className="rcv-empty"><CircleHelp size={30}/><strong>{title}</strong><span>{detail}</span></div>
}
function MapFit({ features }) {
  const map = useMap()
  useEffect(() => {
    if (!features?.features?.length) return
    const bounds = L.geoJSON(features).getBounds()
    if (bounds.isValid()) map.fitBounds(bounds, { padding: [24, 24], maxZoom: 13 })
  }, [features, map])
  return null
}
function FocusOnReach({ data, chosen }) {
  const map = useMap()
  useEffect(() => {
    if (!chosen || !data?.features) return
    const feature = data.features.find(f => f.properties?.reach_id === chosen)
    if (!feature) return
    const bounds = L.geoJSON(feature).getBounds()
    if (bounds.isValid()) map.flyToBounds(bounds.pad(0.38), { duration: 0.7, maxZoom: 16 })
  }, [chosen, data, map])
  return null
}
function ReachMap({
  geometry, centreline, selectedId, onSelect, mode = 'priority',
  comparisonData = [], basemap, setBasemap, waterCatalog,
  maskYear, setMaskYear, osm = null, transects = null, onTransect,
  selectedTransect = '', tall = false, focusToken,
}) {
  const reachData = useMemo(() => Object.fromEntries(comparisonData.map(r => [r.reach_id, r])), [comparisonData])
  const color = feature => {
    const id = feature.properties?.reach_id
    if (mode === 'vegetation') {
      const v = reachData[id]?.net_vegetation_change_ha
      if (v == null) return '#94a3b8'
      return v < -0.05 ? '#ed5666' : v > 0.05 ? '#23c18a' : '#e0b65c'
    }
    if (mode === 'urban') {
      const b = Number(feature.properties?.building_footprint_pct ?? 0)
      return b >= 12 ? '#ed5666' : b >= 8 ? '#ff9653' : b >= 4 ? '#f0bc6b' : '#56bda9'
    }
    if (mode === 'width') return '#55bbeb'
    return CLASS_COLORS[feature.properties?.priority_class] || '#94a3b8'
  }
  function styleFeature(feature) {
    const active = feature.properties?.reach_id === selectedId
    return {
      color: active ? '#ffffff' : color(feature),
      weight: active ? 3.8 : 1.85,
      opacity: 0.98, fillColor: color(feature), fillOpacity: active ? 0.45 : 0.28,
      dashArray: feature.properties?.rcpi_eligible === false && mode === 'priority' ? '5 5' : undefined,
    }
  }
  function bindReach(feature, layer) {
    const p = feature.properties || {}
    layer.bindTooltip(`${p.reach_id || 'Reach'} · ${CLASS_LABELS[p.priority_class] || 'Unclassified'}`, { sticky: true })
    layer.on('click', () => onSelect(p.reach_id))
  }
  const currentWater = waterCatalog.find(x => Number(x.year) === Number(maskYear))
  return <div className={`rcv-map-shell ${tall ? 'rcv-map-tall' : ''}`}>
    <div className="rcv-map-tools">
      <span className="rcv-map-tag"><Satellite size={14}/> Interactive GIS</span>
      <div className="rcv-map-switch"><button className={basemap === 'satellite' ? 'active' : ''} onClick={() => setBasemap('satellite')}>Satellite</button><button className={basemap === 'street' ? 'active' : ''} onClick={() => setBasemap('street')}>Streets</button></div>
      {waterCatalog.length > 0 && <select aria-label="Satellite-derived water-mask overlay year" value={maskYear} onChange={e => setMaskYear(e.target.value)}>
        <option value="">No water mask</option>
        {waterCatalog.map(x => <option key={x.year} value={x.year}>Water mask {x.year}</option>)}
      </select>}
    </div>
    <MapContainer center={[6.93, 79.998]} zoom={12} scrollWheelZoom={true} zoomControl={false} className="rcv-leaflet">
      {basemap === 'satellite' ? <TileLayer
        attribution='Tiles &copy; Esri, Maxar, Earthstar Geographics and the GIS User Community'
        url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
        maxZoom={19}
      /> : <TileLayer attribution='&copy; OpenStreetMap contributors' url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" maxZoom={19}/>}
      {currentWater && <ImageOverlay key={`water-${currentWater.year}`} bounds={currentWater.bounds}
        url={`${API}/intelligence/water-overlay/${currentWater.year}.png`} opacity={0.72} />}
      {geometry && <GeoJSON key={`${mode}-${selectedId}-${comparisonData.length}-${maskYear}`}
        data={geometry} style={styleFeature} onEachFeature={bindReach}/>}
      {centreline && <GeoJSON data={centreline} style={{ color: '#32c7ff', weight: 2.4, opacity: 0.95 }} />}
      {osm?.buildings && <GeoJSON key={`buildings-${selectedId}`} data={osm.buildings}
        style={{ color: '#eee5d4', weight: 0.5, fillColor: '#a5b7c7', fillOpacity: 0.8 }} />}
      {osm?.roads && <GeoJSON key={`roads-${selectedId}`} data={osm.roads}
        style={{ color: '#ffbe68', weight: 2.3, opacity: 0.9 }} />}
      {transects?.features?.length > 0 && <GeoJSON key={`transects-${selectedId}-${selectedTransect}`} data={transects}
        style={f => ({
          color: f.properties?.transect_id === selectedTransect ? '#f9dd64' : (f.properties?.qa_usable_years || 0) >= 4 ? '#b6f2ff' : '#9ba9bc',
          weight: f.properties?.transect_id === selectedTransect ? 3.5 : 1.4,
          opacity: f.properties?.transect_id === selectedTransect ? 1 : 0.75,
        })}
        onEachFeature={(feature, layer) => {
          layer.bindTooltip(`${feature.properties?.transect_id || 'Transect'} · ${feature.properties?.reach_id || ''}`)
          layer.on('click', () => onTransect?.(feature.properties?.transect_id, feature.properties?.reach_id))
        }}/>}
      <MapFit features={geometry}/>
      {focusToken > 0 && <FocusOnReach data={geometry} chosen={selectedId} />}
    </MapContainer>
    <div className="rcv-map-bottom">
      <span><span className="rcv-dot" style={{ background: '#32c7ff' }}/> River centreline</span>
      {mode === 'vegetation' ? <><span><span className="rcv-dot" style={{ background: '#ed5666' }}/> Net decrease</span><span><span className="rcv-dot" style={{ background: '#23c18a' }}/> Net increase</span></> : mode === 'urban' ? <span>Mapped building footprint coverage</span> : mode === 'width' ? <span>Experimental transect widths</span> : <span>Preliminary planning priority · grey = insufficient evidence</span>}
    </div>
  </div>
}

export default function RiverCorridorDashboard() {
  const [page, setPage] = useState('overview')
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [overview, setOverview] = useState(null)
  const [geometry, setGeometry] = useState(null)
  const [centreline, setCentreline] = useState(null)
  const [waterCatalog, setWaterCatalog] = useState([])
  const [selectedId, setSelectedId] = useState('R020')
  const [reachDetails, setReachDetails] = useState(null)
  const [detailLoading, setDetailLoading] = useState(false)
  const [osm, setOsm] = useState(null)
  const [transects, setTransects] = useState(null)
  const [transectHistory, setTransectHistory] = useState(null)
  const [transectHistoryLoading, setTransectHistoryLoading] = useState(false)
  const [transectHistoryError, setTransectHistoryError] = useState('')
  const [selectedTransect, setSelectedTransect] = useState('')
  const [showTransects, setShowTransects] = useState(false)
  const [basemap, setBasemap] = useState('satellite')
  const [maskYear, setMaskYear] = useState('')
  const [periodKey, setPeriodKey] = useState('2020-2024')
  const [targetId, setTargetId] = useState('width_delta_m')
  const [focusToken, setFocusToken] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [reloadKey, setReloadKey] = useState(0)

  useEffect(() => {
    const controller = new AbortController()
    async function load() {
      setLoading(true); setError('')
      try {
        const [research, geog, line, water] = await Promise.all([
          fetchData('/intelligence/overview', controller.signal),
          fetchData('/intelligence/reaches.geojson', controller.signal),
          fetchData('/intelligence/centreline.geojson', controller.signal),
          fetchData('/intelligence/water-catalog', controller.signal),
        ])
        if (!Array.isArray(research?.reaches) || geog.type !== 'FeatureCollection') throw new Error('Unexpected research API response')
        setOverview(research); setGeometry(geog); setCentreline(line)
        setWaterCatalog(water.layers || [])
      } catch (err) { if (err.name !== 'AbortError') setError(err.message) }
      finally { if (!controller.signal.aborted) setLoading(false) }
    }
    load()
    return () => controller.abort()
  }, [reloadKey])

  useEffect(() => {
    if (!overview || !selectedId) return
    const controller = new AbortController()
    setDetailLoading(true); setReachDetails(null)
    fetchData(`/intelligence/reach/${selectedId}`, controller.signal)
      .then(d => { if (!controller.signal.aborted) setReachDetails(d) })
      .catch(err => { if (err.name !== 'AbortError') console.error('River reach details', err) })
      .finally(() => { if (!controller.signal.aborted) setDetailLoading(false) })
    return () => controller.abort()
  }, [selectedId, overview])

  useEffect(() => {
    if (page !== 'urban' || !selectedId) return
    const controller = new AbortController()
    setOsm(null)
    fetchData(`/intelligence/osm/${selectedId}`, controller.signal)
      .then(d => { if (!controller.signal.aborted) setOsm(d) })
      .catch(err => { if (err.name !== 'AbortError') console.error('OSM inspection overlay', err) })
    return () => controller.abort()
  }, [page, selectedId])

  useEffect(() => {
    if (!showTransects || transects) return
    const controller = new AbortController()
    fetchData('/intelligence/transects.geojson', controller.signal).then(setTransects)
      .catch(err => { if (err.name !== 'AbortError') console.error('Transect geometry', err) })
    return () => controller.abort()
  }, [showTransects, transects])

  useEffect(() => {
    if (!selectedTransect) { setTransectHistory(null); return }
    const controller = new AbortController()
    setTransectHistory(null)
    setTransectHistoryError('')
    setTransectHistoryLoading(true)
    fetchData(`/intelligence/transects/${encodeURIComponent(selectedTransect)}/annual`, controller.signal)
      .then(d => { if (!controller.signal.aborted) setTransectHistory(d) })
      .catch(err => {
        if (!controller.signal.aborted && err.name !== 'AbortError') {
          setTransectHistoryError(err.message)
          console.error('Sentinel-2 transect annual history', err)
        }
      })
      .finally(() => { if (!controller.signal.aborted) setTransectHistoryLoading(false) })
    return () => controller.abort()
  }, [selectedTransect])

  const reaches = overview?.reaches || []
  const selected = reaches.find(r => r.reach_id === selectedId) || null
  const period = PERIODS.find(p => p.key === periodKey) || PERIODS[0]
  const vegPeriods = overview?.vegetation_comparisons?.filter(r => Number(r.year_a) === period.a && Number(r.year_b) === period.b) || []
  const selectedVeg = getPeriodRow(reachDetails?.vegetation_pairs, period.a, period.b)
  const selectedWidth = getPeriodRow(reachDetails?.width_pairs, period.a, period.b)
  const classes = overview?.metadata?.priority_counts || {}
  const target = TARGETS.find(t => t.id === targetId) || TARGETS[0]
  const modelMetrics = overview?.models?.metrics || []
  const selectedMethod = overview?.models?.selection?.find(r => r.target === targetId)?.winner_selected_only_on_2024 || 'Not selected'
  const selectedMetrics = useMemo(() => {
    const d = modelMetrics.filter(r => r.target === targetId)
    return [...new Set(d.map(x => x.model))].map(model => ({
      model: model === 'RandomForest' ? 'Random Forest' : model === 'ExtraTrees' ? 'Extra Trees' : model === 'TrainingMean' ? 'Training Mean' : model,
      raw: model,
      validation: d.find(r => r.model === model && r.split === 'validation_2024')?.mae ?? null,
      test: d.find(r => r.model === model && r.split === 'test_2025')?.mae ?? null,
    })).sort((a,b) => (a.validation ?? 1e10)-(b.validation ?? 1e10))
  }, [modelMetrics, targetId])
  const forecastForTarget = reachDetails?.forecasts_2026?.filter(r => r.target === targetId) || []
  const forecastDistinct = forecastForTarget.filter(r => r.forecast_delta != null)
  const forecastValue = forecastDistinct.length ? forecastDistinct.reduce((sum, r) => sum + Number(r.forecast_delta), 0) / forecastDistinct.length : null
  const reportWidth = reachDetails?.width_annual || []
  const reportVegetation = reachDetails?.vegetation_annual || []
  const prioritySorted = [...reaches].filter(r => r.rcpi_eligible).sort((a, b) => (b.equal_weight_baseline ?? 0) - (a.equal_weight_baseline ?? 0))

  function chooseReach(id, focus = false) {
    if (!id) return
    setSelectedId(id)
    if (focus) setFocusToken(n => n + 1)
  }
  function changePage(id) { setPage(id); setSidebarOpen(false); window.scrollTo({ top: 0, behavior: 'smooth' }) }
  function openTransect(id, reach) {
    setSelectedTransect(id); if (reach) chooseReach(reach)
  }
  const mapProps = {
    geometry, centreline, selectedId, onSelect: id => chooseReach(id),
    basemap, setBasemap, waterCatalog, maskYear, setMaskYear, focusToken,
  }
  const navigationTitle = NAV.find(n => n.id === page)
  const graphTooltip = { contentStyle: { background: '#162637', border: '1px solid #31495f', borderRadius: 10, color: '#e5eff8' }, labelStyle: { color: '#e9f4fa' } }

  if (loading) return <main className="rcv-loading"><span className="rcv-loading-spinner"/><h2>Preparing Kelani River Intelligence</h2><p>Loading research evidence, reach geometry and model evaluations…</p></main>
  if (error) return <main className="rcv-loading"><AlertTriangle size={38}/><h2>Research data unavailable</h2><p>{error}</p><p>Start FastAPI on port 8000 and confirm Steps 27–32 outputs exist.</p><button className="rcv-primary-btn" onClick={() => setReloadKey(k => k+1)}><RefreshCw size={16}/> Retry</button></main>

  return <div className="rcv-app">
    {sidebarOpen && <button className="rcv-sidebar-backdrop" aria-label="Close navigation" onClick={() => setSidebarOpen(false)} />}
    <aside className={`rcv-sidebar ${sidebarOpen ? 'is-open' : ''}`}>
      <div className="rcv-brand"><div className="rcv-brand-icon"><Waves size={27}/></div><div><span>KELANI</span><strong>RIVER INTELLIGENCE</strong><small>J26-DS-338 · Research Console</small></div></div>
      <div className="rcv-side-label">RESEARCH WORKSPACE</div>
      <nav className="rcv-side-nav" aria-label="River Corridor sections">
        {NAV.map(({ id, label, icon: Icon, sub }) => <button key={id} onClick={() => changePage(id)} className={`rcv-side-link ${page === id ? 'active' : ''}`}><span className="rcv-side-link-icon"><Icon size={18}/></span><span><strong>{label}</strong><small>{sub}</small></span>{page === id && <ChevronRight className="rcv-side-chevron" size={16}/>}</button>)}
      </nav>
      <div className="rcv-side-bottom"><div className="rcv-side-pulse"><span className="rcv-pulse-dot"/> RESEARCH OUTPUTS CONNECTED</div><p>Satellite-derived evidence · Experimental models · Quality-screened planning intelligence</p><div className="rcv-version">PP1 DEMONSTRATION · v2.0</div></div>
    </aside>
    <div className="rcv-main">
      <header className="rcv-topbar">
        <div className="rcv-top-left"><button className="rcv-mobile-nav" aria-label="Open navigation" onClick={() => setSidebarOpen(true)}><Menu size={22}/></button><div className="rcv-breadcrumb"><span>Kelani Research</span><ChevronRight size={15}/><strong>{navigationTitle?.label}</strong></div></div>
        <div className="rcv-top-actions"><span className="rcv-connected"><span/> Live research API</span><span className="rcv-top-divider"/><span className="rcv-avatar">RC</span></div>
      </header>
      <div className="rcv-content">
        <div className="rcv-hero"><div><div className="rcv-overline"><span className="rcv-indicator"/> ENVIRONMENTAL INTELLIGENCE · KADUWELA STUDY CORRIDOR</div>
          <h1>{page === 'overview' ? 'Kelani River Corridor Intelligence' : navigationTitle?.label}</h1>
          <p>{page === 'overview' ? 'From satellite observations to explainable river-corridor planning evidence.' : navigationTitle?.sub}</p></div>
          <div className="rcv-hero-right"><span><Satellite size={15}/> Satellite + Geospatial</span><span>Experimental · Not survey-validated</span></div></div>
        <div className="rcv-reachbar"><div className="rcv-reachbar-left"><span className="rcv-reach-label"><MapPin size={16}/> FOCUS REACH</span><select aria-label="Select river reach" value={selectedId} onChange={e => chooseReach(e.target.value, true)}>{reaches.map(r => <option key={r.reach_id} value={r.reach_id}>{r.reach_id} · {CLASS_LABELS[r.priority_class]}</option>)}</select><button className="rcv-focus-btn" onClick={() => setFocusToken(t => t + 1)} title="Zoom map to selected reach"><Compass size={16}/> Focus map</button></div><div className="rcv-reachbar-right"><PriorityChip value={selected?.priority_class}/><span className="rcv-bar-caption">Preliminary 2020–2024 RCPI</span></div></div>

        {page === 'overview' && <>
          <div className="rcv-stats-grid">
            <StatCard icon={Waves} label="River corridor" number="13.54 km" detail="Fixed study centreline"/>
            <StatCard icon={MapIcon} label="Analysis reaches" number="28" detail="Approximately 500 m sections" color="blue"/>
            <StatCard icon={Activity} label="Screened RCPI reaches" number="18 / 28" detail="10 with insufficient evidence" color="amber"/>
            <StatCard icon={BrainCircuit} label="Forecasting targets" number="5" detail="2024 validation · 2025 test" color="purple"/>
          </div>
          <IconHeading kicker="01 / GEOSPATIAL COMMAND CENTER" title="A river-wide view of planning evidence" desc="Click an actual reach polygon on the satellite basemap. Switch pages to investigate the same location." icon={Layers}/>
          <div className="rcv-mapgrid"><ReachMap {...mapProps} mode="priority" tall/>
            <div className="rcv-panel rcv-insight"><div className="rcv-insight-top"><span>SELECTED REACH PROFILE</span><PriorityChip value={selected?.priority_class} compact/></div><h2>{selectedId}</h2><p>Evidence-based interpretation of the current selected corridor reach.</p>
              <Field name="RCPI score" value={uiNumber(selected?.equal_weight_baseline, 4)} note="Relative 0–1 composite; not a probability"/>
              <Field name="Vegetation net change" value={fmtSigned(selected?.net_vegetation_change_ha, 2, ' ha')}/>
              <Field name="Mapped building coverage" value={uiNumber(selected?.building_footprint_pct, 2, '%')}/>
              <Field name="Apparent median abs width change" value={uiNumber(selected?.median_abs_width_change_m, 2, ' m')}/>
              <Field name="Width match coverage" value={uiNumber(selected?.width_match_pct, 1, '%')}/>
              {selected?.rcpi_eligible ? <p className="rcv-insight-foot"><CheckCircle2 size={15}/> Eligible for exploratory composite scoring</p> : <p className="rcv-insight-foot is-warning"><AlertTriangle size={15}/> Unscored: {selected?.reason_not_scored?.replaceAll(';', ', ') || 'Insufficient evidence'}</p>}
              <button className="rcv-primary-btn" onClick={() => changePage('planning')}>Explore priority evidence <ArrowRight size={17}/></button>
            </div></div>
          <div className="rcv-double"><article className="rcv-panel rcv-method"><IconHeading kicker="RESEARCH PIPELINE" title="Three domains. One transparent decision framework." icon={ClipboardCheck}/><div className="rcv-pipeline"><div><Waves size={20}/><strong>River morphology</strong><small>Observed water & width change</small></div><ChevronRight size={17}/><div><Leaf size={20}/><strong>Vegetation</strong><small>NDVI threshold transitions</small></div><ChevronRight size={17}/><div><Building2 size={20}/><strong>Urban pressure</strong><small>OSM footprints & road density</small></div></div><div className="rcv-method-outcome"><ShieldCheck size={18}/> Preliminary 3-domain RCPI · Equal weighting · Quality gates</div></article>
            <article className="rcv-panel rcv-qa-summary"><IconHeading kicker="DATA TRANSPARENCY" title="The evidence is screened, not overstated" icon={Eye}/><div className="rcv-qa-columns"><div><strong>18</strong><span>RCPI-scored reaches</span></div><div><strong>10</strong><span>Insufficient-evidence reaches</span></div><div><strong>2026</strong><span>Exploratory baseline forecasts</span></div></div><DataNote>Water edges are not surveyed banks. NDVI changes are not confirmed deforestation. RCPI is not validated legal or disaster risk.</DataNote></article>
          </div>
        </>}

        {page === 'morphology' && <>
          <div className="rcv-stats-grid rcv-stats-3"><StatCard icon={Waves} label="Screened annual transects" number="1,293" detail="Across 2018–2025 · 271/year"/><StatCard icon={Activity} label={`${period.label} matched`} number={shortNum(selectedWidth?.matched)} detail={`Of ${selectedWidth?.total ?? '—'} transects in this reach`} color="blue"/><StatCard icon={TrendingDown} label="Apparent signed width difference" number={fmtSigned(selectedWidth?.median_signed_delta_m, 2, ' m')} detail="Satellite water-width proxy" color="amber"/></div>
          <IconHeading kicker="02 / MORPHOLOGICAL ANALYSIS" title="Historical water width & transects" desc="Satellite-derived 10 m observations. Compare fixed transects; inspect quality before interpreting differences." icon={Waves} trailing={<div className="rcv-inline-controls"><label>Period<select value={periodKey} onChange={e => setPeriodKey(e.target.value)}>{PERIODS.map(p => <option key={p.key} value={p.key}>{p.label}</option>)}</select></label></div>}/>
          <div className="rcv-mapgrid"><ReachMap {...mapProps} mode="width" transects={showTransects ? transects : null} selectedTransect={selectedTransect} onTransect={openTransect} tall/>
            <article className="rcv-panel rcv-insight"><span className="rcv-small-head">TRANSECT INTERPRETATION</span><h2>{selectedId}</h2><div className="rcv-toggle-row"><label><input type="checkbox" checked={showTransects} onChange={e => setShowTransects(e.target.checked)}/> Show transects on map</label></div>
              <Field name="Matched transects" value={`${selectedWidth?.matched ?? '—'} / ${selectedWidth?.total ?? '—'}`}/><Field name="Median signed difference" value={fmtSigned(selectedWidth?.median_signed_delta_m, 2, ' m')}/><Field name="Median absolute difference" value={uiNumber(selectedWidth?.median_abs_delta_m, 2, ' m')}/><Field name="Full 2020–24 RCPI QA" value={`${selected?.matched_transects ?? 0} matched · ${uiNumber(selected?.width_match_pct, 0, '%')}`}/>
              <DataNote level="warn">Detected water-width differences can reflect water stage, seasonal imagery and classification effects. They are not measured bank erosion.</DataNote></article></div>
          <div className="rcv-double"><Plot title={`${selectedId} · annual screened water width`} subtitle="Median of QA-screened transects; missing years remain gaps"><ResponsiveContainer width="100%" height="100%"><LineChart data={reportWidth}><CartesianGrid stroke="#e7edf1" strokeDasharray="3 3"/><XAxis dataKey="year" tick={{fontSize:11}}/><YAxis tick={{fontSize:11}} unit=" m" domain={['auto','auto']}/><Tooltip {...graphTooltip} formatter={v => [uiNumber(v,2,' m'),'Median width']}/><Line type="monotone" dataKey="median_width_m" stroke="#168da4" strokeWidth={3} connectNulls={false} dot={{r:4}}/></LineChart></ResponsiveContainer></Plot>
            <article className="rcv-panel">
              <div className="rcv-panel-head"><div><h3>Sentinel-2 transect explorer</h3><p>Observed water-width proxy · 2018–2025 · 10 m grid</p></div></div>
              <p className="rcv-muted">Click a fixed transect on the satellite map to inspect its QA-screened annual width measurements. Measurements are not surveyed riverbanks.</p>
              {selectedTransect ? <>
                <div className="rcv-inline-highlight">Selected · {selectedTransect}{transectHistory?.reach_id ? ` · ${transectHistory.reach_id}` : ''}</div>
                {transectHistoryLoading ? <EmptyState title="Loading annual observations" detail="Reading the Step 27 Sentinel-2 results..." />
                : transectHistoryError ? <DataNote level="warning">Transect history unavailable: {transectHistoryError}</DataNote>
                : transectHistory ? <>
                  <div className="rcv-field"><div><span>QA-screened years</span><small>Unscreened years remain missing, not zero.</small></div><strong>{transectHistory.qa_usable_years} / {transectHistory.total_years}</strong></div>
                  <div className="rcv-plot-area" style={{ height: 210 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={transectHistory.observations}>
                        <CartesianGrid stroke="#e7edf1" strokeDasharray="3 3" />
                        <XAxis dataKey="year" tick={{fontSize:11}} />
                        <YAxis unit=" m" domain={['auto', 'auto']} tick={{fontSize:11}} />
                        <Tooltip formatter={v => [uiNumber(v, 2, ' m'), 'Screened water width']} labelFormatter={y => `Observation ${y}`} />
                        <Line type="linear" dataKey="screened_width_m" stroke="#168da4" strokeWidth={2.5} connectNulls={false} dot={{r:4}} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                  <p className="rcv-muted">A gap means the transect did not pass the annual satellite screening. Water level, acquisition season and index disagreements may affect the measurements.</p>
                </> : <EmptyState title="No transect response" detail="Select a transect again or check the river API." />}
              </> : <EmptyState title="Select a transect" detail="Click a light-blue transect line to view the actual 2018–2025 Sentinel-2 measurements." />}
            </article>
          </div>
        </>}

        {page === 'vegetation' && <>
          <div className="rcv-stats-grid rcv-stats-3"><StatCard icon={Leaf} label="Apparent vegetation loss" number={uiNumber(selectedVeg?.vegetation_loss_ha, 2, ' ha')} detail={`${selectedId} · ${period.label}`}/><StatCard icon={Leaf} label="Apparent vegetation gain" number={uiNumber(selectedVeg?.vegetation_gain_ha, 2, ' ha')} detail="NDVI threshold transitions" color="blue"/><StatCard icon={Activity} label="Net vegetation-cover change" number={fmtSigned(selectedVeg?.net_vegetation_change_ha, 2, ' ha')} detail="Gain minus loss on matched land" color="amber"/></div>
          <IconHeading kicker="03 / RIPARIAN ENVIRONMENT" title="Where vegetation conditions are changing" desc="Satellite-derived NDVI/EVI analysis using a fixed 2020 river-edge reference and matched nonwater land." icon={Leaf} trailing={<div className="rcv-inline-controls"><label>Comparison<select value={periodKey} onChange={e=>setPeriodKey(e.target.value)}>{PERIODS.map(p=><option key={p.key} value={p.key}>{p.label}</option>)}</select></label></div>}/>
          <div className="rcv-mapgrid"><ReachMap {...mapProps} mode="vegetation" comparisonData={vegPeriods} tall/>
            <article className="rcv-panel rcv-insight"><span className="rcv-small-head">REACH VEGETATION DETAIL</span><h2>{selectedId}</h2><Field name="Matched land coverage" value={uiNumber(selectedVeg?.matched_land_coverage_pct, 1, '%')}/><Field name="Median NDVI difference" value={fmtSigned(selectedVeg?.median_ndvi_delta, 4)}/><Field name="Apparent loss" value={uiNumber(selectedVeg?.vegetation_loss_ha, 2, ' ha')}/><Field name="Apparent gain" value={uiNumber(selectedVeg?.vegetation_gain_ha, 2, ' ha')}/><Field name="Net cover difference" value={fmtSigned(selectedVeg?.net_vegetation_change_ha, 2, ' ha')}/><DataNote level="warn">NDVI ≥ 0.30 is an experimental vegetation proxy. Loss does not prove vegetation removal, and the 2020 reference is not a surveyed riverbank.</DataNote></article></div>
          <div className="rcv-double"><Plot title="Annual riparian NDVI" subtitle={`${selectedId} · 0–100 m reference zone`}><ResponsiveContainer width="100%" height="100%"><LineChart data={reportVegetation}><CartesianGrid stroke="#e7edf1" strokeDasharray="3 3"/><XAxis dataKey="year"/><YAxis domain={['auto','auto']}/><Tooltip {...graphTooltip}/><Line type="monotone" dataKey="median_ndvi" stroke="#21b28c" strokeWidth={3} dot={{r:4}} connectNulls={false} name="Median NDVI"/></LineChart></ResponsiveContainer></Plot>
            <Plot title="Vegetation change balance" subtitle="Matched nonwater land · hectares"><ResponsiveContainer width="100%" height="100%"><BarChart data={[{name:'Loss',value:Number(selectedVeg?.vegetation_loss_ha ?? 0)},{name:'Gain',value:Number(selectedVeg?.vegetation_gain_ha ?? 0)}]}><CartesianGrid stroke="#e7edf1" strokeDasharray="3 3"/><XAxis dataKey="name"/><YAxis/><Tooltip {...graphTooltip} formatter={v=>[uiNumber(v,2,' ha'),'Area']}/><Bar dataKey="value" radius={[5,5,0,0]}>{[0,1].map(i=><Cell key={i} fill={i===0?'#ef6671':'#23b98e'}/>)}</Bar></BarChart></ResponsiveContainer></Plot>
          </div>
        </>}

        {page === 'urban' && <>
          <div className="rcv-stats-grid rcv-stats-3"><StatCard icon={Building2} label="Building footprint coverage" number={uiNumber(selected?.building_footprint_pct, 2, '%')} detail={`${selectedId} · fixed 0–100 m zone`}/><StatCard icon={MapIcon} label="Road density" number={uiNumber(selected?.road_density_km_per_km2,2,' km/km²')} detail="Mapped road length per zone km²" color="blue"/><StatCard icon={Database} label="Intersecting buildings" number={shortNum(selected?.building_features_intersecting)} detail="OSM features, not illegal structures" color="amber"/></div>
          <IconHeading kicker="04 / HUMAN PRESSURE" title="Urban development close to the river" desc="Explore building footprints and mapped roads in the same spatial reference zones used for vegetation analysis." icon={Building2}/>
          <div className="rcv-mapgrid"><ReachMap {...mapProps} mode="urban" osm={osm} tall/><article className="rcv-panel rcv-insight"><span className="rcv-small-head">OPENSTREETMAP INSPECTION</span><h2>{selectedId}</h2><Field name="Mapped building coverage" value={uiNumber(selected?.building_footprint_pct,2,'%')}/><Field name="Road density" value={uiNumber(selected?.road_density_km_per_km2,2,' km/km²')}/><Field name="Mapped intersecting features" value={shortNum(selected?.building_features_intersecting)}/><Field name="Reference zone area" value={uiNumber(selected?.zone_area_ha, 2, ' ha')}/>{osm ? <p className="rcv-mini-success"><CheckCircle2 size={15}/> Nearby OSM geometries loaded</p> : <p className="rcv-muted">Loading selected-reach OSM geometry…</p>}<DataNote level="warn">The mapped features are a current OSM snapshot, not a legally verified encroachment inventory or measured urban expansion.</DataNote></article></div>
          <div className="rcv-double"><Plot title="Selected reach versus other reaches" subtitle="OSM building footprint coverage (%)"><ResponsiveContainer width="100%" height="100%"><BarChart data={[...reaches].sort((a,b)=>b.building_footprint_pct-a.building_footprint_pct).slice(0,12)} margin={{bottom:8}}><CartesianGrid stroke="#e7edf1" strokeDasharray="3 3"/><XAxis dataKey="reach_id" tick={{fontSize:10}}/><YAxis unit="%"/><Tooltip {...graphTooltip}/><Bar dataKey="building_footprint_pct" radius={[4,4,0,0]}>{[...reaches].sort((a,b)=>b.building_footprint_pct-a.building_footprint_pct).slice(0,12).map(r=><Cell key={r.reach_id} fill={r.reach_id===selectedId?'#ed5e65':'#339caf'}/>)}</Bar></BarChart></ResponsiveContainer></Plot>
            <article className="rcv-panel"><IconHeading kicker="INTERPRETATION" title="Mapped pressure ≠ confirmed encroachment" icon={Info}/><div className="rcv-text-block">Building cover and roads indicate nearby mapped development. The GIS buffers here are experimental distances outside the fixed 2020 detected water footprint, not official river reservations.<p>Combining these data with river-width and vegetation observations creates planning context. It does not establish causation, legal violations, or the date when buildings appeared.</p></div></article>
          </div>
        </>}

        {page === 'forecast' && <>
          <div className="rcv-stats-grid rcv-stats-3"><StatCard icon={BrainCircuit} label="Forecast targets" number="5" detail="Width · 2 edges · vegetation loss · NDVI"/><StatCard icon={ClipboardCheck} label="Model selection" number="2024" detail="Validation year · not 2025 test" color="blue"/><StatCard icon={Activity} label="Exploratory output" number="2026" detail="Next observation scenario · not measured" color="amber"/></div>
          <IconHeading kicker="05 / PREDICTIVE RESEARCH" title="Model comparison & exploratory forecasts" desc="Compare trained regression algorithms against transparent persistence and mean-change baselines." icon={BrainCircuit}/>
          <div className="rcv-panel rcv-target-controls"><div><label htmlFor="rcv-target">FORECAST TARGET</label><select id="rcv-target" value={targetId} onChange={e=>setTargetId(e.target.value)}>{TARGETS.map(t=><option key={t.id} value={t.id}>{t.label}</option>)}</select></div><div className="rcv-selection-status"><span>SELECTED ON 2024 VALIDATION</span><strong>{selectedMethod}</strong></div><div className="rcv-selection-status"><span>2025 TEST SAMPLE SIZE</span><strong>{modelMetrics.find(r=>r.target===targetId && r.split==='test_2025')?.n ?? '—'}</strong></div></div>
          <div className="rcv-double rcv-forecast-grid"><Plot title={`${target.label} · MAE by method`} subtitle={`Lower is better · ${target.unit}`} height={340}><ResponsiveContainer width="100%" height="100%"><BarChart data={selectedMetrics} margin={{top:6,right:5,bottom:40,left:0}}><CartesianGrid stroke="#e7edf1" strokeDasharray="3 3"/><XAxis dataKey="model" tick={{fontSize:10}} angle={-26} textAnchor="end" interval={0} height={65}/><YAxis tick={{fontSize:11}}/><Tooltip {...graphTooltip} formatter={v=>[uiNumber(v,4,` ${target.unit}`),'MAE']}/><Legend verticalAlign="top" height={30}/><Bar dataKey="validation" name="2024 validation" fill="#319ead" radius={[4,4,0,0]}/><Bar dataKey="test" name="2025 holdout" fill="#ec9760" radius={[4,4,0,0]}/></BarChart></ResponsiveContainer></Plot>
            <article className="rcv-panel rcv-insight"><span className="rcv-small-head">FORECAST FOR {selectedId} · 2026</span><h2>{forecastValue==null?'Not available':fmtSigned(forecastValue,targetId==='median_ndvi_delta'?5:2, ` ${target.unit==='m'?'m':target.unit==='NDVI units'?'NDVI': 'pp'}`)}</h2><p>{target.label} · mean of {forecastDistinct.length} available selected-location forecasts.</p><Field name="Selected methodology" value={selectedMethod}/><Field name="Eligible forecast locations" value={shortNum(forecastDistinct.length)}/><Field name="Future target QA known?" value="No"/><DataNote level="warn">A 0 m or 0 pp persistence projection means “no-change baseline,” not an AI claim of riverbank or vegetation stability. Water-edge positions are not surveyed banks.</DataNote></article></div>
          <article className="rcv-panel rcv-model-table"><div className="rcv-panel-head"><div><h3>Research model evaluation table</h3><p>Models were selected using 2024 validation. Test 2025 is a subsequent assessment, not an opportunity to reselect the winner.</p></div></div><div className="rcv-table-scroll"><table><thead><tr><th>Method</th><th>Validation MAE</th><th>Test MAE</th><th>Role</th></tr></thead><tbody>{selectedMetrics.map(row=><tr key={row.raw}><td><strong>{row.model}</strong></td><td>{uiNumber(row.validation,4)}</td><td>{uiNumber(row.test,4)}</td><td>{row.raw===selectedMethod?<span className="rcv-cell-winner">Selected on validation</span>:['Persistence','TrainingMean'].includes(row.raw)?'Baseline':'Trained ML'}</td></tr>)}</tbody></table></div></article>
          <DataNote>Some tree-based models performed better on the 2025 vegetation test, but were not selected on the 2024 validation split. Changing model choice after seeing 2025 outcomes would compromise that test's independence.</DataNote>
        </>}

        {page === 'planning' && <>
          <div className="rcv-stats-grid rcv-stats-3"><StatCard icon={ShieldCheck} label="Quality-screened" number={String(overview.metadata.scored_reaches)} detail="3-domain RCPI available"/><StatCard icon={AlertTriangle} label="Insufficient evidence" number={String(overview.metadata.unscored_reaches)} detail="Not wrongly classified as Low" color="amber"/><StatCard icon={MapIcon} label="Very High candidates" number={String(classes.VERY_HIGH || 0)} detail="Exploratory relative priority" color="purple"/></div>
          <IconHeading kicker="06 / PLANNING DECISION SUPPORT" title="Preliminary River Corridor Planning Priority" desc="Three-domain normalized index, Jenks grouping, quality screening and alternative-weight sensitivity." icon={ShieldCheck}/>
          <div className="rcv-mapgrid"><ReachMap {...mapProps} mode="priority" tall/><article className="rcv-panel rcv-insight"><span className="rcv-small-head">EXPLAINABLE PRIORITY</span><h2>{selectedId}</h2><PriorityChip value={selected?.priority_class}/><Field name="Equal-weight RCPI" value={uiNumber(selected?.equal_weight_baseline,4)}/><Field name="Morphology contribution" value={uiNumber(selected?.morphology_normalized,3)}/><Field name="Vegetation contribution" value={uiNumber(selected?.vegetation_normalized,3)}/><Field name="Urban-pressure contribution" value={uiNumber(selected?.urban_normalized,3)}/>{selected?.rcpi_eligible ? <p className="rcv-mini-success"><CheckCircle2 size={16}/> Sufficient for preliminary score</p> : <DataNote level="warn">{selected?.reason_not_scored?.replaceAll(';',' · ') || 'Insufficient evidence'}. No score has been imputed.</DataNote>}<DataNote>RCPI is a relative planning-screening index, not a probability of erosion, flood damage or illegal encroachment.</DataNote></article></div>
          <div className="rcv-double"><article className="rcv-panel"><div className="rcv-panel-head"><div><h3>Reach priority ranking</h3><p>Only 18 eligible reaches, ordered by three-domain equal-weight RCPI.</p></div></div><div className="rcv-rank-list">{prioritySorted.map((r,i)=><button className={`rcv-rank-row ${selectedId===r.reach_id?'selected':''}`} key={r.reach_id} onClick={()=>chooseReach(r.reach_id,true)}><span className="rcv-rank-number">{String(i+1).padStart(2,'0')}</span><strong>{r.reach_id}</strong><div className="rcv-rank-bar"><span style={{width:`${Math.max(2,Number(r.equal_weight_baseline||0)*100)}%`,background:CLASS_COLORS[r.priority_class]}}/></div><span className="rcv-rank-score">{uiNumber(r.equal_weight_baseline,3)}</span></button>)}</div></article>
            <article className="rcv-panel"><div className="rcv-panel-head"><div><h3>Sensitivity to indicator weighting</h3><p>Selected {selectedId}. Alternative weighting scenarios are experimental—not expert-approved weights.</p></div></div><div className="rcv-sensitivity">{[
              ['Equal-weight baseline','equal_weight_baseline'],['Morphology emphasis','morphology_emphasis'],['Vegetation emphasis','vegetation_emphasis'],['Urban emphasis','urban_emphasis'],
            ].map(([name,key])=><div key={key} className="rcv-sensitivity-row"><div><span>{name}</span><strong>{uiNumber(selected?.[key],3)}</strong></div><div className="rcv-sensitivity-track"><span style={{width:`${(selected?.[key] ?? 0)*100}%`}}/></div></div>)}</div><div className="rcv-method-note">The score and class boundaries are sensitive to assumptions. Under urban-emphasis weights, 8 of 18 eligible reach classes changed in the previous experimental audit.</div><DataNote level="warn">The 10 unscored reaches have missing/insufficient morphology evidence. They are intentionally grey rather than treated as safe or low priority.</DataNote></article></div>
        </>}

        <footer className="rcv-footer"><span><Waves size={17}/> KELANI RIVER INTELLIGENCE · J26-DS-338</span><span>Experimental Research</span><span>Satellite imagery © respective provider · OSM contributors</span></footer>
      </div>
    </div>
  </div>
}
