
// import { useEffect, useMemo, useState } from 'react'
// import {
//   ArrowUpRight,
//   Building2,
//   Grid3X3,
//   Leaf,
//   MapPinned,
//   RefreshCcw,
//   ShieldCheck,
// } from 'lucide-react'
// import { Link } from 'react-router-dom'
// import {
//   Bar,
//   BarChart,
//   CartesianGrid,
//   Legend,
//   Line,
//   LineChart,
//   ResponsiveContainer,
//   Tooltip,
//   XAxis,
//   YAxis,
// } from 'recharts'

// import PageHeader from '../../components/shared/PageHeader'
// import MetricCard from '../../components/shared/MetricCard'
// import GreenGridMap, {
//   GREEN_MAP_LAYERS,
//   validScore,
// } from '../../features/urbanGreen/components/GreenGridMap'
// import { urbanGreenApi } from '../../features/urbanGreen/services/urbanGreenApi'
// import { useUrbanGreen } from '../../features/urbanGreen/context/UrbanGreenContext'

// // Pressure categories from the real research backend.
// const pressureLabels = {
//   LOW_RELATIVE: 'Low relative',
//   MODERATE_RELATIVE: 'Moderate',
//   ELEVATED_RELATIVE: 'Elevated',
//   HIGH_RELATIVE: 'High relative',
//   THRESHOLD_FLAGGED: 'Threshold flagged',
// }

// // Formatting helpers.
// // Null and unavailable values must never be treated as zero.

// function formatNumber(value, digits = 2) {
//   return validScore(value)
//     ? value.toLocaleString('en-US', {
//         minimumFractionDigits: digits,
//         maximumFractionDigits: digits,
//       })
//     : 'Unavailable'
// }

// function formatCount(value) {
//   return validScore(value)
//     ? value.toLocaleString('en-US')
//     : 'Unavailable'
// }

// function formatScore(value) {
//   return formatNumber(value, 3)
// }

// function formatChange(value) {
//   return validScore(value)
//     ? `${value > 0 ? '+' : ''}${formatNumber(value)} pp`
//     : 'Unavailable'
// }

// function readable(value) {
//   return typeof value === 'string' && value.trim()
//     ? value.replace(/_/g, ' ')
//     : 'Unavailable'
// }

// // Shared dashboard detail row.
// function DetailRow({ label, value }) {
//   return (
//     <div className="flex items-start justify-between gap-4 border-b border-slate-100 py-3 last:border-0">
//       <span className="text-xs text-slate-500">
//         {label}
//       </span>

//       <span className="max-w-[60%] break-words text-right text-xs font-semibold text-slate-800">
//         {value ?? 'Unavailable'}
//       </span>
//     </div>
//   )
// }

// // Shared section heading.
// function SectionHeading({ title, description }) {
//   return (
//     <div className="mb-4">
//       <h2 className="section-title">
//         {title}
//       </h2>

//       <p className="mt-1 text-xs leading-5 text-slate-500">
//         {description}
//       </p>
//     </div>
//   )
// }

// // Links to detailed Urban Green research pages.
// const pageLinks = [
//   ['/green/green-cover', 'Historical Green Cover'],
//   ['/green/pressure', 'Urbanization Pressure'],
//   ['/green/suitability', 'Development Suitability'],
//   ['/green/planning-conflict', 'Planning Intelligence'],
// ]

// export default function Dashboard() {
//   // Shared selection state across Urban Green pages.
//   const {
//     selectedGridId,
//     setSelectedGridId,
//     selectedGnId,
//     selectGn,
//   } = useUrbanGreen()

//   // Main dashboard state.
//   const [data, setData] = useState(null)
//   const [loading, setLoading] = useState(true)
//   const [error, setError] = useState('')
//   const [refreshIndex, setRefreshIndex] = useState(0)

//   // Interactive GIS map state.
//   const [mapMetric, setMapMetric] = useState('pressure')

//   // Selected grid details.
//   const [detail, setDetail] = useState(null)
//   const [detailLoading, setDetailLoading] = useState(false)

//   // -------------------------------------------------------
//   // 1. LOAD REAL DASHBOARD DATA FROM FASTAPI
//   // -------------------------------------------------------

//   useEffect(() => {
//     const controller = new AbortController()

//     setLoading(true)
//     setError('')

//     Promise.all([
//       urbanGreenApi.historical.summary(controller.signal),
//       urbanGreenApi.historical.trajectory(controller.signal),
//       urbanGreenApi.pressure.overview(controller.signal),
//       urbanGreenApi.suitability.overview(controller.signal),
//       urbanGreenApi.conservation.overview(controller.signal),
//       urbanGreenApi.planning.map(controller.signal),
//       urbanGreenApi.planning.gnMap(controller.signal),
//       urbanGreenApi.planning.gnDivisions(controller.signal),
//       urbanGreenApi.planning.hotspots({}, controller.signal),
//     ])
//       .then((results) => {
//         if (controller.signal.aborted) return

//         const [
//           historical,
//           trajectory,
//           pressure,
//           suitability,
//           conservation,
//           planningMap,
//           gnMap,
//           gnDivisions,
//           hotspots,
//         ] = results

//         setData({
//           historical,
//           trajectory,
//           pressure,
//           suitability,
//           conservation,
//           planningMap,
//           gnMap,
//           gnDivisions,
//           hotspots,
//         })

//         setLoading(false)
//       })
//       .catch((reason) => {
//         if (controller.signal.aborted) return

//         setError(
//           reason?.message ||
//           'Failed to retrieve dashboard data.'
//         )

//         setLoading(false)
//       })

//     return () => controller.abort()
//   }, [refreshIndex])

//   // -------------------------------------------------------
//   // 2. LOAD SELECTED GRID DETAILS
//   // -------------------------------------------------------

//   useEffect(() => {
//     const controller = new AbortController()

//     setDetail(null)
//     setDetailLoading(Boolean(selectedGridId && data))

//     if (selectedGridId && data) {
//       urbanGreenApi.planning
//         .grid(selectedGridId, controller.signal)
//         .then((result) => {
//           if (!controller.signal.aborted) {
//             setDetail(result)
//           }
//         })
//         .catch(() => {
//           if (!controller.signal.aborted) {
//             setDetail(null)
//           }
//         })
//         .finally(() => {
//           if (!controller.signal.aborted) {
//             setDetailLoading(false)
//           }
//         })
//     }

//     return () => controller.abort()
//   }, [selectedGridId, data])

//   // -------------------------------------------------------
//   // 3. PREPARE HISTORICAL TREND CHART
//   // -------------------------------------------------------

//   const trend = useMemo(
//     () =>
//       (data?.trajectory?.data || [])
//         .filter(
//           (row) => row.included_in_core_trend === true
//         )
//         .map((row) => ({
//           year: row.year,
//           green: row.green_percent,
//           built: row.built_percent,
//         }))
//         .sort((a, b) => a.year - b.year),
//     [data]
//   )

//   // -------------------------------------------------------
//   // 4. PREPARE PRESSURE DISTRIBUTION CHART
//   // -------------------------------------------------------

//   const pressureChart = useMemo(
//     () =>
//       Object.entries(pressureLabels).map(
//         ([key, name]) => ({
//           name,
//           count:
//             data?.pressure?.category_counts?.[key] ?? 0,
//         })
//       ),
//     [data]
//   )

//   // -------------------------------------------------------
//   // 5. FIND SELECTED GRID FROM REAL GIS DATA
//   // -------------------------------------------------------

//   const selectedFeature = useMemo(() => {
//     if (!selectedGridId) return null

//     return (
//       data?.planningMap?.features?.find(
//         (feature) =>
//           feature.properties?.grid_id === selectedGridId
//       ) || null
//     )
//   }, [data, selectedGridId])

//   // -------------------------------------------------------
//   // 6. FIND SELECTED GN DIVISION
//   // -------------------------------------------------------

//   const selectedGn = useMemo(
//     () =>
//       (data?.gnDivisions?.data || []).find(
//         (record) => record.gn_id === selectedGnId
//       ) || null,
//     [data, selectedGnId]
//   )

//   // -------------------------------------------------------
//   // 7. MERGE MAP PROPERTIES WITH GRID API DETAILS
//   // -------------------------------------------------------

//   const grid = useMemo(() => {
//     const base = selectedFeature?.properties || {}

//     const record =
//       detail?.data && !Array.isArray(detail.data)
//         ? detail.data
//         : detail?.grid &&
//             typeof detail.grid === 'object'
//           ? detail.grid
//           : detail || {}

//     return {
//       ...base,
//       ...record,
//     }
//   }, [selectedFeature, detail])

//   // -------------------------------------------------------
//   // 8. LOADING STATE
//   // -------------------------------------------------------

//   if (loading) {
//     return (
//       <div className="card-pad">
//         <h2 className="section-title">
//           Loading Urban Green Dashboard
//         </h2>

//         <p className="mt-3 text-sm text-slate-500">
//           Reading historical trends, model summaries,
//           GN divisions and the real GIS grid map...
//         </p>
//       </div>
//     )
//   }

//   // -------------------------------------------------------
//   // 9. ERROR STATE
//   // -------------------------------------------------------

//   if (error || !data) {
//     return (
//       <div className="card-pad">
//         <h2 className="section-title">
//           Dashboard data unavailable
//         </h2>

//         <p className="mt-3 text-sm text-red-700">
//           {error || 'No backend response.'}
//         </p>

//         <p className="mt-2 text-xs text-slate-500">
//           Verify that FastAPI is running at the
//           configured API address.
//         </p>

//         <button
//           className="btn-primary mt-4"
//           onClick={() =>
//             setRefreshIndex((n) => n + 1)
//           }
//         >
//           <RefreshCcw size={15} />
//           Retry
//         </button>
//       </div>
//     )
//   }

//   // -------------------------------------------------------
//   // 10. EXTRACT VERIFIED BACKEND RESULTS
//   // -------------------------------------------------------

//   const {
//     historical,
//     pressure,
//     suitability,
//     conservation,
//     planningMap,
//     gnMap,
//     gnDivisions,
//     hotspots,
//   } = data

//   return (
//     <div>
//       {/* ===============================================
//           DASHBOARD HEADER
//       ================================================ */}

//       <PageHeader
//         eyebrow="Urban Green Intelligence"
//         title="Kaduwela Urban Green Intelligence Dashboard"
//         subtitle="Real historical green-cover trends, relative urbanization pressure, expert-AHP development suitability, conservation and planning evidence."
//         badges={[
//           `Historical core: ${historical.core_period}`,
//           `Future assessment: ${pressure.pressure_period} (combined window)`,
//           '100 m GIS research grids',
//         ]}
//         actions={
//           <button
//             className="btn-secondary"
//             onClick={() =>
//               setRefreshIndex((n) => n + 1)
//             }
//           >
//             <RefreshCcw size={15} />
//             Refresh
//           </button>
//         }
//       />

//       {/* ===============================================
//           RESEARCH KPI CARDS
//       ================================================ */}

//       <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
//         <MetricCard
//           title="Green-cover net change"
//           value={formatChange(
//             historical.green_net_change_percent_points
//           )}
//           subtitle={`${formatNumber(
//             historical.green_start_percent
//           )}% to ${formatNumber(
//             historical.green_end_percent
//           )}% | ${historical.core_period}`}
//           icon={Leaf}
//           tone="green"
//         />

//         <MetricCard
//           title="Built-up net change"
//           value={formatChange(
//             historical.built_net_change_percent_points
//           )}
//           subtitle={`${formatNumber(
//             historical.built_start_percent
//           )}% to ${formatNumber(
//             historical.built_end_percent
//           )}% | ${historical.core_period}`}
//           icon={Building2}
//           tone="orange"
//         />

//         <MetricCard
//           title="Pressure-scored grids"
//           value={formatCount(
//             pressure.eligible_scored_cells
//           )}
//           subtitle={`${formatCount(
//             pressure.outside_inference_domain_cells
//           )} cells outside inference domain`}
//           icon={Grid3X3}
//           tone="teal"
//         />

//         <MetricCard
//           title="Conservation Priority coverage"
//           value={formatCount(
//             conservation.scored_grid_count
//           )}
//           subtitle={`${formatCount(
//             conservation.primary_priority_unscored_count
//           )} cells with unavailable priority scores`}
//           icon={ShieldCheck}
//           tone="blue"
//         />
//       </div>

//       {/* ===============================================
//           HISTORICAL TREND AND PRESSURE CHARTS
//       ================================================ */}

//       <div className="mt-5 grid gap-5 xl:grid-cols-2">
//         {/* Historical chart */}

//         <section className="card-pad min-w-0">
//           <SectionHeading
//             title="Historical land-cover trends"
//             description="Green and built-up shares of valid classified pixels. Only the 2011-2024 core trend is shown."
//           />

//           <ResponsiveContainer
//             width="100%"
//             height={275}
//           >
//             <LineChart data={trend}>
//               <CartesianGrid strokeDasharray="4 5" />

//               <XAxis
//                 dataKey="year"
//                 tick={{ fontSize: 11 }}
//               />

//               <YAxis
//                 domain={[0, 100]}
//                 tick={{ fontSize: 11 }}
//                 tickFormatter={(n) => `${n}%`}
//               />

//               <Tooltip
//                 formatter={(value, name) => [
//                   `${formatNumber(value)}%`,
//                   name,
//                 ]}
//               />

//               <Legend />

//               <Line
//                 name="Green cover"
//                 dataKey="green"
//                 stroke="#26885c"
//                 strokeWidth={3}
//                 dot={false}
//                 isAnimationActive={false}
//               />

//               <Line
//                 name="Built-up"
//                 dataKey="built"
//                 stroke="#e47f54"
//                 strokeWidth={3}
//                 dot={false}
//                 isAnimationActive={false}
//               />
//             </LineChart>
//           </ResponsiveContainer>

//           <p className="mt-2 text-[11px] leading-5 text-slate-500">
//             Net change is measured in percentage points
//             (pp), not relative percent growth.
//           </p>
//         </section>

//         {/* Pressure distribution chart */}

//         <section className="card-pad min-w-0">
//           <SectionHeading
//             title="Future pressure distribution"
//             description={`Frozen relative pressure categories for the single ${pressure.pressure_period} assessment window.`}
//           />

//           <ResponsiveContainer
//             width="100%"
//             height={275}
//           >
//             <BarChart
//               data={pressureChart}
//               layout="vertical"
//               margin={{
//                 left: 14,
//                 right: 18,
//               }}
//             >
//               <CartesianGrid strokeDasharray="4 5" />

//               <XAxis
//                 type="number"
//                 allowDecimals={false}
//                 tick={{ fontSize: 11 }}
//               />

//               <YAxis
//                 type="category"
//                 dataKey="name"
//                 width={112}
//                 tick={{ fontSize: 10 }}
//               />

//               <Tooltip
//                 formatter={(value) => [
//                   formatCount(value),
//                   'Grid cells',
//                 ]}
//               />

//               <Bar
//                 dataKey="count"
//                 fill="#208a76"
//                 radius={[0, 6, 6, 0]}
//                 isAnimationActive={false}
//               />
//             </BarChart>
//           </ResponsiveContainer>

//           <p className="mt-2 text-[11px] leading-5 text-slate-500">
//             {formatCount(
//               pressure.threshold_flagged_cells
//             )} grids are threshold-flagged.
//             Pressure is not a calibrated probability;
//             out-of-domain does not mean low risk.
//           </p>
//         </section>
//       </div>

//       {/* ===============================================
//           INTEGRATED RESEARCH GIS MAP
//       ================================================ */}

//       <div className="mt-5 grid gap-5 2xl:grid-cols-[1.65fr_.85fr]">
//         <section className="card-pad relative isolate z-0 min-w-0">
//           <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
//             <SectionHeading
//               title="Integrated GIS research map"
//               description="Explore real Kaduwela planning grid polygons. Select a layer, choose a GN division, or click a grid."
//             />

//             <span className="pill">
//               {formatCount(
//                 planningMap?.features?.length
//               )} mapped cells
//             </span>
//           </div>

//           {/* Map layer and GN selection */}

//           <div className="mb-4 grid gap-3 md:grid-cols-2">
//             <label className="text-xs font-semibold text-slate-600">
//               Research map layer

//               <select
//                 className="control mt-1"
//                 value={mapMetric}
//                 onChange={(e) =>
//                   setMapMetric(e.target.value)
//                 }
//               >
//                 {Object.entries(GREEN_MAP_LAYERS).map(
//                   ([key, layer]) => (
//                     <option
//                       key={key}
//                       value={key}
//                     >
//                       {layer.label}
//                     </option>
//                   )
//                 )}
//               </select>
//             </label>

//             <label className="text-xs font-semibold text-slate-600">
//               GN division

//               <select
//                 className="control mt-1"
//                 value={selectedGnId || ''}
//                 onChange={(e) =>
//                   selectGn(e.target.value)
//                 }
//               >
//                 <option value="">
//                   All GN divisions
//                 </option>

//                 {(gnDivisions?.data || []).map(
//                   (gn) => (
//                     <option
//                       key={gn.gn_id}
//                       value={gn.gn_id}
//                     >
//                       {gn.gn_name}
//                     </option>
//                   )
//                 )}
//               </select>
//             </label>
//           </div>

//           {/* Actual GeoJSON map */}

//           <div className="relative isolate z-0 overflow-hidden rounded-2xl">
//             <GreenGridMap
//               geojson={planningMap}
//               gnGeojson={gnMap}
//               metric={mapMetric}
//               selectedGridId={selectedGridId}
//               selectedGnId={selectedGnId}
//               onSelect={setSelectedGridId}
//             />
//           </div>

//           {/* GIS guidance */}

//           <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
//             <p className="max-w-2xl text-[11px] leading-5 text-slate-500">
//               GN outlines follow official GN polygons.
//               Grid names use the dominant GN intersection,
//               while GN summaries use area-weighted
//               intersections.
//             </p>

//             {(selectedGnId || selectedGridId) && (
//               <button
//                 type="button"
//                 className="btn-secondary text-xs"
//                 onClick={() => selectGn(null)}
//               >
//                 Clear selections
//               </button>
//             )}
//           </div>
//         </section>

//         {/* ===========================================
//             RIGHT-SIDE INFORMATION PANELS
//         ============================================ */}

//         <div className="space-y-5 min-w-0">
//           {/* Selected grid information */}

//           <section className="card-pad">
//             <div className="flex items-center gap-2">
//               <MapPinned
//                 size={18}
//                 className="text-emerald-700"
//               />

//               <h2 className="section-title">
//                 Selected grid evidence
//               </h2>
//             </div>

//             {!selectedFeature ? (
//               <p className="mt-4 text-sm leading-6 text-slate-500">
//                 Click a grid on the map to inspect
//                 its GN assignment and research scores.
//               </p>
//             ) : (
//               <div className="mt-3">
//                 <DetailRow
//                   label="Grid ID"
//                   value={
//                     grid.grid_id || selectedGridId
//                   }
//                 />

//                 <DetailRow
//                   label="GN division"
//                   value={
//                     grid.gn_name ||
//                     'Unassigned / unavailable'
//                   }
//                 />

//                 <DetailRow
//                   label="Pressure score (relative)"
//                   value={formatScore(
//                     grid.urbanization_pressure_score
//                   )}
//                 />

//                 <DetailRow
//                   label="Expert-AHP suitability"
//                   value={formatScore(
//                     grid.development_suitability_score
//                   )}
//                 />

//                 <DetailRow
//                   label="Conservation Priority"
//                   value={formatScore(
//                     grid.frozen_conservation_priority_0_1
//                   )}
//                 />

//                 <DetailRow
//                   label="Development Conflict Index"
//                   value={formatScore(
//                     grid.development_conflict_index_0_1
//                   )}
//                 />

//                 <DetailRow
//                   label="Review stream"
//                   value={readable(
//                     grid.review_stream
//                   )}
//                 />

//                 <DetailRow
//                   label="Review rule"
//                   value={readable(
//                     grid.review_rule_id
//                   )}
//                 />

//                 {detailLoading && (
//                   <p className="mt-3 text-xs text-slate-500">
//                     Loading additional review details...
//                   </p>
//                 )}

//                 {grid.review_action && (
//                   <div className="mt-4 rounded-xl bg-emerald-50 p-3">
//                     <p className="text-xs font-bold text-emerald-800">
//                       Suggested planning review
//                     </p>

//                     <p className="mt-1 text-xs leading-5 text-emerald-900">
//                       {grid.review_action}
//                     </p>
//                   </div>
//                 )}
//               </div>
//             )}
//           </section>

//           {/* Selected GN division information */}

//           {selectedGn && (
//             <section className="card-pad">
//               <h2 className="section-title">
//                 GN division summary
//               </h2>

//               <p className="mt-1 text-sm font-semibold text-emerald-800">
//                 {selectedGn.gn_name}
//               </p>

//               <div className="mt-3">
//                 <DetailRow
//                   label="Area inside KMC"
//                   value={`${formatNumber(
//                     selectedGn.gn_area_inside_kmc_km2
//                   )} km2`}
//                 />

//                 <DetailRow
//                   label="Area-weighted suitability mean"
//                   value={formatScore(
//                     selectedGn.suitability_area_weighted_mean
//                   )}
//                 />

//                 <DetailRow
//                   label="Area-weighted pressure mean"
//                   value={formatScore(
//                     selectedGn.pressure_area_weighted_mean
//                   )}
//                 />

//                 <DetailRow
//                   label="Area-weighted conservation mean"
//                   value={formatScore(
//                     selectedGn.conservation_area_weighted_mean
//                   )}
//                 />
//               </div>

//               <p className="mt-3 text-[11px] text-slate-500">
//                 Unavailable scores are excluded from
//                 weighted averages.
//               </p>
//             </section>
//           )}

//           {/* Research coverage summary */}

//           <section className="card-pad">
//             <h2 className="section-title">
//               Research coverage
//             </h2>

//             <div className="mt-3">
//               <DetailRow
//                 label="Total planning grids"
//                 value={formatCount(
//                   suitability.grid_count
//                 )}
//               />

//               <DetailRow
//                 label="GN divisions"
//                 value={formatCount(
//                   gnDivisions.total
//                 )}
//               />

//               <DetailRow
//                 label="Expert-AHP suitability mean"
//                 value={formatScore(
//                   suitability.distribution?.mean
//                 )}
//               />

//               <DetailRow
//                 label="Exploratory hotspot candidates"
//                 value={formatCount(
//                   hotspots.total
//                 )}
//               />

//               <DetailRow
//                 label="Conservation scores unavailable"
//                 value={formatCount(
//                   conservation.primary_priority_unscored_count
//                 )}
//               />
//             </div>
//           </section>
//         </div>
//       </div>

//       {/* ===============================================
//           PAGE NAVIGATION CARDS
//       ================================================ */}

//       <section className="card-pad mt-5">
//         <h2 className="section-title">
//           Explore research outputs
//         </h2>

//         <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
//           {pageLinks.map(([path, title]) => (
//             <Link
//               key={path}
//               to={path}
//               className="flex items-center justify-between rounded-xl border border-emerald-100 bg-emerald-50 px-4 py-4 text-sm font-semibold text-emerald-900 transition hover:bg-emerald-100"
//             >
//               <span>{title}</span>

//               <ArrowUpRight
//                 size={16}
//                 className="shrink-0"
//               />
//             </Link>
//           ))}
//         </div>
//       </section>

//       {/* ===============================================
//           SCIENTIFIC RESEARCH LIMITATIONS
//       ================================================ */}

//       <div className="mt-5 rounded-xl border border-amber-200 bg-amber-50 p-4 text-xs leading-6 text-amber-900">
//         <strong>Research interpretation:</strong>
//         {' '}
//         Urbanization Pressure assesses one combined
//         2026-2028 future period, not separate yearly
//         forecasts. Suitability uses three AHP criteria
//         and does not grant development approval.
//         Missing conservation scores remain unavailable.
//         Candidate hotspots are exploratory spatial groups,
//         not legally designated areas or statistically
//         significant hotspots.
//       </div>
//     </div>
//   )
// }


//2nd code

import { useEffect, useMemo, useState } from 'react'
import {
  ArrowUpRight,
  Building2,
  Grid3X3,
  Leaf,
  MapPinned,
  RefreshCcw,
  ShieldCheck,
} from 'lucide-react'

import { Link } from 'react-router-dom'

import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import PageHeader from '../../components/shared/PageHeader'
import MetricCard from '../../components/shared/MetricCard'

import GreenGridMap, {
  GREEN_MAP_LAYERS,
  validScore,
  isOutsidePressureDomain,
  pressureDisplay,
  researchScoreDisplay,
} from '../../features/urbanGreen/components/GreenGridMap'

import { urbanGreenApi } from '../../features/urbanGreen/services/urbanGreenApi'

import { useUrbanGreen } from '../../features/urbanGreen/context/UrbanGreenContext'

// =====================================================
// 1. FORMATTING HELPERS
// =====================================================

function formatNumber(value, digits = 2) {
  return validScore(value)
    ? value.toLocaleString('en-US', {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
      })
    : 'Unavailable'
}

function formatCount(value) {
  return validScore(value)
    ? value.toLocaleString('en-US', {
        maximumFractionDigits: 0,
      })
    : 'Unavailable'
}

function formatScore(value) {
  return formatNumber(value, 3)
}

function formatChange(value) {
  return validScore(value)
    ? `${value > 0 ? '+' : ''}${formatNumber(value)} pp`
    : 'Unavailable'
}

function readable(value) {
  return typeof value === 'string' && value.trim()
    ? value.replace(/_/g, ' ')
    : 'Unavailable'
}

function DetailRow({ label, value }) {
  return (
    <div className="flex items-start justify-between gap-4 border-b border-slate-100 py-3 last:border-0">
      <span className="text-xs text-slate-500">
        {label}
      </span>

      <span className="max-w-[62%] break-words text-right text-xs font-semibold text-slate-800">
        {value ?? 'Unavailable'}
      </span>
    </div>
  )
}

function SectionHeading({ title, description }) {
  return (
    <div className="mb-4">
      <h2 className="section-title">{title}</h2>
      <p className="mt-1 text-xs leading-5 text-slate-500">
        {description}
      </p>
    </div>
  )
}

const pressureLabels = {
  LOW_RELATIVE: 'Low relative',
  MODERATE_RELATIVE: 'Moderate',
  ELEVATED_RELATIVE: 'Elevated',
  HIGH_RELATIVE: 'High relative',
  THRESHOLD_FLAGGED: 'Threshold flagged',
}

const pageLinks = [
  ['/green/green-cover', 'Historical Green Cover'],
  ['/green/pressure', 'Urbanization Pressure'],
  ['/green/suitability', 'Development Suitability'],
  ['/green/planning-conflict', 'Planning Intelligence'],
]

// =====================================================
// 2. MAIN DASHBOARD
// =====================================================

export default function Dashboard() {
  const {
    selectedGridId,
    setSelectedGridId,
    selectedGnId,
    selectGn,
  } = useUrbanGreen()

  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [refreshIndex, setRefreshIndex] = useState(0)

  const [mapMetric, setMapMetric] = useState('pressure')

  const [gridDetail, setGridDetail] = useState(null)
  const [detailLoading, setDetailLoading] = useState(false)

  // ===================================================
  // 3. LOAD REAL BACKEND DATA
  // ===================================================

  useEffect(() => {
    const controller = new AbortController()

    setLoading(true)
    setError('')

    Promise.all([
      urbanGreenApi.historical.summary(controller.signal),
      urbanGreenApi.historical.trajectory(controller.signal),
      urbanGreenApi.pressure.overview(controller.signal),
      urbanGreenApi.suitability.overview(controller.signal),
      urbanGreenApi.conservation.overview(controller.signal),
      urbanGreenApi.planning.map(controller.signal),
      urbanGreenApi.planning.gnMap(controller.signal),
      urbanGreenApi.planning.gnDivisions(controller.signal),
      urbanGreenApi.planning.hotspots({}, controller.signal),
    ])
      .then((results) => {
        if (controller.signal.aborted) return

        const [
          historical,
          trajectory,
          pressure,
          suitability,
          conservation,
          planningMap,
          gnMap,
          gnDivisions,
          hotspots,
        ] = results

        setData({
          historical,
          trajectory,
          pressure,
          suitability,
          conservation,
          planningMap,
          gnMap,
          gnDivisions,
          hotspots,
        })

        setLoading(false)
      })
      .catch((reason) => {
        if (controller.signal.aborted) return

        setError(
          reason?.message ||
          'Failed to load Urban Green research data.'
        )

        setLoading(false)
      })

    return () => controller.abort()
  }, [refreshIndex])

  // ===================================================
  // 4. LOAD SELECTED GRID REVIEW DETAILS
  // ===================================================

  useEffect(() => {
    const controller = new AbortController()

    setGridDetail(null)
    setDetailLoading(Boolean(selectedGridId && data))

    if (selectedGridId && data) {
      urbanGreenApi.planning
        .grid(selectedGridId, controller.signal)
        .then((result) => {
          if (!controller.signal.aborted) {
            setGridDetail(result)
          }
        })
        .catch(() => {
          if (!controller.signal.aborted) {
            setGridDetail(null)
          }
        })
        .finally(() => {
          if (!controller.signal.aborted) {
            setDetailLoading(false)
          }
        })
    }

    return () => controller.abort()
  }, [selectedGridId, data])

  // ===================================================
  // 5. HISTORICAL TREND DATA
  // ===================================================

  const trend = useMemo(
    () =>
      (data?.trajectory?.data || [])
        .filter(
          (row) => row.included_in_core_trend === true
        )
        .map((row) => ({
          year: row.year,
          green: row.green_percent,
          built: row.built_percent,
        }))
        .sort((a, b) => a.year - b.year),
    [data]
  )

  // ===================================================
  // 6. PRESSURE CATEGORY DISTRIBUTION
  // ===================================================

  const pressureChart = useMemo(
    () =>
      Object.entries(pressureLabels).map(
        ([key, name]) => ({
          name,
          count:
            data?.pressure?.category_counts?.[key] ?? 0,
        })
      ),
    [data]
  )

  // ===================================================
  // 7. SELECTED REAL GIS GRID
  // ===================================================

  const selectedFeature = useMemo(() => {
    if (!selectedGridId) return null

    return (
      data?.planningMap?.features?.find(
        (feature) =>
          feature.properties?.grid_id === selectedGridId
      ) || null
    )
  }, [data, selectedGridId])

  const selectedGn = useMemo(
    () =>
      (data?.gnDivisions?.data || []).find(
        (row) => row.gn_id === selectedGnId
      ) || null,
    [data, selectedGnId]
  )

  const grid = useMemo(() => {
    const base = selectedFeature?.properties || {}

    const detailRecord =
      gridDetail?.data &&
      !Array.isArray(gridDetail.data)
        ? gridDetail.data
        : gridDetail?.grid &&
            typeof gridDetail.grid === 'object'
          ? gridDetail.grid
          : gridDetail || {}

    return {
      ...base,
      ...detailRecord,
    }
  }, [selectedFeature, gridDetail])

  // ===================================================
  // 8. LOADING AND ERROR STATES
  // ===================================================

  if (loading) {
    return (
      <div className="card-pad">
        <h2 className="section-title">
          Loading Urban Green Dashboard
        </h2>

        <p className="mt-3 text-sm text-slate-500">
          Retrieving historical trends, model results,
          GN divisions and GIS research polygons...
        </p>
      </div>
    )
  }

  if (error || !data) {
    return (
      <div className="card-pad">
        <h2 className="section-title">
          Dashboard data unavailable
        </h2>

        <p className="mt-3 text-sm text-red-700">
          {error || 'No backend response.'}
        </p>

        <button
          className="btn-primary mt-4"
          onClick={() =>
            setRefreshIndex((n) => n + 1)
          }
        >
          <RefreshCcw size={15} />
          Retry
        </button>
      </div>
    )
  }

  const {
    historical,
    pressure,
    suitability,
    conservation,
    planningMap,
    gnMap,
    gnDivisions,
    hotspots,
  } = data

  const gridOutsideDomain =
    selectedFeature &&
    isOutsidePressureDomain(grid) &&
    !validScore(grid.urbanization_pressure_score)

  return (
    <div>
      {/* ==============================================
          PAGE HEADER
      =============================================== */}

      <PageHeader
        eyebrow="Urban Green Intelligence"
        title="Kaduwela Urban Green Intelligence Dashboard"
        subtitle="Historical green-cover change, relative urbanization pressure, expert-AHP suitability, conservation and planning intelligence."
        badges={[
          `Historical core: ${historical.core_period}`,
          `Future assessment: ${pressure.pressure_period} (combined)`,
          'Original 100 m GIS grids',
        ]}
        actions={
          <button
            className="btn-secondary"
            onClick={() =>
              setRefreshIndex((n) => n + 1)
            }
          >
            <RefreshCcw size={15} />
            Refresh
          </button>
        }
      />

      {/* ==============================================
          RESEARCH KPI CARDS
      =============================================== */}

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          title="Green-cover net change"
          value={formatChange(
            historical.green_net_change_percent_points
          )}
          subtitle={`${formatNumber(
            historical.green_start_percent
          )}% to ${formatNumber(
            historical.green_end_percent
          )}% | ${historical.core_period}`}
          icon={Leaf}
          tone="green"
        />

        <MetricCard
          title="Built-up net change"
          value={formatChange(
            historical.built_net_change_percent_points
          )}
          subtitle={`${formatNumber(
            historical.built_start_percent
          )}% to ${formatNumber(
            historical.built_end_percent
          )}% | ${historical.core_period}`}
          icon={Building2}
          tone="orange"
        />

        <MetricCard
          title="Pressure-scored grids"
          value={formatCount(
            pressure.eligible_scored_cells
          )}
          subtitle={`${formatCount(
            pressure.outside_inference_domain_cells
          )} cells outside prediction domain`}
          icon={Grid3X3}
          tone="teal"
        />

        <MetricCard
          title="Conservation Priority coverage"
          value={formatCount(
            conservation.scored_grid_count
          )}
          subtitle={`${formatCount(
            conservation.primary_priority_unscored_count
          )} cells without a priority score`}
          icon={ShieldCheck}
          tone="blue"
        />
      </div>

      {/* ==============================================
          RESEARCH CHARTS
      =============================================== */}

      <div className="mt-5 grid gap-5 xl:grid-cols-2">
        <section className="card-pad min-w-0">
          <SectionHeading
            title="Historical land-cover trends"
            description="Green and built-up percentages from the 2011-2024 core analysis."
          />

          <ResponsiveContainer width="100%" height={275}>
            <LineChart data={trend}>
              <CartesianGrid strokeDasharray="4 5" />

              <XAxis
                dataKey="year"
                tick={{ fontSize: 11 }}
              />

              <YAxis
                domain={[0, 100]}
                tick={{ fontSize: 11 }}
                tickFormatter={(n) => `${n}%`}
              />

              <Tooltip
                formatter={(value, name) => [
                  `${formatNumber(value)}%`,
                  name,
                ]}
              />

              <Legend />

              <Line
                name="Green cover"
                dataKey="green"
                stroke="#26885c"
                strokeWidth={3}
                dot={false}
                isAnimationActive={false}
              />

              <Line
                name="Built-up"
                dataKey="built"
                stroke="#e47f54"
                strokeWidth={3}
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>

          <p className="mt-2 text-[11px] text-slate-500">
            Net changes are percentage-point differences,
            not relative percent growth.
          </p>
        </section>

        <section className="card-pad min-w-0">
          <SectionHeading
            title="Future pressure distribution"
            description={`Frozen relative pressure categories for the combined ${pressure.pressure_period} assessment period.`}
          />

          <ResponsiveContainer width="100%" height={275}>
            <BarChart
              data={pressureChart}
              layout="vertical"
              margin={{
                left: 14,
                right: 18,
              }}
            >
              <CartesianGrid strokeDasharray="4 5" />

              <XAxis
                type="number"
                allowDecimals={false}
                tick={{ fontSize: 11 }}
              />

              <YAxis
                type="category"
                dataKey="name"
                width={112}
                tick={{ fontSize: 10 }}
              />

              <Tooltip
                formatter={(value) => [
                  formatCount(value),
                  'Grid cells',
                ]}
              />

              <Bar
                dataKey="count"
                fill="#208a76"
                radius={[0, 6, 6, 0]}
                isAnimationActive={false}
              />
            </BarChart>
          </ResponsiveContainer>

          <p className="mt-2 text-[11px] text-slate-500">
            {formatCount(pressure.threshold_flagged_cells)}
            {' '}threshold-flagged grids.
            The pressure model produces relative scores,
            not calibrated development probabilities.
          </p>
        </section>
      </div>

      {/* ==============================================
          GIS MAP AND GRID DETAILS
      =============================================== */}

      <div className="mt-5 grid gap-5 2xl:grid-cols-[1.65fr_.85fr]">
        <section className="card-pad relative isolate z-0 min-w-0">
          <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
            <SectionHeading
              title="Integrated GIS research map"
              description="Explore the original Kaduwela 100 m grids, planning indicators and GN boundaries."
            />

            <span className="pill">
              {formatCount(
                planningMap?.features?.length
              )} mapped cells
            </span>
          </div>

          <div className="mb-4 grid gap-3 md:grid-cols-2">
            <label className="text-xs font-semibold text-slate-600">
              Research map layer

              <select
                className="control mt-1"
                value={mapMetric}
                onChange={(event) =>
                  setMapMetric(event.target.value)
                }
              >
                {Object.entries(GREEN_MAP_LAYERS).map(
                  ([key, layer]) => (
                    <option key={key} value={key}>
                      {layer.label}
                    </option>
                  )
                )}
              </select>
            </label>

            <label className="text-xs font-semibold text-slate-600">
              GN division

              <select
                className="control mt-1"
                value={selectedGnId || ''}
                onChange={(event) =>
                  selectGn(event.target.value)
                }
              >
                <option value="">
                  All GN divisions
                </option>

                {(gnDivisions?.data || []).map(
                  (gn) => (
                    <option
                      key={gn.gn_id}
                      value={gn.gn_id}
                    >
                      {gn.gn_name}
                    </option>
                  )
                )}
              </select>
            </label>
          </div>

          <div className="relative isolate z-0 overflow-hidden rounded-2xl">
            <GreenGridMap
              geojson={planningMap}
              gnGeojson={gnMap}
              metric={mapMetric}
              selectedGridId={selectedGridId}
              selectedGnId={selectedGnId}
              onSelect={setSelectedGridId}
              onClearGn={() => selectGn(null)}
            />
          </div>

          {(selectedGnId || selectedGridId) && (
            <button
              type="button"
              className="btn-secondary mt-3 text-xs"
              onClick={() => selectGn(null)}
            >
              Clear selections
            </button>
          )}
        </section>

        <div className="min-w-0 space-y-5">
          {/* ==========================================
              SELECTED GRID
          ========================================== */}

          <section className="card-pad">
            <div className="flex items-center gap-2">
              <MapPinned
                size={18}
                className="text-emerald-700"
              />

              <h2 className="section-title">
                Selected grid evidence
              </h2>
            </div>

            {!selectedFeature ? (
              <p className="mt-4 text-sm leading-6 text-slate-500">
                Click a research grid on the map to inspect
                its available scores, domain status and
                planning-review recommendations.
              </p>
            ) : (
              <div className="mt-3">
                <DetailRow
                  label="Grid ID"
                  value={grid.grid_id || selectedGridId}
                />

                <DetailRow
                  label="GN division"
                  value={
                    grid.gn_name ||
                    'GN assignment unavailable'
                  }
                />

                <DetailRow
                  label="Urbanization Pressure"
                  value={pressureDisplay(grid)}
                />

                <DetailRow
                  label="Pressure model coverage"
                  value={
                    validScore(grid.urbanization_pressure_score)
                      ? 'Within prediction domain'
                      : gridOutsideDomain
                        ? 'Outside prediction domain'
                        : readable(grid.pressure_domain_status) !== 'Unavailable'
                          ? readable(grid.pressure_domain_status)
                          : 'Reason not confirmed'
                  }
                />

                <DetailRow
                  label="Expert-AHP suitability"
                  value={formatScore(
                    grid.development_suitability_score
                  )}
                />

                <DetailRow
                  label="Conservation Priority"
                  value={researchScoreDisplay(
                    grid,
                    'conservation'
                  )}
                />

                <DetailRow
                  label="Development Conflict Index"
                  value={researchScoreDisplay(
                    grid,
                    'conflict'
                  )}
                />

                <DetailRow
                  label="Review stream"
                  value={readable(grid.review_stream)}
                />

                <DetailRow
                  label="Review rule"
                  value={readable(grid.review_rule_id)}
                />

                {gridOutsideDomain && (
                  <div className="mt-4 rounded-xl border border-amber-200 bg-amber-50 p-3">
                    <p className="text-xs font-bold text-amber-900">
                      Why is pressure unavailable?
                    </p>

                    <p className="mt-2 text-xs leading-5 text-amber-900">
                      This grid is outside the released
                      Urbanization Pressure model's
                      inference domain. No future pressure
                      score was produced for this cell.
                    </p>

                    <p className="mt-2 text-xs leading-5 text-amber-900">
                      This does not mean low or zero
                      pressure. Pressure-dependent
                      conservation and development-conflict
                      indicators cannot be calculated
                      without the required evidence.
                    </p>
                  </div>
                )}

                {!gridOutsideDomain &&
                  !validScore(grid.urbanization_pressure_score) && (
                    <div className="mt-4 rounded-xl border border-slate-200 bg-slate-50 p-3">
                      <p className="text-xs font-semibold text-slate-800">
                        Pressure score unavailable
                      </p>

                      <p className="mt-1 text-xs leading-5 text-slate-600">
                        The current record does not confirm
                        why this score is missing. No value
                        has been estimated or substituted.
                      </p>
                    </div>
                  )}

                {detailLoading && (
                  <p className="mt-3 text-xs text-slate-500">
                    Loading additional grid details...
                  </p>
                )}

                {grid.review_action && (
                  <div className="mt-4 rounded-xl bg-emerald-50 p-3">
                    <p className="text-xs font-bold text-emerald-800">
                      Suggested planning review
                    </p>

                    <p className="mt-1 text-xs leading-5 text-emerald-900">
                      {grid.review_action}
                    </p>
                  </div>
                )}
              </div>
            )}
          </section>

          {/* ==========================================
              SELECTED GN SUMMARY
          ========================================== */}

          {selectedGn && (
            <section className="card-pad">
              <h2 className="section-title">
                GN division summary
              </h2>

              <p className="mt-1 text-sm font-semibold text-emerald-800">
                {selectedGn.gn_name}
              </p>

              <div className="mt-3">
                <DetailRow
                  label="Area inside KMC"
                  value={`${formatNumber(
                    selectedGn.gn_area_inside_kmc_km2
                  )} km2`}
                />

                <DetailRow
                  label="Area-weighted suitability mean"
                  value={formatScore(
                    selectedGn.suitability_area_weighted_mean
                  )}
                />

                <DetailRow
                  label="Area-weighted pressure mean"
                  value={formatScore(
                    selectedGn.pressure_area_weighted_mean
                  )}
                />

                <DetailRow
                  label="Area-weighted conservation mean"
                  value={formatScore(
                    selectedGn.conservation_area_weighted_mean
                  )}
                />
              </div>

              <p className="mt-3 text-[11px] leading-5 text-slate-500">
                GN statistics use area-weighted polygon
                intersections and exclude unavailable scores.
              </p>
            </section>
          )}

          {/* ==========================================
              RESEARCH COVERAGE
          ========================================== */}

          <section className="card-pad">
            <h2 className="section-title">
              Research coverage
            </h2>

            <div className="mt-3">
              <DetailRow
                label="Total planning grids"
                value={formatCount(suitability.grid_count)}
              />

              <DetailRow
                label="GN divisions"
                value={formatCount(gnDivisions.total)}
              />

              <DetailRow
                label="Expert-AHP suitability mean"
                value={formatScore(
                  suitability.distribution?.mean
                )}
              />

              <DetailRow
                label="Exploratory hotspot candidates"
                value={formatCount(hotspots.total)}
              />

              <DetailRow
                label="Pressure-scored cells"
                value={formatCount(
                  pressure.eligible_scored_cells
                )}
              />

              <DetailRow
                label="Pressure out-of-domain cells"
                value={formatCount(
                  pressure.outside_inference_domain_cells
                )}
              />

              <DetailRow
                label="Conservation scores unavailable"
                value={formatCount(
                  conservation.primary_priority_unscored_count
                )}
              />
            </div>
          </section>
        </div>
      </div>

      {/* ==============================================
          RESEARCH PAGE NAVIGATION
      =============================================== */}

      <section className="card-pad mt-5">
        <h2 className="section-title">
          Explore research outputs
        </h2>

        <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {pageLinks.map(([path, title]) => (
            <Link
              key={path}
              to={path}
              className="flex items-center justify-between rounded-xl border border-emerald-100 bg-emerald-50 px-4 py-4 text-sm font-semibold text-emerald-900 transition hover:bg-emerald-100"
            >
              <span>{title}</span>
              <ArrowUpRight size={16} />
            </Link>
          ))}
        </div>
      </section>

      {/* ==============================================
          SCIENTIFIC LIMITATIONS
      =============================================== */}

      {/* <div className="mt-5 rounded-xl border border-amber-200 bg-amber-50 p-4 text-xs leading-6 text-amber-900">
        <strong>Research interpretation:</strong>
        {' '}Urbanization Pressure assesses a single
        combined 2026-2028 future period. Its scores
        are relative model outputs, not calibrated
        probabilities. The model covers 4,620 eligible
        research cells; the remaining cells are outside
        its inference domain. Unavailable values must
        not be interpreted as zero or low pressure.

        Expert-AHP Development Suitability covers
        selected criteria and is not a legal development
        approval. Conservation Priority requires both
        Green Importance and pressure evidence.
        Planning hotspot groups are exploratory,
        not statistically significant or official
        land-use designations.
      </div> */}
    </div>
  )
}
