
// import { useEffect, useMemo, useRef, useState } from 'react'
// import L from 'leaflet'
// import {
//   MapContainer,
//   TileLayer,
//   ZoomControl,
//   useMap,
// } from 'react-leaflet'

// // =====================================================
// // 1. RESEARCH MAP CONFIGURATION
// // =====================================================

// export const GREEN_MAP_LAYERS = {
//   pressure: {
//     label: 'Urbanization Pressure',
//     field: 'urbanization_pressure_score',
//     colors: [
//       '#268bd2',
//       '#26b9d2',
//       '#f6d747',
//       '#f39c43',
//       '#e84948',
//     ],
//   },

//   suitability: {
//     label: 'Development Suitability',
//     field: 'development_suitability_score',
//     colors: [
//       '#dd9859',
//       '#f0cc69',
//       '#c1d786',
//       '#60b68b',
//       '#147b56',
//     ],
//   },

//   conservation: {
//     label: 'Conservation Priority',
//     field: 'frozen_conservation_priority_0_1',
//     colors: [
//       '#d3e5d9',
//       '#a4d0b8',
//       '#71bc98',
//       '#399c75',
//       '#086145',
//     ],
//   },

//   conflict: {
//     label: 'Development Conflict Index',
//     field: 'development_conflict_index_0_1',
//     colors: [
//       '#2f9bc3',
//       '#70c6ae',
//       '#f2d65b',
//       '#f2a052',
//       '#d84b4d',
//     ],
//   },
// }

// // =====================================================
// // 2. SCIENTIFIC VALUE HANDLING
// // =====================================================

// export function validScore(value) {
//   return typeof value === 'number' && Number.isFinite(value)
// }

// function getValue(feature, metric) {
//   const configuration =
//     GREEN_MAP_LAYERS[metric] || GREEN_MAP_LAYERS.pressure

//   return feature?.properties?.[configuration.field]
// }

// function getColor(feature, metric) {
//   const value = getValue(feature, metric)

//   // A missing value is NOT zero or a low score.
//   if (!validScore(value)) {
//     return '#a8b5c1'
//   }

//   const configuration =
//     GREEN_MAP_LAYERS[metric] || GREEN_MAP_LAYERS.pressure

//   const index = Math.min(
//     configuration.colors.length - 1,
//     Math.max(0, Math.floor(value * configuration.colors.length))
//   )

//   return configuration.colors[index]
// }

// function escapeHtml(value) {
//   return String(value ?? '').replace(
//     /[&<>"']/g,
//     (character) => ({
//       '&': '&amp;',
//       '<': '&lt;',
//       '>': '&gt;',
//       '"': '&quot;',
//       "'": '&#39;',
//     })[character]
//   )
// }

// // =====================================================
// // 3. ORIGINAL 100 M GRID STYLING
// // =====================================================

// // Same native-polygon rendering approach as Urban Heat.
// // No 200 m aggregation.
// // No fabricated larger GIS cells.
// // No original grid is removed.

// function gridStyle(feature, metric, selectedGnId) {
//   const properties = feature?.properties || {}

//   const value = getValue(feature, metric)

//   const outsideSelectedGn =
//     Boolean(selectedGnId) &&
//     properties.gn_id !== selectedGnId

//   return {
//     color: '#ffffff',
//     weight: 0.65,
//     opacity: 0.9,

//     fillColor: getColor(feature, metric),

//     // Every grid remains visible.
//     // GN selection only dims surrounding grids.
//     fillOpacity: outsideSelectedGn
//       ? 0.48
//       : validScore(value)
//         ? 0.90
//         : 0.76,
//   }
// }

// function tooltipContent(feature, metric) {
//   const properties = feature?.properties || {}
//   const value = getValue(feature, metric)

//   const label =
//     GREEN_MAP_LAYERS[metric]?.label ||
//     GREEN_MAP_LAYERS.pressure.label

//   return `
//     <div style="min-width:170px;line-height:1.6">
//       <strong>${escapeHtml(
//         properties.gn_name || 'GN unavailable'
//       )}</strong>

//       <div>
//         Grid: ${escapeHtml(properties.grid_id)}
//       </div>

//       <div>
//         ${escapeHtml(label)}:
//         <strong>
//           ${validScore(value) ? value.toFixed(3) : 'Unavailable'}
//         </strong>
//       </div>

//       <div style="font-size:10px;color:#64748b">
//         Original 100 m research cell
//       </div>
//     </div>
//   `
// }

// // =====================================================
// // 4. GEOMETRY HELPERS FOR RELIABLE CLICK SELECTION
// // =====================================================

// function getGeometryBounds(geometry) {
//   if (!geometry?.coordinates) return null

//   let minX = Infinity
//   let minY = Infinity
//   let maxX = -Infinity
//   let maxY = -Infinity

//   function visit(coordinates) {
//     if (!Array.isArray(coordinates)) return

//     if (
//       typeof coordinates[0] === 'number' &&
//       typeof coordinates[1] === 'number'
//     ) {
//       const x = coordinates[0]
//       const y = coordinates[1]

//       minX = Math.min(minX, x)
//       maxX = Math.max(maxX, x)
//       minY = Math.min(minY, y)
//       maxY = Math.max(maxY, y)

//       return
//     }

//     coordinates.forEach(visit)
//   }

//   visit(geometry.coordinates)

//   if (!Number.isFinite(minX)) return null

//   return { minX, minY, maxX, maxY }
// }

// function pointOnSegment(x, y, ax, ay, bx, by) {
//   const cross =
//     (x - ax) * (by - ay) -
//     (y - ay) * (bx - ax)

//   if (Math.abs(cross) > 1e-11) return false

//   const tolerance = 1e-10

//   return (
//     x >= Math.min(ax, bx) - tolerance &&
//     x <= Math.max(ax, bx) + tolerance &&
//     y >= Math.min(ay, by) - tolerance &&
//     y <= Math.max(ay, by) + tolerance
//   )
// }

// function pointInRing(x, y, ring) {
//   if (!Array.isArray(ring) || ring.length < 3) {
//     return false
//   }

//   let inside = false

//   for (
//     let i = 0, j = ring.length - 1;
//     i < ring.length;
//     j = i++
//   ) {
//     const [ax, ay] = ring[i]
//     const [bx, by] = ring[j]

//     if (pointOnSegment(x, y, ax, ay, bx, by)) {
//       return true
//     }

//     const intersects =
//       (ay > y) !== (by > y) &&
//       x < ((bx - ax) * (y - ay)) / (by - ay) + ax

//     if (intersects) {
//       inside = !inside
//     }
//   }

//   return inside
// }

// function pointInPolygon(x, y, rings) {
//   if (!Array.isArray(rings) || !rings.length) {
//     return false
//   }

//   if (!pointInRing(x, y, rings[0])) {
//     return false
//   }

//   // Respect interior holes.
//   for (const hole of rings.slice(1)) {
//     if (pointInRing(x, y, hole)) {
//       return false
//     }
//   }

//   return true
// }

// function pointInGeometry(x, y, geometry) {
//   if (geometry?.type === 'Polygon') {
//     return pointInPolygon(x, y, geometry.coordinates)
//   }

//   if (geometry?.type === 'MultiPolygon') {
//     return geometry.coordinates.some((polygon) =>
//       pointInPolygon(x, y, polygon)
//     )
//   }

//   return false
// }

// function featureBounds(geojson) {
//   if (!geojson?.features?.length) return null

//   const bounds = L.geoJSON(geojson).getBounds()

//   return bounds.isValid() ? bounds : null
// }

// // =====================================================
// // 5. REAL GIS GRID LAYER
// // =====================================================

// function ResearchGridLayer({
//   geojson,
//   metric,
//   selectedGnId,
//   selectedGridId,
//   onSelect,
// }) {
//   const map = useMap()

//   const layerRef = useRef(null)
//   const styleRef = useRef(null)
//   const onSelectRef = useRef(onSelect)
//   const firstFitRef = useRef(false)

//   styleRef.current = (feature) =>
//     gridStyle(feature, metric, selectedGnId)

//   useEffect(() => {
//     onSelectRef.current = onSelect
//   }, [onSelect])

//   // All 9,240 source features, including unavailable cells.
//   const indexedGrids = useMemo(
//     () =>
//       (geojson?.features || [])
//         .map((feature) => ({
//           feature,
//           bounds: getGeometryBounds(feature.geometry),
//         }))
//         .filter((item) => item.bounds),
//     [geojson]
//   )

//   // -----------------------------------------------------
//   // 5A. CREATE NATIVE LEAFLET GRID POLYGONS
//   // -----------------------------------------------------

//   useEffect(() => {
//     if (!geojson?.features?.length) return undefined

//     const renderer = L.canvas({ padding: 0.4 })

//     const layer = L.geoJSON(geojson, {
//       renderer,
//       interactive: true,

//       style: (feature) => styleRef.current(feature),

//       onEachFeature: (feature, polygon) => {
//         polygon.bindTooltip(
//           tooltipContent(feature, metric),
//           {
//             sticky: true,
//             direction: 'top',
//             opacity: 0.97,
//           }
//         )

//         polygon.on('mouseover', () => {
//           polygon.setStyle({
//             color: '#0f172a',
//             weight: 2,
//             opacity: 1,
//           })
//         })

//         polygon.on('mouseout', () => {
//           polygon.setStyle(
//             styleRef.current(feature)
//           )
//         })
//       },
//     }).addTo(map)

//     layerRef.current = layer

//     // Fit all Kaduwela grids on first load.
//     if (!firstFitRef.current) {
//       const bounds = layer.getBounds()

//       if (bounds.isValid()) {
//         map.fitBounds(bounds, {
//           padding: [18, 18],
//           maxZoom: 15,
//           animate: false,
//         })

//         firstFitRef.current = true
//       }
//     }

//     return () => {
//       if (map.hasLayer(layer)) map.removeLayer(layer)
//       layerRef.current = null
//     }
//   }, [geojson, map])

//   // -----------------------------------------------------
//   // 5B. UPDATE COLORS WITHOUT RECREATING THE GRID
//   // -----------------------------------------------------

//   useEffect(() => {
//     const layer = layerRef.current

//     if (!layer) return

//     layer.setStyle((feature) =>
//       gridStyle(feature, metric, selectedGnId)
//     )

//     layer.eachLayer((polygon) => {
//       if (polygon.feature && polygon.getTooltip()) {
//         polygon.setTooltipContent(
//           tooltipContent(polygon.feature, metric)
//         )
//       }
//     })
//   }, [geojson, metric, selectedGnId])

//   // -----------------------------------------------------
//   // 5C. RELIABLE GRID CLICK DETECTION
//   // -----------------------------------------------------

//   // We use geometry-based click selection because it
//   // already worked with your Urban Green GIS polygons.
//   // Leaflet handles hover tooltips separately.

//   useEffect(() => {
//     if (!indexedGrids.length) return undefined

//     const container = map.getContainer()
//     let pointerStart = null

//     function handlePointerDown(event) {
//       pointerStart = {
//         x: event.clientX,
//         y: event.clientY,
//       }
//     }

//     function handleClick(event) {
//       if (
//         event.target?.closest?.(
//           '.leaflet-control, .leaflet-popup, .leaflet-tooltip, button'
//         )
//       ) {
//         return
//       }

//       // Ignore the click at the end of a drag.
//       if (
//         pointerStart &&
//         Math.hypot(
//           event.clientX - pointerStart.x,
//           event.clientY - pointerStart.y
//         ) > 7
//       ) {
//         return
//       }

//       const location = map.mouseEventToLatLng(event)

//       for (const item of indexedGrids) {
//         const bounds = item.bounds

//         if (
//           location.lng < bounds.minX ||
//           location.lng > bounds.maxX ||
//           location.lat < bounds.minY ||
//           location.lat > bounds.maxY
//         ) {
//           continue
//         }

//         if (
//           pointInGeometry(
//             location.lng,
//             location.lat,
//             item.feature.geometry
//           )
//         ) {
//           const gridId =
//             item.feature.properties?.grid_id

//           if (gridId) {
//             onSelectRef.current?.(gridId)
//           }

//           return
//         }
//       }
//     }

//     container.addEventListener(
//       'pointerdown',
//       handlePointerDown,
//       true
//     )

//     container.addEventListener(
//       'click',
//       handleClick,
//       true
//     )

//     return () => {
//       container.removeEventListener(
//         'pointerdown',
//         handlePointerDown,
//         true
//       )

//       container.removeEventListener(
//         'click',
//         handleClick,
//         true
//       )
//     }
//   }, [indexedGrids, map])

//   // -----------------------------------------------------
//   // 5D. STRONG SELECTED GRID OUTLINE
//   // -----------------------------------------------------

//   useEffect(() => {
//     if (!selectedGridId) return undefined

//     const selectedFeature =
//       indexedGrids.find(
//         (item) =>
//           item.feature.properties?.grid_id ===
//           selectedGridId
//       )?.feature

//     if (!selectedFeature) return undefined

//     const paneName = 'urban-green-selected-grid'

//     let pane = map.getPane(paneName)

//     if (!pane) {
//       pane = map.createPane(paneName)
//     }

//     pane.style.zIndex = '650'
//     pane.style.pointerEvents = 'none'

//     const halo = L.geoJSON(selectedFeature, {
//       pane: paneName,
//       interactive: false,
//       style: {
//         color: '#ffffff',
//         weight: 8,
//         opacity: 1,
//         fillOpacity: 0,
//       },
//     }).addTo(map)

//     const outline = L.geoJSON(selectedFeature, {
//       pane: paneName,
//       interactive: false,
//       style: {
//         color: '#e11d68',
//         weight: 3.5,
//         opacity: 1,
//         fillColor: '#e11d68',
//         fillOpacity: 0.12,
//       },
//     }).addTo(map)

//     return () => {
//       map.removeLayer(outline)
//       map.removeLayer(halo)
//     }
//   }, [indexedGrids, selectedGridId, map])

//   return null
// }

// // =====================================================
// // 6. KMC BOUNDARY
// // =====================================================

// function KMCLayer({ boundary }) {
//   const map = useMap()

//   useEffect(() => {
//     if (!boundary?.features?.length) return undefined

//     const paneName = 'urban-green-kmc-outline'

//     let pane = map.getPane(paneName)

//     if (!pane) pane = map.createPane(paneName)

//     pane.style.zIndex = '450'
//     pane.style.pointerEvents = 'none'

//     const layer = L.geoJSON(boundary, {
//       pane: paneName,
//       interactive: false,

//       style: {
//         color: '#0f2e28',
//         weight: 3,
//         opacity: 0.95,
//         fillOpacity: 0,
//       },
//     }).addTo(map)

//     return () => map.removeLayer(layer)
//   }, [boundary, map])

//   return null
// }

// // =====================================================
// // 7. GN BOUNDARY AND FOCUS
// // =====================================================

// function GNLayer({
//   gnGeojson,
//   selectedGnId,
//   geojson,
//   boundary,
// }) {
//   const map = useMap()
//   const previousGnRef = useRef(null)

//   // Draw selected GN outline without blocking clicks.
//   useEffect(() => {
//     if (
//       !selectedGnId ||
//       !gnGeojson?.features?.length
//     ) {
//       return undefined
//     }

//     const matching = gnGeojson.features.filter(
//       (feature) =>
//         feature.properties?.gn_id === selectedGnId
//     )

//     if (!matching.length) return undefined

//     const paneName = 'urban-green-gn-outline'

//     let pane = map.getPane(paneName)

//     if (!pane) pane = map.createPane(paneName)

//     pane.style.zIndex = '500'
//     pane.style.pointerEvents = 'none'

//     const layer = L.geoJSON(
//       {
//         type: 'FeatureCollection',
//         features: matching,
//       },
//       {
//         pane: paneName,
//         interactive: false,

//         style: {
//           color: '#087f76',
//           weight: 4,
//           opacity: 1,
//           dashArray: '7 5',
//           fillOpacity: 0,
//         },
//       }
//     ).addTo(map)

//     return () => map.removeLayer(layer)
//   }, [gnGeojson, selectedGnId, map])

//   // Focus on selected GN, or return to the full
//   // study area when the GN selection is cleared.
//   useEffect(() => {
//     if (selectedGnId) {
//       const matching = (gnGeojson?.features || []).filter(
//         (feature) =>
//           feature.properties?.gn_id === selectedGnId
//       )

//       if (matching.length) {
//         const bounds = featureBounds({
//           type: 'FeatureCollection',
//           features: matching,
//         })

//         if (bounds) {
//           map.fitBounds(bounds, {
//             padding: [35, 35],
//             maxZoom: 16,
//             animate: false,
//           })
//         }
//       }
//     } else if (previousGnRef.current) {
//       const bounds = featureBounds(
//         boundary?.features?.length
//           ? boundary
//           : geojson
//       )

//       if (bounds) {
//         map.fitBounds(bounds, {
//           padding: [18, 18],
//           maxZoom: 15,
//           animate: false,
//         })
//       }
//     }

//     previousGnRef.current = selectedGnId
//   }, [selectedGnId, gnGeojson, boundary, geojson, map])

//   return null
// }

// // =====================================================
// // 8. FULL KMC VIEW / ZOOM RESET
// // =====================================================

// function ResetView({
//   geojson,
//   boundary,
//   resetKey,
// }) {
//   const map = useMap()

//   useEffect(() => {
//     if (resetKey === 0) return

//     const source =
//       boundary?.features?.length
//         ? boundary
//         : geojson

//     const bounds = featureBounds(source)

//     if (bounds) {
//       map.fitBounds(bounds, {
//         padding: [18, 18],
//         maxZoom: 15,
//         animate: false,
//       })
//     }
//   }, [resetKey, geojson, boundary, map])

//   return null
// }

// // =====================================================
// // 9. MAIN GREEN GRID MAP
// // =====================================================

// export default function GreenGridMap({
//   geojson,
//   gnGeojson,
//   metric = 'pressure',
//   selectedGridId,
//   selectedGnId,
//   onSelect,
// }) {
//   const [showStreets, setShowStreets] = useState(true)
//   const [boundary, setBoundary] = useState(null)
//   const [resetKey, setResetKey] = useState(0)

//   const activeMetric =
//     GREEN_MAP_LAYERS[metric]
//       ? metric
//       : 'pressure'

//   const configuration =
//     GREEN_MAP_LAYERS[activeMetric]

//   // Same boundary source used by Urban Heat.
//   useEffect(() => {
//     const controller = new AbortController()

//     const url =
//       `${import.meta.env.BASE_URL || '/'}data/kaduwela-boundary.geojson`

//     fetch(url, { signal: controller.signal })
//       .then((response) =>
//         response.ok ? response.json() : null
//       )
//       .then((result) => {
//         if (!controller.signal.aborted) {
//           setBoundary(result)
//         }
//       })
//       .catch(() => {})

//     return () => controller.abort()
//   }, [])

//   const selected = useMemo(
//     () =>
//       (geojson?.features || []).find(
//         (feature) =>
//           feature.properties?.grid_id ===
//           selectedGridId
//       )?.properties || null,
//     [geojson, selectedGridId]
//   )

//   const selectedValue =
//     selected?.[configuration.field]

//   const gridCount =
//     geojson?.features?.length || 0

//   return (
//     <div className="relative min-w-0">

//       {/* ===========================================
//           MAP TOOLBAR
//       ============================================ */}

//       <div className="mb-3 flex flex-wrap items-center justify-between gap-3 rounded-xl border border-slate-200 bg-slate-50 px-3 py-3">
//         <div>
//           <p className="text-xs font-semibold text-slate-800">
//             Kaduwela research GIS
//           </p>

//           <p className="mt-1 text-[11px] text-slate-500">
//             {gridCount.toLocaleString()} original 100 m cells
//             {' | '}
//             {selectedGnId
//               ? 'Selected GN highlighted'
//               : 'Full study area'}
//           </p>
//         </div>

//         <div className="flex flex-wrap gap-2">
//           <button
//             type="button"
//             onClick={() =>
//               setResetKey((value) => value + 1)
//             }
//             className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-100"
//           >
//             Full KMC View
//           </button>

//           <button
//             type="button"
//             onClick={() =>
//               setShowStreets((value) => !value)
//             }
//             className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-100"
//           >
//             {showStreets
//               ? 'Hide street map'
//               : 'Show street map'}
//           </button>
//         </div>
//       </div>

//       {/* ===========================================
//           LEAFLET MAP
//       ============================================ */}

//       <div
//         className="relative isolate overflow-hidden rounded-2xl border border-slate-200 bg-[#e9f0ef]"
//         style={{ height: 570 }}
//       >
//         <MapContainer
//           center={[6.902, 79.97]}
//           zoom={12}
//           zoomControl={false}
//           scrollWheelZoom
//           preferCanvas
//           maxZoom={19}
//           className="h-full w-full"
//           style={{ zIndex: 0 }}
//         >
//           {showStreets && (
//             <TileLayer
//               url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
//               attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
//               maxZoom={19}
//             />
//           )}

//           {/* Same zoom controls as friend's Heat map */}
//           <ZoomControl position="topleft" />

//           <ResearchGridLayer
//             geojson={geojson}
//             metric={activeMetric}
//             selectedGnId={selectedGnId}
//             selectedGridId={selectedGridId}
//             onSelect={onSelect}
//           />

//           <KMCLayer boundary={boundary} />

//           <GNLayer
//             gnGeojson={gnGeojson}
//             selectedGnId={selectedGnId}
//             geojson={geojson}
//             boundary={boundary}
//           />

//           <ResetView
//             geojson={geojson}
//             boundary={boundary}
//             resetKey={resetKey}
//           />
//         </MapContainer>

//         {boundary?.features?.length > 0 && (
//           <div className="pointer-events-none absolute right-3 top-3 z-[400] rounded-full border border-slate-200 bg-white/95 px-3 py-1.5 text-[10px] font-semibold text-slate-700 shadow-md">
//             KMC study boundary
//           </div>
//         )}

//         {/* Selected original grid information */}
//         {selected && (
//           <div className="pointer-events-none absolute bottom-4 left-4 z-[400] max-w-[240px] rounded-xl border border-slate-200 bg-white/95 p-3 shadow-lg">
//             <p className="text-[10px] font-bold uppercase text-rose-700">
//               Selected research grid
//             </p>

//             <p className="mt-1 font-mono text-xs font-bold text-slate-900">
//               {selectedGridId}
//             </p>

//             <p className="mt-1 text-xs text-slate-600">
//               {selected.gn_name || 'GN unavailable'}
//             </p>

//             <p className="mt-2 text-xs font-semibold text-emerald-800">
//               {configuration.label}:{' '}
//               {validScore(selectedValue)
//                 ? selectedValue.toFixed(3)
//                 : 'Unavailable'}
//             </p>
//           </div>
//         )}
//       </div>

//       {/* ===========================================
//           COLOR LEGEND
//       ============================================ */}

//       <div className="mt-3 flex flex-wrap items-center gap-3 text-[11px] text-slate-600">
//         <strong>{configuration.label}</strong>

//         <span>Lower relative score</span>

//         <div className="flex overflow-hidden rounded-md border border-slate-200">
//           {configuration.colors.map((color) => (
//             <span
//               key={color}
//               className="h-4 w-8"
//               style={{
//                 backgroundColor: color,
//               }}
//             />
//           ))}
//         </div>

//         <span>Higher relative score</span>

//         <span className="inline-flex items-center gap-1">
//           <span
//             className="h-3.5 w-3.5 rounded-sm"
//             style={{
//               backgroundColor: '#a8b5c1',
//             }}
//           />
//           Score unavailable
//         </span>
//       </div>

//       <p className="mt-2 text-[11px] leading-5 text-slate-500">
//         This map displays all original 100 m research
//         polygons. Hover to inspect a grid and click
//         to view its research evidence. Use the plus
//         and minus controls to adjust cell visibility.
//         Gray represents unavailable or out-of-domain
//         scores, not zero. Colors indicate visualization
//         ranges, not official planning classifications.
//       </p>
//     </div>
//   )
// }


//2nd code



import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import L from 'leaflet'
import { MapContainer, TileLayer, ZoomControl, useMap } from 'react-leaflet'

// =====================================================
// 1. ORIGINAL RESEARCH LAYERS
// =====================================================

export const GREEN_MAP_LAYERS = {
  pressure: {
    label: 'Urbanization Pressure',
    field: 'urbanization_pressure_score',
    colors: ['#268bd2', '#26b9d2', '#f6d747', '#f39c43', '#e84948'],
  },
  suitability: {
    label: 'Development Suitability',
    field: 'development_suitability_score',
    colors: ['#dd9859', '#f0cc69', '#c1d786', '#60b68b', '#147b56'],
  },
  conservation: {
    label: 'Conservation Priority',
    field: 'frozen_conservation_priority_0_1',
    colors: ['#d3e5d9', '#a4d0b8', '#71bc98', '#399c75', '#086145'],
  },
  conflict: {
    label: 'Development Conflict Index',
    field: 'development_conflict_index_0_1',
    colors: ['#2f9bc3', '#70c6ae', '#f2d65b', '#f2a052', '#d84b4d'],
  },
}

// =====================================================
// 2. SCORE AND DOMAIN INTERPRETATION
// =====================================================

export function validScore(value) {
  return typeof value === 'number' && Number.isFinite(value)
}

export function isOutsidePressureDomain(properties = {}) {
  const rule = String(properties?.review_rule_id ?? '').toUpperCase()
  const domain = String(properties?.pressure_domain_status ?? '').toUpperCase()

  return /^R00(?:_|[\s-]|$)/.test(rule) ||
    domain.includes('OUTSIDE') ||
    domain.includes('OUT_OF_DOMAIN') ||
    domain.includes('NOT_IN_PRESSURE_DOMAIN') ||
    domain.includes('NOT_IN_INFERENCE_DOMAIN')
}

export function pressureDisplay(properties = {}) {
  if (validScore(properties.urbanization_pressure_score)) {
    return properties.urbanization_pressure_score.toFixed(3)
  }

  return isOutsidePressureDomain(properties)
    ? 'Outside prediction domain'
    : 'Unavailable - reason not confirmed'
}

export function researchScoreDisplay(properties = {}, metric = 'pressure') {
  if (metric === 'pressure') return pressureDisplay(properties)

  const field = GREEN_MAP_LAYERS[metric]?.field
  if (!field) return 'Unavailable'

  const value = properties[field]
  if (validScore(value)) return value.toFixed(3)

  if (
    ['conservation', 'conflict'].includes(metric) &&
    isOutsidePressureDomain(properties)
  ) {
    return 'Not calculated (outside pressure domain)'
  }

  return 'Unavailable'
}

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, (char) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  }[char]))
}

function scoreColor(value, metric) {
  if (!validScore(value)) return '#a8b5c1'

  const colors = GREEN_MAP_LAYERS[metric].colors

  const index = Math.max(
    0,
    Math.min(colors.length - 1, Math.floor(value * colors.length))
  )

  return colors[index]
}

// =====================================================
// 3. GIS GEOMETRY HELPERS
// =====================================================

function getBounds(data) {
  if (!data?.features?.length) return null

  const bounds = L.geoJSON(data).getBounds()
  return bounds.isValid() ? bounds : null
}

function boundsOfGeometry(geometry) {
  if (!geometry?.coordinates) return null

  let west = Infinity
  let south = Infinity
  let east = -Infinity
  let north = -Infinity

  function walk(coords) {
    if (!Array.isArray(coords)) return

    if (
      typeof coords[0] === 'number' &&
      typeof coords[1] === 'number'
    ) {
      west = Math.min(west, coords[0])
      east = Math.max(east, coords[0])
      south = Math.min(south, coords[1])
      north = Math.max(north, coords[1])
    } else {
      coords.forEach(walk)
    }
  }

  walk(geometry.coordinates)

  return Number.isFinite(west)
    ? { west, south, east, north }
    : null
}

function pointInRing(x, y, ring) {
  if (!Array.isArray(ring) || ring.length < 3) return false

  let inside = false

  for (
    let i = 0, j = ring.length - 1;
    i < ring.length;
    j = i++
  ) {
    const [ax, ay] = ring[i]
    const [bx, by] = ring[j]

    const cross = (x - ax) * (by - ay) - (y - ay) * (bx - ax)

    if (
      Math.abs(cross) < 1e-11 &&
      x >= Math.min(ax, bx) - 1e-10 &&
      x <= Math.max(ax, bx) + 1e-10 &&
      y >= Math.min(ay, by) - 1e-10 &&
      y <= Math.max(ay, by) + 1e-10
    ) {
      return true
    }

    if (
      (ay > y) !== (by > y) &&
      x < ((bx - ax) * (y - ay)) / (by - ay) + ax
    ) {
      inside = !inside
    }
  }

  return inside
}

function containsPoint(geometry, x, y) {
  const polygonContains = (rings) =>
    rings?.length > 0 &&
    pointInRing(x, y, rings[0]) &&
    !rings.slice(1).some((hole) => pointInRing(x, y, hole))

  if (geometry?.type === 'Polygon') {
    return polygonContains(geometry.coordinates)
  }

  if (geometry?.type === 'MultiPolygon') {
    return geometry.coordinates.some(polygonContains)
  }

  return false
}

function findFeatureAt(indexed, latlng) {
  for (const { feature, box } of indexed) {
    if (
      latlng.lng < box.west ||
      latlng.lng > box.east ||
      latlng.lat < box.south ||
      latlng.lat > box.north
    ) {
      continue
    }

    if (containsPoint(feature.geometry, latlng.lng, latlng.lat)) {
      return feature
    }
  }

  return null
}

// =====================================================
// 4. TOOLTIPS FOR BOTH RESOLUTIONS
// =====================================================

function makeTooltip(feature, metric, mode, originalById) {
  const p = feature.properties || {}
  const field = GREEN_MAP_LAYERS[metric].field
  const score = p[field]
  const label = GREEN_MAP_LAYERS[metric].label

  if (mode === 'overview') {
    const count = p.source_cell_count ?? 0
    const valid = p[`${metric}_valid_cell_count`] ?? 0
    const fraction = p[`${metric}_coverage_fraction`]

    const coverage = validScore(fraction)
      ? `${(fraction * 100).toFixed(1)}%`
      : 'Unknown'

    const status = p[`${metric}_coverage_status`] || 'UNKNOWN'

    const gnNames = [
      ...new Set(
        (p.source_grid_ids || [])
          .map((gridId) =>
            originalById.get(String(gridId))
              ?.properties?.gn_name
          )
          .filter((name) => typeof name === 'string' && name.trim())
      ),
    ].sort()

    const gnLabel =
      gnNames.length === 1 ? 'GN Division' : 'GN Divisions'

    const gnText =
      gnNames.length > 0
        ? gnNames.join(', ')
        : 'Not available'

    return `
      <div style="min-width:225px;line-height:1.6">
        <strong>200 m display: ${escapeHtml(p.overview_id)}</strong>
        <br>
        ${gnLabel}:
        <strong>${escapeHtml(gnText)}</strong>
        <br>
        ${escapeHtml(label)}:
        <strong>
          ${validScore(score) ? score.toFixed(3) : 'No available score'}
        </strong>
        <br>
        Scored 100 m cells: ${valid}/${count}
        <br>
        Scored area coverage: ${coverage}
        <br>
        Coverage: ${escapeHtml(status)}
        <br>
        <span style="font-size:10px;color:#64748b">
          Area-weighted display mean, not a new model prediction.
          Click to highlight its original 100 m cells at the same zoom.
        </span>
      </div>
    `
  }

  return `
    <div style="min-width:190px;line-height:1.6">
      <strong>${escapeHtml(p.gn_name || 'GN unavailable')}</strong>
      <br>
      Original grid: ${escapeHtml(p.grid_id)}
      <br>
      ${escapeHtml(label)}:
      <strong>${escapeHtml(researchScoreDisplay(p, metric))}</strong>
      <br>
      <span style="font-size:10px;color:#64748b">
        Original 100 m research cell
      </span>
    </div>
  `
}

// =====================================================
// 5. MAP POLYGON STYLING AND GN SELECTION
// =====================================================

// Selected GN works in BOTH display resolutions.
//
// 100 m Detail:
//   Check the original grid's gn_id.
//
// 200 m Overview:
//   Check GN membership of all source 100 m cells.
//   A 200 m block can contain cells from multiple GN
//   divisions, so do not assign it to only one GN.

function belongsToGn(feature, mode, selectedGnId, originalById) {
  if (!selectedGnId) return true

  const gnId = String(selectedGnId)
  const p = feature.properties || {}

  if (mode === 'detail') {
    return String(p.gn_id ?? '') === gnId
  }

  return (p.source_grid_ids || []).some((gridId) => {
    const child = originalById.get(String(gridId))

    return String(child?.properties?.gn_id ?? '') === gnId
  })
}

function styleFor(feature, metric, mode, selectedGnId, originalById) {
  const p = feature.properties || {}
  const value = p[GREEN_MAP_LAYERS[metric].field]

  const hasScore = validScore(value)

  const partial =
    mode === 'overview' &&
    p[`${metric}_coverage_status`] === 'PARTIAL'

  const outsideGN =
    Boolean(selectedGnId) &&
    !belongsToGn(feature, mode, selectedGnId, originalById)

  return {
    color: partial ? '#344054' : '#ffffff',
    weight: partial ? 0.85 : 0.65,
    dashArray: partial ? '3 3' : undefined,
    opacity: partial ? 0.72 : 0.85,
    fillColor: scoreColor(value, metric),
    fillOpacity: outsideGN
      ? 0.28
      : hasScore
        ? 0.89
        : 0.74,
  }
}

// =====================================================
// 6. LEAFLET GIS LAYERS
// =====================================================

function MapLayers({
  displayedGeojson,
  originalGeojson,
  gnGeojson,
  boundary,
  metric,
  mode,
  selectedGnId,
  selectedGridId,
  highlightedSourceIds,
  onFeatureClick,
}) {
  const map = useMap()

  const originalById = useMemo(
    () =>
      new Map(
        (originalGeojson?.features || []).map((feature) => [
          String(feature.properties?.grid_id),
          feature,
        ])
      ),
    [originalGeojson]
  )

  const callbackRef = useRef(onFeatureClick)

  useEffect(() => {
    callbackRef.current = onFeatureClick
  }, [onFeatureClick])

  const indexed = useMemo(
    () =>
      (displayedGeojson?.features || [])
        .map((feature) => ({
          feature,
          box: boundsOfGeometry(feature.geometry),
        }))
        .filter((entry) => entry.box),
    [displayedGeojson]
  )

  // ---------------------------------------------------
  // 6A. DRAW EITHER 200 M OR 100 M POLYGONS
  // ---------------------------------------------------

  useEffect(() => {
    if (!displayedGeojson?.features?.length) return undefined

    const layer = L.geoJSON(displayedGeojson, {
      renderer: L.canvas({ padding: 0.4 }),
      interactive: true,

      style: (feature) =>
        styleFor(feature, metric, mode, selectedGnId, originalById),

      onEachFeature: (feature, polygon) => {
        polygon.bindTooltip(
          makeTooltip(feature, metric, mode, originalById),
          {
            sticky: true,
            direction: 'top',
            opacity: 0.97,
          }
        )

        polygon.on('mouseover', () =>
          polygon.setStyle({
            color: '#0f172a',
            weight: 2,
            opacity: 1,
          })
        )

        polygon.on('mouseout', () =>
          polygon.setStyle(
            styleFor(feature, metric, mode, selectedGnId, originalById)
          )
        )
      },
    }).addTo(map)

    return () => map.removeLayer(layer)
  }, [
    displayedGeojson,
    metric,
    mode,
    selectedGnId,
    map,
    originalById,
  ])

  // ---------------------------------------------------
  // 6B. RELIABLE GEOGRAPHIC CLICK SELECTION
  // ---------------------------------------------------

  useEffect(() => {
    if (!indexed.length) return undefined

    const container = map.getContainer()
    let pointerStart = null

    function onPointerDown(event) {
      pointerStart = {
        x: event.clientX,
        y: event.clientY,
      }
    }

    function onClick(event) {
      if (
        event.target?.closest?.(
          '.leaflet-control, .leaflet-tooltip, .leaflet-popup, button'
        )
      ) {
        return
      }

      if (
        pointerStart &&
        Math.hypot(
          event.clientX - pointerStart.x,
          event.clientY - pointerStart.y
        ) > 7
      ) {
        return
      }

      const latlng = map.mouseEventToLatLng(event)
      const feature = findFeatureAt(indexed, latlng)

      if (feature) {
        callbackRef.current?.(feature, latlng)
      }
    }

    container.addEventListener('pointerdown', onPointerDown, true)
    container.addEventListener('click', onClick, true)

    return () => {
      container.removeEventListener('pointerdown', onPointerDown, true)
      container.removeEventListener('click', onClick, true)
    }
  }, [indexed, map])

  // ---------------------------------------------------
  // 6C. KMC STUDY AREA BOUNDARY
  // ---------------------------------------------------

  useEffect(() => {
    if (!boundary?.features?.length) return undefined

    const paneName = 'urban-green-kmc-outline'
    const pane = map.getPane(paneName) || map.createPane(paneName)

    pane.style.zIndex = '450'
    pane.style.pointerEvents = 'none'

    const layer = L.geoJSON(boundary, {
      pane: paneName,
      interactive: false,

      style: {
        color: '#0f2e28',
        weight: 3,
        opacity: 0.95,
        fillOpacity: 0,
      },
    }).addTo(map)

    return () => map.removeLayer(layer)
  }, [boundary, map])

  // ---------------------------------------------------
  // 6D. SELECTED GN DIVISION BOUNDARY
  // ---------------------------------------------------

  // This outline works in BOTH 200 m and 100 m modes.
  // No fitBounds, flyTo, setView, or panTo is used here.

  useEffect(() => {
    if (!selectedGnId || !gnGeojson?.features?.length) {
      return undefined
    }

    const matching = gnGeojson.features.filter(
      (f) =>
        String(f.properties?.gn_id ?? '') === String(selectedGnId)
    )

    if (!matching.length) return undefined

    const paneName = 'urban-green-gn-outline'
    const pane = map.getPane(paneName) || map.createPane(paneName)

    pane.style.zIndex = '500'
    pane.style.pointerEvents = 'none'

    const layer = L.geoJSON(
      {
        type: 'FeatureCollection',
        features: matching,
      },
      {
        pane: paneName,
        interactive: false,

        style: {
          color: '#087f76',
          weight: 3.5,
          dashArray: '7 5',
          fillOpacity: 0,
        },
      }
    ).addTo(map)

    return () => map.removeLayer(layer)
  }, [selectedGnId, gnGeojson, map])

  // ---------------------------------------------------
  // 6E. HIGHLIGHT ALL RELATED ORIGINAL 100 M CELLS
  // ---------------------------------------------------

  useEffect(() => {
    if (
      mode !== 'detail' ||
      !highlightedSourceIds?.length ||
      !originalGeojson?.features?.length
    ) {
      return undefined
    }

    const ids = new Set(highlightedSourceIds.map(String))

    const matching = originalGeojson.features.filter(
      (feature) =>
        ids.has(String(feature.properties?.grid_id))
    )

    if (!matching.length) return undefined

    const paneName = 'urban-green-group-selection'

    const pane =
      map.getPane(paneName) ||
      map.createPane(paneName)

    pane.style.zIndex = '610'
    pane.style.pointerEvents = 'none'

    const layer = L.geoJSON(
      {
        type: 'FeatureCollection',
        features: matching,
      },
      {
        pane: paneName,
        interactive: false,

        style: {
          color: '#be185d',
          weight: 1.8,
          opacity: 1,
          fillColor: '#f472b6',
          fillOpacity: 0.28,
        },
      }
    ).addTo(map)

    return () => map.removeLayer(layer)
  }, [
    mode,
    highlightedSourceIds,
    originalGeojson,
    map,
  ])

  // ---------------------------------------------------
  // 6F. SELECTED INDIVIDUAL 100 M GRID OUTLINE
  // ---------------------------------------------------

  useEffect(() => {
    if (mode !== 'detail' || !selectedGridId) {
      return undefined
    }

    const selected = (originalGeojson?.features || []).find(
      (f) =>
        String(f.properties?.grid_id) === String(selectedGridId)
    )

    if (!selected) return undefined

    const paneName = 'urban-green-selected-outline'
    const pane = map.getPane(paneName) || map.createPane(paneName)

    pane.style.zIndex = '650'
    pane.style.pointerEvents = 'none'

    const halo = L.geoJSON(selected, {
      pane: paneName,
      interactive: false,

      style: {
        color: '#ffffff',
        weight: 8,
        fillOpacity: 0,
      },
    }).addTo(map)

    const outline = L.geoJSON(selected, {
      pane: paneName,
      interactive: false,

      style: {
        color: '#e11d68',
        weight: 3.5,
        fillColor: '#e11d68',
        fillOpacity: 0.12,
      },
    }).addTo(map)

    return () => {
      map.removeLayer(outline)
      map.removeLayer(halo)
    }
  }, [mode, selectedGridId, originalGeojson, map])

  return null
}

// =====================================================
// 7. MAP CAMERA CONTROLS
// =====================================================

function MapCamera({
  originalGeojson,
  overviewGeojson,
  boundary,
  fullViewToken,
}) {
  const map = useMap()
  const firstFit = useRef(false)

  // Fit map when the Dashboard initially loads.
  useEffect(() => {
    if (firstFit.current || !originalGeojson?.features?.length) {
      return
    }

    const bounds = getBounds(
      boundary?.features?.length
        ? boundary
        : originalGeojson
    )

    if (bounds) {
      map.fitBounds(bounds, {
        padding: [18, 18],
        maxZoom: 15,
        animate: false,
      })

      firstFit.current = true
    }
  }, [originalGeojson, boundary, map])

  // IMPORTANT:
  // GN dropdown selection does not appear in this
  // camera component. Therefore it cannot trigger
  // automatic zooming, panning, or fitting.

  // Only the Full KMC View button resets map bounds.
  useEffect(() => {
    if (!fullViewToken) return

    const bounds = getBounds(
      boundary?.features?.length
        ? boundary
        : overviewGeojson?.features?.length
          ? overviewGeojson
          : originalGeojson
    )

    if (bounds) {
      map.fitBounds(bounds, {
        padding: [18, 18],
        maxZoom: 15,
        animate: false,
      })
    }
  }, [
    fullViewToken,
    boundary,
    overviewGeojson,
    originalGeojson,
    map,
  ])

  return null
}

// =====================================================
// 8. MAIN REUSABLE GREEN MAP
// =====================================================

export default function GreenGridMap({
  geojson,
  gnGeojson,
  metric = 'pressure',
  selectedGridId,
  selectedGnId,
  onSelect,
  onClearGn,
  height = 570,
}) {
  const [mode, setMode] = useState('overview')
  const [showStreets, setShowStreets] = useState(true)

  const [overviewState, setOverviewState] = useState({
    status: 'loading',
    data: null,
    error: '',
  })

  const [boundary, setBoundary] = useState(null)
  const [fullViewToken, setFullViewToken] = useState(0)
  const [highlightedSourceIds, setHighlightedSourceIds] = useState([])

  const activeMetric = GREEN_MAP_LAYERS[metric]
    ? metric
    : 'pressure'

  const config = GREEN_MAP_LAYERS[activeMetric]

  const originalById = useMemo(
    () =>
      new Map(
        (geojson?.features || []).map(
          (f) => [String(f.properties?.grid_id), f]
        )
      ),
    [geojson]
  )

  // ---------------------------------------------------
  // 8A. LOAD THE VALIDATED 200 M STATIC GEOJSON
  // ---------------------------------------------------

  useEffect(() => {
    const controller = new AbortController()

    const base = import.meta.env.BASE_URL || '/'

    const url =
      `${base.endsWith('/') ? base : `${base}/`}` +
      'data/urban-green/grid_map_200m.geojson'

    fetch(url, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(
            `200 m GeoJSON HTTP ${response.status}`
          )
        }

        return response.json()
      })
      .then((data) => {
        if (
          data?.type !== 'FeatureCollection' ||
          data.features?.length !== 2410
        ) {
          throw new Error(
            '200 m GeoJSON must contain 2,410 validated display polygons'
          )
        }

        const sourceCount = data.features.reduce(
          (n, f) =>
            n + (f.properties?.source_cell_count || 0),
          0
        )

        if (
          sourceCount !== 9240 ||
          !data.features.every(
            (f) =>
              f.properties?.overview_id &&
              Array.isArray(f.properties?.source_grid_ids)
          )
        ) {
          throw new Error(
            '200 m GeoJSON source-grid mapping failed validation'
          )
        }

        if (!controller.signal.aborted) {
          setOverviewState({
            status: 'ready',
            data,
            error: '',
          })
        }
      })
      .catch((error) => {
        if (controller.signal.aborted) return

        setOverviewState({
          status: 'error',
          data: null,
          error: error.message,
        })

        // Fall back to original 100 m mode.
        setMode('detail')
      })

    return () => controller.abort()
  }, [])

  // ---------------------------------------------------
  // 8B. LOAD EXISTING KMC BOUNDARY
  // ---------------------------------------------------

  useEffect(() => {
    const controller = new AbortController()
    const base = import.meta.env.BASE_URL || '/'

    const url =
      `${base.endsWith('/') ? base : `${base}/`}` +
      'data/kaduwela-boundary.geojson'

    fetch(url, { signal: controller.signal })
      .then((response) =>
        response.ok ? response.json() : null
      )
      .then((data) => {
        if (!controller.signal.aborted) {
          setBoundary(data)
        }
      })
      .catch(() => {})

    return () => controller.abort()
  }, [])

  // ---------------------------------------------------
  // 8C. GN SELECTION — NO MODE CHANGE OR ZOOM
  // ---------------------------------------------------

  // When GN selection changes, clear previous group
  // highlighting. Do NOT change the current map mode.
  // Do NOT call fitBounds, flyTo, panTo, or setZoom.

  useEffect(() => {
    setHighlightedSourceIds([])
  }, [selectedGnId])

  const availableOverview = overviewState.data

  const effectiveMode =
    mode === 'overview' && availableOverview
      ? 'overview'
      : 'detail'

  const displayedGeojson =
    effectiveMode === 'overview'
      ? availableOverview
      : geojson

  // ---------------------------------------------------
  // 8D. MAP CELL CLICK SELECTION
  // ---------------------------------------------------

  const selectMapFeature = useCallback(
    (feature) => {
      if (effectiveMode === 'detail') {
        const gridId = feature.properties?.grid_id
        if (!gridId) return

        setHighlightedSourceIds([])
        onSelect?.(String(gridId))
        return
      }

      const ids = [
        ...new Set(
          (feature.properties?.source_grid_ids || [])
            .map((id) => String(id))
        ),
      ].filter((id) => originalById.has(id))

      if (!ids.length) return

      onSelect?.(null)
      setHighlightedSourceIds(ids)
      setMode('detail')

      // Switching to detail keeps exact zoom/center.
      // All original source grids are highlighted.
    },
    [effectiveMode, originalById, onSelect]
  )

  // ---------------------------------------------------
  // 8E. FULL KMC VIEW
  // ---------------------------------------------------

  function returnToKMC() {
    setHighlightedSourceIds([])

    setMode(
      availableOverview ? 'overview' : 'detail'
    )

    onSelect?.(null)

    // Full KMC View is explicitly a reset action.
    if (selectedGnId) {
      onClearGn?.()
    }

    setFullViewToken((n) => n + 1)
  }

  const selected =
    originalById.get(
      String(selectedGridId ?? '')
    )?.properties

  const displayValue = selected
    ? researchScoreDisplay(selected, activeMetric)
    : null

  // ===================================================
  // 9. MAP USER INTERFACE
  // ===================================================

  return (
    <div className="relative min-w-0">

      {/* DISPLAY CONTROLS */}

      <div className="mb-3 flex flex-wrap items-center justify-between gap-3 rounded-xl border border-slate-200 bg-slate-50 px-3 py-3">
        <div>
          <p className="text-xs font-semibold text-slate-800">
            Kaduwela research GIS
          </p>

          <p className="mt-1 text-[11px] text-slate-500">
            {effectiveMode === 'overview'
              ? '2,410 validated ~200 m dashboard display cells'
              : `${(
                  geojson?.features?.length || 0
                ).toLocaleString()} original 100 m research grids`}
          </p>

          {selectedGnId && (
            <p className="mt-1 text-[11px] font-medium text-teal-700">
              GN selection active — map zoom and position unchanged.
              Areas outside the selected GN are lighter.
            </p>
          )}

          {effectiveMode === 'detail' &&
            highlightedSourceIds.length > 0 && (
              <p className="mt-1 max-w-xl text-[11px] font-semibold text-rose-700">
                {highlightedSourceIds.length} original 100 m cells
                highlighted from the clicked 200 m block.
                Use + / − to zoom, then click an individual cell.
              </p>
            )}
        </div>

        <div className="flex flex-wrap gap-2">

          <button
            type="button"
            onClick={returnToKMC}
            className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-100"
          >
            Full KMC View
          </button>

          {/* Always allow 200 m when its data is loaded,
              even when a GN division is selected. */}

          <button
            type="button"
            disabled={!availableOverview}
            onClick={() => {
              setHighlightedSourceIds([])
              onSelect?.(null)
              setMode('overview')
            }}
            className={`rounded-lg border px-3 py-2 text-xs font-semibold ${
              effectiveMode === 'overview'
                ? 'border-emerald-800 bg-emerald-800 text-white'
                : 'border-slate-300 bg-white text-slate-700'
            } disabled:cursor-not-allowed disabled:opacity-40`}
          >
            200 m Overview
          </button>

          <button
            type="button"
            onClick={() => setMode('detail')}
            className={`rounded-lg border px-3 py-2 text-xs font-semibold ${
              effectiveMode === 'detail'
                ? 'border-emerald-800 bg-emerald-800 text-white'
                : 'border-slate-300 bg-white text-slate-700'
            }`}
          >
            100 m Detail
          </button>

          <button
            type="button"
            onClick={() =>
              setShowStreets((v) => !v)
            }
            className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-100"
          >
            {showStreets
              ? 'Hide street map'
              : 'Show street map'}
          </button>

        </div>
      </div>

      {/* LOAD / ERROR INFORMATION */}

      {overviewState.status === 'loading' && (
        <p className="mb-2 text-xs text-slate-500">
          Loading validated 200 m overview map...
        </p>
      )}

      {overviewState.status === 'error' && (
        <p className="mb-2 rounded-lg border border-amber-200 bg-amber-50 p-3 text-xs text-amber-900">
          200 m overview unavailable: {overviewState.error}.
          The original 100 m map remains usable.
          Check that grid_map_200m.geojson was copied
          into frontend/public/data/urban-green/.
        </p>
      )}

      {/* LEAFLET MAP */}

      <div
        className="relative isolate overflow-hidden rounded-2xl border border-slate-200 bg-[#e9f0ef]"
        style={{ height }}
      >
        <MapContainer
          center={[6.902, 79.97]}
          zoom={12}
          zoomControl={false}
          scrollWheelZoom
          preferCanvas
          maxZoom={19}
          className="h-full w-full"
          style={{ zIndex: 0 }}
        >
          {showStreets && (
            <TileLayer
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
              maxZoom={19}
            />
          )}

          <ZoomControl position="topleft" />

          <MapLayers
            displayedGeojson={displayedGeojson}
            originalGeojson={geojson}
            gnGeojson={gnGeojson}
            boundary={boundary}
            metric={activeMetric}
            mode={effectiveMode}
            selectedGnId={selectedGnId}
            selectedGridId={selectedGridId}
            highlightedSourceIds={highlightedSourceIds}
            onFeatureClick={selectMapFeature}
          />

          <MapCamera
            originalGeojson={geojson}
            overviewGeojson={availableOverview}
            boundary={boundary}
            fullViewToken={fullViewToken}
          />
        </MapContainer>

        {boundary?.features?.length > 0 && (
          <div className="pointer-events-none absolute right-3 top-3 z-[400] rounded-full border border-slate-200 bg-white/95 px-3 py-1.5 text-[10px] font-semibold text-slate-700 shadow">
            KMC study boundary
          </div>
        )}

        {/* SELECTED INDIVIDUAL ORIGINAL 100 M GRID */}

        {effectiveMode === 'detail' &&
          selected &&
          highlightedSourceIds.length === 0 && (
            <div className="pointer-events-none absolute bottom-4 left-4 z-[400] max-w-[245px] rounded-xl border border-slate-200 bg-white/95 p-3 shadow-lg">
              <p className="text-[10px] font-bold uppercase text-rose-700">
                Selected original 100 m grid
              </p>

              <p className="mt-1 font-mono text-xs font-bold text-slate-900">
                {selectedGridId}
              </p>

              <p className="mt-1 text-xs text-slate-600">
                {selected.gn_name || 'GN unavailable'}
              </p>

              <p className="mt-2 text-xs font-semibold text-emerald-800">
                {config.label}: {displayValue}
              </p>
            </div>
          )}
      </div>

      {/* MAP LEGEND */}

      <div className="mt-3 flex flex-wrap items-center gap-3 text-[11px] text-slate-600">
        <strong>{config.label}</strong>

        <span>Lower relative score</span>

        <div className="flex overflow-hidden rounded-md border border-slate-200">
          {config.colors.map((color) => (
            <span
              key={color}
              className="h-4 w-8"
              style={{ backgroundColor: color }}
            />
          ))}
        </div>

        <span>Higher relative score</span>

        <span className="inline-flex items-center gap-1">
          <span className="h-3.5 w-3.5 rounded-sm bg-[#a8b5c1]" />
          No score
        </span>

        {effectiveMode === 'overview' && (
          <span className="inline-flex items-center gap-1">
            <span className="h-3.5 w-3.5 rounded-sm border border-dashed border-slate-700 bg-slate-200" />
            Partial coverage
          </span>
        )}
      </div>

      {/* SCIENTIFIC INTERPRETATION */}

      {/* <p className="mt-2 text-[11px] leading-5 text-slate-500">
        {effectiveMode === 'overview'
          ? 'The ~200 m overview summarizes original 100 m research cells using area-weighted means of available scores. It is not a new model prediction. Selected GN divisions are highlighted without changing the zoom. Mixed-GN overview blocks can belong partly to the selected GN. Clicking a block switches to 100 m detail and highlights its original cells without zooming.'
          : highlightedSourceIds.length > 0
            ? 'Pink outlines indicate the original 100 m source cells of the clicked overview block. You can zoom manually with + / − and click one original grid to inspect its exact research evidence.'
            : 'The 100 m detail map shows original research scores. Selecting a GN highlights its boundary and dims cells assigned elsewhere without changing zoom. Gray means no score, not zero pressure.'}

        {' '}Colors are relative visualization ranges, not official planning classifications.
      </p> */}

    </div>
  )
}
