import fs from 'node:fs'
import path from 'node:path'

const frontendRoot = process.cwd()
const sourceKml = path.resolve(frontendRoot, '../Urban_Heat_Research/data/raw/boundary/kaduwela_kmc_boundary.kml')
const publicDir = path.resolve(frontendRoot, 'public/data')
const outputGeoJson = path.join(publicDir, 'kaduwela-boundary.geojson')
const fallbackGeoJson = path.join(publicDir, 'kaduwela-grid-envelope.geojson')

fs.mkdirSync(publicDir, { recursive: true })

function parseCoordinateText(text) {
  return text
    .trim()
    .split(/\s+/)
    .map((token) => token.split(',').slice(0, 2).map(Number))
    .filter((pair) => pair.length === 2 && pair.every(Number.isFinite))
}

function extractRing(block, tagName) {
  const re = new RegExp(`<${tagName}>[\\s\\S]*?<coordinates[^>]*>([\\s\\S]*?)<\\/coordinates>[\\s\\S]*?<\\/${tagName}>`, 'i')
  const match = block.match(re)
  return match ? parseCoordinateText(match[1]) : null
}

function kmlToGeoJson(kml) {
  const polygons = []
  const polygonMatches = kml.match(/<Polygon\b[\s\S]*?<\/Polygon>/gi) || []

  for (const polygonBlock of polygonMatches) {
    const outer = extractRing(polygonBlock, 'outerBoundaryIs')
    if (!outer || outer.length < 4) continue

    const holes = []
    const innerMatches = polygonBlock.match(/<innerBoundaryIs\b[\s\S]*?<\/innerBoundaryIs>/gi) || []
    for (const inner of innerMatches) {
      const ring = extractRing(inner, 'innerBoundaryIs')
      if (ring && ring.length >= 4) holes.push(ring)
    }
    polygons.push([outer, ...holes])
  }

  if (!polygons.length) throw new Error('No polygon geometry could be parsed from the KML boundary.')

  const geometry = polygons.length === 1
    ? { type: 'Polygon', coordinates: polygons[0] }
    : { type: 'MultiPolygon', coordinates: polygons }

  return {
    type: 'FeatureCollection',
    features: [{
      type: 'Feature',
      properties: {
        name: 'Kaduwela Municipal Council Boundary',
        source: 'Urban_Heat_Research/data/raw/boundary/kaduwela_kmc_boundary.kml',
        authoritative_source: true
      },
      geometry
    }]
  }
}

try {
  if (fs.existsSync(sourceKml)) {
    const kml = fs.readFileSync(sourceKml, 'utf8')
    const geojson = kmlToGeoJson(kml)
    fs.writeFileSync(outputGeoJson, JSON.stringify(geojson), 'utf8')
    console.log(`[boundary] Prepared official Kaduwela boundary from ${sourceKml}`)
  } else if (fs.existsSync(fallbackGeoJson)) {
    fs.copyFileSync(fallbackGeoJson, outputGeoJson)
    console.warn('[boundary] Official KML not found in sibling Urban_Heat_Research folder; using the real-grid study-area envelope fallback.')
  } else {
    console.warn('[boundary] No official KML or fallback boundary is available. Maps will still render the real grid polygons.')
  }
} catch (error) {
  console.warn(`[boundary] ${error.message}`)
  if (fs.existsSync(fallbackGeoJson)) fs.copyFileSync(fallbackGeoJson, outputGeoJson)
}
