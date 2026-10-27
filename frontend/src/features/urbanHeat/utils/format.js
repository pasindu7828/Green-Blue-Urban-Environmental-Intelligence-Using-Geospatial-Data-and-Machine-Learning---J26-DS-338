export const fmt = (value, digits = 2, fallback = '—') => {
  const number = Number(value)
  return Number.isFinite(number) ? number.toFixed(digits) : fallback
}

export const fmtSigned = (value, digits = 2, suffix = '') => {
  const number = Number(value)
  if (!Number.isFinite(number)) return '—'
  return `${number > 0 ? '+' : ''}${number.toFixed(digits)}${suffix}`
}

export const fmtInt = (value, fallback = '—') => {
  const number = Number(value)
  return Number.isFinite(number) ? Math.round(number).toLocaleString() : fallback
}

export const titleCaseFeature = (value = '') => String(value)
  .replace(/_/g, ' ')
  .replace(/\b\w/g, (m) => m.toUpperCase())

export const confidenceTone = (label = '') => {
  const text = String(label).toLowerCase()
  if (text.includes('high')) return 'green'
  if (text.includes('moderate')) return 'amber'
  if (text.includes('low')) return 'red'
  return 'slate'
}

export const trajectoryTone = (trajectory = '') => ({
  'Persistent Hot': 'red',
  Intensifying: 'orange',
  Emerging: 'amber',
  Diminishing: 'green',
  'Persistent Cool': 'blue',
}[trajectory] || 'slate')
