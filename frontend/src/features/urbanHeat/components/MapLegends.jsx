export function ThermalLegend({ min, max, title = 'LST (°C)' }) {
  const lo = Number.isFinite(min) ? min : null
  const hi = Number.isFinite(max) ? max : null
  return <div><div className="text-[10px] font-bold uppercase tracking-wide text-slate-600">{title}</div><div className="mt-2 h-2.5 w-44 rounded-full" style={{ background: 'linear-gradient(90deg,#2563eb,#38bdf8,#facc15,#fb923c,#dc2626)' }}/><div className="mt-1 flex justify-between text-[9px] text-slate-500"><span>{lo === null ? 'Cooler' : lo.toFixed(1)}</span><span>{hi === null ? 'Warmer' : hi.toFixed(1)}</span></div></div>
}

const trajectoryColors = {
  'Persistent Hot': '#b91c1c',
  Intensifying: '#ea580c',
  Emerging: '#d97706',
  Diminishing: '#16a34a',
  'Persistent Cool': '#2563eb',
}

export function TrajectoryLegend() {
  return <div><div className="text-[10px] font-bold uppercase tracking-wide text-slate-600">Relative heat trajectory</div><div className="mt-2 space-y-1.5">{Object.entries(trajectoryColors).map(([label, color]) => <div key={label} className="flex items-center gap-2 text-[10px] text-slate-600"><span className="h-2.5 w-5 rounded" style={{ background: color }}/>{label}</div>)}</div></div>
}

export function PriorityLegend() {
  return <div><div className="text-[10px] font-bold uppercase tracking-wide text-slate-600">Cooling priority percentile</div><div className="mt-2 h-2.5 w-44 rounded-full" style={{ background: 'linear-gradient(90deg,#d1fae5,#facc15,#f97316,#b91c1c)' }}/><div className="mt-1 flex justify-between text-[9px] text-slate-500"><span>Lower</span><span>Higher</span></div></div>
}

export const TRAJECTORY_COLORS = trajectoryColors

export function thermalColor(value, min, max) {
  const v = Number(value)
  if (!Number.isFinite(v)) return '#cbd5e1'
  if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) return '#f59e0b'
  const t = Math.max(0, Math.min(1, (v - min) / (max - min)))
  if (t < 0.2) return '#2563eb'
  if (t < 0.4) return '#38bdf8'
  if (t < 0.6) return '#facc15'
  if (t < 0.8) return '#fb923c'
  return '#dc2626'
}

export function priorityColor(percentile) {
  const p = Number(percentile)
  if (!Number.isFinite(p)) return '#cbd5e1'
  if (p >= 90) return '#b91c1c'
  if (p >= 75) return '#f97316'
  if (p >= 50) return '#facc15'
  if (p >= 25) return '#84cc16'
  return '#10b981'
}
