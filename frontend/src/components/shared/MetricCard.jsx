export default function MetricCard({ title, value, subtitle, icon: Icon, tone = 'teal', progress, compact = false }) {
  const tones = {
    teal: 'bg-teal-50 text-teal-700', green: 'bg-emerald-50 text-emerald-700', blue: 'bg-sky-50 text-sky-700', amber: 'bg-amber-50 text-amber-700', orange: 'bg-orange-50 text-orange-700', red: 'bg-red-50 text-red-700', slate: 'bg-slate-100 text-slate-700', purple: 'bg-purple-50 text-purple-700'
  }
  return (
    <div className={`card ${compact ? 'p-4' : 'p-5 md:p-6'}`}>
      <div className="flex items-start justify-between gap-4">
        <div className="text-[12px] font-semibold uppercase tracking-wide text-slate-500">{title}</div>
        {Icon && <div className={`grid h-10 w-10 shrink-0 place-items-center rounded-xl ${tones[tone] || tones.teal}`}><Icon size={18}/></div>}
      </div>
      <div className={`${compact ? 'mt-2 text-[27px]' : 'mt-4 text-[34px]'} font-bold leading-none tracking-[-0.04em] text-slate-950`}>{value}</div>
      {progress !== undefined && <div className="mt-4 h-2 overflow-hidden rounded-full bg-slate-100"><div className="h-full rounded-full bg-emerald-600" style={{ width: `${Math.max(0, Math.min(100, progress))}%` }} /></div>}
      {subtitle && <div className="mt-3 text-[12px] leading-5 text-slate-500">{subtitle}</div>}
    </div>
  )
}
