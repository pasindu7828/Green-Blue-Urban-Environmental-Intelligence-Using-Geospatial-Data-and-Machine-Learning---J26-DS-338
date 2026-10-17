const variants = {
  green: 'border-emerald-200 bg-emerald-50 text-emerald-700',
  teal: 'border-teal-200 bg-teal-50 text-teal-700',
  blue: 'border-sky-200 bg-sky-50 text-sky-700',
  amber: 'border-amber-200 bg-amber-50 text-amber-700',
  orange: 'border-orange-200 bg-orange-50 text-orange-700',
  red: 'border-red-200 bg-red-50 text-red-700',
  purple: 'border-purple-200 bg-purple-50 text-purple-700',
  slate: 'border-slate-200 bg-slate-50 text-slate-600'
}
export default function Badge({ variant = 'slate', children, className = '' }) {
  return <span className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-1 text-[11px] font-semibold ${variants[variant] || variants.slate} ${className}`}>{children}</span>
}
