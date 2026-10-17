export default function DisclaimerBar({ children, tone = 'slate' }) {
  const cls = tone === 'amber' ? 'border-amber-200 bg-amber-50 text-amber-800' : tone === 'red' ? 'border-red-200 bg-red-50 text-red-700' : 'border-slate-200 bg-slate-50 text-slate-500'
  return <div className={`rounded-xl border px-4 py-3 text-[12px] leading-5 ${cls}`}>{children}</div>
}
