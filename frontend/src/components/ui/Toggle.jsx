export default function Toggle({ checked, onChange, label, disabled = false }) {
  return (
    <button type="button" disabled={disabled} onClick={() => onChange?.(!checked)} className={`flex w-full items-center justify-between gap-3 rounded-xl border px-3 py-2.5 text-left text-sm transition ${checked ? 'border-emerald-200 bg-emerald-50/70 text-slate-800' : 'border-slate-200 bg-white text-slate-600'} disabled:opacity-50`}>
      <span>{label}</span>
      <span className={`relative h-6 w-10 shrink-0 rounded-full transition ${checked ? 'bg-emerald-600' : 'bg-slate-300'}`}>
        <span className={`absolute top-1 h-4 w-4 rounded-full bg-white shadow transition ${checked ? 'left-5' : 'left-1'}`} />
      </span>
    </button>
  )
}
