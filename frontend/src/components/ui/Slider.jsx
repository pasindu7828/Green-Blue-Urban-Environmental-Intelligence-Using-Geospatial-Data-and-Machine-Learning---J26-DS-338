export default function Slider({ label, value, onChange, min = 0, max = 100, step = 1, suffix = '%', helper }) {
  return (
    <label className="block">
      <div className="mb-2 flex items-center justify-between gap-4 text-sm"><span className="font-medium text-slate-700">{label}</span><span className="rounded-full bg-[#0F2E28] px-2.5 py-1 text-[11px] font-semibold text-white">{value}{suffix}</span></div>
      <input className="w-full accent-emerald-700" type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
      {helper && <div className="mt-1 text-[11px] text-slate-400">{helper}</div>}
    </label>
  )
}
