export default function ChartPanel({ title, subtitle, children, className = '', height = 300 }) {
  return (
    <div className={`card-pad ${className}`}>
      <h2 className="section-title">{title}</h2>
      {subtitle && <p className="mt-1 text-[12px] text-slate-500">{subtitle}</p>}
      <div className="mt-4" style={{ height }}>{children}</div>
    </div>
  )
}
