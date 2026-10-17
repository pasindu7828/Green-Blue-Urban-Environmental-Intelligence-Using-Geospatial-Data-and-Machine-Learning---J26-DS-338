export default function PageHeader({ eyebrow, title, subtitle, actions, badges = [] }) {
  return (
    <div className="mb-6">
      <div className="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
        <div className="min-w-0">
          {eyebrow && <div className="eyebrow">{eyebrow}</div>}
          <h1 className="page-title">{title}</h1>
          {subtitle && <p className="page-subtitle">{subtitle}</p>}
        </div>
        {actions && <div className="flex shrink-0 flex-wrap gap-2">{actions}</div>}
      </div>
      {badges.length > 0 && <div className="mt-4 flex flex-wrap gap-2">{badges.map((b, i) => <span key={i} className="pill">{b}</span>)}</div>}
    </div>
  )
}
