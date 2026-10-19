import PageHeader from '../../../components/shared/PageHeader'

export default function GreenPageScaffold({ page }) {
  return (
    <div>
      <PageHeader
        eyebrow="Urban Green Intelligence"
        title={page.title}
        subtitle={page.description}
        badges={[
          'Kaduwela Research Component',
          'Frontend data integration in progress',
        ]}
      />

      <div className="grid gap-5 xl:grid-cols-[1.5fr_1fr]">
        <section className="card-pad">
          <h2 className="section-title">Planned research visualizations</h2>
          <p className="mt-2 text-sm leading-6 text-slate-500">
            This page is part of the new Urban Green frontend.
            Its layout and navigation are available, but its
            live research charts and maps have not yet been connected.
          </p>

          <div className="mt-5 space-y-3">
            {page.planned.map((item, index) => (
              <div
                key={item}
                className="flex items-start gap-3 rounded-xl border border-slate-100 bg-slate-50 p-4"
              >
                <span className="grid h-7 w-7 shrink-0 place-items-center rounded-lg bg-emerald-100 text-xs font-bold text-emerald-800">
                  {index + 1}
                </span>
                <span className="text-sm text-slate-700">
                  {item}
                </span>
              </div>
            ))}
          </div>
        </section>

        <section className="card-pad">
          <h2 className="section-title">Research API sources</h2>
          <p className="mt-2 text-xs leading-5 text-slate-500">
            The backend routes below will supply this page.
            Listing them does not mean this page is already fetching data.
          </p>

          <div className="mt-5 space-y-2">
            {page.endpoints.map((endpoint) => (
              <div
                key={endpoint}
                className="overflow-x-auto rounded-lg border border-slate-200 bg-slate-50 px-3 py-2"
              >
                <code className="text-[11px] text-emerald-800">
                  /api/urban-green{endpoint}
                </code>
              </div>
            ))}
          </div>
        </section>
      </div>

      <div className="mt-5 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-xs leading-5 text-amber-900">
        <strong>Integration status:</strong> Navigation shell only.
        Live visualizations will be implemented using the validated
        research backend. No mock research scores are displayed.
      </div>
    </div>
  )
}
