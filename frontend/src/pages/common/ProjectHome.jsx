import { Activity, ArrowRight, Droplets, Leaf, LockKeyhole, LogOut, ShieldCheck, Thermometer, UserRound } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { useApp } from '../../contexts/AppContext'

const components = [
  {
    number: 'Component 1',
    icon: Leaf,
    title: 'Urbanization Intelligence',
    description: 'Frontend implementation will be integrated by the responsible team member.',
    accent: '#2F9E5B',
    meta: 'Member integration pending',
    enabled: false,
  },
  {
    number: 'Component 2',
    icon: Thermometer,
    title: 'Urban Heat Intelligence',
    description: 'LST monitoring, heat trend analysis, ML forecasting, explainability, scenario testing and cooling priority.',
    accent: '#E0472B',
    route: '/heat/dashboard',
    meta: 'Heat map · Forecast · Explainable AI',
    enabled: true,
  },

  {
    number: 'Component 3',
    icon: Droplets,
    title: 'Wetland Intelligence',
    description:
      'Historical wetland change, candidate wetland screening, exploratory future scenarios and seasonal waterlogging susceptibility in Kaduwela.',
    accent: '#0E7C7B',
    route: '/wetlands/dashboard',
    meta: 'Wetland Change · Future Scenarios · Waterlogging',
    enabled: true,
  },

  {
    number: 'Component 4',
    icon: Activity,
    title: 'Kelani River Corridor Intelligence',
    description: 'Frontend implementation will be integrated by the responsible team member.',
    accent: '#2B6CB0',
    meta: 'Member integration pending',
    enabled: false,
  },
]

export default function ProjectHome() {
  const nav = useNavigate()
  const { profile } = useApp()
  const logout = () => {
    localStorage.removeItem('gbuei_auth')
    nav('/')
  }

  return <div className="min-h-screen bg-[#f2f7f5]">
    <header className="sticky top-0 z-30 border-b border-slate-200 bg-white/95 backdrop-blur">
      <div className="mx-auto flex h-[76px] max-w-[1480px] items-center gap-4 px-5 md:px-8">
        <div className="grid h-11 w-11 place-items-center rounded-2xl bg-[#0F2E28] text-[#2DD4BF]"><Leaf size={22}/></div>
        <div className="min-w-0 flex-1">
          <div className="font-bold text-slate-900">Green-Blue UEI</div>
          <div className="truncate text-[10px] text-slate-500">AI-Powered Green-Blue Urban Environmental Intelligence Using Geospatial Data and Machine Learning</div>
        </div>
        <div className="hidden items-center gap-2 rounded-xl border border-slate-200 px-3 py-2 sm:flex">
          <div className="grid h-7 w-7 place-items-center rounded-lg bg-[#0F2E28] text-white"><UserRound size={14}/></div>
          <div><div className="text-[10px] font-semibold">{profile.name}</div><div className="text-[8px] text-slate-400">{profile.role}</div></div>
        </div>
        <button onClick={logout} className="btn-secondary px-3"><LogOut size={15}/><span className="hidden sm:inline">Logout</span></button>
      </div>
    </header>

    <main className="mx-auto max-w-[1480px] px-5 py-8 md:px-8 md:py-10">
      <section className="overflow-hidden rounded-[30px] border border-emerald-100 bg-gradient-to-br from-white via-[#f0faf7] to-[#e8f5f7] p-7 shadow-card md:p-10">
        <div className="max-w-4xl">
          <div className="eyebrow">Integrated Group Capstone</div>
          <h1 className="mt-2 text-[38px] font-bold leading-[1.05] tracking-[-.045em] text-slate-950 md:text-[52px]">Urban Green-Blue Infrastructure — Kaduwela</h1>
          <p className="mt-4 text-[15px] leading-7 text-slate-600">The shared project shell is ready for all four research components. Urban Heat is currently integrated; the other three component frontends are intentionally left for their responsible members to add.</p>
          <div className="mt-5 flex flex-wrap gap-2 text-[11px]"><span className="pill">Kaduwela pilot context</span><span className="pill">4 research components</span><span className="pill">Shared React frontend</span><span className="pill">Urban Heat API connected</span></div>
        </div>
      </section>

      <div className="mt-6 grid gap-5 lg:grid-cols-2">
        {components.map(({ number, icon: Icon, title, description, accent, route, meta, enabled }) => (
          <button
            key={number}
            type="button"
            disabled={!enabled}
            onClick={() => enabled && nav(route)}
            className={`group relative overflow-hidden rounded-[24px] border bg-white p-6 text-left shadow-card transition md:p-7 ${enabled ? 'border-slate-200 hover:-translate-y-1 hover:shadow-xl' : 'cursor-not-allowed border-slate-200 opacity-70'}`}
          >
            <div className="absolute inset-y-0 left-0 w-1.5" style={{ background: accent }}/>
            <div className="flex items-start gap-4">
              <div className="grid h-12 w-12 shrink-0 place-items-center rounded-2xl p-3" style={{ background: `${accent}14`, color: accent }}><Icon size={25}/></div>
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="rounded-full px-2.5 py-1 text-[10px] font-bold" style={{ background: `${accent}12`, color: accent }}>{number}</span>
                  <span className="text-[10px] text-slate-400">{meta}</span>
                </div>
                <h2 className="mt-3 text-[21px] font-bold tracking-[-.025em] text-slate-900">{title}</h2>
                <p className="mt-2 text-[12px] leading-5 text-slate-500">{description}</p>
                <div className="mt-5 flex items-center gap-2 text-[12px] font-bold" style={{ color: enabled ? accent : '#64748b' }}>
                  {enabled ? <><span>Enter Dashboard</span><ArrowRight size={15} className="transition group-hover:translate-x-1"/></> : <><LockKeyhole size={14}/><span>Waiting for member frontend</span></>}
                </div>
              </div>
            </div>
          </button>
        ))}
      </div>

      <section className="mt-6 grid gap-5 md:grid-cols-3">
        <div className="card-pad"><ShieldCheck size={20} className="text-emerald-600"/><h3 className="mt-3 text-sm font-bold">Shared project shell</h3><p className="mt-2 text-[11px] leading-5 text-slate-500">Login, project home, common UI and styling remain available for the whole team.</p></div>
        <div className="card-pad"><Thermometer size={20} className="text-orange-600"/><h3 className="mt-3 text-sm font-bold">Urban Heat integrated</h3><p className="mt-2 text-[11px] leading-5 text-slate-500">Urban Heat is connected to the supplied FastAPI and finalized research dashboard outputs.</p></div>
        <div className="card-pad"><Leaf size={20} className="text-teal-600"/><h3 className="mt-3 text-sm font-bold">Independent member ownership</h3><p className="mt-2 text-[11px] leading-5 text-slate-500">The remaining members can add their own pages and services without inheriting placeholder analytical implementations.</p></div>
      </section>

      <div className="mt-6 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-[11px] leading-5 text-amber-800"><b>Integration status:</b> Urbanization, Wetland and River Corridor prototype pages were intentionally removed. Their cards are disabled until the responsible members integrate their own frontend implementations.</div>
    </main>
  </div>
}
