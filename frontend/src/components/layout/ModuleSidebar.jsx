import { NavLink, Link } from 'react-router-dom'
import { ArrowLeft, Bot, Leaf, X } from 'lucide-react'
import { useApp } from '../../contexts/AppContext'

export default function ModuleSidebar({ open, onClose, navItems, brand, subtitle, icon: BrandIcon = Leaf, footer, assistant = false, accent = '#2DD4BF' }) {
  const { setAssistantOpen } = useApp()
  return (
    <>
      {open && <button className="fixed inset-0 z-40 bg-black/30 lg:hidden" onClick={onClose} aria-label="Close sidebar" />}
      <aside className={`fixed inset-y-0 left-0 z-50 flex w-[260px] flex-col bg-[#0F2E28] text-white transition-transform duration-300 lg:translate-x-0 ${open ? 'translate-x-0' : '-translate-x-full'}`}>
        <div className="flex h-[76px] items-center gap-3 border-b border-white/10 px-5">
          <div className="grid h-11 w-11 place-items-center rounded-2xl bg-white/10" style={{ color: accent }}><BrandIcon size={23}/></div>
          <div className="min-w-0">
            <div className="truncate text-[14px] font-bold">{brand}</div>
            <div className="truncate text-[10px] text-white/55">{subtitle}</div>
          </div>
          <button onClick={onClose} className="ml-auto grid h-8 w-8 place-items-center rounded-lg hover:bg-white/10 lg:hidden"><X size={17}/></button>
        </div>
        <div className="px-4 pt-4">
          <Link to="/project-home" onClick={onClose} className="flex items-center gap-2 rounded-xl px-2 py-2 text-[11px] text-white/60 transition hover:bg-white/5 hover:text-white"><ArrowLeft size={14}/>Back to Project Home</Link>
        </div>
        <nav className="mt-2 flex-1 space-y-1 overflow-y-auto px-3 pb-4">
          {navItems.map(({ to, label, icon: Icon }) => (
            <NavLink key={to} to={to} onClick={onClose} className={({isActive}) => `flex items-center gap-3 rounded-[18px] px-3 py-3 text-[13px] transition ${isActive ? 'bg-[#1A4A3E] font-semibold text-[#2DD4BF]' : 'text-white/72 hover:bg-white/[.055] hover:text-white'}`}>
              <Icon size={17} strokeWidth={1.8}/><span>{label}</span>
            </NavLink>
          ))}
        </nav>
        {assistant && (
          <div className="m-3 rounded-2xl border border-white/5 bg-[#1A4A3E] p-4 shadow-lg">
            <div className="flex items-center gap-2 text-[13px] font-bold"><Bot size={17} style={{ color: accent }}/>AI Planning Assistant</div>
            <p className="mt-2 text-[10px] leading-4 text-white/65">Ask me about heat zones, trends, or interventions.</p>
            <button onClick={() => setAssistantOpen(true)} className="mt-4 w-full rounded-xl px-3 py-2 text-[11px] font-semibold text-[#0F2E28] transition hover:brightness-105" style={{ background: accent }}>Ask Assistant</button>
          </div>
        )}
        {footer && <div className="border-t border-white/10 px-5 py-4 text-[10px] leading-4 text-white/55">{footer}</div>}
      </aside>
    </>
  )
}
