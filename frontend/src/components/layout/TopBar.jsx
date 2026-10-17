import { Bell, ChevronDown, Info, Menu, Search, UserRound } from 'lucide-react'
import { useState } from 'react'
import Modal from '../ui/Modal'
import { useApp } from '../../contexts/AppContext'

export default function TopBar({ onMenu, title, subtitle, search, onSearch, searchPlaceholder = 'Search area (e.g. Malabe)', badge, children }) {
  const { profile, setProfile, notifications, markNotificationsRead, showToast } = useApp()
  const [profileOpen, setProfileOpen] = useState(false)
  const [noticeOpen, setNoticeOpen] = useState(false)
  const [draft, setDraft] = useState(profile)
  const unread = notifications.filter((n) => n.unread).length

  const handlePhoto = (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    setDraft((p) => ({ ...p, photo: URL.createObjectURL(file) }))
  }

  return (
    <>
      <header className="fixed inset-x-0 top-0 z-30 h-[76px] border-b border-slate-200 bg-white/95 backdrop-blur lg:left-[260px]">
        <div className="flex h-full items-center gap-3 px-4 md:px-5 lg:px-6">
          <button onClick={onMenu} className="grid h-10 w-10 shrink-0 place-items-center rounded-xl border border-slate-200 lg:hidden"><Menu size={19}/></button>
          <div className="min-w-0 flex-1">
            <div className="truncate text-[15px] font-bold text-slate-900 md:text-[17px]">{title}</div>
            {subtitle && <div className="mt-0.5 hidden truncate text-[11px] text-slate-500 sm:block">{subtitle}</div>}
          </div>
          {children}
          {search !== undefined && (
            <div className="relative hidden w-[280px] xl:block">
              <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
              <input value={search} onChange={(e) => onSearch?.(e.target.value)} placeholder={searchPlaceholder} className="w-full rounded-xl border border-slate-200 bg-slate-50 py-2.5 pl-9 pr-9 text-[12px] outline-none focus:border-teal-400 focus:bg-white" />
              <div className="group absolute right-3 top-1/2 -translate-y-1/2"><Info size={14} className="text-slate-400"/><div className="pointer-events-none absolute right-0 top-6 hidden w-64 rounded-xl bg-slate-900 p-3 text-[10px] leading-4 text-white shadow-xl group-hover:block">Areas are matched to demo grid zones using prototype locality mappings. Search by locality name instead of grid ID.</div></div>
            </div>
          )}
          {badge && <div className="hidden rounded-full border border-amber-200 bg-amber-50 px-3 py-1.5 text-[11px] font-medium text-amber-700 2xl:block">{badge}</div>}
          <div className="group relative">
            <button onClick={() => { setNoticeOpen((v) => !v); markNotificationsRead() }} className="relative grid h-10 w-10 place-items-center rounded-xl border border-slate-200 bg-white hover:bg-slate-50">
              <Bell size={17}/>{unread > 0 && <span className="absolute right-2 top-2 h-2 w-2 rounded-full bg-red-500 ring-2 ring-white"/>}
            </button>
            {!noticeOpen && notifications[0] && <div className="pointer-events-none absolute right-0 top-11 hidden w-64 rounded-xl border border-slate-200 bg-white p-3 text-[11px] shadow-xl group-hover:block"><div className="font-semibold text-slate-800">{notifications[0].title}</div><div className="mt-1 text-slate-500">{notifications[0].body}</div></div>}
            {noticeOpen && <div className="absolute right-0 top-12 z-50 w-[320px] overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-2xl"><div className="border-b border-slate-100 px-4 py-3 text-sm font-bold">Notifications</div>{notifications.map((n)=><div key={n.id} className="border-b border-slate-100 px-4 py-3 last:border-0"><div className="text-[12px] font-semibold text-slate-800">{n.title}</div><div className="mt-1 text-[11px] leading-4 text-slate-500">{n.body}</div><div className="mt-1 text-[10px] text-slate-400">{n.time}</div></div>)}</div>}
          </div>
          <button onClick={() => { setDraft(profile); setProfileOpen(true) }} className="hidden items-center gap-2 rounded-xl border border-slate-200 bg-white px-2.5 py-1.5 text-left hover:bg-slate-50 sm:flex">
            <div className="grid h-8 w-8 place-items-center overflow-hidden rounded-lg bg-[#0F2E28] text-white">{profile.photo ? <img src={profile.photo} className="h-full w-full object-cover"/> : <UserRound size={16}/>}</div>
            <div className="max-w-[130px]"><div className="truncate text-[11px] font-semibold text-slate-800">{profile.name}</div><div className="truncate text-[9px] text-slate-400">{profile.role}</div></div><ChevronDown size={13} className="text-slate-400"/>
          </button>
        </div>
      </header>

      <Modal open={profileOpen} onClose={() => setProfileOpen(false)} title="Edit prototype profile">
        <div className="grid gap-4 md:grid-cols-2">
          <label className="text-xs font-medium text-slate-600">Name<input className="control mt-1" value={draft.name} onChange={(e)=>setDraft({...draft,name:e.target.value})}/></label>
          <label className="text-xs font-medium text-slate-600">Role<input className="control mt-1" value={draft.role} onChange={(e)=>setDraft({...draft,role:e.target.value})}/></label>
          <label className="text-xs font-medium text-slate-600">Organization<input className="control mt-1" value={draft.organization} onChange={(e)=>setDraft({...draft,organization:e.target.value})}/></label>
          <label className="text-xs font-medium text-slate-600">Email<input className="control mt-1" value={draft.email} onChange={(e)=>setDraft({...draft,email:e.target.value})}/></label>
          <label className="text-xs font-medium text-slate-600 md:col-span-2">Profile photo<input type="file" accept="image/*" onChange={handlePhoto} className="mt-2 block w-full text-xs text-slate-500 file:mr-3 file:rounded-lg file:border-0 file:bg-slate-100 file:px-3 file:py-2 file:text-xs file:font-semibold"/></label>
        </div>
        <div className="mt-5 flex justify-end gap-2"><button onClick={()=>setProfileOpen(false)} className="btn-secondary">Cancel</button><button onClick={()=>{setProfile(draft);setProfileOpen(false);showToast('Profile updated for this prototype session.')}} className="btn-primary">Save changes</button></div>
      </Modal>
    </>
  )
}
