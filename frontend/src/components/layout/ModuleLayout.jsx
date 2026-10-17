import { useState } from 'react'
import { Outlet } from 'react-router-dom'
import ModuleSidebar from './ModuleSidebar'
import TopBar from './TopBar'

export default function ModuleLayout({ navItems, brand, subtitle, icon, headerTitle, headerSubtitle, footer, assistant, accent, searchable = false, headerExtras = null }) {
  const [open, setOpen] = useState(false)
  const [search, setSearch] = useState('')
  return (
    <div className="min-h-screen bg-[#F5F7FA]">
      <ModuleSidebar open={open} onClose={()=>setOpen(false)} navItems={navItems} brand={brand} subtitle={subtitle} icon={icon} footer={footer} assistant={assistant} accent={accent}/>
      <TopBar onMenu={()=>setOpen(true)} title={headerTitle} subtitle={headerSubtitle} search={searchable ? search : undefined} onSearch={setSearch}>{headerExtras}</TopBar>
      <main className="min-h-screen pt-[76px] lg:pl-[260px]">
        <div className="mx-auto max-w-[1540px] px-4 py-6 md:px-6 md:py-7 lg:px-7"><Outlet context={{ globalSearch: search, setGlobalSearch: setSearch }}/></div>
      </main>
    </div>
  )
}
