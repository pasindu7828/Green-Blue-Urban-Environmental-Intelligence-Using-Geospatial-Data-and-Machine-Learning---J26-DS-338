import { CheckCircle2, Info } from 'lucide-react'
import { useApp } from '../../contexts/AppContext'
export default function Toast() {
  const { toast } = useApp()
  if (!toast) return null
  const Icon = toast.type === 'success' ? CheckCircle2 : Info
  return <div className="fixed bottom-6 right-6 z-[130] flex max-w-sm items-center gap-3 rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm font-medium text-slate-700 shadow-2xl"><Icon size={19} className={toast.type === 'success' ? 'text-emerald-600' : 'text-sky-600'}/><span>{toast.message}</span></div>
}
