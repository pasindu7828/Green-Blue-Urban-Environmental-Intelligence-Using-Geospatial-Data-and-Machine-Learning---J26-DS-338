import { X } from 'lucide-react'
export default function Modal({ open, onClose, title, children, width = 'max-w-2xl' }) {
  if (!open) return null
  return (
    <div className="fixed inset-0 z-[100] grid place-items-center bg-slate-950/35 p-4 backdrop-blur-[2px]" onMouseDown={onClose}>
      <div className={`max-h-[90vh] w-full ${width} overflow-auto rounded-2xl bg-white shadow-2xl`} onMouseDown={(e) => e.stopPropagation()}>
        <div className="sticky top-0 z-10 flex items-center justify-between border-b border-slate-100 bg-white px-5 py-4">
          <h2 className="text-lg font-bold text-slate-900">{title}</h2>
          <button onClick={onClose} className="grid h-9 w-9 place-items-center rounded-xl hover:bg-slate-100"><X size={18}/></button>
        </div>
        <div className="p-5">{children}</div>
      </div>
    </div>
  )
}
