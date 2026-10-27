import { AlertCircle, LoaderCircle, RefreshCw } from 'lucide-react'
import Button from '../../../components/ui/Button'

export function LoadingState({ message = 'Loading research data…', minHeight = 220 }) {
  return <div className="card grid place-items-center p-8 text-center" style={{ minHeight }}>
    <div><LoaderCircle className="mx-auto animate-spin text-teal-600" size={28}/><p className="mt-3 text-sm font-semibold text-slate-700">{message}</p></div>
  </div>
}

export function ErrorState({ error, onRetry, minHeight = 220 }) {
  const message = typeof error === 'string' ? error : error?.message || 'Something went wrong.'
  return <div className="card grid place-items-center p-8 text-center" style={{ minHeight }}>
    <div className="max-w-xl"><AlertCircle className="mx-auto text-red-500" size={30}/><h3 className="mt-3 font-bold text-slate-900">Unable to load data</h3><p className="mt-2 text-sm leading-6 text-slate-500">{message}</p>{onRetry && <Button className="mt-4" onClick={onRetry}><RefreshCw size={15}/>Retry</Button>}</div>
  </div>
}

export function EmptyState({ title = 'No data available', message = 'No records matched the current selection.', minHeight = 180 }) {
  return <div className="card grid place-items-center p-8 text-center" style={{ minHeight }}><div><h3 className="font-bold text-slate-800">{title}</h3><p className="mt-2 text-sm text-slate-500">{message}</p></div></div>
}
