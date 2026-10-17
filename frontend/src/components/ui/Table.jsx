export default function Table({ columns, rows, onRowClick, empty = 'No matching data.' }) {
  return (
    <div className="table-shell overflow-x-auto">
      <table className="table-base">
        <thead><tr>{columns.map((c) => <th key={c.key}>{c.label}</th>)}</tr></thead>
        <tbody>
          {rows.length === 0 && <tr><td colSpan={columns.length} className="py-10 text-center text-slate-400">{empty}</td></tr>}
          {rows.map((row, idx) => (
            <tr key={row.id || row.zoneId || row.wetlandId || row.sectionId || idx} onClick={() => onRowClick?.(row)} className={onRowClick ? 'cursor-pointer hover:bg-slate-50' : ''}>
              {columns.map((c) => <td key={c.key}>{c.render ? c.render(row[c.key], row) : row[c.key]}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
