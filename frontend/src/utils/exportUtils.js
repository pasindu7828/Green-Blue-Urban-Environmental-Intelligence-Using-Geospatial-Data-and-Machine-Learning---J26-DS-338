import { jsPDF } from 'jspdf'
import autoTable from 'jspdf-autotable'
import * as XLSX from 'xlsx'

function normalizeRows(rows){
  return rows.map(row=>Object.fromEntries(Object.entries(row).map(([k,v])=>[k,typeof v==='object' ? JSON.stringify(v) : v])))
}

export function exportCSV(rows, filename='export.csv'){
  if(!rows?.length) return
  const data=normalizeRows(rows)
  const headers=Object.keys(data[0])
  const esc=v=>`"${String(v??'').replaceAll('"','""')}"`
  const csv=[headers.map(esc).join(','),...data.map(row=>headers.map(h=>esc(row[h])).join(','))].join('\n')
  const blob=new Blob([csv],{type:'text/csv;charset=utf-8;'})
  const url=URL.createObjectURL(blob)
  const a=document.createElement('a');a.href=url;a.download=filename;a.click();URL.revokeObjectURL(url)
}

export function exportExcel(rows, filename='export.xlsx', sheetName='Data'){
  if(!rows?.length) return
  const ws=XLSX.utils.json_to_sheet(normalizeRows(rows))
  const wb=XLSX.utils.book_new();XLSX.utils.book_append_sheet(wb,ws,sheetName.slice(0,31));XLSX.writeFile(wb,filename)
}

export function exportPDF({title, subtitle='Prototype demo data', rows=[], columns, filename='report.pdf', summary=[]}){
  const doc=new jsPDF({orientation:'landscape'})
  doc.setFontSize(17);doc.text(title,14,16)
  doc.setFontSize(9);doc.setTextColor(95);doc.text(subtitle,14,23)
  let y=30
  if(summary.length){doc.setTextColor(40);doc.setFontSize(10);summary.forEach(line=>{doc.text(String(line),14,y);y+=6})}
  const cols=columns?.length ? columns : (rows[0] ? Object.keys(rows[0]).slice(0,8).map(k=>({header:k,dataKey:k})) : [])
  autoTable(doc,{startY:y+3,head:[cols.map(c=>c.header||c.label||c.dataKey)],body:rows.slice(0,250).map(r=>cols.map(c=>r[c.dataKey||c.key])),styles:{fontSize:7,cellPadding:2},headStyles:{fillColor:[15,46,40]},alternateRowStyles:{fillColor:[246,249,248]}})
  doc.setFontSize(8);doc.setTextColor(110);doc.text('Research proposal prototype — values shown are simulated/demo outputs, not final validated results.',14,doc.internal.pageSize.height-8)
  doc.save(filename)
}
