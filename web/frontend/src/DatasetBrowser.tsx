import { useState, useEffect } from 'react'
import { ChevronLeft, ChevronRight } from 'lucide-react'
import { api } from './api'

export function DatasetBrowser({ datasetName, onBack }: { datasetName: string, onBack: () => void }) {
  const [stats, setStats] = useState<any>(null)
  const [samples, setSamples] = useState<any[]>([])
  const [page, setPage] = useState(1)
  const [total, setTotal] = useState(0)

  useEffect(() => {
    api.getDatasetStats(datasetName).then(setStats)
  }, [datasetName])

  useEffect(() => {
    api.getDatasetSamples(datasetName, page).then(res => {
      setSamples(res.samples)
      setTotal(res.total)
    })
  }, [datasetName, page])

  return (
    <div style={{display: 'flex', flexDirection: 'column', height: '100%', gap: '1rem'}}>
      <div className="page-header" style={{marginBottom: 0}}>
        <div style={{display: 'flex', alignItems: 'center', gap: '1rem'}}>
          <button onClick={onBack} style={{padding: '0.5rem'}}><ChevronLeft size={20} /></button>
          <h2 style={{margin: 0}}>{datasetName}</h2>
        </div>
      </div>

      {stats && (
        <div className="grid-cols-2" style={{height: '200px'}}>
          <div className="glass-panel">
            <h3 style={{fontSize: '1rem', color: 'var(--text-muted)'}}>Splits Distribution</h3>
            <div style={{display: 'flex', gap: '2rem', marginTop: '1rem'}}>
              <div><span style={{color: 'var(--accent)'}}>Train:</span> {stats.splits.train}</div>
              <div><span style={{color: 'var(--success)'}}>Val:</span> {stats.splits.val}</div>
              <div><span style={{color: 'var(--warning)'}}>Test:</span> {stats.splits.test}</div>
              <div><span style={{color: 'var(--text-muted)'}}>Unknown:</span> {stats.splits.unknown}</div>
            </div>
          </div>
          <div className="glass-panel" style={{overflowY: 'auto'}}>
            <h3 style={{fontSize: '1rem', color: 'var(--text-muted)'}}>Top 20 Characters</h3>
            <div style={{display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginTop: '1rem'}}>
              {Object.entries(stats.chars).map(([char, count]: any) => (
                <div key={char} style={{background: 'rgba(255,255,255,0.1)', padding: '0.25rem 0.5rem', borderRadius: '4px'}}>
                  <strong style={{color: 'var(--accent)'}}>{char}</strong>: {count}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      <div className="glass-panel" style={{flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column'}}>
        <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem'}}>
          <h3 style={{fontSize: '1rem', color: 'var(--text-muted)', margin: 0}}>Gallery ({total} total)</h3>
          <div style={{display: 'flex', alignItems: 'center', gap: '1rem'}}>
            <button onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page === 1}><ChevronLeft size={16} /></button>
            <span>Page {page}</span>
            <button onClick={() => setPage(p => p + 1)} disabled={page * 50 >= total}><ChevronRight size={16} /></button>
          </div>
        </div>
        
        <div style={{flex: 1, overflowY: 'auto', display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '1rem', alignContent: 'start'}}>
          {samples.map((s, i) => (
            <div key={i} style={{background: 'rgba(0,0,0,0.2)', borderRadius: '8px', overflow: 'hidden', border: '1px solid var(--border-color)'}}>
              <div style={{height: '80px', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#000'}}>
                <img src={api.getDatasetImageUrl(datasetName, s.filename)} style={{maxWidth: '100%', maxHeight: '100%', objectFit: 'contain'}} loading="lazy" />
              </div>
              <div style={{padding: '0.5rem'}}>
                <div style={{fontWeight: 600, color: 'var(--text-main)', fontSize: '1.2rem', textAlign: 'center'}}>{s.label}</div>
                <div style={{display: 'flex', justifyContent: 'space-between', marginTop: '0.5rem', fontSize: '0.75rem', color: 'var(--text-muted)'}}>
                  <span title={s.filename} style={{overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '120px'}}>{s.filename}</span>
                  <span className={`status-badge status-${s.split === 'train' ? 'done' : (s.split === 'val' ? 'running' : 'pending')}`}>{s.split}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
