import { useState, useEffect } from 'react'
import { api } from './api'

export function ErrorGallery({ jobId }: { jobId: number }) {
  const [errors, setErrors] = useState<any[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api.getJobErrors(jobId).then(res => {
      setErrors(res)
      setLoading(false)
    })
  }, [jobId])

  if (loading) return <div>Loading errors...</div>
  if (errors.length === 0) return <div style={{color: 'var(--text-muted)'}}>No errors found or errors.json not available.</div>

  return (
    <div style={{display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: '1rem'}}>
      {errors.map((err, i) => (
        <div key={i} style={{background: 'rgba(0,0,0,0.2)', borderRadius: '8px', overflow: 'hidden', border: '1px solid var(--danger)'}}>
          <div style={{height: '100px', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#000'}}>
            {/* If the error contains a valid path that is served via dataset static mount, we can show it. 
                Assuming err.file points to something like data_1/label_xxx.png. We can try to load it.
                We might need an endpoint to serve arbitrary workspace files securely if they aren't mounted.
                Since we mounted datasets, but test set could be data_test, we can't reliably load unless we have a specific endpoint.
                For now, display the filename and prediction. */}
            <div style={{color: 'var(--text-muted)', fontSize: '0.8rem', padding: '1rem', wordBreak: 'break-all'}}>
              {err.file}
            </div>
          </div>
          <div style={{padding: '1rem'}}>
            <div style={{display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem'}}>
              <span style={{color: 'var(--text-muted)'}}>GT:</span>
              <span style={{color: 'var(--success)', fontWeight: 600}}>{err.gt}</span>
            </div>
            <div style={{display: 'flex', justifyContent: 'space-between'}}>
              <span style={{color: 'var(--text-muted)'}}>Pred:</span>
              <span style={{color: 'var(--danger)', fontWeight: 600}}>{err.pred}</span>
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}
