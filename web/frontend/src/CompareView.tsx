import { useState, useEffect } from 'react'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend } from 'recharts'
import { api } from './api'

export function CompareView() {
  const [jobs, setJobs] = useState<any[]>([])
  const [selectedIds, setSelectedIds] = useState<number[]>([])
  const [metricsData, setMetricsData] = useState<any[]>([])

  useEffect(() => {
    api.getJobs().then(setJobs)
  }, [])

  useEffect(() => {
    async function loadMetrics() {
      const epochMap: Record<number, any> = {}
      for (const id of selectedIds) {
        const hist = await api.getJobMetricsHistory(id)
        for (const m of hist) {
          if (!epochMap[m.epoch]) epochMap[m.epoch] = { epoch: m.epoch }
          epochMap[m.epoch][`job_${id}_loss`] = m.val_loss
          epochMap[m.epoch][`job_${id}_acc`] = m.val_seq_acc
        }
      }
      setMetricsData(Object.values(epochMap).sort((a: any, b: any) => a.epoch - b.epoch))
    }
    loadMetrics()
  }, [selectedIds])

  const toggleJob = (id: number) => {
    setSelectedIds(prev => prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id])
  }
  
  const colors = ['#6366f1', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']

  return (
    <div style={{height: '100%', display: 'flex', flexDirection: 'column'}}>
      <div className="page-header" style={{marginBottom: '1rem'}}>
        <h2>Compare Runs</h2>
      </div>
      <div className="glass-panel" style={{marginBottom: '1rem', padding: '1rem'}}>
        <div style={{display: 'flex', gap: '1rem', flexWrap: 'wrap'}}>
          {jobs.filter(j => j.status === 'done' || j.status === 'running').map(job => (
            <label key={job.id} style={{display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer', background: 'rgba(255,255,255,0.05)', padding: '0.5rem 1rem', borderRadius: '8px'}}>
              <input type="checkbox" checked={selectedIds.includes(job.id)} onChange={() => toggleJob(job.id)} />
              <span style={{fontWeight: 600}}>#{job.id}</span> {job.name}
            </label>
          ))}
          {jobs.length === 0 && <span style={{color: 'var(--text-muted)'}}>No completed jobs to compare.</span>}
        </div>
      </div>
      
      {selectedIds.length > 0 ? (
        <div className="grid-cols-2" style={{flex: 1, minHeight: 0}}>
          <div className="glass-panel" style={{display: 'flex', flexDirection: 'column'}}>
            <h3 style={{fontSize: '1rem', color: 'var(--text-muted)', marginBottom: '1rem'}}>Validation Loss</h3>
            <div style={{flex: 1, minHeight: 0}}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={metricsData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />
                  <XAxis dataKey="epoch" stroke="var(--text-muted)" />
                  <YAxis stroke="var(--text-muted)" />
                  <Tooltip contentStyle={{background: 'var(--bg-main)', borderColor: 'var(--border-color)', borderRadius: '8px'}} />
                  <Legend />
                  {selectedIds.map((id, i) => (
                    <Line key={id} type="monotone" dataKey={`job_${id}_loss`} name={`Job ${id}`} stroke={colors[i % colors.length]} strokeWidth={2} dot={false} />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div className="glass-panel" style={{display: 'flex', flexDirection: 'column'}}>
            <h3 style={{fontSize: '1rem', color: 'var(--text-muted)', marginBottom: '1rem'}}>Validation Sequence Accuracy</h3>
            <div style={{flex: 1, minHeight: 0}}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={metricsData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />
                  <XAxis dataKey="epoch" stroke="var(--text-muted)" />
                  <YAxis stroke="var(--text-muted)" domain={[0, 1]} />
                  <Tooltip contentStyle={{background: 'var(--bg-main)', borderColor: 'var(--border-color)', borderRadius: '8px'}} />
                  <Legend />
                  {selectedIds.map((id, i) => (
                    <Line key={id} type="monotone" dataKey={`job_${id}_acc`} name={`Job ${id}`} stroke={colors[i % colors.length]} strokeWidth={2} dot={false} />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      ) : (
        <div style={{flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)'}}>
          Select jobs above to compare metrics.
        </div>
      )}
    </div>
  )
}
