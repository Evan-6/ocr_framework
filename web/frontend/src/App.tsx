import { useState, useEffect, useRef } from 'react'
import { Activity, Database, Play, Square, UploadCloud, RefreshCw, ChevronLeft, Plus, BarChart2, Zap, Image as ImageIcon } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'
import { api } from './api'
import { CompareView } from './CompareView'
import { DatasetBrowser } from './DatasetBrowser'
import { AugPreview } from './AugPreview'
import { ErrorGallery } from './ErrorGallery'
import { PlaygroundView } from './PlaygroundView'

function StatusBadge({ status }: { status: string }) {
  return <span className={`status-badge status-${status}`}>{status}</span>
}

function JobsView({ onSelectJob }: { onSelectJob: (id: number) => void }) {
  const [jobs, setJobs] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [showCreate, setShowCreate] = useState(false)

  const fetchJobs = async () => {
    try {
      setJobs(await api.getJobs())
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchJobs()
    const interval = setInterval(fetchJobs, 5000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div style={{height: '100%', display: 'flex', flexDirection: 'column'}}>
      <div className="page-header">
        <h2>Training Jobs</h2>
        <div style={{display: 'flex', gap: '1rem'}}>
          <button onClick={fetchJobs}><RefreshCw size={16} /></button>
          <button className="primary" onClick={() => setShowCreate(true)}>
            <Plus size={16} style={{marginRight: '0.5rem', verticalAlign: 'middle'}}/>
            New Job
          </button>
        </div>
      </div>

      {showCreate && <CreateJobModal onClose={() => setShowCreate(false)} onCreated={fetchJobs} />}

      <div className="glass-panel" style={{flex: 1, overflowY: 'auto'}}>
        {loading ? <p>Loading...</p> : (
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>Name</th>
                <th>Status</th>
                <th>Created</th>
                <th>Validation Acc</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {jobs.map(job => (
                <tr key={job.id}>
                  <td>#{job.id}</td>
                  <td style={{fontWeight: 500}}>{job.name}</td>
                  <td><StatusBadge status={job.status} /></td>
                  <td>{new Date(job.created_at).toLocaleString()}</td>
                  <td>
                    {job.metrics?.val?.seq_acc !== undefined
                      ? (job.metrics.val.seq_acc * 100).toFixed(2) + '%'
                      : '-'}
                  </td>
                  <td style={{display: 'flex', gap: '0.5rem'}}>
                    <button onClick={() => onSelectJob(job.id)}>View Details</button>
                    <button
                      onClick={async () => {
                        if (!confirm(`Delete job #${job.id}? This removes its run folder.`)) return
                        try { await api.deleteJob(job.id); fetchJobs() }
                        catch (e: any) { alert(e.message) }
                      }}
                      style={{color: 'var(--danger)', borderColor: 'var(--danger)'}}
                    >Delete</button>
                  </td>
                </tr>
              ))}
              {jobs.length === 0 && (
                <tr><td colSpan={6} style={{textAlign: 'center'}}>No jobs found.</td></tr>
              )}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}

const PRESETS: Record<string, any> = {
  "Custom": {},
  "Arrow 5MB (color, stretch)": {
    channels: 3, resize_mode: "stretch", img_h: 64, img_w: 224,
    stage_channels: "32,64,96,160", hidden: 112, lstm_layers: 2,
    aug_rotate: 0, aug_shear: 0, aug_translate: 0.04, aug_scale: 0.06,
    aug_photometric: true, select_metric: "val_loss", early_stop_patience: 150,
  },
  "Arrow 10MB (color, stretch)": {
    channels: 3, resize_mode: "stretch", img_h: 64, img_w: 224,
    stage_channels: "48,96,160,224", hidden: 160, lstm_layers: 2,
    aug_rotate: 0, aug_shear: 0, aug_translate: 0.04, aug_scale: 0.06,
    aug_photometric: true, select_metric: "val_loss", early_stop_patience: 150,
  },
  "Text grayscale (full)": {
    channels: 1, resize_mode: "pad", img_h: 48, img_w: 224,
    stage_channels: "64,128,256,512", hidden: 256, lstm_layers: 2,
    aug_rotate: 4, aug_shear: 8, aug_translate: 0.06, aug_scale: 0.1,
    aug_photometric: true, select_metric: "val_loss", early_stop_patience: 40,
  },
}

function CreateJobModal({ onClose, onCreated }: { onClose: () => void, onCreated: () => void }) {
  const [datasets, setDatasets] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [preset, setPreset] = useState("Custom")

  const [cfg, setCfg] = useState({
    name: "experiment_1",
    data_dir: "data",
    epochs: 200,
    batch_size: 64,
    lr: 0.001,
    val_ratio: 0.1,
    test_ratio: 0.1,
    select_metric: "val_loss",
    early_stop_patience: 40,
    img_h: 48,
    img_w: 224,
    channels: 1,
    resize_mode: "pad",
    stage_channels: "64,128,256,512",
    hidden: 256,
    lstm_layers: 2,
    augment: true,
    aug_rotate: 4.0,
    aug_translate: 0.06,
    aug_scale: 0.1,
    aug_shear: 8.0,
    aug_photometric: true
  })

  const applyPreset = (p: string) => {
    setPreset(p)
    setCfg(c => ({ ...c, ...PRESETS[p] }))
  }

  useEffect(() => {
    api.getDatasets().then(res => {
      setDatasets(res)
      if (res.length > 0 && cfg.data_dir === "data") {
        setCfg(c => ({ ...c, data_dir: res[0].path }))
      }
    })
  }, [])

  const handleChange = (k: string, v: any) => setCfg(c => ({ ...c, [k]: v }))

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    
    // Split name and overrides; stage_channels is a "a,b,c,d" string -> number[]
    const { name, stage_channels, ...rest } = cfg
    const parsed = stage_channels.split(",").map(s => parseInt(s.trim())).filter(n => !isNaN(n))
    if (parsed.length !== 4) {
      alert("Stage Channels must be 4 comma-separated integers, e.g. 64,128,256,512")
      setLoading(false)
      return
    }
    const overrides = {
      ...rest,
      stage_channels: parsed,
      select_mode: rest.select_metric === "val_loss" ? "min" : "max",
    }

    try {
      await api.createJob({
        name,
        config_file: "", // use default base config
        config_overrides: JSON.stringify(overrides)
      })
      onCreated()
      onClose()
    } catch (err: any) {
      alert("Error creating job: " + err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.5)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 100 }}>
      <div className="glass-panel" style={{ width: '600px', maxHeight: '90vh', display: 'flex', flexDirection: 'column', background: 'var(--bg-main)' }}>
        <h3 style={{marginBottom: '1.5rem'}}>Create New Job</h3>
        <form onSubmit={handleSubmit} style={{display: 'flex', flexDirection: 'column', gap: '1rem', flex: 1, overflowY: 'auto', paddingRight: '1rem'}}>
          
          <div>
            <label style={{display: 'block', marginBottom: '0.5rem', color: 'var(--text-muted)'}}>Preset</label>
            <select value={preset} onChange={e => applyPreset(e.target.value)}>
              {Object.keys(PRESETS).map(p => <option key={p} value={p}>{p}</option>)}
            </select>
          </div>

          <div className="grid-cols-2">
            <div>
              <label style={{display: 'block', marginBottom: '0.5rem', color: 'var(--text-muted)'}}>Job Name</label>
              <input value={cfg.name} onChange={e => handleChange('name', e.target.value)} required />
            </div>
            <div>
              <label style={{display: 'block', marginBottom: '0.5rem', color: 'var(--text-muted)'}}>Dataset</label>
              <select value={cfg.data_dir} onChange={e => handleChange('data_dir', e.target.value)} required>
                <option value="data" disabled>Select Dataset...</option>
                {datasets.map(d => <option key={d.id} value={d.path}>{d.name}</option>)}
              </select>
            </div>
          </div>

          <h4 style={{color: 'var(--accent)', marginTop: '0.5rem', marginBottom: 0}}>Split & Selection</h4>
          <div className="grid-cols-2">
            <div>
              <label style={{display: 'block', marginBottom: '0.5rem', color: 'var(--text-muted)'}}>Val Ratio</label>
              <input type="number" step="0.01" min="0" max="0.9" value={cfg.val_ratio} onChange={e => handleChange('val_ratio', parseFloat(e.target.value))} required />
            </div>
            <div>
              <label style={{display: 'block', marginBottom: '0.5rem', color: 'var(--text-muted)'}}>Test Ratio</label>
              <input type="number" step="0.01" min="0" max="0.9" value={cfg.test_ratio} onChange={e => handleChange('test_ratio', parseFloat(e.target.value))} required />
            </div>
            <div>
              <label style={{display: 'block', marginBottom: '0.5rem', color: 'var(--text-muted)'}}>Select Metric</label>
              <select value={cfg.select_metric} onChange={e => handleChange('select_metric', e.target.value)}>
                <option value="val_loss">val_loss (min)</option>
                <option value="seq_acc">seq_acc (max)</option>
                <option value="char_acc">char_acc (max)</option>
              </select>
            </div>
            <div>
              <label style={{display: 'block', marginBottom: '0.5rem', color: 'var(--text-muted)'}}>Early Stop Patience</label>
              <input type="number" min="1" value={cfg.early_stop_patience} onChange={e => handleChange('early_stop_patience', parseInt(e.target.value))} required />
            </div>
          </div>

          <h4 style={{color: 'var(--accent)', marginTop: '0.5rem', marginBottom: 0}}>Training</h4>
          <div className="grid-cols-2">
            <div>
              <label style={{display: 'block', marginBottom: '0.5rem', color: 'var(--text-muted)'}}>Epochs</label>
              <input type="number" min="1" value={cfg.epochs} onChange={e => handleChange('epochs', parseInt(e.target.value))} required />
            </div>
            <div>
              <label style={{display: 'block', marginBottom: '0.5rem', color: 'var(--text-muted)'}}>Batch Size</label>
              <input type="number" min="1" value={cfg.batch_size} onChange={e => handleChange('batch_size', parseInt(e.target.value))} required />
            </div>
            <div>
              <label style={{display: 'block', marginBottom: '0.5rem', color: 'var(--text-muted)'}}>Learning Rate</label>
              <input type="number" step="0.0001" value={cfg.lr} onChange={e => handleChange('lr', parseFloat(e.target.value))} required />
            </div>
          </div>

          <h4 style={{color: 'var(--accent)', marginTop: '0.5rem', marginBottom: 0}}>Image Processing</h4>
          <div className="grid-cols-2">
            <div>
              <label style={{display: 'block', marginBottom: '0.5rem', color: 'var(--text-muted)'}}>Image Height</label>
              <input type="number" step="16" value={cfg.img_h} onChange={e => handleChange('img_h', parseInt(e.target.value))} required />
            </div>
            <div>
              <label style={{display: 'block', marginBottom: '0.5rem', color: 'var(--text-muted)'}}>Image Width</label>
              <input type="number" step="4" value={cfg.img_w} onChange={e => handleChange('img_w', parseInt(e.target.value))} required />
            </div>
            <div>
              <label style={{display: 'block', marginBottom: '0.5rem', color: 'var(--text-muted)'}}>Channels</label>
              <select value={cfg.channels} onChange={e => handleChange('channels', parseInt(e.target.value))}>
                <option value={1}>1 (Grayscale)</option>
                <option value={3}>3 (RGB)</option>
              </select>
            </div>
            <div>
              <label style={{display: 'block', marginBottom: '0.5rem', color: 'var(--text-muted)'}}>Resize Mode</label>
              <select value={cfg.resize_mode} onChange={e => handleChange('resize_mode', e.target.value)}>
                <option value="pad">Pad (Maintain Aspect Ratio)</option>
                <option value="stretch">Stretch (Ignore Aspect Ratio)</option>
              </select>
            </div>
          </div>

          <h4 style={{color: 'var(--accent)', marginTop: '0.5rem', marginBottom: 0}}>Model Size</h4>
          <div className="grid-cols-2">
            <div style={{gridColumn: '1 / -1'}}>
              <label style={{display: 'block', marginBottom: '0.5rem', color: 'var(--text-muted)'}}>Stage Channels (4 ints, comma-separated)</label>
              <input value={cfg.stage_channels} onChange={e => handleChange('stage_channels', e.target.value)} placeholder="64,128,256,512" required />
              <div className="muted-label" style={{marginTop: '0.25rem'}}>64,128,256,512 ≈ 35MB · 48,96,160,224 ≈ 10MB · 32,64,96,160 ≈ 5MB</div>
            </div>
            <div>
              <label style={{display: 'block', marginBottom: '0.5rem', color: 'var(--text-muted)'}}>LSTM Hidden</label>
              <input type="number" min="16" step="16" value={cfg.hidden} onChange={e => handleChange('hidden', parseInt(e.target.value))} required />
            </div>
            <div>
              <label style={{display: 'block', marginBottom: '0.5rem', color: 'var(--text-muted)'}}>LSTM Layers</label>
              <input type="number" min="1" max="4" value={cfg.lstm_layers} onChange={e => handleChange('lstm_layers', parseInt(e.target.value))} required />
            </div>
          </div>

          <h4 style={{color: 'var(--accent)', marginTop: '0.5rem', marginBottom: 0}}>Augmentation</h4>
          <div className="grid-cols-2">
            <div>
              <label style={{display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem', color: 'var(--text-muted)', cursor: 'pointer'}}>
                <input type="checkbox" checked={cfg.augment} onChange={e => handleChange('augment', e.target.checked)} style={{width: 'auto'}} />
                Enable Augmentation
              </label>
            </div>
            <div>
              <label style={{display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem', color: 'var(--text-muted)', cursor: 'pointer'}}>
                <input type="checkbox" checked={cfg.aug_photometric} onChange={e => handleChange('aug_photometric', e.target.checked)} disabled={!cfg.augment} style={{width: 'auto'}} />
                Photometric Noise
              </label>
            </div>
            <div>
              <label style={{display: 'block', marginBottom: '0.5rem', color: 'var(--text-muted)'}}>Rotation (deg)</label>
              <input type="number" step="0.1" value={cfg.aug_rotate} onChange={e => handleChange('aug_rotate', parseFloat(e.target.value))} disabled={!cfg.augment} required />
            </div>
            <div>
              <label style={{display: 'block', marginBottom: '0.5rem', color: 'var(--text-muted)'}}>Translation (frac)</label>
              <input type="number" step="0.01" value={cfg.aug_translate} onChange={e => handleChange('aug_translate', parseFloat(e.target.value))} disabled={!cfg.augment} required />
            </div>
            <div>
              <label style={{display: 'block', marginBottom: '0.5rem', color: 'var(--text-muted)'}}>Scale (frac)</label>
              <input type="number" step="0.01" value={cfg.aug_scale} onChange={e => handleChange('aug_scale', parseFloat(e.target.value))} disabled={!cfg.augment} required />
            </div>
            <div>
              <label style={{display: 'block', marginBottom: '0.5rem', color: 'var(--text-muted)'}}>Shear (deg)</label>
              <input type="number" step="0.1" value={cfg.aug_shear} onChange={e => handleChange('aug_shear', parseFloat(e.target.value))} disabled={!cfg.augment} required />
            </div>
          </div>

          <div style={{display: 'flex', justifyContent: 'flex-end', gap: '1rem', marginTop: '1rem', paddingTop: '1rem', borderTop: '1px solid var(--border-color)'}}>
            <button type="button" onClick={onClose}>Cancel</button>
            <button type="submit" className="primary" disabled={loading || datasets.length === 0}>
              {loading ? 'Creating...' : 'Create & Queue'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

function FinalSummary({ metrics }: { metrics: any }) {
  if (!metrics || (!metrics.val && !metrics.test)) return null
  const pct = (x: number | undefined) => x === undefined ? '-' : (x * 100).toFixed(2) + '%'
  const Row = ({ label, m, color }: { label: string, m: any, color: string }) => (
    <div style={{ flex: 1 }}>
      <div style={{ color, fontWeight: 700, marginBottom: '0.5rem' }}>{label} <span style={{ color: 'var(--text-muted)', fontWeight: 400 }}>(n={m.n})</span></div>
      <div style={{ display: 'flex', gap: '1.5rem' }}>
        <div><div className="muted-label">Seq Acc</div><div style={{ fontSize: '1.4rem', fontWeight: 700 }}>{pct(m.seq_acc)}</div></div>
        <div><div className="muted-label">Char Acc</div><div style={{ fontSize: '1.4rem', fontWeight: 700 }}>{pct(m.char_acc)}</div></div>
        <div><div className="muted-label">Loss</div><div style={{ fontSize: '1.4rem', fontWeight: 700 }}>{m.loss?.toFixed(3) ?? '-'}</div></div>
      </div>
    </div>
  )
  const test = metrics.test
  return (
    <div className="glass-panel" style={{ marginBottom: '1rem' }}>
      <h3 style={{ fontSize: '1rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
        Final Results — best checkpoint @ epoch {metrics.best_epoch}
      </h3>
      <div style={{ display: 'flex', gap: '2rem' }}>
        {metrics.val && <Row label="Validation" m={metrics.val} color="var(--success)" />}
        {test && <Row label="Test (held-out)" m={test} color="var(--warning)" />}
      </div>
      {test?.per_position_acc && Object.keys(test.per_position_acc).length > 0 && (
        <div style={{ marginTop: '1rem', display: 'flex', gap: '2rem', flexWrap: 'wrap' }}>
          <div>
            <div className="muted-label">Per-position accuracy</div>
            <div style={{ display: 'flex', gap: '0.75rem', marginTop: '0.25rem' }}>
              {Object.entries(test.per_position_acc).map(([pos, acc]: any) => (
                <span key={pos}>pos {pos}: <strong>{(acc * 100).toFixed(1)}%</strong></span>
              ))}
            </div>
          </div>
          {test.confusion && Object.keys(test.confusion).length > 0 && (
            <div>
              <div className="muted-label">Confusions (gt → pred)</div>
              <div style={{ display: 'flex', gap: '0.75rem', marginTop: '0.25rem', flexWrap: 'wrap' }}>
                {Object.entries(test.confusion).flatMap(([gt, preds]: any) =>
                  Object.entries(preds).map(([p, c]: any) => (
                    <span key={gt + p} style={{ color: 'var(--danger)' }}>{gt}→{p} ×{c}</span>
                  )))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function JobDetailView({ jobId, onBack }: { jobId: number, onBack: () => void }) {
  const [job, setJob] = useState<any>(null)
  const [metrics, setMetrics] = useState<any[]>([])
  const [logs, setLogs] = useState<string[]>([])
  const [tab, setTab] = useState<'metrics'|'errors'>('metrics')
  const [exporting, setExporting] = useState(false)
  const [exportResult, setExportResult] = useState<any>(null)
  const logEndRef = useRef<HTMLDivElement>(null)

  const handleExport = async () => {
    setExporting(true)
    try {
      setExportResult(await api.exportJob(jobId))
    } catch (e: any) {
      alert(e.message)
    } finally {
      setExporting(false)
    }
  }

  useEffect(() => {
    api.getJob(jobId).then(setJob)
    // refetch job so status + final metrics update when training finishes
    const poll = setInterval(() => api.getJob(jobId).then(setJob), 5000)

    const source = new EventSource(api.getEventStreamUrl(jobId))
    
    source.addEventListener('metrics', (e) => {
      const data = JSON.parse(e.data)
      setMetrics(prev => {
        if (prev.find(m => m.epoch === data.epoch)) return prev;
        return [...prev, data]
      })
    })

    source.addEventListener('log', (e) => {
      const data = JSON.parse(e.data)
      setLogs(prev => [...prev.slice(-100), data])
    })

    return () => { source.close(); clearInterval(poll) }
  }, [jobId])

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs])

  const handleStop = async () => {
    if (confirm("Are you sure you want to stop this job?")) {
      await api.stopJob(jobId)
      api.getJob(jobId).then(setJob)
    }
  }

  if (!job) return <div>Loading...</div>

  return (
    <div style={{display: 'flex', flexDirection: 'column', height: '100%', gap: '1.5rem'}}>
      <div className="page-header" style={{marginBottom: 0}}>
        <div style={{display: 'flex', alignItems: 'center', gap: '1rem'}}>
          <button onClick={onBack} style={{padding: '0.5rem'}}><ChevronLeft size={20} /></button>
          <h2 style={{margin: 0}}>Job #{job.id}: {job.name}</h2>
          <StatusBadge status={job.status} />
        </div>
        {job.status === 'running' && (
          <button onClick={handleStop} style={{background: 'rgba(239, 68, 68, 0.1)', color: 'var(--danger)', borderColor: 'var(--danger)'}}>
            <Square size={16} style={{marginRight: '0.5rem', verticalAlign: 'middle'}}/> Stop
          </button>
        )}
      </div>

      <div style={{display: 'flex', gap: '1rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '1rem'}}>
        <button className={tab === 'metrics' ? 'primary' : ''} onClick={() => setTab('metrics')}>Live Metrics</button>
        {(job.status === 'done' || job.status === 'stopped' || job.status === 'failed') && (
          <button className={tab === 'errors' ? 'primary' : ''} onClick={() => setTab('errors')}>Error Gallery</button>
        )}
      </div>

      {job.status === 'done' && (
        <div className="glass-panel" style={{padding: '1rem'}}>
          <div style={{display: 'flex', alignItems: 'center', gap: '1.5rem', flexWrap: 'wrap'}}>
            <button className="primary" onClick={handleExport} disabled={exporting}>
              {exporting ? 'Exporting…' : 'Export ONNX + C++ Header'}
            </button>
            {exportResult && (
              <>
                <span>Size: <strong>{exportResult.onnx_size_mb} MB</strong></span>
                <span style={{color: exportResult.parity_ok ? 'var(--success)' : 'var(--danger)'}}>
                  Parity: {exportResult.parity_ok ? 'OK' : 'FAILED'}
                </span>
                <a href={api.getDownloadUrl(jobId, 'onnx')} download>model.onnx</a>
                <a href={api.getDownloadUrl(jobId, 'meta')} download>model_meta.json</a>
                <a href={api.getDownloadUrl(jobId, 'header')} download>ocr_model_config.h</a>
              </>
            )}
          </div>
        </div>
      )}

      {tab === 'metrics' ? (
        <div style={{flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column'}}>
        <FinalSummary metrics={job.metrics} />
        <div className="grid-cols-2" style={{flex: 1, minHeight: 0}}>
          <div className="glass-panel" style={{display: 'flex', flexDirection: 'column'}}>
            <h3 style={{fontSize: '1rem', color: 'var(--text-muted)'}}>Loss & Accuracy Curves</h3>
            <div style={{flex: 1, minHeight: 0, marginTop: '1rem'}}>
              <ResponsiveContainer width="100%" height="50%">
                <LineChart data={metrics}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />
                  <XAxis dataKey="epoch" stroke="var(--text-muted)" fontSize={12} />
                  <YAxis stroke="var(--text-muted)" fontSize={12} />
                  <Tooltip contentStyle={{background: 'var(--bg-main)', border: '1px solid var(--border-color)', borderRadius: '8px'}} />
                  <Line type="monotone" dataKey="loss" stroke="var(--warning)" dot={false} strokeWidth={2} name="Train Loss" />
                  <Line type="monotone" dataKey="val_loss" stroke="var(--danger)" dot={false} strokeWidth={2} name="Val Loss" />
                </LineChart>
              </ResponsiveContainer>
              <ResponsiveContainer width="100%" height="50%">
                <LineChart data={metrics}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />
                  <XAxis dataKey="epoch" stroke="var(--text-muted)" fontSize={12} />
                  <YAxis stroke="var(--text-muted)" fontSize={12} domain={[0, 1]} />
                  <Tooltip contentStyle={{background: 'var(--bg-main)', border: '1px solid var(--border-color)', borderRadius: '8px'}} />
                  <Line type="monotone" dataKey="val_seq_acc" stroke="var(--success)" dot={false} strokeWidth={2} name="Val Seq Acc" />
                  <Line type="monotone" dataKey="val_char_acc" stroke="var(--accent)" dot={false} strokeWidth={2} name="Val Char Acc" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="glass-panel" style={{display: 'flex', flexDirection: 'column'}}>
            <h3 style={{fontSize: '1rem', color: 'var(--text-muted)'}}>Live Log</h3>
            <div className="font-mono" style={{
              flex: 1, 
              background: 'rgba(0,0,0,0.3)', 
              padding: '1rem', 
              borderRadius: '8px', 
              overflowY: 'auto', 
              fontSize: '0.85rem',
              lineHeight: 1.5,
              marginTop: '1rem'
            }}>
              {logs.map((l, i) => <div key={i}>{l}</div>)}
              {logs.length === 0 && <div style={{color: 'var(--text-muted)'}}>Waiting for logs...</div>}
              <div ref={logEndRef} />
            </div>
          </div>
        </div>
        </div>
      ) : (
        <div className="glass-panel" style={{flex: 1, overflowY: 'auto'}}>
          <ErrorGallery jobId={jobId} />
        </div>
      )}
    </div>
  )
}

function DatasetsView({ onSelectDataset }: { onSelectDataset: (name: string) => void }) {
  const [datasets, setDatasets] = useState<any[]>([])
  const [uploading, setUploading] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const fetchDatasets = async () => setDatasets(await api.getDatasets())

  useEffect(() => { fetchDatasets() }, [])

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files ?? [])
    if (files.length === 0) return

    const name = prompt("Enter dataset name (e.g. data_3):")
    if (!name) return

    setUploading(true)
    try {
      const res = await api.uploadDataset(name, files)
      alert(`Uploaded! Saved ${res.saved} images, skipped ${res.skipped}. Charset: ${res.chars}`)
      fetchDatasets()
    } catch (err: any) {
      alert("Error: " + err.message)
    } finally {
      setUploading(false)
      if (fileInputRef.current) fileInputRef.current.value = ""
    }
  }

  const handleSync = async () => {
    try {
      const res = await api.syncDatasets();
      alert(`Synced: ${res.synced.join(', ')}`);
      fetchDatasets();
    } catch (e: any) { alert(e.message) }
  }

  return (
    <div style={{height: '100%', display: 'flex', flexDirection: 'column'}}>
      <div className="page-header">
        <h2>Datasets</h2>
        <div style={{display: 'flex', gap: '1rem'}}>
          <button onClick={handleSync}><RefreshCw size={16} style={{marginRight: '0.5rem', verticalAlign: 'middle'}}/>Sync Local</button>
          <input type="file" accept=".zip,.png,.jpg,.jpeg,.bmp" multiple ref={fileInputRef} style={{display: 'none'}} onChange={handleUpload} />
          <button className="primary" onClick={() => fileInputRef.current?.click()} disabled={uploading}>
            <UploadCloud size={16} style={{marginRight: '0.5rem', verticalAlign: 'middle'}}/>
            {uploading ? 'Uploading...' : 'Upload Images / ZIP'}
          </button>
        </div>
      </div>

      <div className="glass-panel" style={{flex: 1, overflowY: 'auto'}}>
        <table>
          <thead>
            <tr>
              <th>ID</th>
              <th>Name</th>
              <th>Path</th>
              <th>Samples</th>
              <th>Charset</th>
              <th>Max Length</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {datasets.map(ds => (
              <tr key={ds.id}>
                <td>#{ds.id}</td>
                <td style={{fontWeight: 500}}>{ds.name}</td>
                <td className="font-mono">{ds.path}</td>
                <td>{ds.num_samples}</td>
                <td className="font-mono">{ds.chars.substring(0, 20) + (ds.chars.length > 20 ? '...' : '')}</td>
                <td>{ds.max_length}</td>
                <td style={{display: 'flex', gap: '0.5rem'}}>
                  <button onClick={() => onSelectDataset(ds.name)}>Browse</button>
                  <button
                    onClick={async () => {
                      if (!confirm(`Delete dataset "${ds.name}"? Uploaded files will be removed.`)) return
                      try { await api.deleteDataset(ds.name); fetchDatasets() }
                      catch (e: any) { alert(e.message) }
                    }}
                    style={{color: 'var(--danger)', borderColor: 'var(--danger)'}}
                  >Delete</button>
                </td>
              </tr>
            ))}
            {datasets.length === 0 && (
              <tr><td colSpan={7} style={{textAlign: 'center'}}>No datasets found.</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function App() {
  const [activeTab, setActiveTab] = useState<'jobs' | 'compare' | 'datasets' | 'dataset_browser' | 'job_detail' | 'aug' | 'playground'>('jobs');
  const [selectedJob, setSelectedJob] = useState<number | null>(null);
  const [selectedDataset, setSelectedDataset] = useState<string | null>(null);

  return (
    <div className="app-layout">
      <div className="sidebar">
        <div className="sidebar-header">
          <Activity color="var(--accent)" /> OCR Framework
        </div>
        <div className="sidebar-nav">
          <div className={`nav-item ${(activeTab === 'jobs' || activeTab === 'job_detail') ? 'active' : ''}`} onClick={() => setActiveTab('jobs')}>
            <Play size={18} /> Jobs
          </div>
          <div className={`nav-item ${activeTab === 'compare' ? 'active' : ''}`} onClick={() => setActiveTab('compare')}>
            <BarChart2 size={18} /> Compare Runs
          </div>
          <div className={`nav-item ${(activeTab === 'datasets' || activeTab === 'dataset_browser') ? 'active' : ''}`} onClick={() => setActiveTab('datasets')}>
            <Database size={18} /> Datasets
          </div>
          <div className={`nav-item ${activeTab === 'aug' ? 'active' : ''}`} onClick={() => setActiveTab('aug')}>
            <ImageIcon size={18} /> Augmentations
          </div>
          <div className={`nav-item ${activeTab === 'playground' ? 'active' : ''}`} onClick={() => setActiveTab('playground')}>
            <Zap size={18} /> Playground
          </div>
        </div>
      </div>
      
      <div className="main-content">
        {activeTab === 'jobs' && <JobsView onSelectJob={(id) => { setSelectedJob(id); setActiveTab('job_detail') }} />}
        {activeTab === 'compare' && <CompareView />}
        {activeTab === 'datasets' && <DatasetsView onSelectDataset={(name) => { setSelectedDataset(name); setActiveTab('dataset_browser') }} />}
        {activeTab === 'dataset_browser' && selectedDataset && <DatasetBrowser datasetName={selectedDataset} onBack={() => setActiveTab('datasets')} />}
        {activeTab === 'job_detail' && selectedJob && <JobDetailView jobId={selectedJob} onBack={() => setActiveTab('jobs')} />}
        {activeTab === 'aug' && <AugPreview />}
        {activeTab === 'playground' && <PlaygroundView />}
      </div>
    </div>
  )
}

export default App
