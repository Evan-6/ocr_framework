import { useState, useRef, useEffect } from 'react'
import { UploadCloud } from 'lucide-react'
import { api } from './api'

export function PlaygroundView() {
  const [jobs, setJobs] = useState<any[]>([])
  const [selectedJob, setSelectedJob] = useState<number | ''>('')
  const [, setFile] = useState<File | null>(null)
  const [imageUrl, setImageUrl] = useState<string>('')
  const [result, setResult] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    api.getJobs().then(res => {
      const doneJobs = res.filter((j: any) => j.status === 'done')
      setJobs(doneJobs)
      if (doneJobs.length > 0) setSelectedJob(doneJobs[0].id)
    })
  }, [])

  const handleInfer = async (f: File) => {
    if (!selectedJob) {
      alert("Please select a trained job first.")
      return
    }
    setLoading(true)
    try {
      const res = await api.runInference(selectedJob as number, f)
      setResult(res.text)
    } catch (e: any) {
      alert("Inference failed: " + e.message)
    } finally {
      setLoading(false)
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    if (f) {
      setFile(f)
      setImageUrl(URL.createObjectURL(f))
      handleInfer(f)
    }
  }

  return (
    <div style={{height: '100%', display: 'flex', flexDirection: 'column'}}>
      <div className="page-header" style={{marginBottom: '1rem'}}>
        <h2>Inference Playground</h2>
        <div>
          <select value={selectedJob} onChange={e => setSelectedJob(Number(e.target.value))} style={{marginRight: '1rem', width: '200px'}}>
            <option value="" disabled>Select Model...</option>
            {jobs.map(j => <option key={j.id} value={j.id}>#{j.id} {j.name}</option>)}
          </select>
          <input type="file" accept="image/*" ref={fileInputRef} style={{display: 'none'}} onChange={handleFileChange} />
          <button className="primary" onClick={() => fileInputRef.current?.click()} disabled={loading || !selectedJob}>
            <UploadCloud size={16} style={{marginRight: '0.5rem', verticalAlign: 'middle'}}/>
            {loading ? 'Running...' : 'Upload & Infer'}
          </button>
        </div>
      </div>
      
      <div className="glass-panel" style={{flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '2rem'}}>
        {imageUrl ? (
          <>
            <div style={{height: '200px', padding: '1rem', background: '#000', borderRadius: '8px', border: '1px solid var(--border-color)'}}>
              <img src={imageUrl} style={{maxHeight: '100%', maxWidth: '100%', objectFit: 'contain'}} />
            </div>
            <div style={{fontSize: '2rem', fontWeight: 700, color: 'var(--accent)', background: 'var(--bg-card)', padding: '1rem 3rem', borderRadius: '12px', border: '1px solid var(--border-color)'}}>
              {result !== null ? result : '...'}
            </div>
          </>
        ) : (
          <div style={{color: 'var(--text-muted)'}}>Upload an image to see prediction.</div>
        )}
      </div>
    </div>
  )
}
