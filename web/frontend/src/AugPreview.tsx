import { useState, useRef } from 'react'
import { UploadCloud, Sliders } from 'lucide-react'
import { api } from './api'

export function AugPreview() {
  const [file, setFile] = useState<File | null>(null)
  const [, setOriginalUrl] = useState<string>("")
  const [previewImages, setPreviewImages] = useState<string[]>([])
  const [, setLoading] = useState(false)
  const [params, setParams] = useState({
    aug_rotate: 0.0,
    aug_translate: 0.0,
    aug_scale: 0.0,
    aug_shear: 0.0,
    aug_photometric: false,
    channels: 1
  })

  const fileInputRef = useRef<HTMLInputElement>(null)

  const updatePreview = async (newParams: any, f: File | null = file) => {
    if (!f) return
    setLoading(true)
    try {
      const res = await api.augPreview(f, newParams)
      setPreviewImages(res.images)
    } finally {
      setLoading(false)
    }
  }

  const handleParamChange = (k: string, v: any) => {
    const p = { ...params, [k]: v }
    setParams(p)
    updatePreview(p, file)
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    if (f) {
      setFile(f)
      setOriginalUrl(URL.createObjectURL(f))
      updatePreview(params, f)
    }
  }

  return (
    <div style={{height: '100%', display: 'flex', flexDirection: 'column'}}>
      <div className="page-header" style={{marginBottom: '1rem'}}>
        <h2>Augmentation Preview</h2>
        <div>
          <input type="file" accept="image/*" ref={fileInputRef} style={{display: 'none'}} onChange={handleFileChange} />
          <button className="primary" onClick={() => fileInputRef.current?.click()}>
            <UploadCloud size={16} style={{marginRight: '0.5rem', verticalAlign: 'middle'}}/>
            Upload Test Image
          </button>
        </div>
      </div>
      
      {file ? (
        <div style={{display: 'flex', gap: '1.5rem', flex: 1, minHeight: 0}}>
          <div className="glass-panel" style={{width: '320px', overflowY: 'auto'}}>
            <h3 style={{fontSize: '1rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '0.5rem'}}>
              <Sliders size={16} /> Parameters
            </h3>
            
            <div style={{display: 'flex', flexDirection: 'column', gap: '1.5rem', marginTop: '1.5rem'}}>
              <div>
                <label style={{display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem'}}>
                  Rotation (± deg) <span>{params.aug_rotate}</span>
                </label>
                <input type="range" min="0" max="45" step="1" value={params.aug_rotate} onChange={e => handleParamChange('aug_rotate', parseFloat(e.target.value))} />
              </div>
              <div>
                <label style={{display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem'}}>
                  Translation (frac) <span>{params.aug_translate}</span>
                </label>
                <input type="range" min="0" max="0.5" step="0.05" value={params.aug_translate} onChange={e => handleParamChange('aug_translate', parseFloat(e.target.value))} />
              </div>
              <div>
                <label style={{display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem'}}>
                  Scale (frac) <span>{params.aug_scale}</span>
                </label>
                <input type="range" min="0" max="0.5" step="0.05" value={params.aug_scale} onChange={e => handleParamChange('aug_scale', parseFloat(e.target.value))} />
              </div>
              <div>
                <label style={{display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem'}}>
                  Shear (± deg) <span>{params.aug_shear}</span>
                </label>
                <input type="range" min="0" max="45" step="1" value={params.aug_shear} onChange={e => handleParamChange('aug_shear', parseFloat(e.target.value))} />
              </div>
              <div>
                <label style={{display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer'}}>
                  <input type="checkbox" checked={params.aug_photometric} onChange={e => handleParamChange('aug_photometric', e.target.checked)} style={{width: 'auto'}} />
                  Photometric (Color/Blur/Noise)
                </label>
              </div>
              <div>
                <label style={{display: 'block', marginBottom: '0.5rem'}}>Channels</label>
                <select value={params.channels} onChange={e => handleParamChange('channels', parseInt(e.target.value))}>
                  <option value={1}>Grayscale (1)</option>
                  <option value={3}>RGB (3)</option>
                </select>
              </div>
            </div>
          </div>
          
          <div className="glass-panel" style={{flex: 1, display: 'flex', flexDirection: 'column'}}>
            <h3 style={{fontSize: '1rem', color: 'var(--text-muted)'}}>Preview</h3>
            <div style={{flex: 1, display: 'grid', gridTemplateColumns: '1fr 1fr', gridTemplateRows: '1fr 1fr', gap: '1rem', marginTop: '1rem'}}>
              {previewImages.map((src, i) => (
                <div key={i} style={{background: '#000', borderRadius: '8px', overflow: 'hidden', display: 'flex', alignItems: 'center', justifyContent: 'center'}}>
                  <img src={src} style={{maxWidth: '100%', maxHeight: '100%', objectFit: 'contain'}} />
                </div>
              ))}
            </div>
          </div>
        </div>
      ) : (
        <div style={{flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)'}}>
          Upload an image to preview augmentations.
        </div>
      )}
    </div>
  )
}
