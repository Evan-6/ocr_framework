// API Client
const API_BASE = "http://localhost:8000/api";

export const api = {
  getJobs: async () => {
    const res = await fetch(`${API_BASE}/jobs`);
    return res.json();
  },
  getJob: async (id: number) => {
    const res = await fetch(`${API_BASE}/jobs/${id}`);
    return res.json();
  },
  createJob: async (data: { name: string, config_file: string, config_overrides: string }) => {
    const res = await fetch(`${API_BASE}/jobs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data)
    });
    return res.json();
  },
  stopJob: async (id: number) => {
    const res = await fetch(`${API_BASE}/jobs/${id}/stop`, { method: "POST" });
    return res.json();
  },
  deleteJob: async (id: number) => {
    const res = await fetch(`${API_BASE}/jobs/${id}`, { method: "DELETE" });
    if (!res.ok) throw new Error((await res.json()).detail || "Delete failed");
    return res.json();
  },
  deleteDataset: async (name: string) => {
    const res = await fetch(`${API_BASE}/datasets/${name}`, { method: "DELETE" });
    if (!res.ok) throw new Error((await res.json()).detail || "Delete failed");
    return res.json();
  },
  getDatasets: async () => {
    const res = await fetch(`${API_BASE}/datasets`);
    return res.json();
  },
  uploadDataset: async (name: string, files: File[]) => {
    const formData = new FormData();
    formData.append("name", name);
    files.forEach(f => formData.append("files", f));

    const res = await fetch(`${API_BASE}/datasets/upload`, {
      method: "POST",
      body: formData
    });
    
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Upload failed");
    }
    return res.json();
  },
  exportJob: async (id: number) => {
    const res = await fetch(`${API_BASE}/jobs/${id}/export`, { method: "POST" });
    if (!res.ok) throw new Error((await res.json()).detail || "Export failed");
    return res.json();
  },
  getDownloadUrl: (id: number, artifact: string) => `${API_BASE}/jobs/${id}/download/${artifact}`,
  getEventStreamUrl: (jobId: number) => `${API_BASE}/jobs/${jobId}/events`,
  
  getJobMetricsHistory: async (jobId: number) => {
    const res = await fetch(`${API_BASE}/jobs/${jobId}/metrics_history`);
    return res.json();
  },
  getJobErrors: async (jobId: number) => {
    const res = await fetch(`${API_BASE}/jobs/${jobId}/errors`);
    return res.json();
  },
  getDatasetSamples: async (name: string, page: number = 1) => {
    const res = await fetch(`${API_BASE}/datasets/${name}/samples?page=${page}`);
    return res.json();
  },
  getDatasetStats: async (name: string) => {
    const res = await fetch(`${API_BASE}/datasets/${name}/stats`);
    return res.json();
  },
  getDatasetImageUrl: (name: string, filename: string) => `${API_BASE}/datasets/${name}/images/${filename}`,
  
  syncDatasets: async () => {
    const res = await fetch(`${API_BASE}/datasets/sync`, { method: "POST" });
    return res.json();
  },
  
  augPreview: async (file: File, params: any) => {
    const formData = new FormData();
    formData.append("file", file);
    Object.keys(params).forEach(k => formData.append(k, params[k]));
    const res = await fetch(`${API_BASE}/aug/preview`, {
      method: "POST",
      body: formData
    });
    return res.json();
  },
  
  runInference: async (jobId: number, file: File) => {
    const formData = new FormData();
    formData.append("file", file);
    const res = await fetch(`${API_BASE}/jobs/${jobId}/infer`, {
      method: "POST",
      body: formData
    });
    if (!res.ok) throw new Error("Inference failed");
    return res.json();
  }
};
