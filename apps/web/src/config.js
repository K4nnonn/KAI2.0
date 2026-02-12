// API Configuration
// Use VITE_BACKEND_URL when provided; otherwise default to same-origin in prod
// and Vite proxy in dev.

const isDev = import.meta.env.DEV;
const envBackend = import.meta.env.VITE_BACKEND_URL;

export const API_BASE_URL = envBackend || (isDev ? '' : window.location.origin);

export const SESSION_ID_KEY = 'kai_chat_session_id'

export function getOrCreateSessionId() {
  try {
    if (typeof window === 'undefined') return null
    const key = SESSION_ID_KEY
    const local = window.localStorage
    const session = window.sessionStorage

    // Prefer localStorage to keep one stable session across tabs/reloads (broad beta UX).
    // Preserve backward compatibility with existing sessionStorage-based sessions.
    let existing = (local && local.getItem(key)) || (session && session.getItem(key))
    if (!existing) {
      const rand = window.crypto && typeof window.crypto.randomUUID === 'function'
        ? window.crypto.randomUUID()
        : `${Date.now()}-${Math.random().toString(16).slice(2)}`
      existing = rand
    }
    try { local && local.setItem(key, existing) } catch {}
    try { session && session.setItem(key, existing) } catch {}
    return existing
  } catch {
    return null
  }
}

export const api = {
  auth: {
    verify: `${API_BASE_URL}/api/auth/verify`,
  },
  data: {
    upload: `${API_BASE_URL}/api/data/upload`,
  },
  chat: {
    send: `${API_BASE_URL}/api/chat/send`,
    history: `${API_BASE_URL}/api/chat/history`,
    clear: `${API_BASE_URL}/api/chat/clear`,
    planAndRun: `${API_BASE_URL}/api/chat/plan-and-run`,
    route: `${API_BASE_URL}/api/chat/route`,
  },
  audit: {
    generate: `${API_BASE_URL}/api/audit/generate`,
    upload: `${API_BASE_URL}/api/audit/upload`,
    download: (filename) => `${API_BASE_URL}/api/audit/download/${filename}`,
    businessUnits: `${API_BASE_URL}/api/audit/business-units`,
  },
  creative: {
    generate: `${API_BASE_URL}/api/creative/generate`,
  },
  pmax: {
    analyze: `${API_BASE_URL}/api/pmax/analyze`,
  },
  serp: {
    check: `${API_BASE_URL}/api/serp/check`,
    competitorSignal: `${API_BASE_URL}/api/serp/competitor-signal`,
  },
  trends: {
    seasonality: `${API_BASE_URL}/api/trends/seasonality`,
  },
  jobs: {
    health: `${API_BASE_URL}/api/jobs/health`,
    status: (jobId) => `${API_BASE_URL}/api/jobs/${jobId}`,
    result: (jobId) => `${API_BASE_URL}/api/jobs/${jobId}/result`,
  },
  health: `${API_BASE_URL}/api/health`,
};
