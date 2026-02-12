import { PublicClientApplication } from '@azure/msal-browser'
import axios from 'axios'

const tenantId = import.meta.env.VITE_ENTRA_TENANT_ID || ''
const clientId = import.meta.env.VITE_ENTRA_CLIENT_ID || ''
const ssoMode = String(import.meta.env.VITE_SSO_MODE || 'off').toLowerCase()

const TOKEN_STORAGE_KEY = 'kai_entra_id_token_v1'

export function entraEnabled() {
  return Boolean(tenantId && clientId && ssoMode !== 'off')
}

export function entraMode() {
  return ssoMode || 'off'
}

let _msal = null

function getMsal() {
  if (!entraEnabled()) return null
  if (_msal) return _msal

  _msal = new PublicClientApplication({
    auth: {
      clientId,
      authority: `https://login.microsoftonline.com/${tenantId}`,
      redirectUri: typeof window !== 'undefined' ? window.location.origin : '/',
    },
    cache: {
      cacheLocation: 'localStorage',
      storeAuthStateInCookie: false,
    },
  })
  return _msal
}

export function entraCachedToken() {
  try {
    if (typeof window === 'undefined') return null
    return window.sessionStorage.getItem(TOKEN_STORAGE_KEY) || null
  } catch {
    return null
  }
}

export function entraClearCachedToken() {
  try {
    if (typeof window === 'undefined') return
    window.sessionStorage.removeItem(TOKEN_STORAGE_KEY)
  } catch {
    // ignore
  }
}

function cacheToken(token) {
  try {
    if (typeof window === 'undefined') return
    if (token) {
      window.sessionStorage.setItem(TOKEN_STORAGE_KEY, String(token))
    }
  } catch {
    // ignore
  }
}

export function applyEntraAuthToAxios(token) {
  const t = token || entraCachedToken()
  if (!t) return false
  axios.defaults.headers.common.Authorization = `Bearer ${t}`
  return true
}

async function acquireTokenSilent(msal) {
  const accounts = msal.getAllAccounts()
  if (!accounts || accounts.length === 0) return null
  const account = accounts[0]
  const result = await msal.acquireTokenSilent({
    account,
    scopes: ['openid', 'profile', 'email'],
  })
  return result?.idToken || null
}

export async function entraGetIdToken({ interactive = false } = {}) {
  const msal = getMsal()
  if (!msal) return null

  // Try existing cached token first (fast path).
  const cached = entraCachedToken()
  if (cached) {
    applyEntraAuthToAxios(cached)
    return cached
  }

  try {
    const silent = await acquireTokenSilent(msal)
    if (silent) {
      cacheToken(silent)
      applyEntraAuthToAxios(silent)
      return silent
    }
  } catch {
    // fall through to interactive if allowed
  }

  if (!interactive) return null
  const result = await msal.loginPopup({
    scopes: ['openid', 'profile', 'email'],
    prompt: 'select_account',
  })
  const token = result?.idToken || null
  if (token) {
    cacheToken(token)
    applyEntraAuthToAxios(token)
  }
  return token
}

