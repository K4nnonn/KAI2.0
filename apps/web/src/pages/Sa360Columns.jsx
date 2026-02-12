import { useEffect, useMemo, useState } from 'react'
import {
  Box,
  Container,
  Typography,
  Paper,
  TextField,
  Button,
  Chip,
  CircularProgress,
  Alert,
  Autocomplete,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Tooltip,
} from '@mui/material'
import ContentCopyIcon from '@mui/icons-material/ContentCopy'
import ChatIcon from '@mui/icons-material/Chat'
import RefreshIcon from '@mui/icons-material/Refresh'
import axios from 'axios'
import { API_BASE_URL, getOrCreateSessionId } from '../config'

const storageKeyForActiveAccount = (sessionId) => (sessionId ? `kai_sa360_active_account:${sessionId}` : null)
const chatPrefillKey = 'kai_chat_prefill_input_v1'

export default function Sa360Columns() {
  const [sessionId] = useState(() => getOrCreateSessionId())
  const [sa360Status, setSa360Status] = useState({ connected: false, login_customer_id: null })
  const [sa360StatusLoading, setSa360StatusLoading] = useState(false)
  const [sa360StatusError, setSa360StatusError] = useState(null)
  const [saveNotice, setSaveNotice] = useState(null)
  const [availableManagers, setAvailableManagers] = useState([])
  const [managersLoading, setManagersLoading] = useState(false)
  const [managersError, setManagersError] = useState(null)

  const [accountsLoading, setAccountsLoading] = useState(false)
  const [accountsError, setAccountsError] = useState(null)
  const [availableAccounts, setAvailableAccounts] = useState([])

  const [activeAccount, setActiveAccount] = useState(null)
  const [search, setSearch] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [rows, setRows] = useState([])

  const normalizedAccounts = useMemo(() => {
    const list = Array.isArray(availableAccounts) ? availableAccounts : []
    return list
      .filter((a) => a && a.customer_id)
      .map((a) => ({
        customer_id: String(a.customer_id),
        name: a.name || '',
        manager: !!a.manager,
      }))
      .filter((a) => !a.manager)
      .sort((a, b) => (a.name || a.customer_id).localeCompare(b.name || b.customer_id))
  }, [availableAccounts])

  const normalizedManagers = useMemo(() => {
    const list = Array.isArray(availableManagers) ? availableManagers : []
    return list
      .filter((m) => m && m.customer_id)
      .map((m) => ({
        customer_id: String(m.customer_id),
        name: m.name || '',
        manager: true,
      }))
      .sort((a, b) => (a.name || a.customer_id).localeCompare(b.name || b.customer_id))
  }, [availableManagers])

  const fetchSa360Status = async () => {
    if (!sessionId) return
    setSa360StatusLoading(true)
    setSa360StatusError(null)
    try {
      const resp = await axios.get(`${API_BASE_URL}/api/sa360/oauth/status`, {
        params: { session_id: sessionId },
      })
      const next = resp.data || { connected: false, login_customer_id: null }
      setSa360Status(next)
      if (!activeAccount?.customer_id && next.default_customer_id) {
        setActiveAccount({
          customer_id: String(next.default_customer_id),
          name: next.default_account_name ? String(next.default_account_name) : '',
        })
      }
    } catch (err) {
      setSa360StatusError(err?.response?.data?.detail || err?.message || 'SA360 status check failed')
    } finally {
      setSa360StatusLoading(false)
    }
  }

  const fetchManagers = async () => {
    if (!sessionId) return []
    if (!sa360Status.connected) return []
    setManagersLoading(true)
    setManagersError(null)
    try {
      const resp = await axios.get(`${API_BASE_URL}/api/sa360/managers`, {
        params: { session_id: sessionId },
      })
      const list = Array.isArray(resp.data) ? resp.data : []
      setAvailableManagers(list)
      return list
    } catch (err) {
      setManagersError(err?.response?.data?.detail || err?.message || 'Failed to discover MCCs for this user')
      return []
    } finally {
      setManagersLoading(false)
    }
  }

  const saveLoginCustomerId = async (loginCustomerId) => {
    const cid = String(loginCustomerId || '')
      .replace(/[^\d]/g, '')
      .trim()
    if (!sessionId || !cid) return
    try {
      await axios.post(`${API_BASE_URL}/api/sa360/login-customer`, {
        session_id: sessionId,
        login_customer_id: cid,
      })
      await fetchSa360Status()
      await fetchAccounts()
      setSaveNotice('Manager saved')
      setTimeout(() => setSaveNotice(null), 2500)
    } catch (err) {
      setManagersError(err?.response?.data?.detail || err?.message || 'Failed to save MCC')
    }
  }

  const fetchAccounts = async () => {
    if (!sessionId) return
    setAccountsLoading(true)
    setAccountsError(null)
    try {
      const resp = await axios.get(`${API_BASE_URL}/api/sa360/accounts`, {
        params: { session_id: sessionId },
      })
      const list = Array.isArray(resp.data) ? resp.data : []
      setAvailableAccounts(list)
    } catch (err) {
      setAccountsError(err?.response?.data?.detail || err?.message || 'Failed to load SA360 accounts')
    } finally {
      setAccountsLoading(false)
    }
  }

  const loadColumns = async () => {
    if (!sessionId || !activeAccount?.customer_id) return
    setLoading(true)
    setError(null)
    try {
      const resp = await axios.get(`${API_BASE_URL}/api/sa360/conversion-actions`, {
        params: {
          session_id: sessionId,
          customer_id: activeAccount.customer_id,
          date_range: 'LAST_30_DAYS',
        },
        timeout: 120000,
      })
      const actions = Array.isArray(resp.data?.actions) ? resp.data.actions : []
      setRows(actions)
    } catch (err) {
      setError(err?.response?.data?.detail || err?.message || 'Failed to load conversion columns')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchSa360Status()
  }, [sessionId])

  useEffect(() => {
    if (!sessionId) return
    const key = storageKeyForActiveAccount(sessionId)
    if (!key) return
    try {
      const raw = window.localStorage.getItem(key)
      if (!raw) return
      const parsed = JSON.parse(raw)
      if (parsed && parsed.customer_id) {
        setActiveAccount({ customer_id: String(parsed.customer_id), name: parsed.name || '' })
      }
    } catch {
      // ignore
    }
  }, [sessionId])

  useEffect(() => {
    if (!sessionId) return
    const key = storageKeyForActiveAccount(sessionId)
    if (!key) return
    try {
      if (activeAccount?.customer_id) {
        window.localStorage.setItem(key, JSON.stringify({ customer_id: activeAccount.customer_id, name: activeAccount.name || '' }))
      } else {
        window.localStorage.removeItem(key)
      }
    } catch {
      // ignore
    }
  }, [activeAccount?.customer_id, activeAccount?.name, sessionId])

  useEffect(() => {
    // Automatically load accounts once SA360 is connected.
    if (!sa360Status.connected) return
    if (availableAccounts.length) return
    fetchAccounts()
  }, [sa360Status.connected])

  useEffect(() => {
    // If connected but MCC isn't set, attempt discovery to reduce onboarding friction.
    if (!sa360Status.connected) return
    if (String(sa360Status?.login_customer_id || '').trim()) return
    if (availableManagers.length) return
    fetchManagers().then((managers) => {
      if (Array.isArray(managers) && managers.length === 1 && managers[0]?.customer_id) {
        saveLoginCustomerId(managers[0].customer_id)
      }
    })
  }, [sa360Status.connected, sa360Status?.login_customer_id])

  useEffect(() => {
    // Auto-load columns when account changes (keeps the page "one click" once configured).
    if (!activeAccount?.customer_id) return
    loadColumns()
  }, [activeAccount?.customer_id])

  useEffect(() => {
    // Persist the default account selection server-side for cross-page/cross-device consistency.
    if (!sessionId) return
    if (!sa360Status.connected) return
    if (!String(sa360Status?.login_customer_id || '').trim()) return

    const cid = activeAccount?.customer_id ? String(activeAccount.customer_id) : ''
    // Never clear defaults implicitly; only persist when a real account is selected.
    if (!cid) return
    axios
      .post(`${API_BASE_URL}/api/sa360/default-account`, {
        session_id: sessionId,
        customer_id: cid || null,
        account_name: activeAccount?.name || null,
      })
      .then(() => {
        if (cid) {
          setSaveNotice('Account saved')
          setTimeout(() => setSaveNotice(null), 2500)
        }
      })
      .catch(() => {
        // non-blocking
      })
  }, [activeAccount?.customer_id, activeAccount?.name, sessionId, sa360Status.connected, sa360Status?.login_customer_id])

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase()
    if (!q) return rows
    return rows.filter((r) => {
      const name = String(r.name || '').toLowerCase()
      const key = String(r.metric_key || '').toLowerCase()
      const cat = String(r.category || '').toLowerCase()
      return name.includes(q) || key.includes(q) || cat.includes(q)
    })
  }, [rows, search])

  const copyText = async (text) => {
    try {
      await navigator.clipboard.writeText(String(text || ''))
      setSaveNotice('Copied')
      setTimeout(() => setSaveNotice(null), 1500)
    } catch {
      // no-op; clipboard may be blocked by browser policies
    }
  }

  const buildChatPrompt = (row) => {
    const key = row?.metric_key ? String(row.metric_key) : ''
    if (!key) return ''
    const name = row?.name ? String(row.name) : key
    return `Why did ${name} change week over week? Which campaigns drove it? Use metric key: ${key}.`
  }

  const useInChat = async (row) => {
    const prompt = buildChatPrompt(row)
    if (!prompt) return
    try {
      window.localStorage.setItem(chatPrefillKey, prompt)
    } catch {
      // ignore
    }
    await copyText(prompt)
    setSaveNotice('Opening Kai Chat...')
    setTimeout(() => setSaveNotice(null), 1500)
    window.location.assign('/')
  }

  const connected = !!sa360Status.connected

  return (
    <Container maxWidth="lg" sx={{ py: 3 }}>
      <Paper
        elevation={0}
        sx={{
          p: 3,
          borderRadius: 3,
          background: 'var(--kai-surface)',
          border: '1px solid var(--kai-surface-muted)',
          mb: 2,
        }}
      >
        <Box display="flex" alignItems="center" justifyContent="space-between" gap={2} flexWrap="wrap">
          <Box>
            <Typography variant="h4" sx={{ fontWeight: 800, color: 'var(--kai-text)' }}>
              SA360 Columns (Conversion Actions)
            </Typography>
            <Typography variant="body2" sx={{ color: 'var(--kai-text-soft)', mt: 0.5 }}>
              Browse the conversion action names this user can reference in prompts. Values shown are for LAST_30_DAYS.
            </Typography>
          </Box>
          <Box display="flex" gap={1} alignItems="center">
            <Chip
              label={connected ? 'SA360 connected' : 'SA360 not connected'}
              sx={{
                borderColor: connected ? 'rgba(34,197,94,0.6)' : 'rgba(248,113,113,0.6)',
                color: connected ? '#22c55e' : '#f87171',
                backgroundColor: connected ? 'rgba(34,197,94,0.15)' : 'rgba(248,113,113,0.15)',
              }}
              variant="outlined"
            />
            <Tooltip title="Refresh SA360 status">
              <span>
                <IconButton onClick={fetchSa360Status} disabled={sa360StatusLoading}>
                  <RefreshIcon />
                </IconButton>
              </span>
            </Tooltip>
          </Box>
        </Box>

        {sa360StatusError && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {sa360StatusError}
          </Alert>
        )}

        {!connected && (
          <Alert severity="warning" sx={{ mt: 2 }}>
            SA360 isn&apos;t connected for this session. Go to Kai Chat and click <b>Connect SA360</b>.
          </Alert>
        )}

        {connected && !String(sa360Status?.login_customer_id || '').trim() && (
          <Alert severity="info" sx={{ mt: 2 }}>
            Manager (MCC) isn&apos;t set for this user yet. Pick one below (or go to Kai Chat and paste your MCC).
          </Alert>
        )}

        <Box display="flex" gap={2} alignItems="center" flexWrap="wrap" mt={2}>
          {connected && !String(sa360Status?.login_customer_id || '').trim() && (
            <Autocomplete
              value={null}
              onChange={(_e, next) => {
                if (next?.customer_id) {
                  saveLoginCustomerId(next.customer_id)
                }
              }}
              options={normalizedManagers}
              loading={managersLoading}
              getOptionLabel={(opt) => (opt?.name ? `${opt.name} (${opt.customer_id})` : String(opt?.customer_id || ''))}
              isOptionEqualToValue={(a, b) => String(a?.customer_id) === String(b?.customer_id)}
              sx={{ minWidth: 420, flex: 1, maxWidth: 720 }}
              renderInput={(params) => (
                <TextField
                  {...params}
                  label="Manager (MCC) by name"
                  placeholder={connected ? 'Select your manager account' : 'Connect SA360 first'}
                  InputProps={{
                    ...params.InputProps,
                    endAdornment: (
                      <>
                        {managersLoading ? <CircularProgress color="inherit" size={18} /> : null}
                        {params.InputProps.endAdornment}
                      </>
                    ),
                  }}
                  disabled={!connected}
                />
              )}
            />
          )}
          <Autocomplete
            value={activeAccount}
            onChange={(_e, next) => setActiveAccount(next)}
            options={normalizedAccounts}
            loading={accountsLoading}
            getOptionLabel={(opt) => (opt?.name ? `${opt.name} (${opt.customer_id})` : String(opt?.customer_id || ''))}
            isOptionEqualToValue={(a, b) => String(a?.customer_id) === String(b?.customer_id)}
            sx={{ minWidth: 420, flex: 1, maxWidth: 720 }}
            renderInput={(params) => (
              <TextField
                {...params}
                label="Account (by name)"
                placeholder={connected ? 'Select an account to browse columns' : 'Connect SA360 first'}
                InputProps={{
                  ...params.InputProps,
                  endAdornment: (
                    <>
                      {accountsLoading ? <CircularProgress color="inherit" size={18} /> : null}
                      {params.InputProps.endAdornment}
                    </>
                  ),
                }}
                disabled={!connected}
              />
            )}
          />
          <Button
            variant="outlined"
            onClick={fetchAccounts}
            disabled={!connected || accountsLoading}
            sx={{ whiteSpace: 'nowrap' }}
          >
            Reload accounts
          </Button>
          {connected && !String(sa360Status?.login_customer_id || '').trim() && (
            <Button
              variant="outlined"
              onClick={fetchManagers}
              disabled={managersLoading}
              sx={{ whiteSpace: 'nowrap' }}
            >
              Discover MCCs
            </Button>
          )}
          <TextField
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            label="Search columns"
            placeholder="e.g., store visits, directions, fuel rewards"
            sx={{ minWidth: 280, flex: 0.5, maxWidth: 520 }}
            disabled={!connected || !activeAccount?.customer_id}
          />
        </Box>

        {accountsError && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {accountsError}
          </Alert>
        )}

        {managersError && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {managersError}
          </Alert>
        )}

        {saveNotice && (
          <Alert severity="success" sx={{ mt: 2 }}>
            {saveNotice}
          </Alert>
        )}
      </Paper>

      <Paper
        elevation={0}
        sx={{
          p: 0,
          borderRadius: 3,
          background: 'var(--kai-surface)',
          border: '1px solid var(--kai-surface-muted)',
        }}
      >
        {!activeAccount?.customer_id ? (
          <Box sx={{ p: 3 }}>
            <Alert severity="info">
              Select an account to load conversion columns.
            </Alert>
          </Box>
        ) : (
          <Box>
            {error && (
              <Box sx={{ p: 3, pb: 0 }}>
                <Alert severity="error">{error}</Alert>
              </Box>
            )}
            <Box display="flex" alignItems="center" justifyContent="space-between" gap={2} flexWrap="wrap" sx={{ p: 3, pb: 1 }}>
              <Typography variant="subtitle2" sx={{ color: 'var(--kai-text-soft)' }}>
                Showing <b>{filtered.length}</b> of <b>{rows.length}</b> columns for <b>{activeAccount?.name || activeAccount?.customer_id}</b>
              </Typography>
              <Button
                variant="outlined"
                onClick={loadColumns}
                disabled={loading}
                sx={{ whiteSpace: 'nowrap' }}
              >
                {loading ? 'Loading...' : 'Refresh columns'}
              </Button>
            </Box>
            <TableContainer sx={{ maxHeight: 'calc(100vh - 340px)' }}>
              <Table stickyHeader size="small" aria-label="SA360 conversion columns">
                <TableHead>
                  <TableRow>
                    <TableCell>Name</TableCell>
                    <TableCell>Category</TableCell>
                    <TableCell align="right">Conversions</TableCell>
                    <TableCell align="right">All conversions</TableCell>
                    <TableCell align="right">All conv value</TableCell>
                    <TableCell>Metric key</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {filtered.map((r) => (
                    <TableRow key={r.metric_key || r.name}>
                      <TableCell sx={{ maxWidth: 420 }}>
                        <Typography variant="body2" sx={{ fontWeight: 600, color: 'var(--kai-text)' }}>
                          {r.name || ''}
                        </Typography>
                        {r.status && (
                          <Typography variant="caption" sx={{ color: 'var(--kai-text-soft)' }}>
                            {String(r.status).toLowerCase()}
                          </Typography>
                        )}
                      </TableCell>
                      <TableCell>{r.category || ''}</TableCell>
                      <TableCell align="right">{r.conversions ?? ''}</TableCell>
                      <TableCell align="right">{r.all_conversions ?? ''}</TableCell>
                      <TableCell align="right">{r.all_conversions_value ?? ''}</TableCell>
                      <TableCell sx={{ maxWidth: 360 }}>
                        <Box display="flex" alignItems="center" gap={1}>
                          <Typography variant="caption" sx={{ color: 'var(--kai-text-soft)' }}>
                            {r.metric_key || ''}
                          </Typography>
                          <Tooltip title="Copy metric key">
                            <span>
                              <IconButton size="small" onClick={() => copyText(r.metric_key)} disabled={!r.metric_key}>
                                <ContentCopyIcon fontSize="inherit" />
                              </IconButton>
                            </span>
                          </Tooltip>
                          <Tooltip title="Use in chat">
                            <span>
                              <IconButton size="small" onClick={() => useInChat(r)} disabled={!r.metric_key}>
                                <ChatIcon fontSize="inherit" />
                              </IconButton>
                            </span>
                          </Tooltip>
                        </Box>
                      </TableCell>
                    </TableRow>
                  ))}
                  {loading && (
                    <TableRow>
                      <TableCell colSpan={6}>
                        <Box display="flex" alignItems="center" gap={1}>
                          <CircularProgress size={18} />
                          <Typography variant="body2" sx={{ color: 'var(--kai-text-soft)' }}>
                            Loading...
                          </Typography>
                        </Box>
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </TableContainer>
          </Box>
        )}
      </Paper>
    </Container>
  )
}
