import { useEffect, useState } from 'react'
import {
  Box,
  Container,
  Typography,
  Paper,
  Table,
  TableHead,
  TableRow,
  TableCell,
  TableBody,
  TextField,
  Button,
  Alert,
  Stack,
  Divider,
  Chip,
  Grid,
} from '@mui/material'
import { Key as KeyIcon, Save as SaveIcon, Refresh as RefreshIcon } from '@mui/icons-material'
import axios from 'axios'
import { API_BASE_URL } from '../config'

export default function EnvManager() {
  const [authPass, setAuthPass] = useState('')
  const [authed, setAuthed] = useState(false)
  const [envList, setEnvList] = useState([])
  const [selectedKey, setSelectedKey] = useState('')
  const [newValue, setNewValue] = useState('')
  const [adminPassword, setAdminPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [wiringLoading, setWiringLoading] = useState(false)
  const [message, setMessage] = useState(null)
  const [error, setError] = useState(null)
  const [wiring, setWiring] = useState(null)
  const [wiringError, setWiringError] = useState(null)

  const SA360_KEYS = [
    'SA360_CLIENT_ID',
    'SA360_CLIENT_SECRET',
    'SA360_OAUTH_REDIRECT_URI',
  ]
  const ADS_KEYS = [
    'ADS_CLIENT_ID',
    'ADS_CLIENT_SECRET',
    'ADS_REFRESH_TOKEN',
    'ADS_DEVELOPER_TOKEN',
  ]
  const ADS_FLAGS = ['ADS_FETCH_ENABLED']
  const SEARCH_KEYS = [
    'CONCIERGE_SEARCH_ENDPOINT',
    'CONCIERGE_SEARCH_KEY',
    'CONCIERGE_SEARCH_INDEX',
    'AZURE_OPENAI_EMBEDDING_DEPLOYMENT',
  ]
  const SEARCH_FLAGS = [
    'ENABLE_VECTOR_INDEXING',
    'CONCIERGE_SEARCH_VECTOR_FIELD',
    'CONCIERGE_SEARCH_VECTOR_K',
    'CONCIERGE_SEARCH_HYBRID',
    'CONCIERGE_SEARCH_VECTOR_FILTER_MODE',
    'AZURE_OPENAI_EMBEDDING_DIMENSIONS',
    'AZURE_OPENAI_EMBEDDING_API_VERSION',
  ]

  const fetchEnv = async () => {
    if (!authed) return
    if (!authPass) {
      setError('Env GUI passphrase required to load environment variables.')
      return
    }
    setLoading(true)
    setError(null)
    try {
      const resp = await axios.get(`${API_BASE_URL}/api/settings/env`, {
        params: { admin_password: authPass },
      })
      const env = resp.data?.env || []
      setEnvList(env)
      if (!selectedKey && env.length > 0) {
        setSelectedKey(env[0].key)
      }
    } catch (err) {
      if (err?.response?.status === 403) {
        setAuthed(false)
        setError('Invalid env GUI passphrase.')
      } else {
        setError(err?.response?.data?.error || err?.message || 'Failed to load environment variables')
      }
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchEnv()
  }, [authed])

  const envValue = (key) => {
    const item = envList.find((entry) => entry.key === key)
    return item ? item.value : null
  }

  const isKeySet = (key) => {
    const value = envValue(key)
    return value !== null && value !== undefined && value !== ''
  }

  const summarizeKeys = (keys) => {
    const present = keys.filter((key) => isKeySet(key)).length
    return { present, total: keys.length, all: present === keys.length, any: present > 0 }
  }

  const sa360Summary = summarizeKeys(SA360_KEYS)
  const adsSummary = summarizeKeys(ADS_KEYS)
  const searchSummary = summarizeKeys(SEARCH_KEYS)

  const runWiringCheck = async () => {
    if (!authed) return
    setWiringLoading(true)
    setWiringError(null)
    try {
      const resp = await axios.get(`${API_BASE_URL}/api/diagnostics/health`)
      setWiring(resp.data || null)
    } catch (err) {
      setWiringError(err?.response?.data?.error || err?.message || 'Wiring check failed')
    } finally {
      setWiringLoading(false)
    }
  }

  const handleUpdate = async () => {
    if (!selectedKey || !adminPassword) {
      setError('Select a key and provide the admin password.')
      return
    }
    setLoading(true)
    setError(null)
    setMessage(null)
    try {
      const resp = await axios.post(`${API_BASE_URL}/api/settings/env/update`, {
        key: selectedKey,
        value: newValue,
        admin_password: adminPassword,
      })
      setMessage(`Updated ${resp.data?.updated}; value now set (masked).`)
      setNewValue('')
      // refresh list to show masked value
      await fetchEnv()
    } catch (err) {
      setError(err?.response?.data?.detail || err?.response?.data?.error || err?.message || 'Update failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <Container maxWidth="md">
      {!authed && (
        <Paper elevation={3} sx={{ p: 3, borderRadius: 3, mt: 4 }}>
          <Box mb={2}>
            <Typography variant="h5" fontWeight={800} gutterBottom>
              Env & Keys Access
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Enter the Env GUI passphrase to view variables. Admin passphrase is required to update.
            </Typography>
          </Box>
          {error && <Alert severity="error" onClose={() => setError(null)} sx={{ mb: 2 }}>{error}</Alert>}
          <TextField
            label="Passphrase"
            type="password"
            fullWidth
            value={authPass}
            onChange={(e) => setAuthPass(e.target.value)}
            sx={{ mb: 2 }}
          />
          <Button
            variant="contained"
            onClick={() => {
              if (!authPass) {
                setError('Enter the Env GUI passphrase.')
                return
              }
              setAuthed(true)
              setError(null)
            }}
            disabled={!authPass}
          >
            Unlock
          </Button>
        </Paper>
      )}

      {authed && (
        <>
          <Box mb={4} display="flex" alignItems="center" gap={2}>
            <KeyIcon sx={{ fontSize: 36, color: '#E60000' }} />
            <Box>
              <Typography variant="h3" fontWeight={800} gutterBottom>
                Environment & Keys
              </Typography>
              <Typography variant="body1" color="text.secondary">
                View and update allowed environment variables with admin password (runtime only).
              </Typography>
            </Box>
          </Box>

          <Paper elevation={3} sx={{ p: 3, borderRadius: 3, mb: 3 }}>
            <Typography variant="h6" fontWeight={700} gutterBottom>
              Integration Key Status
            </Typography>
            <Typography variant="body2" color="text.secondary" mb={2}>
              Search Ads 360 is required for performance and audits. Google/Microsoft Ads keys are optional and can remain empty until connected.
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <Paper variant="outlined" sx={{ p: 2, borderRadius: 2, borderColor: 'rgba(148,163,184,0.3)' }}>
                  <Stack direction="row" alignItems="center" justifyContent="space-between" mb={1}>
                    <Typography variant="subtitle1" fontWeight={700}>
                      Search Ads 360
                    </Typography>
                    <Chip
                      size="small"
                      label={sa360Summary.all ? 'Configured' : 'Missing keys'}
                      sx={{
                        borderColor: sa360Summary.all ? 'rgba(34,197,94,0.6)' : 'rgba(248,113,113,0.6)',
                        color: sa360Summary.all ? '#22c55e' : '#f87171',
                        backgroundColor: sa360Summary.all ? 'rgba(34,197,94,0.15)' : 'rgba(248,113,113,0.15)',
                      }}
                      variant="outlined"
                    />
                  </Stack>
                  <Stack direction="row" flexWrap="wrap" gap={1}>
                    {SA360_KEYS.map((key) => (
                      <Chip
                        key={key}
                        size="small"
                        label={key}
                        sx={{
                          borderColor: isKeySet(key) ? 'rgba(34,197,94,0.5)' : 'rgba(148,163,184,0.3)',
                          color: isKeySet(key) ? '#86efac' : 'var(--kai-text-soft)',
                          backgroundColor: isKeySet(key) ? 'rgba(34,197,94,0.08)' : 'transparent',
                        }}
                        variant="outlined"
                      />
                    ))}
                  </Stack>
                </Paper>
              </Grid>
              <Grid item xs={12} md={6}>
                <Paper variant="outlined" sx={{ p: 2, borderRadius: 2, borderColor: 'rgba(148,163,184,0.3)' }}>
                  <Stack direction="row" alignItems="center" justifyContent="space-between" mb={1}>
                    <Typography variant="subtitle1" fontWeight={700}>
                      Ads Integrations (Optional)
                    </Typography>
                    <Chip
                      size="small"
                      label={
                        adsSummary.all
                          ? 'Configured'
                          : adsSummary.any
                            ? 'Partial keys'
                            : 'Not configured'
                      }
                      sx={{
                        borderColor: adsSummary.all ? 'rgba(34,197,94,0.6)' : 'rgba(148,163,184,0.3)',
                        color: adsSummary.all ? '#22c55e' : '#cbd5f5',
                        backgroundColor: adsSummary.all ? 'rgba(34,197,94,0.15)' : 'rgba(148,163,184,0.1)',
                      }}
                      variant="outlined"
                    />
                  </Stack>
                  <Stack direction="row" flexWrap="wrap" gap={1} mb={1}>
                    {ADS_KEYS.map((key) => (
                      <Chip
                        key={key}
                        size="small"
                        label={key}
                        sx={{
                          borderColor: isKeySet(key) ? 'rgba(34,197,94,0.5)' : 'rgba(148,163,184,0.3)',
                          color: isKeySet(key) ? '#86efac' : 'var(--kai-text-soft)',
                          backgroundColor: isKeySet(key) ? 'rgba(34,197,94,0.08)' : 'transparent',
                        }}
                        variant="outlined"
                      />
                    ))}
                  </Stack>
                  <Stack direction="row" flexWrap="wrap" gap={1}>
                    {ADS_FLAGS.map((key) => {
                      const flagSet = isKeySet(key)
                      return (
                      <Chip
                        key={key}
                        size="small"
                        label={`${key}${flagSet ? ' (set)' : ''}`}
                        sx={{
                          borderColor: flagSet ? 'rgba(34,197,94,0.5)' : 'rgba(148,163,184,0.3)',
                          color: flagSet ? '#86efac' : 'var(--kai-text-soft)',
                          backgroundColor: flagSet ? 'rgba(34,197,94,0.08)' : 'transparent',
                        }}
                        variant="outlined"
                      />
                      )
                    })}
                  </Stack>
                  <Typography variant="caption" color="text.secondary" display="block" mt={1}>
                    Ads APIs are not connected yet. Keep keys empty until you are ready to enable them.
                  </Typography>
                </Paper>
              </Grid>
              <Grid item xs={12} md={12}>
                <Paper variant="outlined" sx={{ p: 2, borderRadius: 2, borderColor: 'rgba(148,163,184,0.3)' }}>
                  <Stack direction="row" alignItems="center" justifyContent="space-between" mb={1}>
                    <Typography variant="subtitle1" fontWeight={700}>
                      Retrieval + Vector Search (Azure)
                    </Typography>
                    <Chip
                      size="small"
                      label={searchSummary.all ? 'Configured' : 'Missing keys'}
                      sx={{
                        borderColor: searchSummary.all ? 'rgba(34,197,94,0.6)' : 'rgba(248,113,113,0.6)',
                        color: searchSummary.all ? '#22c55e' : '#f87171',
                        backgroundColor: searchSummary.all ? 'rgba(34,197,94,0.15)' : 'rgba(248,113,113,0.15)',
                      }}
                      variant="outlined"
                    />
                  </Stack>
                  <Stack direction="row" flexWrap="wrap" gap={1} mb={1}>
                    {SEARCH_KEYS.map((key) => (
                      <Chip
                        key={key}
                        size="small"
                        label={key}
                        sx={{
                          borderColor: isKeySet(key) ? 'rgba(34,197,94,0.5)' : 'rgba(148,163,184,0.3)',
                          color: isKeySet(key) ? '#86efac' : 'var(--kai-text-soft)',
                          backgroundColor: isKeySet(key) ? 'rgba(34,197,94,0.08)' : 'transparent',
                        }}
                        variant="outlined"
                      />
                    ))}
                  </Stack>
                  <Stack direction="row" flexWrap="wrap" gap={1}>
                    {SEARCH_FLAGS.map((key) => (
                      <Chip
                        key={key}
                        size="small"
                        label={key}
                        sx={{
                          borderColor: isKeySet(key) ? 'rgba(34,197,94,0.5)' : 'rgba(148,163,184,0.3)',
                          color: isKeySet(key) ? '#86efac' : 'var(--kai-text-soft)',
                          backgroundColor: isKeySet(key) ? 'rgba(34,197,94,0.08)' : 'transparent',
                        }}
                        variant="outlined"
                      />
                    ))}
                  </Stack>
                </Paper>
              </Grid>
            </Grid>
          </Paper>

          <Paper elevation={3} sx={{ p: 3, borderRadius: 3, mb: 3 }}>
            <Stack direction="row" alignItems="center" justifyContent="space-between" mb={2} flexWrap="wrap" gap={2}>
              <Box>
                <Typography variant="h6" fontWeight={700}>
                  Wiring Check
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Verifies SA360 reachability and core dependency health.
                </Typography>
              </Box>
              <Button variant="outlined" onClick={runWiringCheck} disabled={wiringLoading}>
                {wiringLoading ? 'Checking...' : 'Run Check'}
              </Button>
            </Stack>
            {wiringError && <Alert severity="error" onClose={() => setWiringError(null)} sx={{ mb: 2 }}>{wiringError}</Alert>}
            {wiring && (
              <Stack spacing={1}>
                <Stack direction="row" spacing={1} flexWrap="wrap">
                  <Chip
                    size="small"
                    label={`Status: ${wiring.status || 'unknown'}`}
                    sx={{
                      borderColor: 'var(--kai-border)',
                      color: 'var(--kai-text)',
                      backgroundColor: 'var(--kai-surface-alt)',
                    }}
                    variant="outlined"
                  />
                  <Chip
                    size="small"
                    label={`Accounts: ${wiring.accounts?.count ?? 'n/a'}`}
                    sx={{
                      borderColor: 'var(--kai-border)',
                      color: 'var(--kai-text)',
                      backgroundColor: 'var(--kai-surface-alt)',
                    }}
                    variant="outlined"
                  />
                  <Chip
                    size="small"
                    label={`Errors: ${(wiring.errors || []).length}`}
                    sx={{
                      borderColor: 'var(--kai-border)',
                      color: 'var(--kai-text)',
                      backgroundColor: 'var(--kai-surface-alt)',
                    }}
                    variant="outlined"
                  />
                </Stack>
                <Typography variant="caption" color="text.secondary">
                  Uses /api/diagnostics/health. No keys are exposed.
                </Typography>
              </Stack>
            )}
          </Paper>

          <Paper elevation={3} sx={{ p: 3, borderRadius: 3 }}>
            <Stack direction="row" spacing={2} alignItems="center" mb={2}>
              <Button
                variant="outlined"
                startIcon={<RefreshIcon />}
                onClick={fetchEnv}
                disabled={loading}
              >
                Refresh
              </Button>
              {message && <Alert severity="success" onClose={() => setMessage(null)}>{message}</Alert>}
              {error && <Alert severity="error" onClose={() => setError(null)}>{error}</Alert>}
            </Stack>

            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Key</TableCell>
                  <TableCell>Value (masked)</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {envList.map((item) => (
                  <TableRow
                    key={item.key}
                    hover
                    selected={item.key === selectedKey}
                    onClick={() => {
                      setSelectedKey(item.key)
                      setNewValue('')
                    }}
                    sx={{ cursor: 'pointer' }}
                  >
                    <TableCell>{item.key}</TableCell>
                    <TableCell>{item.value || '-'}</TableCell>
                  </TableRow>
                ))}
                {envList.length === 0 && (
                  <TableRow>
                    <TableCell colSpan={2}>No variables loaded.</TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>

            <Divider sx={{ my: 3 }} />

            <Box display="grid" gridTemplateColumns="1fr" gap={2}>
              <TextField
                label="Selected Key"
                value={selectedKey}
                disabled
                fullWidth
              />
              <TextField
                label="New Value"
                placeholder="Enter new value"
                value={newValue}
                onChange={(e) => setNewValue(e.target.value)}
                fullWidth
              />
              <TextField
                label="Admin Password"
                type="password"
                placeholder="KAI_ACCESS_PASSWORD"
                value={adminPassword}
                onChange={(e) => setAdminPassword(e.target.value)}
                fullWidth
              />
              <Button
                variant="contained"
                startIcon={<SaveIcon />}
                onClick={handleUpdate}
                disabled={loading || !selectedKey}
                sx={{
                  mt: 1,
                  background: 'linear-gradient(135deg, #E60000, #C50000)',
                  fontWeight: 700,
                }}
              >
                Update
              </Button>
              <Typography variant="caption" color="text.secondary">
                Changes are applied at runtime only; container restarts may revert to deployment-time values.
              </Typography>
            </Box>
          </Paper>
        </>
      )}
    </Container>
  )
}
