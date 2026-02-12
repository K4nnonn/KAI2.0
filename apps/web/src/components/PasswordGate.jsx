import { useState, useEffect } from 'react'
import { Box, TextField, Button, Typography, Paper, CircularProgress } from '@mui/material'
import { Lock } from '@mui/icons-material'
import axios from 'axios'
import KaiLogo from './KaiLogo'
import { api } from '../config'
import { entraEnabled, entraGetIdToken, entraMode } from '../auth/entra'

// Server-side password validation - password is NOT stored in client code
const STORAGE_KEY = 'kai_access_granted_v2'

export default function PasswordGate({ children }) {
  const [password, setPassword] = useState('')
  const [error, setError] = useState(false)
  const [errorMessage, setErrorMessage] = useState('')
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [ssoLoading, setSsoLoading] = useState(false)
  const [ssoError, setSsoError] = useState('')

  const ssoIsEnabled = entraEnabled()
  const ssoIsRequired = ssoIsEnabled && entraMode() === 'required'

  useEffect(() => {
    // Check if already authenticated in this session (password gate)
    try {
      const granted = sessionStorage.getItem(STORAGE_KEY)
      if (granted === 'true') {
        setIsAuthenticated(true)
      }
    } catch {
      // ignore
    }
  }, [])

  useEffect(() => {
    // If Entra SSO is configured, attempt silent token acquisition.
    if (!ssoIsEnabled) return
    setSsoError('')
    entraGetIdToken({ interactive: false })
      .then((tok) => {
        if (tok) setIsAuthenticated(true)
      })
      .catch(() => {
        // Non-blocking; user can click Sign in.
      })
  }, [ssoIsEnabled])

  const handleSsoLogin = async () => {
    setSsoLoading(true)
    setSsoError('')
    try {
      const tok = await entraGetIdToken({ interactive: true })
      if (tok) {
        setIsAuthenticated(true)
      } else {
        setSsoError('Unable to sign in. Please retry.')
      }
    } catch (err) {
      setSsoError(err?.message || 'Unable to sign in. Please retry.')
    } finally {
      setSsoLoading(false)
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setIsLoading(true)
    setError(false)
    setErrorMessage('')

    try {
      // Validate password server-side
      const response = await axios.post(api.auth.verify, { password })

      if (response.data.authenticated) {
        try {
          sessionStorage.setItem(STORAGE_KEY, 'true')
        } catch {
          // ignore
        }
        setIsAuthenticated(true)
      }
    } catch (err) {
      setError(true)
      if (err.response?.status === 401) {
        setErrorMessage('Incorrect password')
      } else {
        setErrorMessage('Unable to verify. Please try again.')
      }
    } finally {
      setIsLoading(false)
    }
  }

  if (isAuthenticated) {
    return children
  }

  return (
    <Box
      sx={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background:
          'radial-gradient(circle at 20% 20%, rgba(34, 211, 238, 0.06), transparent 35%), radial-gradient(circle at 80% 10%, rgba(245, 158, 11, 0.05), transparent 40%), var(--kai-bg)',
      }}
    >
      <Paper
        elevation={0}
        sx={{
          p: 4,
          width: '100%',
          maxWidth: 420,
          background: 'var(--kai-surface)',
          border: '1px solid var(--kai-border-strong)',
          borderRadius: 3,
          boxShadow: '0 24px 60px rgba(15, 23, 42, 0.12)',
        }}
      >
        <Box sx={{ textAlign: 'center', mb: 4 }}>
          <KaiLogo size={60} />
          <Typography
            variant="h5"
            sx={{
              mt: 2,
              fontWeight: 700,
              background: 'linear-gradient(135deg, #22d3ee 0%, #3b82f6 50%, #f59e0b 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            KAI Platform
          </Typography>
          <Typography variant="body2" sx={{ color: 'var(--kai-text-soft)', mt: 1 }}>
            {ssoIsRequired ? 'Sign in to access' : 'Enter password or sign in to access'}
          </Typography>
        </Box>

        {ssoIsEnabled && (
          <Box sx={{ mb: ssoIsRequired ? 0 : 3 }}>
            <Button
              fullWidth
              variant="contained"
              size="large"
              onClick={handleSsoLogin}
              disabled={ssoLoading}
              sx={{
                py: 1.5,
                background: 'linear-gradient(135deg, #0ea5e9, #2563eb)',
                '&:hover': {
                  background: 'linear-gradient(135deg, #0284c7, #1d4ed8)',
                },
                '&:disabled': {
                  background: 'var(--kai-border-strong)',
                },
              }}
            >
              {ssoLoading ? <CircularProgress size={24} sx={{ color: '#fff' }} /> : 'Sign in with Microsoft'}
            </Button>
            {ssoError && (
              <Typography variant="caption" color="error" sx={{ display: 'block', mt: 1 }}>
                {ssoError}
              </Typography>
            )}
          </Box>
        )}

        {!ssoIsRequired && (
          <form onSubmit={handleSubmit}>
            <TextField
              fullWidth
              type="password"
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              error={error}
              helperText={error ? errorMessage : ''}
              disabled={isLoading}
              InputProps={{
                startAdornment: <Lock sx={{ color: '#64748b', mr: 1 }} />,
                sx: {
                  color: 'var(--kai-text)',
                  '& fieldset': { borderColor: 'var(--kai-border-strong)' },
                  '&:hover fieldset': { borderColor: '#22d3ee' },
                  '&.Mui-focused fieldset': { borderColor: '#22d3ee' },
                },
              }}
              sx={{ mb: 3 }}
            />
            <Button
              fullWidth
              type="submit"
              variant="contained"
              size="large"
              disabled={isLoading}
              sx={{
                py: 1.5,
                background: 'linear-gradient(135deg, #22d3ee, #3b82f6)',
                '&:hover': {
                  background: 'linear-gradient(135deg, #0ea5e9, #2563eb)',
                },
                '&:disabled': {
                  background: 'var(--kai-border-strong)',
                },
              }}
            >
              {isLoading ? <CircularProgress size={24} sx={{ color: '#fff' }} /> : 'Access KAI'}
            </Button>
          </form>
        )}
      </Paper>
    </Box>
  )
}

