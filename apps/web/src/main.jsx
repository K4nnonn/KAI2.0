import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { ThemeProvider, createTheme, CssBaseline } from '@mui/material'
import App from './App'
import './index.css'

const buildInfo = {
  buildSha: import.meta.env.VITE_BUILD_SHA,
  buildTime: import.meta.env.VITE_BUILD_TIME,
  appVersion: import.meta.env.VITE_APP_VERSION,
}

if (typeof window !== 'undefined') {
  window.__BUILD__ = buildInfo
}

// Modern premium theme
const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      // Cool-side brand (Cyan/Blue)
      main: '#3b82f6',
      light: '#22d3ee',
      dark: '#2563eb',
    },
    secondary: {
      // Warm-side brand (Amber/Orange)
      main: '#f59e0b',
      light: '#fbbf24',
      dark: '#d97706',
    },
    background: {
      // Let the global body gradient show through; use Paper surfaces for content.
      default: 'transparent',
      paper: '#ffffff',
    },
    text: {
      primary: '#0f172a',
      secondary: '#475569',
    },
    divider: '#d7dee7',
    success: { main: '#10b981' },
    warning: { main: '#f59e0b' },
    error: { main: '#ef4444' },
    info: { main: '#3b82f6' },
  },
  typography: {
    fontFamily: '"Manrope", "Segoe UI", system-ui, -apple-system, sans-serif',
    h1: {
      fontFamily: '"Space Grotesk", "Manrope", "Segoe UI", system-ui, sans-serif',
      fontWeight: 800,
      fontSize: '2.75rem',
      letterSpacing: '-0.03em',
    },
    h2: {
      fontFamily: '"Space Grotesk", "Manrope", "Segoe UI", system-ui, sans-serif',
      fontWeight: 700,
      fontSize: '2rem',
      letterSpacing: '-0.02em',
    },
    h3: {
      fontFamily: '"Space Grotesk", "Manrope", "Segoe UI", system-ui, sans-serif',
      fontWeight: 600,
      fontSize: '1.5rem',
    },
  },
  shape: {
    borderRadius: 18,
  },
  shadows: [
    'none',
    '0 1px 3px rgba(15, 23, 42, 0.08)',
    '0 6px 18px rgba(15, 23, 42, 0.10)',
    '0 12px 32px rgba(15, 23, 42, 0.12)',
    '0 20px 44px rgba(15, 23, 42, 0.14)',
    ...Array(20).fill('0 0 0 0 rgba(0,0,0,0)'),
  ],
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 650,
          borderRadius: 999,
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: 999,
        },
      },
    },
  },
})

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </ThemeProvider>
  </React.StrictMode>,
)
