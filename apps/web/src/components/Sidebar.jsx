import { useLocation, useNavigate } from 'react-router-dom'
import { Tooltip } from '@mui/material'
import KaiLogo from './KaiLogo'
import './Sidebar.css'

// Tooltip descriptions for each nav item
const tooltipDescriptions = {
  '/': 'AI assistant for PPC tasks - I can help with audits, creative, analysis, and strategic insights',
  '/klaudit': '100+ point Google Ads audit with industry benchmarks and actionable recommendations',
  '/creative-studio': 'Generate RSA headlines and descriptions with Google Ads character validation',
  '/pmax': 'Analyze Performance Max campaigns for channel mix, efficiency, and mobile app waste',
  '/serp': 'Check URL health, detect soft 404s, and monitor search rankings',
  '/sa360-columns': 'Browse SA360 conversion action columns available to your connected account(s)',
  '/info': 'Full technical architecture map with system flows and external dependencies',
  '/settings': 'Configure your preferences, API keys, and platform settings',
}

export default function Sidebar({ items, activePath }) {
  const location = useLocation()
  const navigate = useNavigate()
  const current = activePath || location.pathname

  return (
    <aside className="sb-container">
      <div className="sb-logo">
        <KaiLogo size={36} className="sb-logo-icon" />
        <div className="sb-logo-text-container">
          <span className="sb-logo-text">KAI</span>
          <span className="sb-logo-subtext">kelvin AI</span>
        </div>
      </div>
      <nav className="sb-nav">
        {items.map((item) => {
          const active = current === item.path
          const tooltip = tooltipDescriptions[item.path] || item.label

          return (
            <Tooltip
              key={item.label}
              title={tooltip}
              placement="right"
              arrow
              componentsProps={{
                tooltip: {
                  sx: {
                    bgcolor: 'var(--kai-surface-muted)',
                    border: '1px solid var(--kai-border-strong)',
                    boxShadow: '0 10px 30px rgba(0,0,0,0.5)',
                    fontSize: '0.875rem',
                    maxWidth: '280px',
                    '& .MuiTooltip-arrow': {
                      color: 'var(--kai-surface-muted)',
                      '&::before': {
                        border: '1px solid var(--kai-border-strong)',
                      },
                    },
                  },
                },
              }}
            >
              <button
                onClick={() => navigate(item.path)}
                className={`sb-item ${active ? 'sb-item-active' : ''}`}
              >
                <span className="sb-icon">{item.icon}</span>
                <span>{item.label}</span>
              </button>
            </Tooltip>
          )
        })}
      </nav>
    </aside>
  )
}
