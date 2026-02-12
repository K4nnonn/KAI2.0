/**
 * Data Source Citation Badge
 * Implements the "Source Citation" pattern from elite-level architecture
 *
 * Pattern: Every data point shown in the UI includes metadata tag
 * Purpose: Allows users to verify data provenance and maintain trust
 * Reference: "source: google_ads_api" or "source: semrush_sensor"
 */
import { Chip, Tooltip } from '@mui/material'
import { Storage, Cloud, Assessment } from '@mui/icons-material'

const SOURCE_CONFIG = {
  'google_ads_api': {
    label: 'Google Ads API',
    icon: <Cloud sx={{ fontSize: 14 }} />,
    color: '#4285f4',
    description: 'Real-time data from Google Ads Reporting API'
  },
  'backend_engine': {
    label: 'KAI Engine',
    icon: <Assessment sx={{ fontSize: 14 }} />,
    color: '#22d3ee',
    description: 'Processed through KAI_UNIFIED_ENGINE analytics'
  },
  'pmax_channel_splitter': {
    label: 'PMax Deduction',
    icon: <Storage sx={{ fontSize: 14 }} />,
    color: '#10b981',
    description: 'Channel spend calculated using Mike Rhodes deduction logic'
  },
  'semantic_layer': {
    label: 'Semantic Layer',
    icon: <Storage sx={{ fontSize: 14 }} />,
    color: '#8b5cf6',
    description: 'Standardized metrics from dbt transformations'
  }
}

/**
 * Renders a small badge showing data source with hover tooltip
 *
 * @param {string} source - Source identifier (e.g., 'google_ads_api')
 * @param {string} timestamp - Optional ISO timestamp of data freshness
 * @param {string} size - Badge size: 'small' | 'medium'
 */
export default function DataSourceBadge({ source, timestamp, size = 'small' }) {
  const config = SOURCE_CONFIG[source] || {
    label: source,
    icon: <Storage sx={{ fontSize: 14 }} />,
    color: '#64748b',
    description: 'Data source'
  }

  const tooltipContent = (
    <>
      <div style={{ fontWeight: 600, marginBottom: 4 }}>{config.label}</div>
      <div style={{ fontSize: '0.75rem', color: '#cbd5e1' }}>{config.description}</div>
      {timestamp && (
        <div style={{ fontSize: '0.7rem', color: 'var(--kai-text-soft)', marginTop: 4 }}>
          Last updated: {new Date(timestamp).toLocaleString()}
        </div>
      )}
    </>
  )

  return (
    <Tooltip
      title={tooltipContent}
      placement="top"
      arrow
      componentsProps={{
        tooltip: {
          sx: {
            bgcolor: 'var(--kai-surface-muted)',
            border: '1px solid var(--kai-border-strong)',
            boxShadow: '0 4px 12px rgba(0,0,0,0.4)',
            maxWidth: '280px',
            '& .MuiTooltip-arrow': {
              color: 'var(--kai-surface-muted)',
              '&::before': {
                border: '1px solid var(--kai-border-strong)'
              }
            }
          }
        }
      }}
    >
      <Chip
        icon={config.icon}
        label={config.label}
        size={size}
        sx={{
          background: `${config.color}15`,
          color: config.color,
          border: `1px solid ${config.color}40`,
          fontSize: size === 'small' ? '0.7rem' : '0.75rem',
          height: size === 'small' ? 20 : 24,
          cursor: 'help',
          '& .MuiChip-icon': {
            color: config.color
          }
        }}
      />
    </Tooltip>
  )
}

/**
 * Multiple source badges for data that combines multiple sources
 */
export function DataSourceGroup({ sources, timestamp }) {
  return (
    <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
      {sources.map((source, idx) => (
        <DataSourceBadge key={idx} source={source} timestamp={timestamp} size="small" />
      ))}
    </div>
  )
}

