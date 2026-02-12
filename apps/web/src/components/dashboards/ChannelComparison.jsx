/**
 * Channel Comparison Bar Chart
 * Visualizes performance metrics across PMax channels
 *
 * Data Source: /api/pmax/analyze -> channel_breakout
 * Architecture: Additive visualization that supplements AI text responses
 *
 * Features:
 * - Real-time backend data with observability
 * - Metric switching (cost, conversions, efficiency)
 * - Schema validation to prevent rendering bad data
 * - Responsive design with Material-UI integration
 */
import { Box, Typography, Skeleton, Alert, Paper, ToggleButton, ToggleButtonGroup, Chip } from '@mui/material'
import { ErrorOutline, CheckCircle, TrendingUp, TrendingDown } from '@mui/icons-material'
import { useState } from 'react'
import { useDashboardData, isDataUsable, getHealthMessage } from '../../hooks/useDashboardData'
import DataSourceBadge from './DataSourceBadge'

// Demo data for preview mode (shows what dashboards will look like)
const DEMO_CHANNEL_BREAKOUT = {
  search_cost: 2500,
  search_conversions: 45,
  shopping_cost: 3200,
  shopping_conversions: 72,
  video_cost: 1800,
  video_conversions: 28,
  display_cost: 1200,
  display_conversions: 18,
  remainder_cost: 300,
  remainder_conversions: 5,
  total_cost: 9000
}

export default function ChannelComparison({
  placements = [],
  spend = null,
  conversions = null,
  theme,
  defaultMetric = 'cost',
  isDemoData = false,  // When true, show demo data instead of making API call
  channelBreakout = null, // Optional: bypass fetch and render provided breakout
}) {
  const [selectedMetric, setSelectedMetric] = useState(defaultMetric)

  // Skip API call when in demo mode
  const shouldFetch = !isDemoData && !channelBreakout
  const { data, loading, health } = useDashboardData(
    shouldFetch ? '/api/pmax/analyze' : null,
    shouldFetch ? { placements, spend, conversions } : null,
    'channel_comparison'
  )

  const safeHealth = health || {
    fresh: true,
    valid: true,
    lastUpdate: Date.now(),
    warnings: [],
    error: null
  }

  // Loading state (not applicable in demo mode)
  if (loading && !isDemoData && shouldFetch) {
    return (
      <Box sx={{ p: 3, borderRadius: 2, border: '1px solid var(--kai-border-strong)' }}>
        <Skeleton variant="text" width="40%" height={40} sx={{ mb: 2 }} />
        <Skeleton variant="rectangular" height={350} />
        <Box sx={{ display: 'flex', gap: 1, mt: 2 }}>
          <Skeleton variant="rectangular" width={80} height={32} />
          <Skeleton variant="rectangular" width={80} height={32} />
          <Skeleton variant="rectangular" width={80} height={32} />
        </Box>
      </Box>
    )
  }

  // Error state (not applicable in demo mode)
  if (!isDemoData && !channelBreakout && !isDataUsable(safeHealth)) {
    return (
      <Alert
        severity="error"
        icon={<ErrorOutline />}
        sx={{
          borderRadius: 2,
          background: 'var(--kai-surface-alt)',
          border: '1px solid #f44336',
          color: '#fca5a5'
        }}
      >
        <Typography variant="subtitle2" fontWeight={600}>
          Dashboard Data Error
        </Typography>
        <Typography variant="body2" sx={{ mt: 0.5 }}>
          {getHealthMessage(safeHealth)}
        </Typography>
        {safeHealth.warnings?.length > 0 && (
          <ul style={{ marginTop: 8, paddingLeft: 20 }}>
            {safeHealth.warnings.map((warning, idx) => (
              <li key={idx}>{warning}</li>
            ))}
          </ul>
        )}
      </Alert>
    )
  }

  // Extract channel breakout data - use demo data in demo mode
  const breakout = channelBreakout
    || (isDemoData
      ? DEMO_CHANNEL_BREAKOUT
      : (data?.result?.channel_breakout || data?.channel_breakout))

  if (!breakout) {
    return (
      <Alert severity="warning" sx={{ borderRadius: 2 }}>
        No channel data available. Upload placement data to see channel comparison.
      </Alert>
    )
  }

  // Transform backend data into visualization format
  const channels = transformChannelData(breakout)
  const maxValue = Math.max(...channels.map(c => c[selectedMetric]))

  return (
    <Paper
      elevation={3}
      sx={{
        p: 3,
        borderRadius: 3,
        border: `1px solid ${theme?.borderColor || 'var(--kai-border-strong)'}`,
        background: 'var(--kai-bg)',
        boxShadow: '0 10px 30px rgba(0,0,0,0.3)'
      }}
    >
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Box>
          <Typography variant="h6" sx={{ color: 'var(--kai-text)', fontWeight: 700 }}>
            PMax Channel Performance
          </Typography>
          <Typography variant="body2" sx={{ color: 'var(--kai-text-soft)', mt: 0.5 }}>
            Compare channels by cost, conversions, and efficiency
          </Typography>
          {/* Source Citation: Elite-level pattern for data provenance */}
          <Box sx={{ mt: 1, display: 'flex', gap: 0.5 }}>
            <DataSourceBadge source="pmax_channel_splitter" timestamp={safeHealth.lastUpdate} />
            <DataSourceBadge source="backend_engine" timestamp={safeHealth.lastUpdate} />
          </Box>
        </Box>

        <Chip
          icon={<CheckCircle sx={{ fontSize: 16 }} />}
          label={isDemoData ? "Demo Data" : "Live Data"}
          size="small"
          sx={{
            background: isDemoData ? '#f59e0b20' : `${theme?.accentColor}20`,
            color: isDemoData ? '#f59e0b' : (theme?.accentColor || '#22d3ee'),
            border: `1px solid ${isDemoData ? '#f59e0b50' : (theme?.accentColor + '50')}`
          }}
        />
      </Box>

      {/* Metric Selector */}
      <Box sx={{ mb: 3 }}>
        <ToggleButtonGroup
          value={selectedMetric}
          exclusive
          onChange={(e, newMetric) => {
            if (newMetric !== null) {
              setSelectedMetric(newMetric)
            }
          }}
          size="small"
          sx={{
            '& .MuiToggleButton-root': {
              color: 'var(--kai-text-soft)',
              borderColor: 'var(--kai-border-strong)',
              '&.Mui-selected': {
                background: `${theme?.accentColor}20`,
                color: theme?.accentColor || '#22d3ee',
                borderColor: theme?.accentColor || '#22d3ee',
              }
            }
          }}
        >
          <ToggleButton value="cost">Cost</ToggleButton>
          <ToggleButton value="conversions">Conversions</ToggleButton>
          <ToggleButton value="efficiency">Efficiency</ToggleButton>
        </ToggleButtonGroup>
      </Box>

      {/* Bar Chart */}
      <Box sx={{ position: 'relative', minHeight: 350 }}>
        <BarChart
          channels={channels}
          metric={selectedMetric}
          maxValue={maxValue}
          theme={theme}
        />
      </Box>

      {/* Summary Statistics */}
      <Box sx={{ mt: 3, p: 2, borderRadius: 2, background: 'var(--kai-surface-alt)' }}>
        <Typography variant="subtitle2" sx={{ color: '#cbd5e1', mb: 1.5, fontWeight: 600 }}>
          Channel Summary
        </Typography>
        <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 2 }}>
          <SummaryCard
            label="Top Performer"
            value={getTopPerformer(channels, selectedMetric)}
            icon={<TrendingUp sx={{ color: '#10b981' }} />}
          />
          <SummaryCard
            label="Lowest Performer"
            value={getLowestPerformer(channels, selectedMetric)}
            icon={<TrendingDown sx={{ color: '#f59e0b' }} />}
          />
          <SummaryCard
            label="Total Channels"
            value={`${channels.length} active`}
            icon={null}
          />
        </Box>
      </Box>

      {/* Data health indicator */}
      <Typography
        variant="caption"
        sx={{ color: '#64748b', mt: 2, display: 'block', textAlign: 'right' }}
      >
        Last updated: {new Date(safeHealth.lastUpdate).toLocaleTimeString()}
      </Typography>
    </Paper>
  )
}

/**
 * Transform backend channel breakout into visualization format
 */
function transformChannelData(breakout) {
  const total = breakout.total_cost || 1

  const channels = [
    {
      name: 'Search',
      cost: breakout.search_cost || 0,
      conversions: breakout.search_conversions || 0,
      efficiency: calculateEfficiency(breakout.search_cost, breakout.search_conversions),
      color: '#3b82f6' // Blue
    },
    {
      name: 'Shopping',
      cost: breakout.shopping_cost || 0,
      conversions: breakout.shopping_conversions || 0,
      efficiency: calculateEfficiency(breakout.shopping_cost, breakout.shopping_conversions),
      color: '#10b981' // Green
    },
    {
      name: 'Video',
      cost: breakout.video_cost || 0,
      conversions: breakout.video_conversions || 0,
      efficiency: calculateEfficiency(breakout.video_cost, breakout.video_conversions),
      color: '#f59e0b' // Amber
    },
    {
      name: 'Display',
      cost: breakout.display_cost || 0,
      conversions: breakout.display_conversions || 0,
      efficiency: calculateEfficiency(breakout.display_cost, breakout.display_conversions),
      color: '#8b5cf6' // Purple
    },
    {
      name: 'Other',
      cost: breakout.remainder_cost || 0,
      conversions: breakout.remainder_conversions || 0,
      efficiency: calculateEfficiency(breakout.remainder_cost, breakout.remainder_conversions),
      color: '#64748b' // Gray
    }
  ]

  // Filter out zero-value channels
  return channels.filter(c => c.cost > 0)
}

/**
 * Calculate efficiency score (conversions per dollar spent)
 */
function calculateEfficiency(cost, conversions) {
  if (!cost || cost === 0) return 0
  return (conversions / cost).toFixed(4)
}

/**
 * Get top performing channel for selected metric
 */
function getTopPerformer(channels, metric) {
  if (channels.length === 0) return 'N/A'
  const top = channels.reduce((max, channel) =>
    channel[metric] > max[metric] ? channel : max
  )
  return top.name
}

/**
 * Get lowest performing channel for selected metric
 */
function getLowestPerformer(channels, metric) {
  if (channels.length === 0) return 'N/A'
  const lowest = channels.reduce((min, channel) =>
    channel[metric] < min[metric] ? channel : min
  )
  return lowest.name
}

/**
 * Bar Chart Component (Custom SVG)
 * Lightweight implementation using SVG rectangles
 */
function BarChart({ channels, metric, maxValue, theme }) {
  const chartWidth = 600
  const chartHeight = 300
  const barSpacing = 80
  const barWidth = 50
  const leftMargin = 100
  const bottomMargin = 50

  // Calculate scale
  const scale = (chartHeight - bottomMargin) / (maxValue || 1)

  return (
    <svg
      width="100%"
      height="100%"
      viewBox={`0 0 ${chartWidth} ${chartHeight}`}
      preserveAspectRatio="xMidYMid meet"
    >
      {/* Y-axis */}
      <line
        x1={leftMargin}
        y1={0}
        x2={leftMargin}
        y2={chartHeight - bottomMargin}
        stroke="var(--kai-border-strong)"
        strokeWidth={2}
      />

      {/* X-axis */}
      <line
        x1={leftMargin}
        y1={chartHeight - bottomMargin}
        x2={chartWidth}
        y2={chartHeight - bottomMargin}
        stroke="var(--kai-border-strong)"
        strokeWidth={2}
      />

      {/* Bars */}
      {channels.map((channel, idx) => {
        const barHeight = channel[metric] * scale
        const x = leftMargin + 20 + (idx * barSpacing)
        const y = chartHeight - bottomMargin - barHeight

        return (
          <g key={channel.name}>
            {/* Bar */}
            <rect
              x={x}
              y={y}
              width={barWidth}
              height={barHeight}
              fill={channel.color}
              opacity={0.8}
              rx={4}
            />

            {/* Value label */}
            <text
              x={x + barWidth / 2}
              y={y - 10}
              fill="var(--kai-text)"
              fontSize="12"
              fontWeight="600"
              textAnchor="middle"
            >
              {formatMetricValue(channel[metric], metric)}
            </text>

            {/* Channel label */}
            <text
              x={x + barWidth / 2}
              y={chartHeight - bottomMargin + 20}
              fill="var(--kai-text-soft)"
              fontSize="13"
              fontWeight="500"
              textAnchor="middle"
            >
              {channel.name}
            </text>
          </g>
        )
      })}

      {/* Y-axis label */}
      <text
        x={leftMargin - 60}
        y={chartHeight / 2}
        fill="#cbd5e1"
        fontSize="13"
        fontWeight="600"
        textAnchor="middle"
        transform={`rotate(-90, ${leftMargin - 60}, ${chartHeight / 2})`}
      >
        {getMetricLabel(metric)}
      </text>
    </svg>
  )
}

/**
 * Format metric values for display
 */
function formatMetricValue(value, metric) {
  switch (metric) {
    case 'cost':
      return `$${value.toFixed(0)}`
    case 'conversions':
      return value.toFixed(0)
    case 'efficiency':
      return value.toFixed(4)
    default:
      return value.toString()
  }
}

/**
 * Get human-readable metric label
 */
function getMetricLabel(metric) {
  switch (metric) {
    case 'cost':
      return 'Total Spend ($)'
    case 'conversions':
      return 'Conversions'
    case 'efficiency':
      return 'Conversions per $'
    default:
      return metric
  }
}

/**
 * Summary Card Component
 */
function SummaryCard({ label, value, icon }) {
  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
      {icon}
      <Box>
        <Typography variant="caption" sx={{ color: 'var(--kai-text-soft)', display: 'block' }}>
          {label}
        </Typography>
        <Typography variant="body2" sx={{ color: 'var(--kai-text)', fontWeight: 600 }}>
          {value}
        </Typography>
      </Box>
    </Box>
  )
}

