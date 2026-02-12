/**
 * PMax Spend Sankey Diagram
 * Visualizes budget flow from total spend into specific channels
 *
 * Data Source: /api/pmax/analyze -> channel_breakout
 * Architecture: Additive visualization that supplements AI text responses
 *
 * Features:
 * - Real-time backend data with observability
 * - Interactive click handling for agent workflows
 * - Schema validation to prevent rendering bad data
 * - Responsive design with Material-UI integration
 */
import { Box, Typography, Skeleton, Alert, Paper, Chip } from '@mui/material'
import { ErrorOutline, CheckCircle } from '@mui/icons-material'
import { useDashboardData, isDataUsable, getHealthMessage } from '../../hooks/useDashboardData'
import DataSourceBadge from './DataSourceBadge'

// Demo data for preview mode (shows what dashboards will look like)
const DEMO_CHANNEL_BREAKOUT = {
  search_cost: 2500,
  shopping_cost: 3200,
  video_cost: 1800,
  display_cost: 1200,
  remainder_cost: 300,
  total_cost: 9000
}

export default function PMaxSpendSankey({
  placements = [],
  spend = null,
  conversions = null,
  theme,
  onNodeClick,
  interactive = true,
  isDemoData = false,  // When true, show demo data instead of making API call
  channelBreakout = null, // Optional: bypass fetch and render provided breakout
}) {
  // Skip API call when in demo mode; otherwise fetch live data
  const shouldFetch = !isDemoData && !channelBreakout
  const { data, loading, health } = useDashboardData(
    shouldFetch ? '/api/pmax/analyze' : null,
    shouldFetch ? { placements, spend, conversions } : null,
    'pmax_spend_sankey'
  )

  // Use demo data health when in demo mode to avoid null timestamps
  const healthState = isDemoData
    ? {
        fresh: true,
        valid: true,
        lastUpdate: Date.now(),
        error: null,
        warnings: [],
      }
    : health || {
        fresh: true,
        valid: true,
        lastUpdate: Date.now(),
        error: null,
        warnings: [],
      }

  // Loading state (not applicable in demo mode)
  if (loading && !isDemoData && shouldFetch) {
    return (
      <Box sx={{ p: 3, borderRadius: 2, border: '1px solid var(--kai-border-strong)' }}>
        <Skeleton variant="text" width="40%" height={40} sx={{ mb: 2 }} />
        <Skeleton variant="rectangular" height={400} />
        <Skeleton variant="text" width="30%" height={20} sx={{ mt: 1 }} />
      </Box>
    )
  }

  // Error state (not applicable in demo mode)
  if (!isDemoData && !isDataUsable(healthState)) {
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
          {getHealthMessage(healthState)}
        </Typography>
        {healthState.warnings?.length > 0 && (
          <ul style={{ marginTop: 8, paddingLeft: 20 }}>
            {healthState.warnings.map((warning, idx) => (
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
        No channel breakout data available. Upload placement data to see spend allocation.
      </Alert>
    )
  }

  // Transform backend data into visualization format
  const channels = transformChannelData(breakout)
  const totalSpend = breakout.total_cost || 0

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
            PMax Channel Spend Allocation
          </Typography>
          <Typography variant="body2" sx={{ color: 'var(--kai-text-soft)', mt: 0.5 }}>
            Budget flow analysis across Search, Shopping, Video, and Display
          </Typography>
          {/* Source Citation: Elite-level pattern for data provenance */}
          <Box sx={{ mt: 1, display: 'flex', gap: 0.5 }}>
            <DataSourceBadge source="pmax_channel_splitter" timestamp={healthState.lastUpdate} />
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

      {/* Sankey Visualization */}
      <Box sx={{ position: 'relative', height: 400, mb: 2 }}>
        <SankeyFlowDiagram
          channels={channels}
          totalSpend={totalSpend}
          theme={theme}
          onNodeClick={interactive ? onNodeClick : null}
        />
      </Box>

      {/* Channel Breakdown Table */}
      <Box sx={{ mt: 3, p: 2, borderRadius: 2, background: 'var(--kai-surface-alt)' }}>
        <Typography variant="subtitle2" sx={{ color: '#cbd5e1', mb: 1.5, fontWeight: 600 }}>
          Spend Breakdown
        </Typography>
        {channels.map((channel) => (
          <Box
            key={channel.name}
            sx={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              py: 1,
              borderBottom: '1px solid var(--kai-border-strong)',
              '&:last-child': { borderBottom: 'none' }
            }}
          >
            <Box display="flex" alignItems="center" gap={1}>
              <Box
                sx={{
                  width: 12,
                  height: 12,
                  borderRadius: '50%',
                  background: channel.color
                }}
              />
              <Typography variant="body2" sx={{ color: 'var(--kai-text)' }}>
                {channel.name}
              </Typography>
            </Box>
            <Box display="flex" alignItems="baseline" gap={2}>
              <Typography variant="body2" sx={{ color: 'var(--kai-text-soft)' }}>
                ${channel.value.toFixed(2)}
              </Typography>
              <Typography variant="caption" sx={{ color: '#64748b' }}>
                ({channel.percentage}%)
              </Typography>
            </Box>
          </Box>
        ))}
      </Box>

      {/* Data health indicator */}
      <Typography
        variant="caption"
        sx={{ color: '#64748b', mt: 2, display: 'block', textAlign: 'right' }}
      >
        Last updated: {new Date(healthState.lastUpdate || Date.now()).toLocaleTimeString()}
      </Typography>
    </Paper>
  )
}

/**
 * Transform backend channel breakout into visualization format
 */
function transformChannelData(breakout) {
  const total = breakout.total_cost || 1 // Prevent division by zero

  const channels = [
    {
      name: 'Search',
      value: breakout.search_cost || 0,
      color: '#3b82f6', // Blue
      percentage: ((breakout.search_cost / total) * 100).toFixed(1)
    },
    {
      name: 'Shopping',
      value: breakout.shopping_cost || 0,
      color: '#10b981', // Green
      percentage: ((breakout.shopping_cost / total) * 100).toFixed(1)
    },
    {
      name: 'Video',
      value: breakout.video_cost || 0,
      color: '#f59e0b', // Amber
      percentage: ((breakout.video_cost / total) * 100).toFixed(1)
    },
    {
      name: 'Display',
      value: breakout.display_cost || 0,
      color: '#8b5cf6', // Purple
      percentage: ((breakout.display_cost / total) * 100).toFixed(1)
    },
    {
      name: 'Other',
      value: breakout.remainder_cost || 0,
      color: '#64748b', // Gray
      percentage: ((breakout.remainder_cost / total) * 100).toFixed(1)
    }
  ]

  // Filter out zero-value channels
  return channels.filter(c => c.value > 0)
}

/**
 * Sankey Flow Diagram Component (Custom SVG)
 * Lightweight implementation using SVG paths
 */
function SankeyFlowDiagram({ channels, totalSpend, theme, onNodeClick }) {
  const width = 600
  const height = 400
  const sourceX = 100
  const targetX = 500
  const nodeWidth = 20

  // Calculate y-positions based on proportions
  let sourceY = 50
  let targetY = 50

  return (
    <svg
      width="100%"
      height="100%"
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="xMidYMid meet"
    >
      <defs>
        <linearGradient id="flowGradient" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor={theme?.accentColor || '#22d3ee'} stopOpacity="0.3" />
          <stop offset="100%" stopColor={theme?.accentColor || '#22d3ee'} stopOpacity="0.1" />
        </linearGradient>
      </defs>

      {/* Source Node (Total Spend) */}
      <rect
        x={sourceX}
        y={height / 2 - 100}
        width={nodeWidth}
        height={200}
        fill={theme?.accentColor || '#22d3ee'}
        rx={4}
        style={{ cursor: onNodeClick ? 'pointer' : 'default' }}
        onClick={() => onNodeClick?.({ name: 'Total Spend', value: totalSpend })}
      />
      <text
        x={sourceX - 10}
        y={height / 2}
        fill="var(--kai-text)"
        fontSize="14"
        fontWeight="600"
        textAnchor="end"
      >
        Total: ${totalSpend.toFixed(0)}
      </text>

      {/* Flow Links and Target Nodes */}
      {channels.map((channel, idx) => {
        const linkHeight = (channel.value / totalSpend) * 200
        const nodeHeight = linkHeight

        // Calculate positions
        const sourceLinkY = (height / 2 - 100) + sourceY
        const targetNodeY = (height / 2 - 100) + targetY

        // Bezier curve path
        const midX = (sourceX + targetX) / 2
        const path = `
          M ${sourceX + nodeWidth} ${sourceLinkY}
          C ${midX} ${sourceLinkY},
            ${midX} ${targetNodeY + nodeHeight / 2},
            ${targetX} ${targetNodeY + nodeHeight / 2}
        `

        // Increment positions for next channel
        sourceY += linkHeight
        targetY += nodeHeight + 10 // Add gap between target nodes

        return (
          <g key={channel.name}>
            {/* Flow Link */}
            <path
              d={path}
              fill="none"
              stroke={channel.color}
              strokeWidth={linkHeight}
              opacity={0.4}
              strokeLinecap="round"
            />

            {/* Target Node */}
            <rect
              x={targetX}
              y={targetNodeY}
              width={nodeWidth}
              height={nodeHeight}
              fill={channel.color}
              rx={4}
              style={{ cursor: onNodeClick ? 'pointer' : 'default' }}
              onClick={() => onNodeClick?.(channel)}
            />

            {/* Channel Label */}
            <text
              x={targetX + nodeWidth + 10}
              y={targetNodeY + nodeHeight / 2 + 5}
              fill="var(--kai-text)"
              fontSize="13"
              fontWeight="500"
            >
              {channel.name}: ${channel.value.toFixed(0)} ({channel.percentage}%)
            </text>
          </g>
        )
      })}
    </svg>
  )
}

