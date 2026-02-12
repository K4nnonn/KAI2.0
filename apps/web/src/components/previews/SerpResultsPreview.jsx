/**
 * SerpResultsPreview - URL Health Status Dashboard
 *
 * Shows URL health check results with visual status indicators,
 * grouping by domain, and health summary.
 *
 * Features:
 * - Status indicators (green/yellow/red)
 * - Domain grouping
 * - Health summary statistics
 * - Demo mode with sample data
 */
import { Box, Typography, Paper, Chip, Stack, LinearProgress, Avatar } from '@mui/material'
import {
  CheckCircle,
  Warning,
  Error as ErrorIcon,
  Link as LinkIcon,
  TrendingUp,
  TrendingDown,
} from '@mui/icons-material'

// Demo data for preview mode
const DEMO_RESULTS = [
  { url: 'https://example.com/products', status: 'healthy', statusCode: 200, responseTime: 245 },
  { url: 'https://example.com/about', status: 'healthy', statusCode: 200, responseTime: 180 },
  { url: 'https://example.com/old-page', status: 'soft_404', statusCode: 200, responseTime: 890 },
  { url: 'https://example.com/pricing', status: 'healthy', statusCode: 200, responseTime: 320 },
  { url: 'https://example.com/broken-link', status: 'error', statusCode: 404, responseTime: 0 },
]

// Status configurations
const STATUS_CONFIG = {
  healthy: {
    label: 'Healthy',
    color: '#10b981',
    icon: CheckCircle,
    description: 'Page is accessible and indexed',
  },
  soft_404: {
    label: 'Soft 404',
    color: '#f59e0b',
    icon: Warning,
    description: 'Returns 200 but appears to be an error page',
  },
  error: {
    label: 'Error',
    color: '#ef4444',
    icon: ErrorIcon,
    description: 'Page returned an error status',
  },
  redirect: {
    label: 'Redirect',
    color: '#3b82f6',
    icon: LinkIcon,
    description: 'Page redirects to another URL',
  },
}

export default function SerpResultsPreview({
  results = [],
  theme,
  isDemoData = false,
}) {
  // Normalize API results to expected shape
  const normalizeResult = (result) => {
    if (!result) return null
    // If backend returns numeric status + soft_404 boolean, map to statuses
    if (typeof result.status === 'number') {
      const isHealthy = result.status === 200 && result.soft_404 === false
      const mappedStatus = isHealthy ? 'healthy' : (result.soft_404 ? 'soft_404' : 'error')
      return {
        ...result,
        status: mappedStatus,
        statusCode: result.status,
      }
    }
    return result
  }

  // Use demo data if no real data provided
  const normalized = (results || []).map(normalizeResult).filter(Boolean)
  const displayResults = isDemoData || normalized.length === 0 ? DEMO_RESULTS : normalized

  // Calculate summary stats
  const healthyCount = displayResults.filter(r => r.status === 'healthy').length
  const warningCount = displayResults.filter(r => r.status === 'soft_404' || r.status === 'redirect').length
  const errorCount = displayResults.filter(r => r.status === 'error').length
  const healthPercentage = displayResults.length > 0
    ? Math.round((healthyCount / displayResults.length) * 100)
    : 0

  return (
    <Box>
      {/* Health Summary */}
      <Paper
        elevation={0}
        sx={{
          p: 3,
          mb: 3,
          borderRadius: 2,
          background: 'var(--kai-bg)',
          border: `1px solid ${theme?.borderColor || 'var(--kai-border-strong)'}`,
        }}
      >
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Typography variant="h6" sx={{ color: 'var(--kai-text)', fontWeight: 700 }}>
            Health Summary
          </Typography>
          <Chip
            icon={isDemoData ? <Warning sx={{ fontSize: 14 }} /> : <CheckCircle sx={{ fontSize: 14 }} />}
            label={isDemoData ? 'Demo Data' : 'Live Results'}
            size="small"
            sx={{
              background: isDemoData ? '#f59e0b20' : `${theme?.accentColor || '#a78bfa'}20`,
              color: isDemoData ? '#f59e0b' : (theme?.accentColor || '#a78bfa'),
              border: `1px solid ${isDemoData ? '#f59e0b50' : (theme?.accentColor || '#a78bfa') + '50'}`,
            }}
          />
        </Box>

        {/* Health Score */}
        <Box sx={{ mb: 3 }}>
          <Box display="flex" justifyContent="space-between" alignItems="baseline" mb={1}>
            <Typography variant="h3" sx={{ color: 'var(--kai-text)', fontWeight: 700 }}>
              {healthPercentage}%
            </Typography>
            <Typography variant="body2" sx={{ color: 'var(--kai-text-soft)' }}>
              URLs Healthy
            </Typography>
          </Box>
          <LinearProgress
            variant="determinate"
            value={healthPercentage}
            sx={{
              height: 8,
              borderRadius: 4,
              background: 'var(--kai-border-strong)',
              '& .MuiLinearProgress-bar': {
                background: healthPercentage >= 80
                  ? '#10b981'
                  : healthPercentage >= 50
                  ? '#f59e0b'
                  : '#ef4444',
                borderRadius: 4,
              },
            }}
          />
        </Box>

        {/* Stats Grid */}
        <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 2 }}>
          <StatCard
            label="Healthy"
            value={healthyCount}
            color="#10b981"
            icon={<CheckCircle sx={{ fontSize: 20 }} />}
          />
          <StatCard
            label="Warnings"
            value={warningCount}
            color="#f59e0b"
            icon={<Warning sx={{ fontSize: 20 }} />}
          />
          <StatCard
            label="Errors"
            value={errorCount}
            color="#ef4444"
            icon={<ErrorIcon sx={{ fontSize: 20 }} />}
          />
        </Box>
      </Paper>

      {/* URL Results List */}
      <Paper
        elevation={0}
        sx={{
          p: 3,
          borderRadius: 2,
          background: 'var(--kai-bg)',
          border: `1px solid ${theme?.borderColor || 'var(--kai-border-strong)'}`,
        }}
      >
        <Typography variant="h6" sx={{ color: 'var(--kai-text)', fontWeight: 700, mb: 2 }}>
          URL Status ({displayResults.length})
        </Typography>

        <Stack spacing={1.5}>
          {displayResults.map((result, idx) => (
            <UrlStatusItem key={idx} result={result} theme={theme} />
          ))}
        </Stack>
      </Paper>
    </Box>
  )
}

/**
 * Stat Card Component
 */
function StatCard({ label, value, color, icon }) {
  return (
    <Box
      sx={{
        p: 2,
        borderRadius: 1.5,
        background: 'var(--kai-surface-alt)',
        border: `1px solid ${color}33`,
        textAlign: 'center',
      }}
    >
      <Box sx={{ color, mb: 1 }}>{icon}</Box>
      <Typography variant="h5" sx={{ color: 'var(--kai-text)', fontWeight: 700 }}>
        {value}
      </Typography>
      <Typography variant="caption" sx={{ color: 'var(--kai-text-soft)' }}>
        {label}
      </Typography>
    </Box>
  )
}

/**
 * URL Status Item Component
 */
function UrlStatusItem({ result, theme }) {
  const status = STATUS_CONFIG[result.status] || STATUS_CONFIG.error
  const StatusIcon = status.icon

  return (
    <Box
      sx={{
        p: 2,
        borderRadius: 1.5,
        background: 'var(--kai-surface-alt)',
        border: `1px solid ${status.color}33`,
        display: 'flex',
        alignItems: 'center',
        gap: 2,
      }}
    >
      {/* Status Icon */}
      <Avatar
        sx={{
          width: 36,
          height: 36,
          background: `${status.color}20`,
          color: status.color,
        }}
      >
        <StatusIcon sx={{ fontSize: 20 }} />
      </Avatar>

      {/* URL and Status */}
      <Box sx={{ flex: 1, minWidth: 0 }}>
        <Typography
          variant="body2"
          sx={{
            color: 'var(--kai-text)',
            fontWeight: 500,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
        >
          {result.url}
        </Typography>
        <Typography variant="caption" sx={{ color: 'var(--kai-text-soft)' }}>
          {status.description}
        </Typography>
      </Box>

      {/* Status Badge */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
        {typeof result.statusCode === 'number' && (
          <Chip
            label={`Status: ${result.statusCode}`}
            size="small"
            sx={{
              height: 24,
              fontSize: '0.7rem',
              background: 'var(--kai-surface-alt)',
              color: '#cbd5e1',
              border: '1px solid var(--kai-border-strong)',
            }}
          />
        )}
        {typeof result.soft_404 === 'boolean' && (
          <Chip
            label={`soft_404: ${String(result.soft_404)}`}
            size="small"
            sx={{
              height: 24,
              fontSize: '0.7rem',
              background: result.soft_404 ? '#f59e0b20' : '#10b98120',
              color: result.soft_404 ? '#f59e0b' : '#10b981',
              border: `1px solid ${result.soft_404 ? '#f59e0b50' : '#10b98150'}`,
            }}
          />
        )}
        {result.responseTime > 0 && (
          <Chip
            icon={result.responseTime < 500 ? <TrendingUp sx={{ fontSize: 12 }} /> : <TrendingDown sx={{ fontSize: 12 }} />}
            label={`${result.responseTime}ms`}
            size="small"
            sx={{
              height: 24,
              fontSize: '0.7rem',
              background: result.responseTime < 500 ? '#10b98120' : '#f59e0b20',
              color: result.responseTime < 500 ? '#10b981' : '#f59e0b',
              border: `1px solid ${result.responseTime < 500 ? '#10b98150' : '#f59e0b50'}`,
            }}
          />
        )}
        <Chip
          label={status.label}
          size="small"
          sx={{
            height: 24,
            fontSize: '0.7rem',
            fontWeight: 600,
            background: `${status.color}20`,
            color: status.color,
            border: `1px solid ${status.color}50`,
          }}
        />
      </Box>
    </Box>
  )
}

