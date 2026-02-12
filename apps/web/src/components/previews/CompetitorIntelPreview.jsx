/**
 * CompetitorIntelPreview - Competitor Investment Intelligence Dashboard
 *
 * Shows competitor investment signals with visual indicators,
 * metric comparison bars, and AI interpretation.
 *
 * Features:
 * - Investment signal indicators (Ramping Up, Stable, Declining)
 * - Metric comparison bars (impression share, outranking rate)
 * - AI interpretation text
 * - Demo mode with sample data
 */
import { Box, Typography, Paper, Chip, Stack, LinearProgress, Avatar } from '@mui/material'
import {
  TrendingUp,
  TrendingDown,
  TrendingFlat,
  Business,
  Insights,
  Psychology,
} from '@mui/icons-material'

// Demo data for preview mode
const DEMO_COMPETITORS = [
  {
    competitor: 'homedepot.com',
    signal: 'ramping_up',
    confidence: 0.85,
    impression_share_current: 45,
    impression_share_previous: 25,
    outranking_rate: 68,
    interpretation: 'Home Depot appears to be significantly increasing their paid search investment. The 20-point jump in impression share combined with 68% outranking rate suggests aggressive bidding. Consider defensive bid increases on your core brand terms.',
  },
  {
    competitor: 'lowes.com',
    signal: 'stable',
    confidence: 0.72,
    impression_share_current: 32,
    impression_share_previous: 30,
    outranking_rate: 45,
    interpretation: "Lowe's is maintaining stable paid search investment with only minor fluctuations. Continue monitoring for any shifts in competitive behavior.",
  },
  {
    competitor: 'menards.com',
    signal: 'declining',
    confidence: 0.78,
    impression_share_current: 15,
    impression_share_previous: 28,
    outranking_rate: 22,
    interpretation: 'Menards appears to be reducing their paid search investment. This may be an opportunity to capture additional market share at lower cost.',
  },
]

// Signal configurations
const SIGNAL_CONFIG = {
  ramping_up: {
    label: 'Ramping Up',
    icon: TrendingUp,
    color: '#ef4444',  // Red - threat
    bgColor: '#ef444420',
    description: 'Increasing paid search investment',
  },
  stable: {
    label: 'Stable',
    icon: TrendingFlat,
    color: '#f59e0b',  // Amber - neutral
    bgColor: '#f59e0b20',
    description: 'Maintaining current investment levels',
  },
  declining: {
    label: 'Declining',
    icon: TrendingDown,
    color: '#10b981',  // Green - opportunity
    bgColor: '#10b98120',
    description: 'Reducing paid search investment',
  },
}

export default function CompetitorIntelPreview({
  competitors = [],
  theme,
  isDemoData = false,
}) {
  // Use demo data if no real data provided
  const displayCompetitors = isDemoData || competitors.length === 0 ? DEMO_COMPETITORS : competitors

  // Calculate summary stats
  const rampingCount = displayCompetitors.filter(c => c.signal === 'ramping_up').length
  const stableCount = displayCompetitors.filter(c => c.signal === 'stable').length
  const decliningCount = displayCompetitors.filter(c => c.signal === 'declining').length

  // Overall threat level
  const threatLevel = rampingCount > 0 ? 'high' : stableCount > decliningCount ? 'moderate' : 'low'

  return (
    <Box>
      {/* Market Summary */}
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
            Competitive Intelligence
          </Typography>
          <Chip
            icon={isDemoData ? <Psychology sx={{ fontSize: 14 }} /> : <Insights sx={{ fontSize: 14 }} />}
            label={isDemoData ? 'Demo Data' : 'Live Analysis'}
            size="small"
            sx={{
              background: isDemoData ? '#f59e0b20' : `${theme?.accentColor || '#a78bfa'}20`,
              color: isDemoData ? '#f59e0b' : (theme?.accentColor || '#a78bfa'),
              border: `1px solid ${isDemoData ? '#f59e0b50' : (theme?.accentColor || '#a78bfa') + '50'}`,
            }}
          />
        </Box>

        {/* Threat Level Indicator */}
        <Box sx={{ mb: 3 }}>
          <Box display="flex" justifyContent="space-between" alignItems="baseline" mb={1}>
            <Typography variant="body2" sx={{ color: 'var(--kai-text-soft)' }}>
              Competitive Pressure
            </Typography>
            <Chip
              label={threatLevel.toUpperCase()}
              size="small"
              sx={{
                height: 20,
                fontSize: '0.65rem',
                fontWeight: 700,
                background: threatLevel === 'high' ? '#ef444420' : threatLevel === 'moderate' ? '#f59e0b20' : '#10b98120',
                color: threatLevel === 'high' ? '#ef4444' : threatLevel === 'moderate' ? '#f59e0b' : '#10b981',
                border: `1px solid ${threatLevel === 'high' ? '#ef444450' : threatLevel === 'moderate' ? '#f59e0b50' : '#10b98150'}`,
              }}
            />
          </Box>
          <LinearProgress
            variant="determinate"
            value={threatLevel === 'high' ? 85 : threatLevel === 'moderate' ? 50 : 25}
            sx={{
              height: 8,
              borderRadius: 4,
              background: 'var(--kai-border-strong)',
              '& .MuiLinearProgress-bar': {
                background: threatLevel === 'high' ? '#ef4444' : threatLevel === 'moderate' ? '#f59e0b' : '#10b981',
                borderRadius: 4,
              },
            }}
          />
        </Box>

        {/* Signal Stats Grid */}
        <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 2 }}>
          <SignalStatCard
            label="Ramping Up"
            value={rampingCount}
            config={SIGNAL_CONFIG.ramping_up}
          />
          <SignalStatCard
            label="Stable"
            value={stableCount}
            config={SIGNAL_CONFIG.stable}
          />
          <SignalStatCard
            label="Declining"
            value={decliningCount}
            config={SIGNAL_CONFIG.declining}
          />
        </Box>
      </Paper>

      {/* Competitor Cards */}
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
          Competitor Analysis ({displayCompetitors.length})
        </Typography>

        <Stack spacing={2}>
          {displayCompetitors.map((competitor, idx) => (
            <CompetitorCard key={idx} competitor={competitor} theme={theme} />
          ))}
        </Stack>
      </Paper>
    </Box>
  )
}

/**
 * Signal Stat Card Component
 */
function SignalStatCard({ label, value, config }) {
  const Icon = config.icon

  return (
    <Box
      sx={{
        p: 2,
        borderRadius: 1.5,
        background: 'var(--kai-surface-alt)',
        border: `1px solid ${config.color}33`,
        textAlign: 'center',
      }}
    >
      <Box sx={{ color: config.color, mb: 1 }}>
        <Icon sx={{ fontSize: 24 }} />
      </Box>
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
 * Competitor Card Component
 */
function CompetitorCard({ competitor, theme }) {
  const signalConfig = SIGNAL_CONFIG[competitor.signal] || SIGNAL_CONFIG.stable
  const SignalIcon = signalConfig.icon

  // Calculate impression share delta
  const impressionDelta = competitor.impression_share_current && competitor.impression_share_previous
    ? competitor.impression_share_current - competitor.impression_share_previous
    : null

  return (
    <Box
      sx={{
        p: 3,
        borderRadius: 2,
        background: 'var(--kai-surface-alt)',
        border: `1px solid ${signalConfig.color}33`,
      }}
    >
      {/* Header: Competitor + Signal */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Box display="flex" alignItems="center" gap={1.5}>
          <Avatar
            sx={{
              width: 40,
              height: 40,
              background: `${signalConfig.color}20`,
              color: signalConfig.color,
            }}
          >
            <Business sx={{ fontSize: 22 }} />
          </Avatar>
          <Box>
            <Typography variant="subtitle1" sx={{ color: 'var(--kai-text)', fontWeight: 600 }}>
              {competitor.competitor}
            </Typography>
            <Typography variant="caption" sx={{ color: 'var(--kai-text-soft)' }}>
              {signalConfig.description}
            </Typography>
          </Box>
        </Box>

        <Chip
          icon={<SignalIcon sx={{ fontSize: 14 }} />}
          label={signalConfig.label}
          size="small"
          sx={{
            background: signalConfig.bgColor,
            color: signalConfig.color,
            border: `1px solid ${signalConfig.color}50`,
            fontWeight: 600,
            '& .MuiChip-icon': {
              color: signalConfig.color,
            },
          }}
        />
      </Box>

      {/* Metrics */}
      <Box sx={{ mb: 2 }}>
        {/* Impression Share */}
        {competitor.impression_share_current !== undefined && (
          <Box sx={{ mb: 2 }}>
            <Box display="flex" justifyContent="space-between" alignItems="baseline" mb={0.5}>
              <Typography variant="body2" sx={{ color: 'var(--kai-text-soft)' }}>
                Impression Share
              </Typography>
              <Box display="flex" alignItems="center" gap={1}>
                <Typography variant="body2" sx={{ color: 'var(--kai-text)', fontWeight: 600 }}>
                  {competitor.impression_share_current}%
                </Typography>
                {impressionDelta !== null && (
                  <Chip
                    label={`${impressionDelta > 0 ? '+' : ''}${impressionDelta}%`}
                    size="small"
                    sx={{
                      height: 18,
                      fontSize: '0.65rem',
                      fontWeight: 600,
                      background: impressionDelta > 0 ? '#ef444420' : '#10b98120',
                      color: impressionDelta > 0 ? '#ef4444' : '#10b981',
                      border: `1px solid ${impressionDelta > 0 ? '#ef444450' : '#10b98150'}`,
                    }}
                  />
                )}
              </Box>
            </Box>
            <LinearProgress
              variant="determinate"
              value={Math.min(competitor.impression_share_current, 100)}
              sx={{
                height: 6,
                borderRadius: 3,
                background: 'var(--kai-border-strong)',
                '& .MuiLinearProgress-bar': {
                  background: signalConfig.color,
                  borderRadius: 3,
                },
              }}
            />
          </Box>
        )}

        {/* Outranking Rate */}
        {competitor.outranking_rate !== undefined && (
          <Box>
            <Box display="flex" justifyContent="space-between" alignItems="baseline" mb={0.5}>
              <Typography variant="body2" sx={{ color: 'var(--kai-text-soft)' }}>
                Outranking Rate
              </Typography>
              <Typography variant="body2" sx={{ color: 'var(--kai-text)', fontWeight: 600 }}>
                {competitor.outranking_rate}%
              </Typography>
            </Box>
            <LinearProgress
              variant="determinate"
              value={Math.min(competitor.outranking_rate, 100)}
              sx={{
                height: 6,
                borderRadius: 3,
                background: 'var(--kai-border-strong)',
                '& .MuiLinearProgress-bar': {
                  background: theme?.accentColor || '#a78bfa',
                  borderRadius: 3,
                },
              }}
            />
          </Box>
        )}
      </Box>

      {/* AI Interpretation */}
      {competitor.interpretation && (
        <Box
          sx={{
            p: 2,
            borderRadius: 1.5,
            background: 'var(--kai-bg)',
            border: '1px solid var(--kai-border-strong)',
          }}
        >
          <Box display="flex" alignItems="center" gap={1} mb={1}>
            <Psychology sx={{ fontSize: 16, color: theme?.accentColor || '#a78bfa' }} />
            <Typography variant="caption" sx={{ color: theme?.accentColor || '#a78bfa', fontWeight: 600 }}>
              AI Analysis
            </Typography>
          </Box>
          <Typography variant="body2" sx={{ color: 'var(--kai-text-soft)', lineHeight: 1.6 }}>
            {competitor.interpretation}
          </Typography>
        </Box>
      )}

      {/* Confidence */}
      {competitor.confidence !== undefined && (
        <Box display="flex" justifyContent="flex-end" mt={1.5}>
          <Typography variant="caption" sx={{ color: '#64748b' }}>
            Confidence: {Math.round(competitor.confidence * 100)}%
          </Typography>
        </Box>
      )}
    </Box>
  )
}

