/**
 * CreativePreview - Google Ads Format Mockup
 *
 * Shows RSA headlines and descriptions in Google Ads format
 * with character counts and quality indicators.
 *
 * Features:
 * - Live character count with limits
 * - Quality score estimation
 * - Google Ads visual mockup
 * - Demo mode with sample data
 */
import { Box, Typography, Paper, Chip, LinearProgress, Stack } from '@mui/material'
import { CheckCircle, Warning, Error as ErrorIcon } from '@mui/icons-material'

// Demo data for preview mode
const DEMO_HEADLINES = [
  'Transform Your Marketing Today',
  'AI-Powered Ad Solutions',
  'Get 3X More Conversions',
  'Free Trial Available Now',
  'Trusted by 10,000+ Businesses',
]

const DEMO_DESCRIPTIONS = [
  'Our AI platform analyzes your campaigns and delivers actionable insights. Start optimizing today.',
  'Save hours of manual work with automated ad generation. Character-perfect, policy-compliant copy.',
]

// Character limits for Google Ads
const HEADLINE_LIMIT = 30
const DESCRIPTION_LIMIT = 90

export default function CreativePreview({
  headlines = [],
  descriptions = [],
  theme,
  isDemoData = false,
}) {
  // Use demo data if no real data provided
  const displayHeadlines = isDemoData || headlines.length === 0 ? DEMO_HEADLINES : headlines
  const displayDescriptions = isDemoData || descriptions.length === 0 ? DEMO_DESCRIPTIONS : descriptions

  return (
    <Box>
      {/* Google Ads Visual Mockup */}
      <Paper
        elevation={0}
        sx={{
          p: 3,
          mb: 3,
          borderRadius: 2,
          background: '#ffffff',
          border: '1px solid #e0e0e0',
        }}
      >
        <Typography variant="caption" sx={{ color: '#70757a', mb: 1, display: 'block' }}>
          Ad Preview (Google Search)
        </Typography>

        {/* Ad Preview Box */}
        <Box sx={{ p: 2, background: '#fff', borderRadius: 1 }}>
          {/* Ad Label */}
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
            <Chip
              label="Ad"
              size="small"
              sx={{
                height: 18,
                fontSize: '0.7rem',
                fontWeight: 700,
                background: 'transparent',
                border: '1px solid #1a73e8',
                color: '#1a73e8',
                borderRadius: 0.5,
              }}
            />
            <Typography variant="caption" sx={{ color: '#202124' }}>
              www.example.com
            </Typography>
          </Box>

          {/* Headlines */}
          <Typography
            variant="h6"
            sx={{
              color: '#1a0dab',
              fontWeight: 400,
              fontSize: '1.2rem',
              lineHeight: 1.3,
              mb: 0.5,
              cursor: 'pointer',
              '&:hover': { textDecoration: 'underline' },
            }}
          >
            {displayHeadlines.slice(0, 3).join(' | ')}
          </Typography>

          {/* Descriptions */}
          <Typography variant="body2" sx={{ color: '#4d5156', lineHeight: 1.58 }}>
            {displayDescriptions[0]}
          </Typography>
        </Box>
      </Paper>

      {/* Headlines with character counts */}
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
            Headlines ({displayHeadlines.length}/15)
          </Typography>
          <Chip
            icon={isDemoData ? <Warning sx={{ fontSize: 14 }} /> : <CheckCircle sx={{ fontSize: 14 }} />}
            label={isDemoData ? 'Demo Data' : 'Your Copy'}
            size="small"
            sx={{
              background: isDemoData ? '#f59e0b20' : `${theme?.accentColor || '#34d399'}20`,
              color: isDemoData ? '#f59e0b' : (theme?.accentColor || '#34d399'),
              border: `1px solid ${isDemoData ? '#f59e0b50' : (theme?.accentColor || '#34d399') + '50'}`,
            }}
          />
        </Box>

        <Stack spacing={1.5}>
          {displayHeadlines.map((headline, idx) => (
            <HeadlineItem key={idx} text={headline} limit={HEADLINE_LIMIT} theme={theme} />
          ))}
        </Stack>
      </Paper>

      {/* Descriptions with character counts */}
      <Paper
        elevation={0}
        sx={{
          p: 3,
          borderRadius: 2,
          background: 'var(--kai-bg)',
          border: `1px solid ${theme?.borderColor || 'var(--kai-border-strong)'}`,
        }}
      >
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Typography variant="h6" sx={{ color: 'var(--kai-text)', fontWeight: 700 }}>
            Descriptions ({displayDescriptions.length}/4)
          </Typography>
        </Box>

        <Stack spacing={1.5}>
          {displayDescriptions.map((desc, idx) => (
            <DescriptionItem key={idx} text={desc} limit={DESCRIPTION_LIMIT} theme={theme} />
          ))}
        </Stack>
      </Paper>
    </Box>
  )
}

/**
 * Headline Item with character count indicator
 */
function HeadlineItem({ text, limit, theme }) {
  const charCount = text.length
  const isOverLimit = charCount > limit
  const percentage = Math.min((charCount / limit) * 100, 100)

  return (
    <Box
      sx={{
        p: 1.5,
        borderRadius: 1.5,
        background: 'var(--kai-surface-alt)',
        border: `1px solid ${isOverLimit ? '#ef4444' : 'var(--kai-border-strong)'}`,
      }}
    >
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
        <Typography variant="body2" sx={{ color: 'var(--kai-text)', fontWeight: 500 }}>
          {text}
        </Typography>
        <Chip
          icon={isOverLimit ? <ErrorIcon sx={{ fontSize: 12 }} /> : null}
          label={`${charCount}/${limit}`}
          size="small"
          sx={{
            height: 20,
            fontSize: '0.7rem',
            background: isOverLimit ? '#ef444420' : '#10b98120',
            color: isOverLimit ? '#ef4444' : '#10b981',
            border: `1px solid ${isOverLimit ? '#ef444450' : '#10b98150'}`,
          }}
        />
      </Box>
      <LinearProgress
        variant="determinate"
        value={percentage}
        sx={{
          height: 4,
          borderRadius: 2,
          background: 'var(--kai-border-strong)',
          '& .MuiLinearProgress-bar': {
            background: isOverLimit
              ? '#ef4444'
              : percentage > 80
              ? '#f59e0b'
              : (theme?.accentColor || '#34d399'),
          },
        }}
      />
    </Box>
  )
}

/**
 * Description Item with character count indicator
 */
function DescriptionItem({ text, limit, theme }) {
  const charCount = text.length
  const isOverLimit = charCount > limit
  const percentage = Math.min((charCount / limit) * 100, 100)

  return (
    <Box
      sx={{
        p: 1.5,
        borderRadius: 1.5,
        background: 'var(--kai-surface-alt)',
        border: `1px solid ${isOverLimit ? '#ef4444' : 'var(--kai-border-strong)'}`,
      }}
    >
      <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={1}>
        <Typography
          variant="body2"
          sx={{ color: 'var(--kai-text)', fontWeight: 400, flex: 1, mr: 2, lineHeight: 1.5 }}
        >
          {text}
        </Typography>
        <Chip
          icon={isOverLimit ? <ErrorIcon sx={{ fontSize: 12 }} /> : null}
          label={`${charCount}/${limit}`}
          size="small"
          sx={{
            height: 20,
            fontSize: '0.7rem',
            flexShrink: 0,
            background: isOverLimit ? '#ef444420' : '#10b98120',
            color: isOverLimit ? '#ef4444' : '#10b981',
            border: `1px solid ${isOverLimit ? '#ef444450' : '#10b98150'}`,
          }}
        />
      </Box>
      <LinearProgress
        variant="determinate"
        value={percentage}
        sx={{
          height: 4,
          borderRadius: 2,
          background: 'var(--kai-border-strong)',
          '& .MuiLinearProgress-bar': {
            background: isOverLimit
              ? '#ef4444'
              : percentage > 80
              ? '#f59e0b'
              : (theme?.accentColor || '#34d399'),
          },
        }}
      />
    </Box>
  )
}

