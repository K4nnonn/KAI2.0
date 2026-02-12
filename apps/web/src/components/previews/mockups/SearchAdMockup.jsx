/**
 * SearchAdMockup - Google Search Ad Preview
 *
 * Shows how RSA ads appear in Google Search results
 * with desktop and mobile variants.
 */
import { Box, Typography, Paper, Chip } from '@mui/material'

export default function SearchAdMockup({
  headlines = [],
  descriptions = [],
  displayUrl = 'example.com',
  device = 'desktop',
  theme,
}) {
  const isMobile = device === 'mobile'

  // Take first 3 headlines and first description
  const displayedHeadlines = headlines.slice(0, 3)
  const displayedDescription = descriptions[0] || ''

  return (
    <Box
      sx={{
        display: 'flex',
        justifyContent: 'center',
        p: 2,
      }}
    >
      {/* Device Frame */}
      <Box
        sx={{
          width: isMobile ? 375 : '100%',
          maxWidth: isMobile ? 375 : 650,
          background: isMobile ? 'var(--kai-surface-alt)' : 'transparent',
          borderRadius: isMobile ? 4 : 0,
          p: isMobile ? 2 : 0,
          border: isMobile ? '8px solid #374151' : 'none',
        }}
      >
        {/* Mobile notch */}
        {isMobile && (
          <Box
            sx={{
              width: 120,
              height: 24,
              background: '#374151',
              borderRadius: 2,
              mx: 'auto',
              mb: 2,
            }}
          />
        )}

        {/* Search bar mockup */}
        <Paper
          elevation={0}
          sx={{
            p: 1.5,
            mb: 2,
            borderRadius: 3,
            background: '#fff',
            border: '1px solid #dfe1e5',
            display: 'flex',
            alignItems: 'center',
            gap: 1,
          }}
        >
          <Box
            component="span"
            sx={{
              width: 20,
              height: 20,
              borderRadius: '50%',
              background: 'linear-gradient(135deg, #4285f4, #34a853, #fbbc05, #ea4335)',
            }}
          />
          <Typography
            variant="body2"
            sx={{ color: '#5f6368', flex: 1, fontSize: isMobile ? '0.85rem' : '1rem' }}
          >
            marketing automation software
          </Typography>
        </Paper>

        {/* Ad Result */}
        <Paper
          elevation={0}
          sx={{
            p: isMobile ? 2 : 2.5,
            borderRadius: 2,
            background: '#fff',
            border: '1px solid #e0e0e0',
          }}
        >
          {/* Sponsored label and URL */}
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
            <Typography
              variant="caption"
              sx={{
                color: '#202124',
                fontWeight: 500,
                fontSize: isMobile ? '0.7rem' : '0.75rem',
              }}
            >
              Sponsored
            </Typography>
          </Box>

          {/* Display URL with breadcrumb */}
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 0.5 }}>
            <Box
              sx={{
                width: 20,
                height: 20,
                borderRadius: '50%',
                background: '#f1f3f4',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '0.65rem',
                fontWeight: 700,
                color: '#5f6368',
              }}
            >
              {displayUrl.charAt(0).toUpperCase()}
            </Box>
            <Typography
              variant="caption"
              sx={{ color: '#202124', fontSize: isMobile ? '0.75rem' : '0.8rem' }}
            >
              {displayUrl}
            </Typography>
          </Box>

          {/* Headlines */}
          <Typography
            variant={isMobile ? 'body1' : 'h6'}
            sx={{
              color: '#1a0dab',
              fontWeight: 400,
              fontSize: isMobile ? '1rem' : '1.25rem',
              lineHeight: 1.3,
              mb: 0.5,
              cursor: 'pointer',
              '&:hover': { textDecoration: 'underline' },
            }}
          >
            {displayedHeadlines.join(' | ')}
          </Typography>

          {/* Description */}
          <Typography
            variant="body2"
            sx={{
              color: '#4d5156',
              lineHeight: 1.58,
              fontSize: isMobile ? '0.8rem' : '0.875rem',
            }}
          >
            {displayedDescription}
          </Typography>

          {/* Sitelinks (desktop only) */}
          {!isMobile && (
            <Box sx={{ display: 'flex', gap: 3, mt: 2, flexWrap: 'wrap' }}>
              {['Features', 'Pricing', 'Free Trial', 'Contact'].map((link) => (
                <Typography
                  key={link}
                  variant="body2"
                  sx={{
                    color: '#1a0dab',
                    cursor: 'pointer',
                    '&:hover': { textDecoration: 'underline' },
                  }}
                >
                  {link}
                </Typography>
              ))}
            </Box>
          )}
        </Paper>

        {/* Organic result below (for context) */}
        <Paper
          elevation={0}
          sx={{
            p: isMobile ? 2 : 2.5,
            mt: 2,
            borderRadius: 2,
            background: '#fff',
            border: '1px solid #e0e0e0',
            opacity: 0.6,
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 0.5 }}>
            <Box
              sx={{
                width: 20,
                height: 20,
                borderRadius: '50%',
                background: '#e8f0fe',
                fontSize: '0.65rem',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              O
            </Box>
            <Typography variant="caption" sx={{ color: '#202124' }}>
              organic-result.com
            </Typography>
          </Box>
          <Typography
            variant={isMobile ? 'body1' : 'h6'}
            sx={{ color: '#1a0dab', fontWeight: 400, fontSize: isMobile ? '1rem' : '1.1rem' }}
          >
            Organic Search Result
          </Typography>
          <Typography variant="body2" sx={{ color: '#4d5156', fontSize: isMobile ? '0.8rem' : '0.875rem' }}>
            This is an example organic search result appearing below your ad...
          </Typography>
        </Paper>
      </Box>
    </Box>
  )
}

