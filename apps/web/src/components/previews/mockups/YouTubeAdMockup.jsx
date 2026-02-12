/**
 * YouTubeAdMockup - YouTube Video Ad Preview
 *
 * Shows skippable in-stream ad format with:
 * - Video player frame
 * - Ad headline overlay
 * - CTA button
 * - Skip button
 * - Companion banner
 */
import { Box, Typography, Paper, Chip, IconButton } from '@mui/material'
import {
  PlayArrow,
  VolumeUp,
  Fullscreen,
  SkipNext,
  Info as InfoIcon,
} from '@mui/icons-material'

export default function YouTubeAdMockup({
  headline = '',
  description = '',
  businessName = 'Example Company',
  displayUrl = 'example.com',
  theme,
}) {
  return (
    <Box>
      {/* YouTube Player Frame */}
      <Paper
        elevation={0}
        sx={{
          background: '#000',
          borderRadius: 2,
          overflow: 'hidden',
          position: 'relative',
        }}
      >
        {/* Video Area */}
        <Box
          sx={{
            position: 'relative',
            paddingBottom: '56.25%', // 16:9 aspect ratio
            background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)',
          }}
        >
          {/* Video content placeholder */}
          <Box
            sx={{
              position: 'absolute',
              inset: 0,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <Box
              sx={{
                width: 80,
                height: 80,
                borderRadius: '50%',
                background: 'rgba(255,255,255,0.1)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <PlayArrow sx={{ fontSize: 48, color: 'rgba(255,255,255,0.5)' }} />
            </Box>
          </Box>

          {/* Ad Label */}
          <Box
            sx={{
              position: 'absolute',
              top: 12,
              left: 12,
              display: 'flex',
              alignItems: 'center',
              gap: 1,
            }}
          >
            <Chip
              label="Ad"
              size="small"
              sx={{
                height: 20,
                fontSize: '0.7rem',
                fontWeight: 700,
                background: 'rgba(255,204,0,0.9)',
                color: '#000',
                borderRadius: 0.5,
              }}
            />
            <Typography variant="caption" sx={{ color: '#fff', opacity: 0.8 }}>
              0:05 / 0:30
            </Typography>
          </Box>

          {/* Ad Info Panel (bottom left) */}
          <Box
            sx={{
              position: 'absolute',
              bottom: 60,
              left: 12,
              right: 200,
              background: 'rgba(0,0,0,0.7)',
              borderRadius: 1,
              p: 1.5,
              display: 'flex',
              alignItems: 'center',
              gap: 1.5,
            }}
          >
            <InfoIcon sx={{ color: '#fff', fontSize: 20 }} />
            <Box sx={{ flex: 1, minWidth: 0 }}>
              <Typography
                sx={{
                  color: '#fff',
                  fontWeight: 600,
                  fontSize: '0.9rem',
                  lineHeight: 1.2,
                  whiteSpace: 'nowrap',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                }}
              >
                {headline || 'Your Ad Headline Here'}
              </Typography>
              <Typography
                sx={{
                  color: 'rgba(255,255,255,0.7)',
                  fontSize: '0.75rem',
                }}
              >
                {displayUrl}
              </Typography>
            </Box>
            <Box
              sx={{
                px: 2,
                py: 0.75,
                background: '#3ea6ff',
                borderRadius: 1,
                fontSize: '0.85rem',
                fontWeight: 600,
                color: '#000',
                whiteSpace: 'nowrap',
                cursor: 'pointer',
                '&:hover': { background: '#65b8ff' },
              }}
            >
              Learn More
            </Box>
          </Box>

          {/* Skip Button */}
          <Box
            sx={{
              position: 'absolute',
              bottom: 60,
              right: 0,
              background: 'rgba(0,0,0,0.8)',
              borderTopLeftRadius: 4,
              borderBottomLeftRadius: 4,
              px: 2,
              py: 1,
              display: 'flex',
              alignItems: 'center',
              gap: 1,
              cursor: 'pointer',
              '&:hover': { background: 'rgba(0,0,0,0.9)' },
            }}
          >
            <Typography sx={{ color: '#fff', fontSize: '0.9rem' }}>Skip Ad</Typography>
            <SkipNext sx={{ color: '#fff', fontSize: 20 }} />
          </Box>

          {/* Video Controls Bar */}
          <Box
            sx={{
              position: 'absolute',
              bottom: 0,
              left: 0,
              right: 0,
              height: 48,
              background: 'linear-gradient(transparent, rgba(0,0,0,0.7))',
              display: 'flex',
              alignItems: 'center',
              px: 1,
            }}
          >
            <IconButton size="small" sx={{ color: '#fff' }}>
              <PlayArrow />
            </IconButton>

            {/* Progress bar */}
            <Box sx={{ flex: 1, mx: 2, height: 4, background: 'rgba(255,255,255,0.3)', borderRadius: 2 }}>
              <Box
                sx={{
                  width: '15%',
                  height: '100%',
                  background: '#ff0000',
                  borderRadius: 2,
                }}
              />
            </Box>

            <Typography variant="caption" sx={{ color: '#fff', mr: 2 }}>
              0:05 / 0:30
            </Typography>

            <IconButton size="small" sx={{ color: '#fff' }}>
              <VolumeUp />
            </IconButton>
            <IconButton size="small" sx={{ color: '#fff' }}>
              <Fullscreen />
            </IconButton>
          </Box>
        </Box>
      </Paper>

      {/* Companion Banner (below video) */}
      <Paper
        elevation={0}
        sx={{
          mt: 2,
          p: 2,
          borderRadius: 2,
          background: 'var(--kai-bg)',
          border: `1px solid ${theme?.borderColor || 'var(--kai-border-strong)'}`,
        }}
      >
        <Typography variant="caption" sx={{ color: 'var(--kai-text-soft)', mb: 1, display: 'block' }}>
          Companion Banner (300x60)
        </Typography>
        <Box
          sx={{
            width: 300,
            height: 60,
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            borderRadius: 1,
            display: 'flex',
            alignItems: 'center',
            px: 2,
            gap: 1.5,
          }}
        >
          <Box
            sx={{
              width: 40,
              height: 40,
              borderRadius: 1,
              background: 'rgba(255,255,255,0.2)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontWeight: 700,
              color: '#fff',
            }}
          >
            {businessName.charAt(0)}
          </Box>
          <Box sx={{ flex: 1 }}>
            <Typography sx={{ color: '#fff', fontWeight: 600, fontSize: '0.85rem', lineHeight: 1.2 }}>
              {headline.length > 25 ? headline.substring(0, 25) + '...' : headline || 'Your Headline'}
            </Typography>
            <Typography sx={{ color: 'rgba(255,255,255,0.7)', fontSize: '0.7rem' }}>
              {displayUrl}
            </Typography>
          </Box>
          <Box
            sx={{
              px: 1.5,
              py: 0.5,
              background: '#fff',
              borderRadius: 0.5,
              fontSize: '0.75rem',
              fontWeight: 600,
              color: '#333',
            }}
          >
            Visit
          </Box>
        </Box>
      </Paper>

      <Typography
        variant="caption"
        sx={{ display: 'block', textAlign: 'center', mt: 2, color: '#64748b' }}
      >
        Skippable In-Stream Ad preview - shown before or during YouTube videos
      </Typography>
    </Box>
  )
}

