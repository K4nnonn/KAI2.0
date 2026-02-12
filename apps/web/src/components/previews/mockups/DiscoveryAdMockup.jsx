/**
 * DiscoveryAdMockup - Discovery/Demand Gen Ad Preview
 *
 * Shows feed-style card ads as they appear in:
 * - YouTube Home feed
 * - Gmail Promotions
 * - Google Discover feed
 */
import { useState } from 'react'
import { Box, Typography, Paper, ToggleButtonGroup, ToggleButton, Chip, IconButton } from '@mui/material'
import {
  Home as HomeIcon,
  Email as EmailIcon,
  Explore as ExploreIcon,
  MoreVert,
  ThumbUp,
  ThumbDown,
  Share,
} from '@mui/icons-material'

const PLACEMENTS = [
  { id: 'youtube', label: 'YouTube Home', icon: HomeIcon },
  { id: 'discover', label: 'Discover', icon: ExploreIcon },
  { id: 'gmail', label: 'Gmail', icon: EmailIcon },
]

export default function DiscoveryAdMockup({
  headline = '',
  description = '',
  displayUrl = 'example.com',
  imageUrl = null,
  theme,
}) {
  const [placement, setPlacement] = useState('youtube')

  const handlePlacementChange = (event, newPlacement) => {
    if (newPlacement !== null) {
      setPlacement(newPlacement)
    }
  }

  return (
    <Box>
      {/* Placement selector */}
      <Paper
        elevation={0}
        sx={{
          p: 2,
          mb: 2,
          borderRadius: 2,
          background: 'var(--kai-bg)',
          border: `1px solid ${theme?.borderColor || 'var(--kai-border-strong)'}`,
        }}
      >
        <Typography variant="caption" sx={{ color: 'var(--kai-text-soft)', mb: 1, display: 'block' }}>
          Placement Preview
        </Typography>
        <ToggleButtonGroup
          value={placement}
          exclusive
          onChange={handlePlacementChange}
          size="small"
          sx={{
            '& .MuiToggleButton-root': {
              color: 'var(--kai-text-soft)',
              borderColor: 'var(--kai-border-strong)',
              px: 2,
              py: 0.5,
              textTransform: 'none',
              '&.Mui-selected': {
                color: theme?.accentColor || '#34d399',
                backgroundColor: `${theme?.accentColor || '#34d399'}15`,
              },
            },
          }}
        >
          {PLACEMENTS.map((p) => (
            <ToggleButton key={p.id} value={p.id}>
              <p.icon sx={{ fontSize: 18, mr: 0.5 }} />
              {p.label}
            </ToggleButton>
          ))}
        </ToggleButtonGroup>
      </Paper>

      {/* Feed Mockup */}
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'center',
          p: 3,
          background: placement === 'youtube' ? '#0f0f0f' : placement === 'discover' ? '#fff' : '#f6f8fc',
          borderRadius: 2,
        }}
      >
        {placement === 'youtube' && (
          <YouTubeHomeFeed
            headline={headline}
            description={description}
            displayUrl={displayUrl}
            imageUrl={imageUrl}
          />
        )}
        {placement === 'discover' && (
          <DiscoverFeed
            headline={headline}
            description={description}
            displayUrl={displayUrl}
            imageUrl={imageUrl}
          />
        )}
        {placement === 'gmail' && (
          <GmailPromotions
            headline={headline}
            description={description}
            displayUrl={displayUrl}
            imageUrl={imageUrl}
          />
        )}
      </Box>

      <Typography
        variant="caption"
        sx={{ display: 'block', textAlign: 'center', mt: 2, color: '#64748b' }}
      >
        Discovery/Demand Gen Ad preview - native feed placements
      </Typography>
    </Box>
  )
}

/**
 * YouTube Home Feed Card
 */
function YouTubeHomeFeed({ headline, description, displayUrl, imageUrl }) {
  return (
    <Box sx={{ width: 360 }}>
      {/* Video card */}
      <Box sx={{ mb: 2 }}>
        {/* Thumbnail */}
        <Box
          sx={{
            width: '100%',
            aspectRatio: '16/9',
            background: imageUrl
              ? `url(${imageUrl}) center/cover`
              : 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            borderRadius: 2,
            position: 'relative',
            mb: 1.5,
          }}
        >
          {/* Ad label */}
          <Chip
            label="Ad"
            size="small"
            sx={{
              position: 'absolute',
              bottom: 8,
              left: 8,
              height: 20,
              fontSize: '0.7rem',
              fontWeight: 700,
              background: 'rgba(0,0,0,0.8)',
              color: '#fff',
            }}
          />
          {/* Duration */}
          <Box
            sx={{
              position: 'absolute',
              bottom: 8,
              right: 8,
              px: 1,
              py: 0.25,
              background: 'rgba(0,0,0,0.8)',
              borderRadius: 0.5,
              fontSize: '0.75rem',
              color: '#fff',
            }}
          >
            0:30
          </Box>
        </Box>

        {/* Info */}
        <Box sx={{ display: 'flex', gap: 1.5 }}>
          <Box
            sx={{
              width: 36,
              height: 36,
              borderRadius: '50%',
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              flexShrink: 0,
            }}
          />
          <Box sx={{ flex: 1, minWidth: 0 }}>
            <Typography
              sx={{
                color: '#f1f1f1',
                fontWeight: 500,
                fontSize: '0.95rem',
                lineHeight: 1.3,
                mb: 0.5,
                display: '-webkit-box',
                WebkitLineClamp: 2,
                WebkitBoxOrient: 'vertical',
                overflow: 'hidden',
              }}
            >
              {headline || 'Your Ad Headline Here'}
            </Typography>
            <Typography sx={{ color: '#aaa', fontSize: '0.8rem' }}>
              {displayUrl} Â· Sponsored
            </Typography>
          </Box>
          <IconButton size="small" sx={{ color: '#aaa' }}>
            <MoreVert fontSize="small" />
          </IconButton>
        </Box>
      </Box>

      {/* Organic video for context */}
      <Box sx={{ opacity: 0.5 }}>
        <Box
          sx={{
            width: '100%',
            aspectRatio: '16/9',
            background: '#272727',
            borderRadius: 2,
            mb: 1.5,
          }}
        />
        <Box sx={{ display: 'flex', gap: 1.5 }}>
          <Box sx={{ width: 36, height: 36, borderRadius: '50%', background: '#383838' }} />
          <Box>
            <Typography sx={{ color: '#f1f1f1', fontSize: '0.9rem', mb: 0.5 }}>
              Regular Video Title
            </Typography>
            <Typography sx={{ color: '#aaa', fontSize: '0.75rem' }}>Channel Â· 123K views</Typography>
          </Box>
        </Box>
      </Box>
    </Box>
  )
}

/**
 * Google Discover Feed Card
 */
function DiscoverFeed({ headline, description, displayUrl, imageUrl }) {
  return (
    <Box sx={{ width: 400 }}>
      {/* Ad card */}
      <Paper
        elevation={1}
        sx={{
          borderRadius: 3,
          overflow: 'hidden',
          mb: 2,
        }}
      >
        {/* Image */}
        <Box
          sx={{
            width: '100%',
            aspectRatio: '16/9',
            background: imageUrl
              ? `url(${imageUrl}) center/cover`
              : 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          }}
        />

        {/* Content */}
        <Box sx={{ p: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
            <Box
              sx={{
                width: 20,
                height: 20,
                borderRadius: '50%',
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              }}
            />
            <Typography sx={{ color: '#70757a', fontSize: '0.8rem' }}>{displayUrl}</Typography>
            <Chip
              label="Ad"
              size="small"
              sx={{
                height: 18,
                fontSize: '0.65rem',
                fontWeight: 700,
                background: '#e8f0fe',
                color: '#1a73e8',
              }}
            />
          </Box>

          <Typography
            sx={{
              color: '#202124',
              fontWeight: 500,
              fontSize: '1.1rem',
              lineHeight: 1.3,
              mb: 1,
            }}
          >
            {headline || 'Your Ad Headline Here'}
          </Typography>

          <Typography
            sx={{
              color: '#5f6368',
              fontSize: '0.9rem',
              lineHeight: 1.4,
              display: '-webkit-box',
              WebkitLineClamp: 2,
              WebkitBoxOrient: 'vertical',
              overflow: 'hidden',
            }}
          >
            {description || 'Your description text appears here...'}
          </Typography>
        </Box>
      </Paper>

      {/* Organic card for context */}
      <Paper elevation={1} sx={{ borderRadius: 3, overflow: 'hidden', opacity: 0.5 }}>
        <Box sx={{ width: '100%', aspectRatio: '16/9', background: '#e8eaed' }} />
        <Box sx={{ p: 2 }}>
          <Typography sx={{ color: '#70757a', fontSize: '0.8rem', mb: 0.5 }}>
            news-source.com
          </Typography>
          <Typography sx={{ color: '#202124', fontWeight: 500 }}>Regular Article Title</Typography>
        </Box>
      </Paper>
    </Box>
  )
}

/**
 * Gmail Promotions Tab Card
 */
function GmailPromotions({ headline, description, displayUrl, imageUrl }) {
  return (
    <Box sx={{ width: 400 }}>
      {/* Promotions header */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2, px: 1 }}>
        <EmailIcon sx={{ color: '#34a853', fontSize: 20 }} />
        <Typography sx={{ color: '#202124', fontWeight: 500 }}>Promotions</Typography>
        <Chip
          label="1 new"
          size="small"
          sx={{
            height: 18,
            fontSize: '0.65rem',
            background: '#e8f0fe',
            color: '#1a73e8',
          }}
        />
      </Box>

      {/* Ad card */}
      <Paper
        elevation={0}
        sx={{
          borderRadius: 2,
          overflow: 'hidden',
          border: '1px solid #e0e0e0',
          mb: 2,
          background: '#fff',
        }}
      >
        {/* Image banner */}
        <Box
          sx={{
            width: '100%',
            height: 120,
            background: imageUrl
              ? `url(${imageUrl}) center/cover`
              : 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          }}
        />

        {/* Content */}
        <Box sx={{ p: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
            <Box
              sx={{
                width: 24,
                height: 24,
                borderRadius: '50%',
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '0.7rem',
                fontWeight: 700,
                color: '#fff',
              }}
            >
              {displayUrl.charAt(0).toUpperCase()}
            </Box>
            <Typography sx={{ color: '#202124', fontWeight: 600, fontSize: '0.9rem', flex: 1 }}>
              {displayUrl}
            </Typography>
            <Chip
              label="Ad"
              size="small"
              sx={{
                height: 18,
                fontSize: '0.65rem',
                fontWeight: 700,
                background: '#34a853',
                color: '#fff',
              }}
            />
          </Box>

          <Typography
            sx={{
              color: '#202124',
              fontWeight: 500,
              fontSize: '1rem',
              lineHeight: 1.3,
              mb: 0.5,
            }}
          >
            {headline || 'Your Ad Headline Here'}
          </Typography>

          <Typography
            sx={{
              color: '#5f6368',
              fontSize: '0.85rem',
              display: '-webkit-box',
              WebkitLineClamp: 2,
              WebkitBoxOrient: 'vertical',
              overflow: 'hidden',
            }}
          >
            {description || 'Your description text appears here...'}
          </Typography>

          {/* CTA */}
          <Box
            sx={{
              display: 'inline-block',
              mt: 1.5,
              px: 2,
              py: 0.75,
              background: '#1a73e8',
              borderRadius: 1,
              color: '#fff',
              fontWeight: 500,
              fontSize: '0.85rem',
            }}
          >
            Learn More
          </Box>
        </Box>
      </Paper>

      {/* Regular promotion for context */}
      <Paper
        elevation={0}
        sx={{
          borderRadius: 2,
          border: '1px solid #e0e0e0',
          p: 2,
          opacity: 0.5,
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
          <Box sx={{ width: 24, height: 24, borderRadius: '50%', background: '#e8eaed' }} />
          <Typography sx={{ color: '#5f6368', fontSize: '0.9rem' }}>other-store.com</Typography>
        </Box>
        <Typography sx={{ color: '#202124', fontWeight: 500 }}>Regular Promotion Email</Typography>
      </Paper>
    </Box>
  )
}

