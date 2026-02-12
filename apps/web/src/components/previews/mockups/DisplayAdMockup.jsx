/**
 * DisplayAdMockup - Google Display Network Ad Preview
 *
 * Shows responsive display ads in various standard sizes:
 * - 300x250 (Medium Rectangle)
 * - 728x90 (Leaderboard)
 * - 160x600 (Wide Skyscraper)
 * - 320x50 (Mobile Banner)
 */
import { useState } from 'react'
import { Box, Typography, Paper, ToggleButtonGroup, ToggleButton, Chip } from '@mui/material'
import { Image as ImageIcon } from '@mui/icons-material'

const AD_SIZES = [
  { id: '300x250', label: '300x250', width: 300, height: 250, name: 'Medium Rectangle' },
  { id: '728x90', label: '728x90', width: 728, height: 90, name: 'Leaderboard' },
  { id: '160x600', label: '160x600', width: 160, height: 600, name: 'Skyscraper' },
  { id: '320x50', label: '320x50', width: 320, height: 50, name: 'Mobile Banner' },
]

export default function DisplayAdMockup({
  headline = '',
  description = '',
  businessName = 'Example Company',
  displayUrl = 'example.com',
  logoUrl = null,
  imageUrl = null,
  device = 'desktop',
  theme,
}) {
  const [selectedSize, setSelectedSize] = useState('300x250')
  const sizeConfig = AD_SIZES.find((s) => s.id === selectedSize) || AD_SIZES[0]

  const handleSizeChange = (event, newSize) => {
    if (newSize !== null) {
      setSelectedSize(newSize)
    }
  }

  // Truncate text based on ad size
  const truncateText = (text, maxLength) => {
    if (text.length <= maxLength) return text
    return text.substring(0, maxLength - 3) + '...'
  }

  const getHeadlineLength = () => {
    switch (selectedSize) {
      case '728x90':
      case '320x50':
        return 40
      case '160x600':
        return 20
      default:
        return 30
    }
  }

  const getDescriptionLength = () => {
    switch (selectedSize) {
      case '728x90':
      case '320x50':
        return 60
      case '160x600':
        return 50
      default:
        return 80
    }
  }

  return (
    <Box>
      {/* Size selector */}
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
          Select Ad Size
        </Typography>
        <ToggleButtonGroup
          value={selectedSize}
          exclusive
          onChange={handleSizeChange}
          size="small"
          sx={{
            flexWrap: 'wrap',
            gap: 0.5,
            '& .MuiToggleButton-root': {
              color: 'var(--kai-text-soft)',
              borderColor: 'var(--kai-border-strong)',
              fontSize: '0.75rem',
              px: 1.5,
              py: 0.5,
              '&.Mui-selected': {
                color: theme?.accentColor || '#34d399',
                backgroundColor: `${theme?.accentColor || '#34d399'}15`,
              },
            },
          }}
        >
          {AD_SIZES.map((size) => (
            <ToggleButton key={size.id} value={size.id}>
              {size.label}
            </ToggleButton>
          ))}
        </ToggleButtonGroup>
        <Typography variant="caption" sx={{ color: '#64748b', mt: 1, display: 'block' }}>
          {sizeConfig.name}
        </Typography>
      </Paper>

      {/* Ad Preview */}
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'center',
          p: 3,
          background: 'var(--kai-surface-muted)',
          borderRadius: 2,
          overflow: 'auto',
        }}
      >
        {/* The actual ad mockup */}
        <Box
          sx={{
            width: sizeConfig.width,
            height: sizeConfig.height,
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            borderRadius: 1,
            overflow: 'hidden',
            position: 'relative',
            display: 'flex',
            flexDirection: selectedSize === '728x90' || selectedSize === '320x50' ? 'row' : 'column',
            boxShadow: '0 4px 20px rgba(0,0,0,0.3)',
          }}
        >
          {/* Ad badge */}
          <Chip
            label="Ad"
            size="small"
            sx={{
              position: 'absolute',
              top: 4,
              left: 4,
              height: 16,
              fontSize: '0.6rem',
              fontWeight: 700,
              background: 'rgba(255,255,255,0.9)',
              color: '#333',
              zIndex: 10,
              '& .MuiChip-label': { px: 0.5 },
            }}
          />

          {/* Image area (for 300x250 and 160x600) */}
          {(selectedSize === '300x250' || selectedSize === '160x600') && (
            <Box
              sx={{
                flex: selectedSize === '300x250' ? '0 0 55%' : '0 0 40%',
                background: imageUrl
                  ? `url(${imageUrl}) center/cover`
                  : 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              {!imageUrl && (
                <ImageIcon sx={{ fontSize: selectedSize === '160x600' ? 40 : 60, color: 'rgba(255,255,255,0.3)' }} />
              )}
            </Box>
          )}

          {/* Content area */}
          <Box
            sx={{
              flex: 1,
              p: selectedSize === '320x50' ? 0.75 : selectedSize === '728x90' ? 1.5 : 2,
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'center',
              background: 'rgba(0,0,0,0.4)',
            }}
          >
            {/* Logo for horizontal formats */}
            {(selectedSize === '728x90' || selectedSize === '320x50') && (
              <Box
                sx={{
                  width: selectedSize === '320x50' ? 24 : 40,
                  height: selectedSize === '320x50' ? 24 : 40,
                  borderRadius: 1,
                  background: logoUrl ? `url(${logoUrl}) center/cover` : 'rgba(255,255,255,0.2)',
                  mr: 1.5,
                  flexShrink: 0,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '0.8rem',
                  fontWeight: 700,
                  color: '#fff',
                }}
              >
                {!logoUrl && businessName.charAt(0)}
              </Box>
            )}

            {/* Text content */}
            <Box sx={{ flex: 1, minWidth: 0 }}>
              <Typography
                sx={{
                  color: '#fff',
                  fontWeight: 700,
                  fontSize:
                    selectedSize === '320x50'
                      ? '0.75rem'
                      : selectedSize === '728x90'
                      ? '1rem'
                      : selectedSize === '160x600'
                      ? '0.85rem'
                      : '1.1rem',
                  lineHeight: 1.2,
                  mb: selectedSize === '320x50' ? 0 : 0.5,
                }}
              >
                {truncateText(headline, getHeadlineLength())}
              </Typography>

              {selectedSize !== '320x50' && (
                <Typography
                  sx={{
                    color: 'rgba(255,255,255,0.8)',
                    fontSize: selectedSize === '160x600' ? '0.7rem' : '0.8rem',
                    lineHeight: 1.3,
                    mb: 1,
                  }}
                >
                  {truncateText(description, getDescriptionLength())}
                </Typography>
              )}

              {/* CTA Button */}
              <Box
                sx={{
                  display: 'inline-block',
                  px: selectedSize === '320x50' ? 1 : 2,
                  py: selectedSize === '320x50' ? 0.25 : 0.5,
                  background: '#fff',
                  borderRadius: 1,
                  fontSize: selectedSize === '320x50' ? '0.65rem' : '0.75rem',
                  fontWeight: 600,
                  color: '#333',
                }}
              >
                Learn More
              </Box>
            </Box>
          </Box>
        </Box>
      </Box>

      {/* Info text */}
      <Typography
        variant="caption"
        sx={{ display: 'block', textAlign: 'center', mt: 2, color: '#64748b' }}
      >
        Responsive Display Ad preview - actual appearance may vary by placement
      </Typography>
    </Box>
  )
}

