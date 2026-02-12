/**
 * GmailAdMockup - Gmail Ad Preview
 *
 * Shows Gmail ads in both collapsed (inbox) and expanded states:
 * - Collapsed: Looks like an email in the inbox
 * - Expanded: Full promotional format
 */
import { useState } from 'react'
import { Box, Typography, Paper, IconButton, Chip, ToggleButtonGroup, ToggleButton } from '@mui/material'
import {
  Star,
  StarBorder,
  Delete,
  Archive,
  MoreVert,
  KeyboardArrowDown,
  AttachFile,
} from '@mui/icons-material'

export default function GmailAdMockup({
  headline = '',
  description = '',
  businessName = 'Example Company',
  theme,
}) {
  const [viewState, setViewState] = useState('collapsed')
  const [starred, setStarred] = useState(false)

  const handleViewChange = (event, newView) => {
    if (newView !== null) {
      setViewState(newView)
    }
  }

  return (
    <Box>
      {/* View toggle */}
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
          View State
        </Typography>
        <ToggleButtonGroup
          value={viewState}
          exclusive
          onChange={handleViewChange}
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
          <ToggleButton value="collapsed">Collapsed (Inbox)</ToggleButton>
          <ToggleButton value="expanded">Expanded</ToggleButton>
        </ToggleButtonGroup>
      </Paper>

      {/* Gmail Interface Mockup */}
      <Paper
        elevation={0}
        sx={{
          background: '#fff',
          borderRadius: 2,
          overflow: 'hidden',
          border: '1px solid #e0e0e0',
        }}
      >
        {/* Gmail header bar */}
        <Box
          sx={{
            px: 2,
            py: 1,
            background: '#f6f8fc',
            borderBottom: '1px solid #e0e0e0',
            display: 'flex',
            alignItems: 'center',
            gap: 2,
          }}
        >
          <Typography sx={{ color: '#5f6368', fontSize: '0.85rem' }}>Promotions</Typography>
          <Chip
            label="1 new"
            size="small"
            sx={{
              height: 20,
              fontSize: '0.7rem',
              background: '#e8f0fe',
              color: '#1a73e8',
            }}
          />
        </Box>

        {viewState === 'collapsed' ? (
          /* Collapsed (Inbox) View */
          <Box>
            {/* The ad email row */}
            <Box
              sx={{
                display: 'flex',
                alignItems: 'center',
                px: 2,
                py: 1,
                borderBottom: '1px solid #f1f3f4',
                background: '#fffaf3',
                cursor: 'pointer',
                '&:hover': { boxShadow: 'inset 0 -1px 0 #e0e0e0, inset 0 1px 0 #e0e0e0' },
              }}
              onClick={() => setViewState('expanded')}
            >
              {/* Checkbox area */}
              <Box sx={{ width: 24, mr: 1 }}>
                <input type="checkbox" style={{ cursor: 'pointer' }} />
              </Box>

              {/* Star */}
              <IconButton
                size="small"
                onClick={(e) => {
                  e.stopPropagation()
                  setStarred(!starred)
                }}
              >
                {starred ? (
                  <Star sx={{ color: '#f4b400', fontSize: 20 }} />
                ) : (
                  <StarBorder sx={{ color: '#c5c5c5', fontSize: 20 }} />
                )}
              </IconButton>

              {/* Ad badge + Sender */}
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, width: 200, flexShrink: 0 }}>
                <Chip
                  label="Ad"
                  size="small"
                  sx={{
                    height: 16,
                    fontSize: '0.6rem',
                    fontWeight: 700,
                    background: '#34a853',
                    color: '#fff',
                    borderRadius: 0.5,
                  }}
                />
                <Typography
                  sx={{
                    fontWeight: 600,
                    fontSize: '0.9rem',
                    color: '#202124',
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                  }}
                >
                  {businessName}
                </Typography>
              </Box>

              {/* Subject + snippet */}
              <Box sx={{ flex: 1, minWidth: 0, mx: 2 }}>
                <Typography
                  sx={{
                    fontSize: '0.9rem',
                    color: '#202124',
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                  }}
                >
                  <Box component="span" sx={{ fontWeight: 600 }}>
                    {headline || 'Your Headline'}
                  </Box>
                  <Box component="span" sx={{ color: '#5f6368' }}>
                    {' - '}
                    {description || 'Your description text appears here...'}
                  </Box>
                </Typography>
              </Box>

              {/* Time */}
              <Typography sx={{ color: '#5f6368', fontSize: '0.75rem', flexShrink: 0 }}>
                Sponsored
              </Typography>
            </Box>

            {/* Regular email rows (for context) */}
            {[1, 2].map((i) => (
              <Box
                key={i}
                sx={{
                  display: 'flex',
                  alignItems: 'center',
                  px: 2,
                  py: 1,
                  borderBottom: '1px solid #f1f3f4',
                  opacity: 0.6,
                }}
              >
                <Box sx={{ width: 24, mr: 1 }}>
                  <input type="checkbox" disabled />
                </Box>
                <IconButton size="small" disabled>
                  <StarBorder sx={{ color: '#c5c5c5', fontSize: 20 }} />
                </IconButton>
                <Box sx={{ width: 200, flexShrink: 0 }}>
                  <Typography sx={{ fontSize: '0.9rem', color: '#5f6368' }}>Regular Sender</Typography>
                </Box>
                <Box sx={{ flex: 1 }}>
                  <Typography sx={{ fontSize: '0.9rem', color: '#5f6368' }}>
                    Regular email subject line...
                  </Typography>
                </Box>
                <Typography sx={{ color: '#5f6368', fontSize: '0.75rem' }}>Dec {10 - i}</Typography>
              </Box>
            ))}
          </Box>
        ) : (
          /* Expanded View */
          <Box sx={{ p: 3 }}>
            {/* Email header */}
            <Box sx={{ display: 'flex', alignItems: 'flex-start', mb: 3 }}>
              <Box
                sx={{
                  width: 48,
                  height: 48,
                  borderRadius: '50%',
                  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontWeight: 700,
                  color: '#fff',
                  fontSize: '1.2rem',
                  mr: 2,
                }}
              >
                {businessName.charAt(0)}
              </Box>
              <Box sx={{ flex: 1 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Typography sx={{ fontWeight: 600, fontSize: '1rem', color: '#202124' }}>
                    {businessName}
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
                <Typography variant="caption" sx={{ color: '#5f6368' }}>
                  to me
                </Typography>
              </Box>
              <Box sx={{ display: 'flex', gap: 1 }}>
                <IconButton size="small">
                  <Archive sx={{ fontSize: 20, color: '#5f6368' }} />
                </IconButton>
                <IconButton size="small">
                  <Delete sx={{ fontSize: 20, color: '#5f6368' }} />
                </IconButton>
                <IconButton size="small">
                  <MoreVert sx={{ fontSize: 20, color: '#5f6368' }} />
                </IconButton>
              </Box>
            </Box>

            {/* Subject */}
            <Typography
              variant="h5"
              sx={{ fontWeight: 400, color: '#202124', mb: 3, lineHeight: 1.3 }}
            >
              {headline || 'Your Headline Here'}
            </Typography>

            {/* Header image placeholder */}
            <Box
              sx={{
                width: '100%',
                height: 200,
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                borderRadius: 2,
                mb: 3,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <Typography sx={{ color: 'rgba(255,255,255,0.5)', fontSize: '1rem' }}>
                [Header Image]
              </Typography>
            </Box>

            {/* Description */}
            <Typography sx={{ color: '#3c4043', lineHeight: 1.8, mb: 3, fontSize: '0.95rem' }}>
              {description || 'Your description text appears here. This is where you can elaborate on your offer and convince users to take action.'}
            </Typography>

            {/* CTA Button */}
            <Box
              sx={{
                display: 'inline-block',
                px: 4,
                py: 1.5,
                background: '#1a73e8',
                borderRadius: 1,
                color: '#fff',
                fontWeight: 600,
                cursor: 'pointer',
                '&:hover': { background: '#1557b0' },
              }}
            >
              Visit Website
            </Box>
          </Box>
        )}
      </Paper>

      <Typography
        variant="caption"
        sx={{ display: 'block', textAlign: 'center', mt: 2, color: '#64748b' }}
      >
        Gmail Ad preview - appears in Promotions tab
      </Typography>
    </Box>
  )
}

