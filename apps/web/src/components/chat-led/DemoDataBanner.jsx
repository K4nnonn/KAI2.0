/**
 * DemoDataBanner - Indicates preview mode with sample data
 */
import { Box, Typography, Chip } from '@mui/material'
import { Visibility } from '@mui/icons-material'

export default function DemoDataBanner({ themeColor, message }) {
  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'center',
        gap: 1.5,
        p: 1.5,
        mb: 2,
        borderRadius: 2,
        background: `${themeColor}10`,
        border: `1px dashed ${themeColor}44`,
      }}
    >
      <Visibility sx={{ fontSize: 18, color: themeColor }} />
      <Typography variant="caption" sx={{ color: 'var(--kai-text-soft)', flex: 1 }}>
        {message || 'Showing example data. Chat with AI to analyze your own data.'}
      </Typography>
      <Chip
        label="Preview"
        size="small"
        sx={{
          height: 20,
          fontSize: '0.65rem',
          background: `${themeColor}22`,
          color: themeColor,
          border: `1px solid ${themeColor}33`,
        }}
      />
    </Box>
  )
}

