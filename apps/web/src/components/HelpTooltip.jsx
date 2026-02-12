import { Tooltip, IconButton, Box, Typography } from '@mui/material'
import { InfoOutlined } from '@mui/icons-material'

export default function HelpTooltip({ title, content, icon, placement = 'top', theme }) {
  const tooltipContent = (
    <Box sx={{ maxWidth: 350, p: 1 }}>
      {title && (
        <Typography variant="subtitle2" fontWeight={700} mb={1} color="#fff">
          {title}
        </Typography>
      )}
      {typeof content === 'string' ? (
        <Typography variant="body2" sx={{ whiteSpace: 'pre-line', color: 'var(--kai-text)' }}>
          {content}
        </Typography>
      ) : (
        content
      )}
    </Box>
  )

  return (
    <Tooltip
      title={tooltipContent}
      placement={placement}
      arrow
      componentsProps={{
        tooltip: {
          sx: {
            bgcolor: 'var(--kai-surface-muted)',
            border: `1px solid ${theme?.borderColor || 'var(--kai-border-strong)'}`,
            boxShadow: '0 10px 30px rgba(0,0,0,0.5)',
            '& .MuiTooltip-arrow': {
              color: 'var(--kai-surface-muted)',
              '&::before': {
                border: `1px solid ${theme?.borderColor || 'var(--kai-border-strong)'}`,
              },
            },
          },
        },
      }}
    >
      <IconButton
        size="small"
        sx={{
          color: theme?.accentColor || '#64748b',
          '&:hover': {
            color: theme?.focusColor || 'var(--kai-text-soft)',
            background: `${theme?.accentColor || '#64748b'}15`,
          },
        }}
      >
        {icon || <InfoOutlined fontSize="small" />}
      </IconButton>
    </Tooltip>
  )
}

