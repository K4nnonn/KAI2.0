/**
 * ChatLedLayout - Split-panel layout for AI-chat-led tool pages
 *
 * Provides consistent structure across PMaxDeepDive, CreativeStudio, and SerpMonitor
 * Left panel: AI chat interface
 * Right panel: Live preview / dashboard
 */
import { Box, Container, Typography, Paper, Avatar, Chip, useMediaQuery, useTheme } from '@mui/material'
import { motion } from 'framer-motion'

export default function ChatLedLayout({
  // Tool configuration
  toolId,           // 'pmax' | 'creative' | 'serp'
  toolName,         // Display name
  toolDescription,  // Subtitle
  toolIcon: ToolIcon,
  themeColor,       // Tool-specific accent color

  // Content
  chatPanel,        // AI chat interface component
  previewPanel,     // Dashboard/preview component

  // Optional hero badges
  badges = [],

  // Optional extra content in header (e.g., tabs)
  headerExtra = null,
}) {
  const muiTheme = useTheme()
  const isMobile = useMediaQuery(muiTheme.breakpoints.down('md'))

  return (
    <Container
      maxWidth="xl"
      sx={{
        minHeight: 'calc(100vh - 64px)',
        display: 'flex',
        flexDirection: 'column',
        py: 2,
        px: { xs: 1, md: 2 },
      }}
    >
      {/* Hero Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Paper
          elevation={0}
          sx={{
            p: 3,
            mb: 2,
            borderRadius: 3,
            background: `linear-gradient(135deg, ${themeColor}15 0%, ${themeColor}08 100%)`,
            border: `1px solid ${themeColor}33`,
            position: 'relative',
            overflow: 'hidden',
          }}
        >
          {/* Decorative orb */}
          <Box
            sx={{
              position: 'absolute',
              top: -30,
              right: -30,
              width: 120,
              height: 120,
              borderRadius: '50%',
              background: `radial-gradient(circle, ${themeColor}20 0%, transparent 70%)`,
            }}
          />

          <Box display="flex" alignItems="center" gap={2} position="relative" zIndex={1}>
            <Avatar
              sx={{
                width: 48,
                height: 48,
                background: `linear-gradient(135deg, ${themeColor}, ${themeColor}cc)`,
                boxShadow: `0 4px 20px ${themeColor}40`,
              }}
            >
              {ToolIcon && <ToolIcon sx={{ fontSize: 24 }} />}
            </Avatar>

            <Box flex={1}>
              <Typography
                variant="h5"
                sx={{
                  fontWeight: 700,
                  color: 'var(--kai-text)',
                  letterSpacing: '-0.01em',
                }}
              >
                {toolName}
              </Typography>
              <Typography variant="body2" sx={{ color: 'var(--kai-text-soft)', mt: 0.5 }}>
                {toolDescription}
              </Typography>
            </Box>

            <Box display="flex" gap={1}>
              {badges.map((badge, idx) => (
                <Chip
                  key={idx}
                  icon={badge.icon}
                  label={badge.label}
                  size="small"
                  sx={{
                    background: `${badge.color || themeColor}20`,
                    color: badge.color || themeColor,
                    border: `1px solid ${badge.color || themeColor}33`,
                    '& .MuiChip-icon': { color: badge.color || themeColor },
                  }}
                />
              ))}
            </Box>
          </Box>

          {/* Optional extra header content (e.g., tabs) */}
          {headerExtra && (
            <Box sx={{ mt: 2, position: 'relative', zIndex: 1 }}>
              {headerExtra}
            </Box>
          )}
        </Paper>
      </motion.div>

      {/* Split Panel Content */}
      <Box
        sx={{
          flex: 1,
          display: 'flex',
          flexDirection: isMobile ? 'column' : 'row',
          gap: 2,
          minHeight: 0, // Allow flex shrink
        }}
      >
        {/* Left Panel: Chat Interface */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          style={{
            flex: isMobile ? 'none' : '0 0 55%',
            display: 'flex',
            flexDirection: 'column',
            minHeight: isMobile ? '50vh' : 0,
          }}
        >
          <Paper
            elevation={0}
            sx={{
              flex: 1,
              display: 'flex',
              flexDirection: 'column',
              borderRadius: 3,
              border: '1px solid var(--kai-surface-muted)',
              background: 'var(--kai-bg)',
              overflow: 'hidden',
            }}
          >
            {chatPanel}
          </Paper>
        </motion.div>

        {/* Right Panel: Preview / Dashboard */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          style={{
            flex: isMobile ? 'none' : '1',
            display: 'flex',
            flexDirection: 'column',
            minHeight: isMobile ? '40vh' : 0,
          }}
        >
          <Paper
            elevation={0}
            sx={{
              flex: 1,
              display: 'flex',
              flexDirection: 'column',
              borderRadius: 3,
              border: `1px solid ${themeColor}22`,
              background: 'var(--kai-bg)',
              overflow: 'auto',
            }}
          >
            {previewPanel}
          </Paper>
        </motion.div>
      </Box>
    </Container>
  )
}

