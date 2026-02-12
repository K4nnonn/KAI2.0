/**
 * PreviewPanel - Right panel wrapper with demo/live data toggle
 *
 * Shows demo data initially to demonstrate value
 * Transitions to live data when analysis completes
 */
import { Box, Typography, Skeleton } from '@mui/material'
import { motion, AnimatePresence } from 'framer-motion'
import DemoDataBanner from './DemoDataBanner'

export default function PreviewPanel({
  toolId,
  themeColor,
  liveData,
  isLoading,
  children,       // Dashboard/visualization components
  emptyMessage = 'Chat with the AI to see your analysis here',
  forceLive = false,
}) {
  const isDemoMode = !forceLive && !liveData && !isLoading

  return (
    <Box
      sx={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        p: 2,
      }}
    >
      {/* Demo Data Banner */}
      <AnimatePresence>
        {isDemoMode && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.3 }}
          >
            <DemoDataBanner themeColor={themeColor} />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Loading State */}
      {isLoading && (
        <Box sx={{ p: 2 }}>
          <Skeleton
            variant="text"
            width="40%"
            height={32}
            sx={{ bgcolor: 'var(--kai-surface-muted)', mb: 2 }}
          />
          <Skeleton
            variant="rectangular"
            height={300}
            sx={{ bgcolor: 'var(--kai-surface-muted)', borderRadius: 2, mb: 2 }}
          />
          <Box display="flex" gap={1}>
            <Skeleton variant="rectangular" width={80} height={28} sx={{ bgcolor: 'var(--kai-surface-muted)', borderRadius: 1 }} />
            <Skeleton variant="rectangular" width={80} height={28} sx={{ bgcolor: 'var(--kai-surface-muted)', borderRadius: 1 }} />
            <Skeleton variant="rectangular" width={80} height={28} sx={{ bgcolor: 'var(--kai-surface-muted)', borderRadius: 1 }} />
          </Box>
        </Box>
      )}

      {/* Content Area */}
      <Box
        sx={{
          flex: 1,
          overflowY: 'auto',
          opacity: isDemoMode ? 0.7 : 1,
          transition: 'opacity 0.3s ease',
          filter: isDemoMode ? 'saturate(0.7)' : 'none',
        }}
      >
        <AnimatePresence mode="wait">
          <motion.div
            key={liveData ? 'live' : 'demo'}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3 }}
          >
            {children}
          </motion.div>
        </AnimatePresence>
      </Box>

      {/* Empty State (no demo data provided) */}
      {!children && !isLoading && (
        <Box
          sx={{
            flex: 1,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'var(--kai-border)',
            textAlign: 'center',
            p: 4,
          }}
        >
          <Box
            sx={{
              width: 80,
              height: 80,
              borderRadius: '50%',
              background: `${themeColor}15`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              mb: 2,
            }}
          >
            <Typography sx={{ fontSize: 32 }}>
              {toolId === 'pmax' ? 'ðŸ“Š' : toolId === 'creative' ? 'âœ¨' : 'ðŸ”'}
            </Typography>
          </Box>
          <Typography variant="body2" sx={{ maxWidth: 250 }}>
            {emptyMessage}
          </Typography>
        </Box>
      )}
    </Box>
  )
}

