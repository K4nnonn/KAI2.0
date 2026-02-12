import { Box, Typography, LinearProgress, Stack } from '@mui/material'
import { motion, AnimatePresence } from 'framer-motion'
import { CheckCircle, HourglassEmpty } from '@mui/icons-material'

export default function ProgressStages({ stages, currentStage, theme }) {
  if (!stages || currentStage === null || currentStage === undefined) return null

  const current = stages[currentStage]
  const totalStages = stages.length
  const progress = ((currentStage + 1) / totalStages) * 100
  const isComplete = currentStage >= totalStages - 1

  // Calculate total estimated time remaining
  const remainingTime = stages
    .slice(currentStage + 1)
    .reduce((total, stage) => {
      const time = stage.duration
      if (typeof time === 'string') {
        // Parse durations like "5-10s", "2-3 min", "1-2 min"
        const match = time.match(/(\d+)-?(\d+)?\s*(s|sec|min|minute)/)
        if (match) {
          const avg = match[2] ? (parseInt(match[1]) + parseInt(match[2])) / 2 : parseInt(match[1])
          const unit = match[3]
          return total + (unit.startsWith('min') ? avg * 60 : avg)
        }
      }
      return total
    }, 0)

  const formatTime = (seconds) => {
    if (seconds < 60) return `${Math.round(seconds)}s`
    const mins = Math.floor(seconds / 60)
    const secs = Math.round(seconds % 60)
    return secs > 0 ? `${mins}m ${secs}s` : `${mins}m`
  }

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -10 }}
        transition={{ duration: 0.3 }}
      >
        <Box
          sx={{
            p: 3,
            borderRadius: 3,
            background: 'var(--kai-bg)',
            border: `1px solid ${theme.borderColor}33`,
          }}
        >
          {/* Progress Bar */}
          <LinearProgress
            variant="determinate"
            value={progress}
            sx={{
              height: 8,
              borderRadius: 2,
              mb: 2,
              background: 'var(--kai-surface-alt)',
              '& .MuiLinearProgress-bar': {
                background: theme.primaryGradient,
                borderRadius: 2,
              },
            }}
          />

          {/* Stage Info */}
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
            <Typography variant="body2" color="var(--kai-text-soft)" fontWeight={600}>
              Step {currentStage + 1} of {totalStages}
            </Typography>
            <Typography variant="body2" color="#64748b">
              {Math.round(progress)}% complete
              {remainingTime > 0 && ` â€¢ ${formatTime(remainingTime)} remaining`}
            </Typography>
          </Box>

          {/* Current Stage Message */}
          <Box display="flex" alignItems="center" gap={2} mb={3}>
            {isComplete ? (
              <CheckCircle sx={{ color: '#10b981', fontSize: 24 }} />
            ) : (
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
              >
                <HourglassEmpty sx={{ color: theme.accentColor, fontSize: 24 }} />
              </motion.div>
            )}
            <Typography variant="body1" color="var(--kai-text)" fontWeight={600}>
              {current.message}
            </Typography>
          </Box>

          {/* Stage List */}
          <Stack spacing={1}>
            {stages.map((stage, idx) => {
              const isPast = idx < currentStage
              const isCurrent = idx === currentStage
              const isFuture = idx > currentStage

              return (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.05 }}
                >
                  <Box
                    display="flex"
                    alignItems="center"
                    gap={1.5}
                    sx={{
                      opacity: isPast ? 0.6 : isCurrent ? 1 : 0.4,
                      transition: 'opacity 0.3s ease',
                    }}
                  >
                    {/* Status Icon */}
                    <Box
                      sx={{
                        width: 20,
                        height: 20,
                        borderRadius: '50%',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        background: isPast
                          ? '#10b981'
                          : isCurrent
                          ? theme.accentColor
                          : 'var(--kai-surface-alt)',
                        border: isCurrent ? `2px solid ${theme.focusColor}` : 'none',
                      }}
                    >
                      {isPast ? (
                        <CheckCircle sx={{ fontSize: 14, color: '#fff' }} />
                      ) : (
                        <Typography variant="caption" color="#fff" fontWeight={700}>
                          {idx + 1}
                        </Typography>
                      )}
                    </Box>

                    {/* Stage Text */}
                    <Typography
                      variant="caption"
                      color={isCurrent ? 'var(--kai-text)' : 'var(--kai-text-soft)'}
                      fontWeight={isCurrent ? 600 : 400}
                    >
                      {stage.message}
                    </Typography>

                    {/* Duration */}
                    {isFuture && stage.duration && (
                      <Typography variant="caption" color="#64748b">
                        ({stage.duration})
                      </Typography>
                    )}
                  </Box>
                </motion.div>
              )
            })}
          </Stack>
        </Box>
      </motion.div>
    </AnimatePresence>
  )
}

