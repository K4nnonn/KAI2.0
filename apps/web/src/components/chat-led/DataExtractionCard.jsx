/**
 * DataExtractionCard - Shows AI-extracted data with inline editing
 *
 * Displays extracted values from natural language input
 * Allows users to edit/confirm before running analysis
 */
import { useState } from 'react'
import {
  Box,
  Typography,
  Paper,
  TextField,
  Button,
  Stack,
  Chip,
  IconButton,
  InputAdornment,
} from '@mui/material'
import { motion } from 'framer-motion'
import { Check, Edit, Close, AutoAwesome } from '@mui/icons-material'

export default function DataExtractionCard({
  data,           // Object of extracted key-value pairs
  schema,         // Schema defining field types and labels
  themeColor,
  onConfirm,      // Called with confirmed data
  onCancel,       // Called to dismiss card
  onEdit,         // Called with (field, value) on inline edit
}) {
  const [editingField, setEditingField] = useState(null)
  const [editValue, setEditValue] = useState('')

  const startEdit = (field, value) => {
    setEditingField(field)
    setEditValue(value?.toString() || '')
  }

  const saveEdit = (field) => {
    onEdit?.(field, editValue)
    setEditingField(null)
  }

  const cancelEdit = () => {
    setEditingField(null)
    setEditValue('')
  }

  const formatValue = (value, type) => {
    if (value === null || value === undefined) return 'â€”'
    if (type === 'currency') return `$${Number(value).toLocaleString()}`
    if (type === 'number') return Number(value).toLocaleString()
    if (type === 'array') return Array.isArray(value) ? value.join(', ') : value
    return String(value)
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.3 }}
    >
      <Paper
        elevation={0}
        sx={{
          p: 2,
          background: 'var(--kai-surface)',
          border: `1px solid ${themeColor}55`,
          borderRadius: 2,
          boxShadow: `0 4px 20px ${themeColor}15`,
        }}
      >
        {/* Header */}
        <Box display="flex" alignItems="center" gap={1.5} mb={2}>
          <Box
            sx={{
              width: 28,
              height: 28,
              borderRadius: '50%',
              background: `${themeColor}22`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <AutoAwesome sx={{ fontSize: 14, color: themeColor }} />
          </Box>
          <Typography variant="subtitle2" sx={{ color: themeColor, fontWeight: 600 }}>
            I extracted this from your message:
          </Typography>
        </Box>

        {/* Extracted Fields */}
        <Stack spacing={1.5}>
          {Object.entries(data).map(([field, value]) => {
            const fieldSchema = schema?.fields?.[field] || {}
            const isEditing = editingField === field

            return (
              <Box
                key={field}
                sx={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 1.5,
                  p: 1,
                  borderRadius: 1.5,
                  background: 'var(--kai-bg)',
                  border: '1px solid var(--kai-surface-muted)',
                }}
              >
                <Typography
                  variant="caption"
                  sx={{
                    color: 'var(--kai-text-soft)',
                    minWidth: 100,
                    fontWeight: 500,
                  }}
                >
                  {fieldSchema.label || field}:
                </Typography>

                {isEditing ? (
                  <Box display="flex" alignItems="center" gap={1} flex={1}>
                    <TextField
                      value={editValue}
                      onChange={(e) => setEditValue(e.target.value)}
                      size="small"
                      autoFocus
                      sx={{
                        flex: 1,
                        '& .MuiOutlinedInput-root': {
                          color: 'var(--kai-text)',
                          '& fieldset': { borderColor: themeColor },
                        },
                      }}
                      InputProps={{
                        startAdornment: fieldSchema.type === 'currency' ? (
                          <InputAdornment position="start">
                            <Typography color="#64748b">$</Typography>
                          </InputAdornment>
                        ) : null,
                      }}
                    />
                    <IconButton
                      size="small"
                      onClick={() => saveEdit(field)}
                      sx={{ color: '#10b981' }}
                    >
                      <Check sx={{ fontSize: 16 }} />
                    </IconButton>
                    <IconButton
                      size="small"
                      onClick={cancelEdit}
                      sx={{ color: '#64748b' }}
                    >
                      <Close sx={{ fontSize: 16 }} />
                    </IconButton>
                  </Box>
                ) : (
                  <>
                    <Typography
                      variant="body2"
                      sx={{ color: 'var(--kai-text)', fontWeight: 600, flex: 1 }}
                    >
                      {formatValue(value, fieldSchema.type)}
                    </Typography>
                    {fieldSchema.unit && (
                      <Typography variant="caption" sx={{ color: '#64748b' }}>
                        {fieldSchema.unit}
                      </Typography>
                    )}
                    <IconButton
                      size="small"
                      onClick={() => startEdit(field, value)}
                      sx={{ color: '#64748b', '&:hover': { color: themeColor } }}
                    >
                      <Edit sx={{ fontSize: 14 }} />
                    </IconButton>
                  </>
                )}
              </Box>
            )
          })}
        </Stack>

        {/* Action Buttons */}
        <Box display="flex" gap={1.5} mt={2.5}>
          <Button
            variant="contained"
            onClick={() => onConfirm?.(data)}
            startIcon={<Check />}
            sx={{
              background: themeColor,
              textTransform: 'none',
              fontWeight: 600,
              '&:hover': { background: `${themeColor}dd` },
            }}
          >
            Looks good, analyze!
          </Button>
          <Button
            variant="outlined"
            onClick={onCancel}
            sx={{
              borderColor: 'var(--kai-border-strong)',
              color: 'var(--kai-text-soft)',
              textTransform: 'none',
              '&:hover': { borderColor: '#64748b', background: 'var(--kai-surface-muted)' },
            }}
          >
            Let me add more details
          </Button>
        </Box>
      </Paper>
    </motion.div>
  )
}

