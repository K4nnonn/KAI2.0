import { useState } from 'react'
import {
  Box,
  Container,
  Typography,
  Paper,
  Switch,
  FormControlLabel,
  Divider,
  Button,
} from '@mui/material'
import { Settings as SettingsIcon, Save } from '@mui/icons-material'
import { motion } from 'framer-motion'

export default function Settings() {
  const [settings, setSettings] = useState({
    aiChatEnabled: true,
    aiInsightsEnabled: true,
    darkMode: false,
    notifications: true,
  })

  const handleChange = (key) => (event) => {
    setSettings({ ...settings, [key]: event.target.checked })
  }

  const saveSettings = () => {
    // API call would go here
    alert('Settings saved!')
  }

  return (
    <Container maxWidth="md">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Box mb={4}>
          <Typography variant="h3" fontWeight={800} gutterBottom>
            Settings
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Manage your preferences and configurations
          </Typography>
        </Box>

        <Paper elevation={3} sx={{ p: 4, borderRadius: 3 }}>
          <Box display="flex" alignItems="center" gap={2} mb={3}>
            <SettingsIcon sx={{ fontSize: 32, color: '#E60000' }} />
            <Typography variant="h5" fontWeight={700}>
              AI Configuration
            </Typography>
          </Box>

          <FormControlLabel
            control={
              <Switch checked={settings.aiChatEnabled} onChange={handleChange('aiChatEnabled')} />
            }
            label={
              <Box>
                <Typography variant="body1" fontWeight={600}>
                  Enable AI Chat
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Use Azure OpenAI GPT-4 for intelligent chat responses
                </Typography>
              </Box>
            }
            sx={{ mb: 2, alignItems: 'flex-start' }}
          />

          <FormControlLabel
            control={
              <Switch
                checked={settings.aiInsightsEnabled}
                onChange={handleChange('aiInsightsEnabled')}
              />
            }
            label={
              <Box>
                <Typography variant="body1" fontWeight={600}>
                  Enable AI Insights
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Generate AI-powered strategic insights in audit reports
                </Typography>
              </Box>
            }
            sx={{ mb: 3, alignItems: 'flex-start' }}
          />

          <Divider sx={{ my: 3 }} />

          <Typography variant="h6" fontWeight={700} mb={2}>
            General Settings
          </Typography>

          <FormControlLabel
            control={<Switch checked={settings.darkMode} onChange={handleChange('darkMode')} />}
            label={
              <Box>
                <Typography variant="body1" fontWeight={600}>
                  Dark Mode
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Switch to dark theme (coming soon)
                </Typography>
              </Box>
            }
            sx={{ mb: 2, alignItems: 'flex-start' }}
            disabled
          />

          <FormControlLabel
            control={
              <Switch checked={settings.notifications} onChange={handleChange('notifications')} />
            }
            label={
              <Box>
                <Typography variant="body1" fontWeight={600}>
                  Notifications
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Receive updates about audits and reports
                </Typography>
              </Box>
            }
            sx={{ mb: 3, alignItems: 'flex-start' }}
          />

          <Button
            variant="contained"
            startIcon={<Save />}
            onClick={saveSettings}
            sx={{
              mt: 2,
              px: 4,
              py: 1.5,
              borderRadius: 2,
              background: 'linear-gradient(135deg, #E60000, #C50000)',
              fontWeight: 700,
              '&:hover': {
                background: 'linear-gradient(135deg, #FF1A1A, #E60000)',
              },
            }}
          >
            Save Settings
          </Button>
        </Paper>
      </motion.div>
    </Container>
  )
}
