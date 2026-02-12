/**
 * AdMockupGenerator - Multi-Format Ad Preview System
 *
 * Shows how ads would look across different Google ad placements:
 * - Google Search (desktop/mobile)
 * - Display (various sizes)
 * - YouTube (video ads)
 * - Gmail (collapsed/expanded)
 * - Discovery/Demand Gen (feed cards)
 *
 * Similar to Karooya's ad preview tool.
 */
import { useState } from 'react'
import { Box, Typography, Tabs, Tab, ToggleButtonGroup, ToggleButton, Paper } from '@mui/material'
import {
  Search as SearchIcon,
  ViewModule as DisplayIcon,
  YouTube as YouTubeIcon,
  Email as GmailIcon,
  Explore as DiscoveryIcon,
  DesktopWindows as DesktopIcon,
  PhoneIphone as MobileIcon,
} from '@mui/icons-material'
import SearchAdMockup from './mockups/SearchAdMockup'
import DisplayAdMockup from './mockups/DisplayAdMockup'
import YouTubeAdMockup from './mockups/YouTubeAdMockup'
import GmailAdMockup from './mockups/GmailAdMockup'
import DiscoveryAdMockup from './mockups/DiscoveryAdMockup'

// Demo data for preview mode
const DEMO_HEADLINES = [
  'Transform Your Marketing Today',
  'AI-Powered Ad Solutions',
  'Get 3X More Conversions',
  'Free Trial Available Now',
  'Trusted by 10,000+ Businesses',
]

const DEMO_DESCRIPTIONS = [
  'Our AI platform analyzes your campaigns and delivers actionable insights. Start optimizing today.',
  'Save hours of manual work with automated ad generation. Character-perfect, policy-compliant copy.',
]

const AD_FORMATS = [
  { id: 'search', label: 'Search', icon: SearchIcon },
  { id: 'display', label: 'Display', icon: DisplayIcon },
  { id: 'youtube', label: 'YouTube', icon: YouTubeIcon },
  { id: 'gmail', label: 'Gmail', icon: GmailIcon },
  { id: 'discovery', label: 'Discovery', icon: DiscoveryIcon },
]

export default function AdMockupGenerator({
  headlines = [],
  descriptions = [],
  businessName = 'Example Company',
  displayUrl = 'example.com',
  logoUrl = null,
  imageUrl = null,
  theme,
  isDemoData = false,
}) {
  const [activeFormat, setActiveFormat] = useState('search')
  const [device, setDevice] = useState('desktop')

  // Use demo data if no real data provided
  const displayHeadlines = isDemoData || headlines.length === 0 ? DEMO_HEADLINES : headlines
  const displayDescriptions = isDemoData || descriptions.length === 0 ? DEMO_DESCRIPTIONS : descriptions
  const isDemo = isDemoData || headlines.length === 0

  const handleFormatChange = (event, newValue) => {
    if (newValue !== null) {
      setActiveFormat(newValue)
    }
  }

  const handleDeviceChange = (event, newDevice) => {
    if (newDevice !== null) {
      setDevice(newDevice)
    }
  }

  // Determine which formats support device toggle
  const supportsDeviceToggle = ['search', 'display'].includes(activeFormat)

  return (
    <Box>
      {/* Header with format tabs and device toggle */}
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
        <Box display="flex" justifyContent="space-between" alignItems="center" flexWrap="wrap" gap={2}>
          {/* Format Tabs */}
          <Tabs
            value={activeFormat}
            onChange={handleFormatChange}
            variant="scrollable"
            scrollButtons="auto"
            sx={{
              minHeight: 40,
              '& .MuiTab-root': {
                minHeight: 40,
                minWidth: 'auto',
                px: 2,
                color: 'var(--kai-text-soft)',
                textTransform: 'none',
                fontSize: '0.875rem',
                '&.Mui-selected': {
                  color: theme?.accentColor || '#34d399',
                },
              },
              '& .MuiTabs-indicator': {
                backgroundColor: theme?.accentColor || '#34d399',
              },
            }}
          >
            {AD_FORMATS.map((format) => (
              <Tab
                key={format.id}
                value={format.id}
                icon={<format.icon sx={{ fontSize: 18 }} />}
                iconPosition="start"
                label={format.label}
              />
            ))}
          </Tabs>

          {/* Device Toggle */}
          {supportsDeviceToggle && (
            <ToggleButtonGroup
              value={device}
              exclusive
              onChange={handleDeviceChange}
              size="small"
              sx={{
                '& .MuiToggleButton-root': {
                  color: 'var(--kai-text-soft)',
                  borderColor: 'var(--kai-border-strong)',
                  px: 1.5,
                  py: 0.5,
                  '&.Mui-selected': {
                    color: theme?.accentColor || '#34d399',
                    backgroundColor: `${theme?.accentColor || '#34d399'}15`,
                  },
                },
              }}
            >
              <ToggleButton value="desktop">
                <DesktopIcon sx={{ fontSize: 18, mr: 0.5 }} />
                Desktop
              </ToggleButton>
              <ToggleButton value="mobile">
                <MobileIcon sx={{ fontSize: 18, mr: 0.5 }} />
                Mobile
              </ToggleButton>
            </ToggleButtonGroup>
          )}
        </Box>

        {/* Demo indicator */}
        {isDemo && (
          <Typography
            variant="caption"
            sx={{
              display: 'block',
              mt: 1.5,
              color: '#f59e0b',
              fontStyle: 'italic',
            }}
          >
            Showing example data - generate your own headlines to see your ads
          </Typography>
        )}
      </Paper>

      {/* Mockup Content */}
      <Box>
        {activeFormat === 'search' && (
          <SearchAdMockup
            headlines={displayHeadlines}
            descriptions={displayDescriptions}
            displayUrl={displayUrl}
            device={device}
            theme={theme}
          />
        )}
        {activeFormat === 'display' && (
          <DisplayAdMockup
            headline={displayHeadlines[0]}
            description={displayDescriptions[0]}
            businessName={businessName}
            displayUrl={displayUrl}
            logoUrl={logoUrl}
            imageUrl={imageUrl}
            device={device}
            theme={theme}
          />
        )}
        {activeFormat === 'youtube' && (
          <YouTubeAdMockup
            headline={displayHeadlines[0]}
            description={displayDescriptions[0]}
            businessName={businessName}
            displayUrl={displayUrl}
            theme={theme}
          />
        )}
        {activeFormat === 'gmail' && (
          <GmailAdMockup
            headline={displayHeadlines[0]}
            description={displayDescriptions[0]}
            businessName={businessName}
            theme={theme}
          />
        )}
        {activeFormat === 'discovery' && (
          <DiscoveryAdMockup
            headline={displayHeadlines[0]}
            description={displayDescriptions[0]}
            displayUrl={displayUrl}
            imageUrl={imageUrl}
            theme={theme}
          />
        )}
      </Box>
    </Box>
  )
}

