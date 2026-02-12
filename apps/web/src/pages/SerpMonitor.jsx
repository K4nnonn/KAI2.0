/**
 * SERP Monitor - AI-Chat-Led URL Health & Competitor Intelligence
 *
 * Enhanced with two modes:
 * - URL Health: Check landing page health and detect soft 404s
 * - Competitor Intelligence: Analyze competitor investment signals
 *
 * Both modes use the intelligent mapping pattern where data is
 * extracted from conversation rather than requiring file uploads.
 */
import { useState } from 'react'
import {
  Box,
  TextField,
  Button,
  Stack,
  Alert,
  Tabs,
  Tab,
  Typography,
} from '@mui/material'
import {
  Search,
  Psychology,
  HealthAndSafety,
  Link as LinkIcon,
  BugReport,
  Speed,
  TrendingUp,
  Business,
  Insights,
} from '@mui/icons-material'
import axios from 'axios'
import { api } from '../config'

// Chat-led components
import {
  ChatLedLayout,
  ToolChatInterface,
  PreviewPanel,
  DemoDataBanner,
} from '../components/chat-led'

// Preview components
import { SerpResultsPreview, CompetitorIntelPreview } from '../components/previews'

// SERP-specific theme
const SERP_THEME = {
  color: '#a78bfa',  // Purple - matches KaiChat system
  accentColor: '#a78bfa',
  borderColor: 'var(--kai-border-strong)',
}

// URL Health capabilities for the AI chat
const URL_HEALTH_CAPABILITIES = [
  {
    id: 'health_check',
    name: 'URL Health Check',
    icon: HealthAndSafety,
    description: 'Check if your landing pages are accessible and returning proper status codes',
    color: '#10b981',
    examples: ['Check the health of my landing pages', 'Are my URLs working?'],
  },
  {
    id: 'soft_404_detection',
    name: 'Soft 404 Detection',
    icon: BugReport,
    description: 'Find pages that return 200 but look like error pages to search engines',
    color: '#f59e0b',
    examples: ['Find soft 404s on my site', 'Which pages look like errors?'],
  },
  {
    id: 'batch_checking',
    name: 'Batch URL Checking',
    icon: LinkIcon,
    description: 'Check multiple URLs at once to find issues across your site',
    color: '#3b82f6',
    examples: ['Check all my product pages', 'Monitor my key landing pages'],
  },
  {
    id: 'performance_insights',
    name: 'Performance Insights',
    icon: Speed,
    description: 'Get response time and performance metrics for your URLs',
    color: '#8b5cf6',
    examples: ['Which pages are slow?', 'Check my page response times'],
  },
]

// Competitor Intelligence capabilities
const COMPETITOR_CAPABILITIES = [
  {
    id: 'investment_analysis',
    name: 'Investment Analysis',
    icon: TrendingUp,
    description: 'Analyze if competitors are ramping up or reducing their paid search investment',
    color: '#ef4444',
    examples: ['Is HomeDepot ramping up their ad spend?', 'Are my competitors investing more?'],
  },
  {
    id: 'competitor_tracking',
    name: 'Competitor Tracking',
    icon: Business,
    description: 'Track multiple competitors and their auction behavior over time',
    color: '#3b82f6',
    examples: ['Track Lowes and HomeDepot', 'Who is my biggest threat?'],
  },
  {
    id: 'market_pressure',
    name: 'Market Pressure',
    icon: Insights,
    description: 'Understand overall competitive pressure in your market',
    color: '#f59e0b',
    examples: ['How competitive is my market?', 'Is competitive pressure increasing?'],
  },
]

// Example prompts for URL Health
const URL_HEALTH_PROMPTS = [
  'Check the health of my URLs',
  'Find any broken pages',
  'Are there soft 404s on my site?',
]

// Example prompts for Competitor Intelligence
const COMPETITOR_PROMPTS = [
  'Is my competitor increasing their ad spend?',
  'Analyze CompetitorX investment signals',
  'Who is ramping up in my market?',
]

export default function SerpMonitor() {
  // Tab state
  const [activeTab, setActiveTab] = useState(0)

  // URL Health state
  const [urlResults, setUrlResults] = useState([])
  const [urlLoading, setUrlLoading] = useState(false)
  const [urlError, setUrlError] = useState(null)
  const [urls, setUrls] = useState('')

  // Competitor Intelligence state
  const [competitors, setCompetitors] = useState([])
  const [competitorLoading, setCompetitorLoading] = useState(false)
  const [competitorError, setCompetitorError] = useState(null)
  const [compDomain, setCompDomain] = useState('example.com')
  const [compCurrent, setCompCurrent] = useState('0.2')
  const [compPrevious, setCompPrevious] = useState('0.1')
  const [compOutranking, setCompOutranking] = useState('0.15')
  const [compTopPage, setCompTopPage] = useState('0.5')
  const [compPosAbove, setCompPosAbove] = useState('0.4')
  const [compNotes, setCompNotes] = useState('Competitor ramping up brand ads')

  // Handle tab change
  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue)
  }

  // ============== URL Health Functions ==============

  // Handle data extracted by AI from natural language (URL Health)
  const handleUrlDataExtracted = (data) => {
    if (data?.urls && data.urls.length > 0) {
      runUrlCheck(data.urls)
    }
  }

  // Handle check completion from AI chat (URL Health)
  const handleUrlAnalysisComplete = (analysisResult) => {
    console.log('[SerpMonitor] URL check complete:', analysisResult)
    if (analysisResult && Array.isArray(analysisResult)) {
      setUrlResults(analysisResult)
    } else if (analysisResult?.results) {
      setUrlResults(analysisResult.results || [])
    }
  }

  // Run URL check with provided data
  const runUrlCheck = async (urlList) => {
    setUrlError(null)
    setUrlLoading(true)
    setUrlResults([])
    try {
      const resp = await axios.post(api.serp.check, { urls: urlList })
      console.log('[SerpMonitor] API response:', resp.data)
      const results = resp.data?.results || []
      setUrlResults(results)
      if (!results.length) {
        setUrlError('No results returned for the provided URLs')
      }
    } catch (err) {
      setUrlError(err.response?.data?.detail || 'SERP check failed')
    } finally {
      setUrlLoading(false)
    }
  }

  // Handle manual check from advanced mode
  const handleManualUrlCheck = async () => {
    const list = urls
      .split('\n')
      .map((u) => u.trim())
      .filter(Boolean)
    if (!list.length) {
      setUrlError('Add one or more URLs (one per line)')
      return
    }
    setUrlError(null)
    setUrlLoading(true)
    setUrlResults([])
    try {
      const resp = await axios.post(api.serp.check, { urls: list })
      console.log('[SerpMonitor] Manual check response:', resp.data)
      const results = resp.data?.results || []
      setUrlResults(results)
      if (!results.length) {
        setUrlError('No results returned for the provided URLs')
      }
    } catch (err) {
      setUrlError(err.response?.data?.detail || 'SERP check failed')
    } finally {
      setUrlLoading(false)
    }
  }

  // ============== Competitor Intelligence Functions ==============

  // Handle data extracted by AI from natural language (Competitor)
  const handleCompetitorDataExtracted = (data) => {
    if (data?.competitor_domain) {
      runCompetitorAnalysis(data)
    }
  }

  // Handle analysis completion from AI chat (Competitor)
  const handleCompetitorAnalysisComplete = (analysisResult) => {
    console.log('[SerpMonitor] Competitor analysis complete:', analysisResult)
    if (analysisResult?.result) {
      // Add to competitors list
      setCompetitors(prev => {
        // Update existing or add new
        const existing = prev.findIndex(c => c.competitor === analysisResult.result.competitor)
        if (existing >= 0) {
          const updated = [...prev]
          updated[existing] = analysisResult.result
          return updated
        }
        return [...prev, analysisResult.result]
      })
    }
  }

  // Run competitor analysis
  const runCompetitorAnalysis = async (data) => {
    setCompetitorError(null)
    setCompetitorLoading(true)
    try {
      const resp = await axios.post(api.serp.competitorSignal, data)
      console.log('[SerpMonitor] Competitor API response:', resp.data)
      if (resp.data.result) {
        setCompetitors(prev => {
          const existing = prev.findIndex(c => c.competitor === resp.data.result.competitor)
          if (existing >= 0) {
            const updated = [...prev]
            updated[existing] = resp.data.result
            return updated
          }
          return [...prev, resp.data.result]
        })
      }
    } catch (err) {
      setCompetitorError(err.response?.data?.detail || 'Competitor analysis failed')
    } finally {
      setCompetitorLoading(false)
    }
  }

  const handleManualCompetitor = async () => {
    if (!compDomain.trim()) {
      setCompetitorError('Add a competitor domain')
      return
    }
    setCompetitorError(null)
    setCompetitorLoading(true)
    const payload = {
      competitor_domain: compDomain.trim(),
      impression_share_current: parseFloat(compCurrent) || null,
      impression_share_previous: parseFloat(compPrevious) || null,
      outranking_rate: parseFloat(compOutranking) || null,
      top_of_page_rate: parseFloat(compTopPage) || null,
      position_above_rate: parseFloat(compPosAbove) || null,
      raw_description: compNotes,
    }
    try {
      const resp = await axios.post(api.serp.competitorSignal, payload)
      if (resp.data.result) {
        setCompetitors((prev) => {
          const existing = prev.findIndex((c) => c.competitor === resp.data.result.competitor)
          if (existing >= 0) {
            const updated = [...prev]
            updated[existing] = resp.data.result
            return updated
          }
          return [...prev, resp.data.result]
        })
      }
    } catch (err) {
      setCompetitorError(err.response?.data?.detail || 'Competitor analysis failed')
    } finally {
      setCompetitorLoading(false)
    }
  }

  // ============== UI Components ==============

  // URL Health advanced mode content
  const urlAdvancedModeContent = (
    <Stack spacing={2}>
      {urlError && (
        <Alert severity="error" onClose={() => setUrlError(null)}>
          {urlError}
        </Alert>
      )}

      <TextField
        label="URLs (one per line)"
        value={urls}
        onChange={(e) => setUrls(e.target.value)}
        fullWidth
        multiline
        minRows={4}
        placeholder="https://example.com/page-1
https://example.com/page-2
https://example.com/page-3"
        InputLabelProps={{ style: { color: '#cbd5e1' } }}
        InputProps={{ style: { color: 'var(--kai-text)' } }}
        sx={{
          '& .MuiOutlinedInput-root': {
            '& fieldset': { borderColor: 'var(--kai-border-strong)' },
            '&:hover fieldset': { borderColor: SERP_THEME.color },
            '&.Mui-focused fieldset': { borderColor: SERP_THEME.color },
          },
        }}
      />
      <Button
        variant="contained"
        size="large"
        onClick={handleManualUrlCheck}
        disabled={urlLoading}
        startIcon={<Search />}
        sx={{
          alignSelf: 'flex-start',
          textTransform: 'none',
          background: `linear-gradient(135deg, ${SERP_THEME.color}, ${SERP_THEME.color}cc)`,
          '&:hover': {
            background: `linear-gradient(135deg, ${SERP_THEME.color}ee, ${SERP_THEME.color})`,
          },
        }}
      >
        {urlLoading ? 'Checking...' : 'Run SERP Check'}
      </Button>
    </Stack>
  )

  // Competitor advanced mode content (placeholder for manual entry)
  const competitorAdvancedModeContent = (
    <Stack spacing={2}>
      {competitorError && (
        <Alert severity="error" onClose={() => setCompetitorError(null)}>
          {competitorError}
        </Alert>
      )}
      <Typography variant="body2" sx={{ color: 'var(--kai-text-soft)' }}>
        Enter a competitor to run a quick investment signal. Kai will extract metrics from chat as well.
      </Typography>
      <TextField
        label="Competitor domain"
        inputProps={{ 'data-testid': 'competitor-domain' }}
        value={compDomain}
        onChange={(e) => setCompDomain(e.target.value)}
        fullWidth
        InputLabelProps={{ style: { color: '#cbd5e1' } }}
        InputProps={{ style: { color: 'var(--kai-text)' } }}
        sx={{ '& .MuiOutlinedInput-root': { '& fieldset': { borderColor: 'var(--kai-border-strong)' }, '&:hover fieldset': { borderColor: SERP_THEME.color }, '&.Mui-focused fieldset': { borderColor: SERP_THEME.color } } }}
      />
      <Stack direction="row" spacing={2} flexWrap="wrap">
        <TextField label="Impression share (current)" value={compCurrent} onChange={(e) => setCompCurrent(e.target.value)} size="small" sx={{ minWidth: 220 }} />
        <TextField label="Impression share (previous)" value={compPrevious} onChange={(e) => setCompPrevious(e.target.value)} size="small" sx={{ minWidth: 220 }} />
        <TextField label="Outranking rate" value={compOutranking} onChange={(e) => setCompOutranking(e.target.value)} size="small" sx={{ minWidth: 180 }} />
        <TextField label="Top of page rate" value={compTopPage} onChange={(e) => setCompTopPage(e.target.value)} size="small" sx={{ minWidth: 180 }} />
        <TextField label="Position above rate" value={compPosAbove} onChange={(e) => setCompPosAbove(e.target.value)} size="small" sx={{ minWidth: 180 }} />
      </Stack>
      <TextField
        label="Notes / description"
        value={compNotes}
        onChange={(e) => setCompNotes(e.target.value)}
        fullWidth
        multiline
        minRows={2}
        InputLabelProps={{ style: { color: '#cbd5e1' } }}
        InputProps={{ style: { color: 'var(--kai-text)' } }}
        sx={{ '& .MuiOutlinedInput-root': { '& fieldset': { borderColor: 'var(--kai-border-strong)' }, '&:hover fieldset': { borderColor: SERP_THEME.color }, '&.Mui-focused fieldset': { borderColor: SERP_THEME.color } } }}
      />
      <Button
        variant="contained"
        size="large"
        onClick={handleManualCompetitor}
        disabled={competitorLoading}
        startIcon={<TrendingUp />}
        sx={{
          alignSelf: 'flex-start',
          textTransform: 'none',
          background: `linear-gradient(135deg, ${SERP_THEME.color}, ${SERP_THEME.color}cc)`,
          '&:hover': { background: `linear-gradient(135deg, ${SERP_THEME.color}ee, ${SERP_THEME.color})` },
        }}
      >
        {competitorLoading ? 'Analyzing...' : 'Run Competitor Signal'}
      </Button>
    </Stack>
  )

  // Determine if we have live data
  const hasUrlData = Array.isArray(urlResults) && urlResults.length > 0
  const hasCompetitorData = competitors && competitors.length > 0

  // URL Health preview content
  const urlPreviewContent = (
    <Box sx={{ p: 2 }}>
      <SerpResultsPreview
        results={hasUrlData ? urlResults : []}
        theme={SERP_THEME}
        isDemoData={!hasUrlData}
      />
    </Box>
  )

  // Competitor Intelligence preview content
  const competitorPreviewContent = (
    <Box sx={{ p: 2 }}>
      {!hasCompetitorData && (
        <DemoDataBanner
          themeColor={SERP_THEME.color}
          message="This is example data. Ask Kai about your competitors to see real analysis."
        />
      )}
      <CompetitorIntelPreview
        competitors={hasCompetitorData ? competitors : []}
        theme={SERP_THEME}
        isDemoData={!hasCompetitorData}
      />
    </Box>
  )

  // Tab-specific content
  const isUrlHealthTab = activeTab === 0
  const currentCapabilities = isUrlHealthTab ? URL_HEALTH_CAPABILITIES : COMPETITOR_CAPABILITIES
  const currentPrompts = isUrlHealthTab ? URL_HEALTH_PROMPTS : COMPETITOR_PROMPTS
  const currentPreview = isUrlHealthTab ? urlPreviewContent : competitorPreviewContent
  const currentAdvanced = isUrlHealthTab ? urlAdvancedModeContent : competitorAdvancedModeContent
  const currentLoading = isUrlHealthTab ? urlLoading : competitorLoading
  const currentLiveData = isUrlHealthTab ? (hasUrlData ? urlResults : null) : (hasCompetitorData ? competitors : null)
  const currentEmptyMessage = isUrlHealthTab
    ? 'URL health check results will appear here'
    : 'Competitor analysis results will appear here'

  // Tab header component
  const tabHeader = (
    <Box sx={{ borderBottom: 1, borderColor: 'var(--kai-border-strong)', mb: 2 }}>
      <Tabs
        value={activeTab}
        onChange={handleTabChange}
        sx={{
          '& .MuiTabs-indicator': {
            backgroundColor: SERP_THEME.color,
          },
          '& .MuiTab-root': {
            color: 'var(--kai-text-soft)',
            textTransform: 'none',
            fontWeight: 600,
            '&.Mui-selected': {
              color: SERP_THEME.color,
            },
          },
        }}
      >
        <Tab
          icon={<HealthAndSafety sx={{ fontSize: 18 }} />}
          iconPosition="start"
          label="URL Health"
        />
        <Tab
          icon={<TrendingUp sx={{ fontSize: 18 }} />}
          iconPosition="start"
          label="Competitor Intelligence"
        />
      </Tabs>
    </Box>
  )

  return (
    <ChatLedLayout
      toolId="serp"
      toolName="SERP Monitor"
      toolDescription={isUrlHealthTab
        ? "Check landing page health and detect soft 404s via SERP scanning"
        : "Analyze competitor investment signals from auction behavior"
      }
      toolIcon={Search}
      themeColor={SERP_THEME.color}
      badges={isUrlHealthTab
        ? [
            { label: 'AI Analysis', icon: <Psychology sx={{ fontSize: 16 }} />, color: '#8b5cf6' },
            { label: 'Health Checks', icon: <HealthAndSafety sx={{ fontSize: 16 }} />, color: '#10b981' },
          ]
        : [
            { label: 'AI Analysis', icon: <Psychology sx={{ fontSize: 16 }} />, color: '#8b5cf6' },
            { label: 'Competitor Intel', icon: <TrendingUp sx={{ fontSize: 16 }} />, color: '#ef4444' },
          ]
      }
      headerExtra={tabHeader}
      chatPanel={
        <ToolChatInterface
          key={activeTab} // Force re-render on tab change to reset chat
          toolId="serp"
          toolName={isUrlHealthTab ? 'URL Health Monitor' : 'Competitor Intelligence'}
          themeColor={SERP_THEME.color}
          capabilities={currentCapabilities}
          examplePrompts={currentPrompts}
          onDataExtracted={isUrlHealthTab ? handleUrlDataExtracted : handleCompetitorDataExtracted}
          onAnalysisComplete={isUrlHealthTab ? handleUrlAnalysisComplete : handleCompetitorAnalysisComplete}
          advancedModeContent={currentAdvanced}
          mode={isUrlHealthTab ? 'url_health' : 'competitor'}
        />
      }
      previewPanel={
        <PreviewPanel
          toolId="serp"
          themeColor={SERP_THEME.color}
          liveData={currentLiveData}
          isLoading={currentLoading}
          emptyMessage={currentEmptyMessage}
          forceLive={isUrlHealthTab ? (urlLoading || hasUrlData) : hasCompetitorData}
        >
          {currentPreview}
        </PreviewPanel>
      }
    />
  )
}

