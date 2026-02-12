/**
 * PMax Deep Dive - AI-Chat-Led Analysis Page
 *
 * Redesigned with ChatLedLayout for an AI-first experience:
 * - Left panel: AI chat for natural language PMax analysis
 * - Right panel: Live preview dashboards (demo data by default)
 * - Advanced mode: Original form for power users
 */
import { useState, useEffect, Suspense, lazy } from 'react'
import {
  Box,
  Typography,
  Paper,
  TextField,
  Button,
  Stack,
  Alert,
  Chip,
  CircularProgress,
} from '@mui/material'
import {
  TrendingUp,
  UploadFile,
  Psychology,
  Analytics,
  PieChart,
  Speed,
} from '@mui/icons-material'
import axios from 'axios'
import { api, getOrCreateSessionId } from '../config'

// Chat-led components
import {
  ChatLedLayout,
  ToolChatInterface,
  PreviewPanel,
  DemoDataBanner,
} from '../components/chat-led'

// Lazy load dashboard components
const PMaxSpendSankey = lazy(() => import('../components/dashboards/PMaxSpendSankey'))
const ChannelComparison = lazy(() => import('../components/dashboards/ChannelComparison'))

// PMax-specific theme
const PMAX_THEME = {
  color: '#f472b6',  // Pink - matches KaiChat system
  accentColor: '#f472b6',
  borderColor: 'var(--kai-border-strong)',
}

// PMax capabilities for the AI chat
const PMAX_CAPABILITIES = [
  {
    id: 'spend_analysis',
    name: 'Spend Analysis',
    icon: PieChart,
    description: 'Analyze where your PMax budget is going across Search, Shopping, Video, and Display',
    color: '#3b82f6',
    examples: ['Where is my PMax budget going?', 'My PMax campaign spent $5000 last month'],
  },
  {
    id: 'channel_performance',
    name: 'Channel Performance',
    icon: Analytics,
    description: 'Compare performance metrics across channels to find winners and losers',
    color: '#10b981',
    examples: ['Which channel has the best ROI?', 'Compare my channel performance'],
  },
  {
    id: 'placement_insights',
    name: 'Placement Insights',
    icon: TrendingUp,
    description: 'Deep dive into specific placements and their contribution to your goals',
    color: '#f59e0b',
    examples: ['Show me my top placements', 'How is YouTube performing?'],
  },
  {
    id: 'optimization_tips',
    name: 'Optimization Tips',
    icon: Speed,
    description: 'Get AI-powered recommendations to improve your PMax campaigns',
    color: '#8b5cf6',
    examples: ['How can I improve my PMax?', 'What should I optimize first?'],
  },
]

// Example prompts for the chat
const PMAX_EXAMPLE_PROMPTS = [
  'Analyze my PMax spend allocation',
  'Which channels are performing best?',
  'Show me where my budget is going',
]

const storageKeyForActiveAccount = (sessionId) => (sessionId ? `kai_sa360_active_account:${sessionId}` : null)

export default function PMaxDeepDive() {
  const [sessionId] = useState(() => getOrCreateSessionId())
  const [activeAccount, setActiveAccount] = useState(null)
  // State for analysis results
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // State for extracted data (from AI or manual entry)
  const [extractedData, setExtractedData] = useState(null)

  // Advanced mode state (original form)
  const [placementsJson, setPlacementsJson] = useState('')
  const [spend, setSpend] = useState('')
  const [conversions, setConversions] = useState('')

  useEffect(() => {
    if (!sessionId) return
    const key = storageKeyForActiveAccount(sessionId)
    if (!key) return
    try {
      const raw = window.localStorage.getItem(key)
      if (!raw) return
      const parsed = JSON.parse(raw)
      if (parsed && parsed.customer_id) {
        setActiveAccount({ customer_id: String(parsed.customer_id), name: parsed.name || '' })
      }
    } catch {
      // ignore
    }
  }, [sessionId])

  // Handle data extracted by AI from natural language
  const handleDataExtracted = (data) => {
    setExtractedData(data)
    // If data is complete, auto-analyze
    if (data?.placements || data?.spend) {
      runAnalysis(data)
    }
  }

  // Handle analysis completion from AI chat
  const handleAnalysisComplete = (analysisResult) => {
    console.log('[PMaxDeepDive] Analysis complete:', analysisResult)
    setResult(analysisResult)
  }

  // Run analysis with provided data
  const runAnalysis = async (data) => {
    setError(null)
    setLoading(true)
    setResult(null)
    try {
      const payload = {
        placements: data?.placements || [],
        spend: data?.spend || null,
        conversions: data?.conversions || null,
      }
      const resp = await axios.post(api.pmax.analyze, payload)
      console.log('[PMaxDeepDive] API response:', resp.data)
      setResult(resp.data.result)
    } catch (err) {
      setError(err.response?.data?.detail || 'Analysis failed')
    } finally {
      setLoading(false)
    }
  }

  // Handle manual analyze from advanced mode
  const handleManualAnalyze = async () => {
    setError(null)
    setLoading(true)
    setResult(null)
    try {
      let placements = []
      if (placementsJson.trim()) {
        placements = JSON.parse(placementsJson)
      }
      const payload = {
        placements,
        spend: spend ? Number(spend) : null,
        conversions: conversions ? Number(conversions) : null,
      }
      const resp = await axios.post(api.pmax.analyze, payload)
      console.log('[PMaxDeepDive] Manual analysis response:', resp.data)
      setResult(resp.data.result)
    } catch (err) {
      setError(err.response?.data?.detail || 'Analysis failed (ensure placements JSON is valid)')
    } finally {
      setLoading(false)
    }
  }

  // Advanced mode content (original form)
  const advancedModeContent = (
    <Stack spacing={2}>
      {error && (
        <Alert severity="error" onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <TextField
        label="Placements JSON (array of objects)"
        value={placementsJson}
        onChange={(e) => setPlacementsJson(e.target.value)}
        fullWidth
        multiline
        minRows={4}
        placeholder='[{"placement":"YouTube","cost":1200,"conversions":24}]'
        InputLabelProps={{ style: { color: '#cbd5e1' } }}
        InputProps={{ style: { color: 'var(--kai-text)' } }}
        sx={{
          '& .MuiOutlinedInput-root': {
            '& fieldset': { borderColor: 'var(--kai-border-strong)' },
            '&:hover fieldset': { borderColor: PMAX_THEME.color },
            '&.Mui-focused fieldset': { borderColor: PMAX_THEME.color },
          },
        }}
      />
      <Stack direction="row" spacing={2}>
        <TextField
          label="Total spend"
          value={spend}
          onChange={(e) => setSpend(e.target.value)}
          type="number"
          fullWidth
          InputLabelProps={{ style: { color: '#cbd5e1' } }}
          InputProps={{ style: { color: 'var(--kai-text)' } }}
          sx={{
            '& .MuiOutlinedInput-root': {
              '& fieldset': { borderColor: 'var(--kai-border-strong)' },
              '&:hover fieldset': { borderColor: PMAX_THEME.color },
              '&.Mui-focused fieldset': { borderColor: PMAX_THEME.color },
            },
          }}
        />
        <TextField
          label="Total conversions"
          value={conversions}
          onChange={(e) => setConversions(e.target.value)}
          type="number"
          fullWidth
          InputLabelProps={{ style: { color: '#cbd5e1' } }}
          InputProps={{ style: { color: 'var(--kai-text)' } }}
          sx={{
            '& .MuiOutlinedInput-root': {
              '& fieldset': { borderColor: 'var(--kai-border-strong)' },
              '&:hover fieldset': { borderColor: PMAX_THEME.color },
              '&.Mui-focused fieldset': { borderColor: PMAX_THEME.color },
            },
          }}
        />
      </Stack>
      <Button
        variant="contained"
        size="large"
        onClick={handleManualAnalyze}
        disabled={loading}
        startIcon={<TrendingUp />}
        sx={{
          alignSelf: 'flex-start',
          textTransform: 'none',
          background: `linear-gradient(135deg, ${PMAX_THEME.color}, ${PMAX_THEME.color}cc)`,
          '&:hover': {
            background: `linear-gradient(135deg, ${PMAX_THEME.color}ee, ${PMAX_THEME.color})`,
          },
        }}
      >
        {loading ? 'Analyzingâ€¦' : 'Run Analysis'}
      </Button>
    </Stack>
  )

  // Determine if we have live data to show
  const hasLiveData = result && result.channel_breakout

  // Preview panel content - shows dashboards
  const previewContent = (
    <Box sx={{ p: 2 }}>
      {/* Show demo banner when no live data */}
      {!hasLiveData && (
        <DemoDataBanner
          themeColor={PMAX_THEME.color}
          message="This is example data. Chat with Kai to analyze your PMax campaigns."
        />
      )}

      {/* AI Findings section - only show with live results */}
      {result && (
        <Paper
          elevation={0}
          sx={{
            p: 2,
            mb: 3,
            borderRadius: 2,
            background: 'var(--kai-bg)',
            border: `1px solid ${PMAX_THEME.color}33`,
          }}
        >
          <Typography variant="h6" fontWeight={700} gutterBottom sx={{ color: 'var(--kai-text)' }}>
            AI Findings
          </Typography>
          <Stack spacing={1}>
            {Array.isArray(result.findings)
              ? result.findings.map((f, idx) => (
                  <Chip
                    key={idx}
                    label={f}
                    variant="outlined"
                    sx={{
                      color: '#cbd5e1',
                      borderColor: 'var(--kai-border-strong)',
                      justifyContent: 'flex-start',
                      height: 'auto',
                      '& .MuiChip-label': {
                        whiteSpace: 'normal',
                        py: 1,
                      },
                    }}
                  />
                ))
              : Object.entries(result)
                  .filter(([k]) => k !== 'channel_breakout' && k !== 'spend_split')
                  .map(([k, v]) => (
                    <Chip
                      key={k}
                      label={`${k}: ${JSON.stringify(v)}`}
                      variant="outlined"
                      sx={{ color: '#cbd5e1', borderColor: 'var(--kai-border-strong)' }}
                    />
                  ))}
          </Stack>
        </Paper>
      )}

      {/* Dashboard visualizations */}
      <Stack spacing={3}>
        {/* Sankey Diagram - Shows spend flow */}
        <Suspense fallback={<CircularProgress sx={{ color: PMAX_THEME.color }} />}>
          <PMaxSpendSankey
            placements={hasLiveData ? JSON.parse(placementsJson || '[]') : []}
            spend={hasLiveData && spend ? Number(spend) : null}
            conversions={hasLiveData && conversions ? Number(conversions) : null}
            theme={PMAX_THEME}
            interactive={true}
            isDemoData={!hasLiveData}
            channelBreakout={hasLiveData ? result.channel_breakout : null}
          />
        </Suspense>

        {/* Channel Comparison - Performance metrics */}
        <Suspense fallback={<CircularProgress sx={{ color: PMAX_THEME.color }} />}>
          <ChannelComparison
            placements={hasLiveData ? JSON.parse(placementsJson || '[]') : []}
            spend={hasLiveData && spend ? Number(spend) : null}
            conversions={hasLiveData && conversions ? Number(conversions) : null}
            theme={PMAX_THEME}
            defaultMetric="cost"
            isDemoData={!hasLiveData}
            channelBreakout={hasLiveData ? result.channel_breakout : null}
          />
        </Suspense>
      </Stack>
    </Box>
  )

  return (
    <ChatLedLayout
      toolId="pmax"
      toolName="PMax Deep Dive"
      toolDescription="Analyze Performance Max campaigns with AI-powered insights and visualizations"
      toolIcon={TrendingUp}
      themeColor={PMAX_THEME.color}
      badges={[
        { label: 'AI Analysis', icon: <Psychology sx={{ fontSize: 16 }} />, color: '#8b5cf6' },
        { label: 'Live Dashboards', icon: <Analytics sx={{ fontSize: 16 }} />, color: '#10b981' },
      ]}
      chatPanel={
        <ToolChatInterface
          toolId="pmax"
          toolName="PMax Deep Dive"
          themeColor={PMAX_THEME.color}
          capabilities={PMAX_CAPABILITIES}
          examplePrompts={PMAX_EXAMPLE_PROMPTS}
          accountName={activeAccount?.name || null}
          customerIds={activeAccount?.customer_id ? [String(activeAccount.customer_id)] : []}
          onDataExtracted={handleDataExtracted}
          onAnalysisComplete={handleAnalysisComplete}
          advancedModeContent={advancedModeContent}
        />
      }
      previewPanel={
        <PreviewPanel
          toolId="pmax"
          themeColor={PMAX_THEME.color}
          liveData={hasLiveData ? result : null}
          isLoading={loading}
          emptyMessage="Dashboards will show your PMax analysis results"
        >
          {previewContent}
        </PreviewPanel>
      }
    />
  )
}
