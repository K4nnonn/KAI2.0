/**
 * Creative Studio - AI-Chat-Led Ad Copy Generation
 *
 * Redesigned with ChatLedLayout for an AI-first experience:
 * - Left panel: AI chat for natural language ad copy generation
 * - Right panel: Live preview with Google Ads format mockup
 * - Advanced mode: Original form for power users
 */
import { useState } from 'react'
import {
  Box,
  TextField,
  Button,
  Stack,
  Alert,
} from '@mui/material'
import {
  Brush,
  RocketLaunch,
  Psychology,
  AutoAwesome,
  Edit,
  Speed,
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
import { CreativePreview, AdMockupGenerator } from '../components/previews'

// Creative-specific theme - Amber (warm side of brand spectrum)
const CREATIVE_THEME = {
  color: '#f59e0b',
  accentColor: '#f59e0b',
  borderColor: 'var(--kai-border-strong)',
}

// Creative capabilities for the AI chat
const CREATIVE_CAPABILITIES = [
  {
    id: 'headline_generation',
    name: 'Headline Generation',
    icon: Edit,
    description: 'Create compelling headlines that capture attention and drive clicks',
    color: '#3b82f6',
    examples: ['Create headlines for my SaaS product', 'I need punchy headlines for a sale'],
  },
  {
    id: 'description_writing',
    name: 'Description Writing',
    icon: AutoAwesome,
    description: 'Write persuasive descriptions that convert browsers to buyers',
    color: '#10b981',
    examples: ['Write descriptions highlighting free shipping', 'Create urgency in my ad copy'],
  },
  {
    id: 'refinement',
    name: 'Copy Refinement',
    icon: Brush,
    description: 'Improve existing copy with suggestions for better performance',
    color: '#f59e0b',
    examples: ['Make these headlines shorter', 'Add more urgency to this copy'],
  },
  {
    id: 'compliance_check',
    name: 'Policy Compliance',
    icon: Speed,
    description: 'Ensure your ad copy meets Google Ads policy requirements',
    color: '#8b5cf6',
    examples: ['Is this copy compliant?', 'Check my headlines for policy issues'],
  },
]

// Example prompts for the chat
const CREATIVE_EXAMPLE_PROMPTS = [
  'Generate headlines for my business',
  'Create ad copy for a limited-time sale',
  'Make my descriptions more compelling',
]

export default function CreativeStudio() {
  // State for generation results
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // Advanced mode state (original form)
  const [businessName, setBusinessName] = useState('')
  const [url, setUrl] = useState('')
  const [keywords, setKeywords] = useState('')
  const [usps, setUsps] = useState('')

  // Handle data extracted by AI from natural language
  const handleDataExtracted = (data) => {
    // If data is complete, auto-generate
    if (data?.businessName) {
      runGenerate(data)
    }
  }

  // Handle generation completion from AI chat
  const handleAnalysisComplete = (analysisResult) => {
    console.log('[CreativeStudio] Generation complete:', analysisResult)
    setResult(analysisResult)
  }

  // Run generation with provided data
  const runGenerate = async (data) => {
    setError(null)
    setLoading(true)
    setResult(null)
    try {
      const payload = {
        business_name: data?.businessName || data?.business_name || '',
        url: data?.url || null,
        keywords: data?.keywords || [],
        usps: data?.usps || [],
      }
      const resp = await axios.post(api.creative.generate, payload)
      console.log('[CreativeStudio] API response:', resp.data)
      setResult(resp.data.result)
    } catch (err) {
      setError(err.response?.data?.detail || 'Generation failed')
    } finally {
      setLoading(false)
    }
  }

  // Handle manual generate from advanced mode
  const handleManualGenerate = async () => {
    if (!businessName.trim()) {
      setError('Business name is required')
      return
    }
    setError(null)
    setLoading(true)
    setResult(null)
    try {
      const payload = {
        business_name: businessName.trim(),
        url: url.trim() || null,
        keywords: keywords
          .split(',')
          .map((k) => k.trim())
          .filter(Boolean),
        usps: usps
          .split(',')
          .map((u) => u.trim())
          .filter(Boolean),
      }
      const resp = await axios.post(api.creative.generate, payload)
      console.log('[CreativeStudio] Manual generation response:', resp.data)
      setResult(resp.data.result)
    } catch (err) {
      setError(err.response?.data?.detail || 'Generation failed')
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
        label="Business name"
        value={businessName}
        onChange={(e) => setBusinessName(e.target.value)}
        fullWidth
        required
        InputLabelProps={{ style: { color: '#cbd5e1' } }}
        InputProps={{ style: { color: 'var(--kai-text)' } }}
        sx={{
          '& .MuiOutlinedInput-root': {
            '& fieldset': { borderColor: 'var(--kai-border-strong)' },
            '&:hover fieldset': { borderColor: CREATIVE_THEME.color },
            '&.Mui-focused fieldset': { borderColor: CREATIVE_THEME.color },
          },
        }}
      />
      <TextField
        label="Landing page URL"
        value={url}
        onChange={(e) => setUrl(e.target.value)}
        fullWidth
        placeholder="https://example.com"
        InputLabelProps={{ style: { color: '#cbd5e1' } }}
        InputProps={{ style: { color: 'var(--kai-text)' } }}
        sx={{
          '& .MuiOutlinedInput-root': {
            '& fieldset': { borderColor: 'var(--kai-border-strong)' },
            '&:hover fieldset': { borderColor: CREATIVE_THEME.color },
            '&.Mui-focused fieldset': { borderColor: CREATIVE_THEME.color },
          },
        }}
      />
      <TextField
        label="Keywords (comma-separated)"
        value={keywords}
        onChange={(e) => setKeywords(e.target.value)}
        fullWidth
        InputLabelProps={{ style: { color: '#cbd5e1' } }}
        InputProps={{ style: { color: 'var(--kai-text)' } }}
        sx={{
          '& .MuiOutlinedInput-root': {
            '& fieldset': { borderColor: 'var(--kai-border-strong)' },
            '&:hover fieldset': { borderColor: CREATIVE_THEME.color },
            '&.Mui-focused fieldset': { borderColor: CREATIVE_THEME.color },
          },
        }}
      />
      <TextField
        label="USPs (comma-separated)"
        value={usps}
        onChange={(e) => setUsps(e.target.value)}
        fullWidth
        InputLabelProps={{ style: { color: '#cbd5e1' } }}
        InputProps={{ style: { color: 'var(--kai-text)' } }}
        sx={{
          '& .MuiOutlinedInput-root': {
            '& fieldset': { borderColor: 'var(--kai-border-strong)' },
            '&:hover fieldset': { borderColor: CREATIVE_THEME.color },
            '&.Mui-focused fieldset': { borderColor: CREATIVE_THEME.color },
          },
        }}
      />
      <Button
        variant="contained"
        size="large"
        onClick={handleManualGenerate}
        disabled={loading}
        startIcon={<Brush />}
        sx={{
          alignSelf: 'flex-start',
          textTransform: 'none',
          background: `linear-gradient(135deg, ${CREATIVE_THEME.color}, ${CREATIVE_THEME.color}cc)`,
          '&:hover': {
            background: `linear-gradient(135deg, ${CREATIVE_THEME.color}ee, ${CREATIVE_THEME.color})`,
          },
        }}
      >
        {loading ? 'Generatingâ€¦' : 'Generate Ad Copy'}
      </Button>
    </Stack>
  )

  // Determine if we have live data to show
  const hasLiveData = result && (result.headlines || result.descriptions)

  // Preview panel content
  const previewContent = (
    <Box sx={{ p: 2 }}>
      {/* Show demo banner when no live data */}
      {!hasLiveData && (
        <DemoDataBanner
          themeColor={CREATIVE_THEME.color}
          message="This is example ad copy. Chat with Kai to generate custom copy for your business."
        />
      )}

      {/* Ad Mockup Generator - Multi-format previews */}
      <AdMockupGenerator
        headlines={hasLiveData ? result.headlines : []}
        descriptions={hasLiveData ? result.descriptions : []}
        businessName={businessName || 'Example Company'}
        displayUrl={url ? (() => { try { return new URL(url.startsWith('http') ? url : `https://${url}`).hostname } catch { return url } })() : 'example.com'}
        theme={CREATIVE_THEME}
        isDemoData={!hasLiveData}
      />

      {/* Divider */}
      <Box sx={{ my: 4, borderTop: '1px solid var(--kai-border-strong)' }} />

      {/* Creative Preview Component - Character counts */}
      <CreativePreview
        headlines={hasLiveData ? result.headlines : []}
        descriptions={hasLiveData ? result.descriptions : []}
        theme={CREATIVE_THEME}
        isDemoData={!hasLiveData}
      />
    </Box>
  )

  return (
    <ChatLedLayout
      toolId="creative"
      toolName="Creative Studio"
      toolDescription="Generate RSA headlines and descriptions with AI-powered copywriting"
      toolIcon={Brush}
      themeColor={CREATIVE_THEME.color}
      badges={[
        { label: 'AI Copywriting', icon: <Psychology sx={{ fontSize: 16 }} />, color: '#8b5cf6' },
        { label: 'Google Ads Ready', icon: <RocketLaunch sx={{ fontSize: 16 }} />, color: '#f59e0b' },
      ]}
      chatPanel={
        <ToolChatInterface
          toolId="creative"
          toolName="Creative Studio"
          themeColor={CREATIVE_THEME.color}
          capabilities={CREATIVE_CAPABILITIES}
          examplePrompts={CREATIVE_EXAMPLE_PROMPTS}
          onDataExtracted={handleDataExtracted}
          onAnalysisComplete={handleAnalysisComplete}
          advancedModeContent={advancedModeContent}
        />
      }
      previewPanel={
        <PreviewPanel
          toolId="creative"
          themeColor={CREATIVE_THEME.color}
          liveData={hasLiveData ? result : null}
          isLoading={loading}
          emptyMessage="Your generated ad copy will appear here"
        >
          {previewContent}
        </PreviewPanel>
      }
    />
  )
}

