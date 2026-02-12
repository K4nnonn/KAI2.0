import { useState, useEffect, useRef } from 'react'
import {
  Box,
  Container,
  Typography,
  Paper,
  TextField,
  IconButton,
  LinearProgress,
  Alert,
  Chip,
  Avatar,
  Stack,
  Tooltip,
} from '@mui/material'
import { motion, AnimatePresence } from 'framer-motion'
import { Send, SmartToy, Person, Download, CheckCircle, CloudUpload } from '@mui/icons-material'
import axios from 'axios'
import { api } from '../config'
import EmptyState from '../components/EmptyState'
import ProgressStages from '../components/ProgressStages'
import OnboardingTour from '../components/OnboardingTour'
import HelpTooltip from '../components/HelpTooltip'
import useFirstVisit from '../hooks/useFirstVisit'

export default function KlauditAudit({
  title = 'Klaudit Audit',
  subtitle = "Upload your exports or describe the account. I'll generate a clean, AI-enhanced audit.",
  showUpload = true,
  inputPlaceholder = "Tell me what account you'd like audited...",
  variant = 'audit', // 'chat' | 'audit'
}) {
  // Variant-specific theming
  const variantThemes = {
    chat: {
      primaryGradient: 'linear-gradient(135deg, #8b5cf6, #3b82f6)',
      headerBg: 'linear-gradient(135deg, #1e1b4b, #1e3a8a)',
      userBubbleGradient: 'linear-gradient(135deg, #8b5cf6, #6366f1)',
      aiBubbleBg: '#1e1b4b',
      avatarBg: '#5b21b6',
      borderColor: '#6366f1',
      accentColor: '#8b5cf6',
      focusColor: '#a78bfa',
    },
    audit: {
      primaryGradient: 'linear-gradient(135deg, #22d3ee, #6366f1)',
      headerBg: 'var(--kai-bg)',
      userBubbleGradient: 'linear-gradient(135deg, #0ea5e9, #6366f1)',
      aiBubbleBg: 'var(--kai-surface)',
      avatarBg: '#0e7490',
      borderColor: '#22d3ee',
      accentColor: '#22d3ee',
      focusColor: '#22d3ee',
    },
  }

  const theme = variantThemes[variant]
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content:
        'Hi! I can help you generate a comprehensive PPC audit. What account would you like me to audit? Just describe what you need, and I\'ll guide you through the process.',
    },
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [generating, setGenerating] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [auditQueue, setAuditQueue] = useState(null)
  const [files, setFiles] = useState([])
  const [accountName, setAccountName] = useState('Kelvin Co')
  const [currentStage, setCurrentStage] = useState(0)
  const [placeholderIndex, setPlaceholderIndex] = useState(0)
  const messagesEndRef = useRef(null)

  // Progress stage definitions
  const auditStages = [
    { stage: 1, message: 'Uploading files...', duration: '5-10s' },
    { stage: 2, message: 'Validating data structure (14 file types)...', duration: '5-10s' },
    { stage: 3, message: 'Mapping data sources (intelligent discovery)...', duration: '10-15s' },
    { stage: 4, message: 'Analyzing 100+ audit points across 9 dimensions...', duration: '2-3 min' },
    { stage: 5, message: 'Comparing to industry benchmarks...', duration: '30s' },
    { stage: 6, message: 'Scoring and prioritizing recommendations...', duration: '30s' },
    { stage: 7, message: 'Generating Excel report with color-coding...', duration: '1-2 min' },
    { stage: 8, message: 'Complete! Your audit is ready.', duration: '0s' },
  ]

  const creativeStages = [
    { stage: 1, message: 'Analyzing business context...', duration: '1s' },
    { stage: 2, message: 'Generating AI ad copy (Azure OpenAI)...', duration: '2-3s' },
    { stage: 3, message: 'Validating Google Ads character limits...', duration: '0.5s' },
    { stage: 4, message: 'Complete! 3 headlines + 2 descriptions ready.', duration: '0s' },
  ]

  // Rotating placeholder examples
  const chatPlaceholders = [
    'Tell me what you need (audit, PMax review, SERP check, creative help)...',
    'Generate a Brand audit with demo data',
    'Create headlines for my SaaS company',
    'Why is my PMax campaign underperforming?',
    'What\'s the benchmark CTR for Financial Services?',
    'Check URL health for my landing pages',
  ]

  const auditPlaceholders = [
    "Tell me what account you'd like audited...",
    'Generate an audit for my Brand campaigns',
    'Audit my account with uploaded files',
    'I need a PMax performance analysis',
    'Run a full audit with industry benchmarks',
  ]

  const placeholders = variant === 'chat' ? chatPlaceholders : auditPlaceholders
  const currentPlaceholder = inputPlaceholder || placeholders[placeholderIndex]
  const downloadUrl = result?.download_url || (result?.file_name ? api.audit.download(result.file_name) : null)
  const queueing = Boolean(auditQueue)

  // Rotate placeholder every 3 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      setPlaceholderIndex((prev) => (prev + 1) % placeholders.length)
    }, 3000)
    return () => clearInterval(interval)
  }, [placeholders.length])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Progress stage advancement logic
  useEffect(() => {
    if (!generating) {
      setCurrentStage(0)
      return
    }

    const stages = showUpload && files.length > 0 ? auditStages : creativeStages

    // Calculate cumulative durations for each stage
    const stageDurations = stages.map((stage) => {
      const duration = stage.duration
      if (typeof duration === 'string') {
        const match = duration.match(/(\d+)-?(\d+)?\s*(s|sec|min|minute)/)
        if (match) {
          const avg = match[2] ? (parseInt(match[1]) + parseInt(match[2])) / 2 : parseInt(match[1])
          const unit = match[3]
          return unit.startsWith('min') ? avg * 60 * 1000 : avg * 1000 // Convert to milliseconds
        }
      }
      return 0
    })

    const intervals = []

    // Set up timers for each stage
    stageDurations.forEach((_, idx) => {
      if (idx < stages.length - 1) { // Don't advance past the final stage
        const timeout = setTimeout(() => {
          setCurrentStage(idx + 1)
        }, stageDurations.slice(0, idx + 1).reduce((a, b) => a + b, 0))
        intervals.push(timeout)
      }
    })

    return () => {
      intervals.forEach(clearTimeout)
    }
  }, [generating, files.length, showUpload])

  const handleFiles = (newFiles) => {
    if (!newFiles?.length) return
    setFiles((prev) => [...prev, ...Array.from(newFiles)])
  }

  const handleDrop = (e) => {
    e.preventDefault()
    handleFiles(e.dataTransfer.files)
  }

  const handleDragOver = (e) => {
    e.preventDefault()
  }

  const removeFile = (name) => {
    setFiles((prev) => prev.filter((f) => f.name !== name))
  }

  const pollAuditJob = async (jobId, businessUnit) => {
    const attempts = 90
    const intervalMs = 10000
    for (let i = 0; i < attempts; i += 1) {
      try {
        const statusResp = await axios.get(api.jobs.status(jobId))
        const job = statusResp.data?.job
        const jobStatus = job?.status
        if (jobStatus === 'failed') {
          const detail = job?.error ? `Audit job failed: ${job.error}` : 'Audit job failed.'
          setError(detail)
          setMessages((prev) => [...prev, { role: 'assistant', content: detail }])
          setAuditQueue(null)
          setGenerating(false)
          return
        }
        if (jobStatus === 'succeeded') {
          const resultResp = await axios.get(api.jobs.result(jobId))
          const jobResult = resultResp.data?.result ?? resultResp.data
          const auditResult = jobResult?.file_name ? jobResult : jobResult?.result ?? jobResult
          if (auditResult) {
            setResult(auditResult)
            setMessages((prev) => [
              ...prev,
              { role: 'assistant', content: `Audit complete! Your ${businessUnit || 'audit'} report is ready for download.` },
            ])
            setAuditQueue(null)
            setGenerating(false)
            return
          }
        }
      } catch (err) {
        const status = err?.response?.status
        if (status && status !== 404) {
          const detail = err?.response?.data?.detail || 'Failed to retrieve audit job status.'
          setError(detail)
          setMessages((prev) => [...prev, { role: 'assistant', content: detail }])
          setAuditQueue(null)
          setGenerating(false)
          return
        }
      }
      await new Promise((resolve) => setTimeout(resolve, intervalMs))
    }
    const timeoutMsg = 'Audit job is still running. Please check again shortly.'
    setError(timeoutMsg)
    setMessages((prev) => [...prev, { role: 'assistant', content: timeoutMsg }])
    setAuditQueue(null)
    setGenerating(false)
  }

  const queueAuditJob = (jobId, businessUnit) => {
    if (!jobId) return
    setError(null)
    setAuditQueue({ jobId, businessUnit })
    const shortId = jobId.slice(0, 8)
    setMessages((prev) => [
      ...prev,
      { role: 'assistant', content: `Audit queued (job ${shortId}). I'll update you when it's ready.` },
    ])
    setGenerating(false)
    void pollAuditJob(jobId, businessUnit)
  }

  const generateAuditFromFiles = async () => {
    if (!files.length || generating || queueing) return
    setGenerating(true)
    setError(null)
    setMessages((prev) => [...prev, { role: 'assistant', content: `Uploading ${files.length} file(s) and generating audit...` }])

    try {
      const formData = new FormData()
      files.forEach((f) => formData.append('files', f))
      formData.append('account_name', accountName || 'Uploaded')

      // Upload to blob storage
      await axios.post(api.data.upload, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })

      // Trigger audit generation pulling from storage
      const auditResponse = await axios.post(api.audit.generate, {
        business_unit: accountName || undefined,
        account_name: accountName || 'Uploaded',
        use_mock_data: false,
        async_mode: true,
      })
      if (auditResponse.data?.status === 'queued' && auditResponse.data?.job_id) {
        setFiles([])
        queueAuditJob(auditResponse.data.job_id, 'audit')
      } else {
        setResult(auditResponse.data)
        setMessages((prev) => [...prev, { role: 'assistant', content: `Audit complete! Your file is ready for download.` }])
        setFiles([])
        setGenerating(false)
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to upload and generate audit')
      setMessages((prev) => [...prev, { role: 'assistant', content: 'Upload failed. Please try again or check your files.' }])
      setGenerating(false)
    }
  }

  const sendMessage = async () => {
    if (showUpload && files.length) {
      await generateAuditFromFiles()
      return
    }
    if (!input.trim() || loading || generating || queueing) return

    const userMessage = input.trim()
    setInput('')
    setMessages((prev) => [...prev, { role: 'user', content: userMessage }])
    setLoading(true)
    setError(null)

    try {
      const chatResponse = await axios.post(api.chat.send, {
        message: `[AUDIT REQUEST] ${userMessage}`,
        ai_enabled: true,
      })
      const aiReply = chatResponse.data.reply

      const shouldGenerate =
        aiReply.toLowerCase().includes('generating') ||
        aiReply.toLowerCase().includes('will generate') ||
        userMessage.toLowerCase().includes('generate') ||
        userMessage.toLowerCase().includes('audit') ||
        userMessage.toLowerCase().includes('brand') ||
        userMessage.toLowerCase().includes('nonbrand') ||
        userMessage.toLowerCase().includes('pmax')

      setMessages((prev) => [...prev, { role: 'assistant', content: aiReply }])

      if (shouldGenerate) {
        let businessUnit = 'Brand'
        const lowerMsg = userMessage.toLowerCase()
        if (lowerMsg.includes('nonbrand') || lowerMsg.includes('non-brand')) {
          businessUnit = 'NonBrand'
        } else if (lowerMsg.includes('pmax') || lowerMsg.includes('performance max')) {
          businessUnit = 'PMax'
        }

        setGenerating(true)
        setMessages((prev) => [...prev, { role: 'assistant', content: `Starting audit generation for ${businessUnit}... This may take a few moments.` }])

        const auditResponse = await axios.post(api.audit.generate, {
          business_unit: businessUnit,
          account_name: businessUnit,
          use_mock_data: lowerMsg.includes('demo') || lowerMsg.includes('test'),
          async_mode: true,
        })
        if (auditResponse.data?.status === 'queued' && auditResponse.data?.job_id) {
          queueAuditJob(auditResponse.data.job_id, businessUnit)
        } else {
          setResult(auditResponse.data)
          setMessages((prev) => [...prev, { role: 'assistant', content: `Audit complete! Your ${businessUnit} audit report is ready for download.` }])
          setGenerating(false)
        }
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to process request')
      setMessages((prev) => [...prev, { role: 'assistant', content: 'Sorry, I encountered an error. Please try again or rephrase your request.' }])
      setGenerating(false)
    } finally {
      setLoading(false)
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const handleSuggestedPrompt = (prompt) => {
    setInput(prompt)
  }

  const showEmptyState = messages.length === 1 && !loading && !generating && !result

  // First visit detection for onboarding tour
  const { isFirstVisit, markAsVisited } = useFirstVisit(`kai-${variant}-visited`)

  return (
    <>
      {/* Onboarding Tour */}
      {isFirstVisit && (
        <OnboardingTour variant={variant} onComplete={markAsVisited} theme={theme} />
      )}

      {/* Main Content */}
    <Container
      maxWidth="lg"
      sx={{
        minHeight: 'calc(100vh - 64px)',
        display: 'flex',
        flexDirection: 'column',
        gap: 2,
        color: 'var(--kai-text)',
      }}
    >
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }} style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
        <Paper
          elevation={0}
          sx={{
            p: 3,
            mb: 2,
            borderRadius: 3,
            border: `1px solid ${theme.borderColor}33`,
            background: theme.headerBg,
            boxShadow: '0 20px 45px rgba(0, 0, 0, 0.35)',
          }}
        >
          <Box display="flex" justifyContent="space-between" alignItems="center" flexWrap="wrap" gap={2}>
            <Box display="flex" alignItems="center" gap={1}>
              <Box>
                <Typography variant="h4" fontWeight={800} gutterBottom color="var(--kai-text)">
                  {title}
                </Typography>
                <Typography variant="body1" sx={{ color: '#cbd5e1' }}>
                  {subtitle}
                </Typography>
              </Box>
              <HelpTooltip
                theme={theme}
                title={variant === 'chat' ? 'What can I ask Kai?' : 'How does the audit work?'}
                content={
                  variant === 'chat'
                    ? `AUDIT REQUESTS:
â€¢ "Generate audit for [Brand/NonBrand/PMax]"
â€¢ "Audit my account with uploaded files"

CREATIVE HELP:
â€¢ "Create headlines for [business name]"
â€¢ "Generate RSA copy for [keywords]"

ANALYSIS:
â€¢ "Analyze PMax placements for waste"
â€¢ "Check URL health for [URLs]"
â€¢ "Why is performance down?" (root-cause)

STRATEGY:
â€¢ "What's the benchmark CTR for Finance?"
â€¢ "How many conversions for Target CPA?"`
                    : `The audit analyzes 100+ points across 9 dimensions:

1. Structure & Hygiene
2. Keywords & Queries
3. Audiences & Remarketing
4. Ads & Creative
5. Landing Pages & CX
6. Measurement & Tracking
7. Bid Strategy & Budgets
8. Cross-Channel Integration
9. Strategic Recommendations

Each item is scored 1-5 with industry benchmarks, business impact ratings, and actionable recommendations.

Processing time: 5-10 minutes
Supports: 50K+ rows, 14 file types`
                }
              />
            </Box>
            <Chip label="Live AI Model" color="primary" variant="outlined" />
          </Box>
        </Paper>

        <Paper
          elevation={3}
          sx={{
            flex: 1,
            p: 3,
            mb: 2,
            borderRadius: 3,
            background: 'var(--kai-bg)',
            border: '1px solid var(--kai-surface-alt)',
            boxShadow: '0 20px 45px rgba(0, 0, 0, 0.35)',
            overflowY: 'auto',
            display: 'flex',
            flexDirection: 'column',
            gap: 2,
          }}
        >
          {showEmptyState && (
            <EmptyState variant={variant} onSuggestedPrompt={handleSuggestedPrompt} theme={theme} />
          )}

          <AnimatePresence>
            {!showEmptyState && messages.map((msg, idx) => (
              <motion.div key={idx} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.2 }}>
                <Box display="flex" gap={2} alignItems="flex-start" justifyContent={msg.role === 'user' ? 'flex-end' : 'flex-start'}>
                  {msg.role === 'assistant' && (
                    <Avatar sx={{ bgcolor: theme.aiBubbleBg, width: 36, height: 36 }}>
                      <SmartToy fontSize="small" />
                    </Avatar>
                  )}
                  <Paper
                    elevation={1}
                    sx={{
                      p: 2,
                      maxWidth: '70%',
                      borderRadius: 3,
                      background: msg.role === 'user' ? theme.userBubbleGradient : theme.aiBubbleBg,
                      color: 'var(--kai-text)',
                      border: msg.role === 'user' ? `1px solid ${theme.accentColor}` : '1px solid var(--kai-surface-alt)',
                    }}
                  >
                    <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap' }}>
                      {msg.content}
                    </Typography>
                  </Paper>
                  {msg.role === 'user' && (
                    <Avatar sx={{ bgcolor: theme.avatarBg, width: 36, height: 36 }}>
                      <Person fontSize="small" />
                    </Avatar>
                  )}
                </Box>
              </motion.div>
            ))}
          </AnimatePresence>

          {generating && (
            <ProgressStages
              stages={showUpload && files.length > 0 ? auditStages : creativeStages}
              currentStage={currentStage}
              theme={theme}
            />
          )}

          {auditQueue && (
            <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.25 }}>
              <Paper
                elevation={2}
                sx={{
                  p: 2.5,
                  borderRadius: 3,
                  background: 'var(--kai-bg)',
                  border: `1px solid ${theme.borderColor}33`,
                }}
              >
                <Typography variant="body1" color="var(--kai-text)" fontWeight={600} mb={1}>
                  Audit queued - waiting for the worker to finish.
                </Typography>
                <LinearProgress
                  variant="indeterminate"
                  sx={{
                    height: 6,
                    borderRadius: 2,
                    background: 'var(--kai-surface-alt)',
                    '& .MuiLinearProgress-bar': {
                      background: theme.primaryGradient,
                      borderRadius: 2,
                    },
                  }}
                />
                <Typography variant="caption" color="var(--kai-text-soft)" sx={{ mt: 1, display: 'block' }}>
                  Job ID: {auditQueue.jobId}
                </Typography>
              </Paper>
            </motion.div>
          )}

          {result && (
            <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.3 }}>
              <Paper
                elevation={3}
                sx={{
                  p: 3,
                  borderRadius: 3,
                  background: 'linear-gradient(135deg, rgba(16,185,129,0.15), rgba(59,130,246,0.12))',
                  border: '1px solid rgba(16,185,129,0.35)',
                }}
              >
                <Box display="flex" alignItems="center" gap={2} mb={2}>
                  <CheckCircle sx={{ fontSize: 32, color: '#34d399' }} />
                  <Typography variant="h6" fontWeight={700} color="var(--kai-text)">
                    Audit Ready
                  </Typography>
                </Box>
                <Box display="flex" gap={2} mb={2} flexWrap="wrap">
                  <Chip label={`File: ${result.file_name}`} color="success" size="small" />
                  {result.result?.overall_score && <Chip label={`Score: ${result.result.overall_score}`} color="info" size="small" />}
                </Box>
                <Box display="flex" gap={2}>
                  <IconButton
                    color="success"
                    href={downloadUrl || '#'}
                    target={downloadUrl ? '_blank' : undefined}
                    rel={downloadUrl ? 'noreferrer' : undefined}
                    disabled={!downloadUrl}
                    sx={{
                      background: '#10b981',
                      color: '#fff',
                      '&:hover': { background: '#059669' },
                    }}
                  >
                    <Download />
                  </IconButton>
                  <Typography variant="body2" sx={{ color: '#d1fae5' }} alignSelf="center">
                    Click to download your audit report
                  </Typography>
                </Box>
              </Paper>
            </motion.div>
          )}

          {error && (
            <Alert severity="error" onClose={() => setError(null)} sx={{ borderRadius: 2, background: 'var(--kai-surface)', color: '#fca5a5', border: '1px solid #dc2626' }}>
              {error}
            </Alert>
          )}

          <div ref={messagesEndRef} />
        </Paper>

        <Paper
          elevation={0}
          sx={{
            p: 2.5,
            mb: 2,
            borderRadius: 3,
            border: '1px dashed var(--kai-border-strong)',
            background: 'var(--kai-surface-alt)',
          }}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
        >
          <Stack direction="row" alignItems="center" spacing={1.5} flexWrap="wrap">
            {showUpload ? (
              <>
                <TextField
                  label="Account name"
                  inputProps={{ 'data-testid': 'audit-account' }}
                  value={accountName}
                  onChange={(e) => setAccountName(e.target.value)}
                  size="small"
                  sx={{
                    minWidth: 200,
                    '& .MuiOutlinedInput-root': {
                      color: 'var(--kai-text)',
                      '& fieldset': { borderColor: 'var(--kai-border-strong)' },
                      '&:hover fieldset': { borderColor: theme.accentColor },
                      '&.Mui-focused fieldset': { borderColor: theme.focusColor },
                    },
                    '& .MuiInputLabel-root': { color: '#cbd5e1' },
                  }}
                />
                <Tooltip title="Upload CSV/XLSX exports to generate a fresh audit">
                  <Avatar sx={{ bgcolor: 'var(--kai-bg)', width: 36, height: 36 }}>
                    <CloudUpload fontSize="small" />
                  </Avatar>
                </Tooltip>
                <Typography variant="body2" sx={{ color: '#cbd5e1' }}>
                  Drop CSV/XLSX here or browse to start a new audit.
                </Typography>
                <input
                  type="file"
                  multiple
                  accept=".csv,.xlsx,.xls"
                  id="file-upload-input"
                  data-testid="audit-file-input"
                  style={{ position: 'absolute', opacity: 0, width: 1, height: 1, pointerEvents: 'auto' }}
                  onChange={(e) => handleFiles(e.target.files)}
                />
                <label htmlFor="file-upload-input">
                  <Chip label="Browse files" component="span" clickable sx={{ color: 'var(--kai-text)' }} data-testid="audit-browse" />
                </label>
                {files.map((f) => (
                  <Chip key={f.name} label={f.name} onDelete={() => removeFile(f.name)} sx={{ color: 'var(--kai-text)', borderColor: 'var(--kai-border-strong)' }} />
                ))}
                <Chip
                  label="Generate from files"
                  color="success"
                  clickable
                  data-testid="audit-run"
                  onClick={generateAuditFromFiles}
                  disabled={!files.length || generating || queueing}
                />
              </>
            ) : (
              <Typography variant="body2" sx={{ color: '#cbd5e1' }}>
                Ask me for audits, insights, or any taskâ€”Iâ€™ll route it to the right system.
              </Typography>
            )}
          </Stack>
        </Paper>

        <Paper
          elevation={3}
          sx={{
            p: 2,
            borderRadius: 3,
            background: 'var(--kai-bg)',
            border: '1px solid var(--kai-surface-alt)',
            boxShadow: '0 10px 30px rgba(0,0,0,0.3)',
          }}
        >
          <Box display="flex" gap={2} alignItems="center">
            <TextField
              fullWidth
              multiline
              maxRows={3}
              placeholder={currentPlaceholder}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              disabled={loading || generating || queueing}
              variant="outlined"
              sx={{
                '& .MuiOutlinedInput-root': {
                  borderRadius: 2,
                  color: 'var(--kai-text)',
                  background: 'var(--kai-bg)',
                  '& fieldset': { borderColor: 'var(--kai-border-strong)' },
                  '&:hover fieldset': { borderColor: theme.accentColor },
                  '&.Mui-focused fieldset': { borderColor: theme.focusColor },
                },
              }}
            />
            <IconButton
              color="primary"
              onClick={sendMessage}
              disabled={!input.trim() || loading || generating || queueing}
              sx={{
                background: theme.primaryGradient,
                color: '#fff',
                width: 48,
                height: 48,
                '&:hover': {
                  background: theme.userBubbleGradient,
                },
                '&:disabled': {
                  background: 'var(--kai-border)',
                },
              }}
            >
              <Send />
            </IconButton>
          </Box>
        </Paper>
      </motion.div>
    </Container>
    </>
  )
}

