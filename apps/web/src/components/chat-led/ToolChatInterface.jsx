/**
 * ToolChatInterface - AI chat component for tool pages
 *
 * Features:
 * - Capability cards (hide after first message)
 * - Suggested prompts
 * - Message bubbles with avatars
 * - Loading animation
 * - Advanced mode toggle for original forms
 */
import { useState, useEffect, useRef } from 'react'
import {
  Box,
  Typography,
  Paper,
  TextField,
  IconButton,
  Avatar,
  Chip,
  Stack,
  Alert,
  Fade,
  Collapse,
  Button,
  Card,
  CardContent,
  CardActionArea,
  Grid,
} from '@mui/material'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Send,
  AutoAwesome,
  Person,
  ExpandMore,
  ExpandLess,
  Settings,
} from '@mui/icons-material'
import axios from 'axios'
import { api, getOrCreateSessionId } from '../../config'

export default function ToolChatInterface({
  toolId,           // 'pmax' | 'creative' | 'serp'
  toolName,
  themeColor,
  capabilities,     // Array of { id, name, icon, description, examples }
  examplePrompts,   // Array of strings
  onDataExtracted,  // Callback when AI extracts structured data
  onAnalysisComplete, // Callback when backend returns results
  advancedModeContent, // Optional: original form for advanced mode
  mode = null,      // Optional: sub-mode for multi-mode tools (e.g., 'url_health' | 'competitor')
  accountName = null,   // Optional: account context for planner/chat
  customerIds = [],     // Optional: known customer ids to pass through
  topK = null,          // Optional: top_k for web grounding
}) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [showCapabilities, setShowCapabilities] = useState(true)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [lastPlannerContext, setLastPlannerContext] = useState(null)
  const messagesEndRef = useRef(null)
  const [sessionId] = useState(() => {
    return getOrCreateSessionId()
  })

  const metricTerms = [
    'impression',
    'impressions',
    'click',
    'clicks',
    'cpc',
    'ctr',
    'cpm',
    'roas',
    'cpa',
    'cost',
    'spend',
    'conversion',
    'conversions',
    'conv',
    'performance',
    'audit',
    'ppc',
    'sa360',
    'google ads',
  ]

  const timeframeHints = [
    'last week',
    'this week',
    'last month',
    'this month',
    'yesterday',
    'today',
    'week before last',
    'two weeks ago',
    'last 7 days',
    'last 14 days',
    'last 30 days',
    'last 90 days',
    'year over year',
    'yoy',
    'q1',
    'q2',
    'q3',
    'q4',
  ]

  const isMetricIntent = (text) => {
    const t = (text || '').toLowerCase()
    return metricTerms.some((k) => t.includes(k)) || /\b\d{8,12}\b/.test(t)
  }

  const isFollowupPrompt = (text) => {
    const t = (text || '').trim().toLowerCase()
    if (!t) return false
    const words = t.split(/\s+/).filter(Boolean)
    if (words.length <= 4) return true
    const cues = [
      'explain',
      'why',
      'what does that mean',
      'what happened',
      'tell me more',
      'break down',
      'breakdown',
      'driver',
      'drivers',
      'cause',
      'reason',
      'so what',
      'interpret',
      'summary',
      'summarize',
      'recap',
      'next action',
      'next step',
      'next steps',
      'next actions',
      'action',
      'actions',
      'recommend',
      'recommendation',
      'recommendations',
      'what should i do',
      'what should we do',
      'what would you do',
      'what do you suggest',
      'optimize',
      'optimise',
      'optimization',
      'optimizations',
      'improve',
      'improved',
      'improvement',
      'improvements',
      'suggest',
      'suggestion',
      'prioritize',
      'priority',
      'impact',
      'takeaway',
      'what now',
      'need more data',
      'what data',
      'what else do you need',
      'what would you need',
      'what additional data',
    ]
    return cues.some((cue) => t.includes(cue))
  }

  const hasTimeframeHint = (text) => {
    const t = (text || '').toLowerCase()
    if (!t) return false
    if (/\b\d{4}-\d{2}-\d{2}\b/.test(t)) return true
    if (/\b(last|past|previous)\s+\d+\s+(day|week|month|quarter|year)s?\b/.test(t)) return true
    if (/\bLAST_(7|14|30|90)_DAYS\b/.test(t.toUpperCase())) return true
    return timeframeHints.some((k) => t.includes(k))
  }

  const extractCustomerIds = (text) => {
    const t = (text || '')
    const matches = t.match(/\b\d{8,12}\b/g) || []
    return Array.from(new Set(matches))
  }

  const mergeCustomerIds = (textIds) => {
    const base = Array.isArray(customerIds) ? customerIds : []
    const inferred = Array.isArray(textIds) ? textIds : []
    return Array.from(new Set([...base, ...inferred].filter(Boolean)))
  }

  const stripInternalNotes = (notes) => {
    if (!notes || typeof notes !== 'string') return ''
    const parts = notes.split(';').map((p) => p.trim()).filter(Boolean)
    const filtered = parts.filter((note) => {
      const lower = note.toLowerCase()
      return !(
        lower.startsWith('resolved account to') ||
        lower.startsWith('detected customer_id') ||
        lower.startsWith('identified account') ||
        lower.startsWith('multiple account matches found') ||
        lower.startsWith('no date specified') ||
        lower.startsWith('defaulting to') ||
        lower.startsWith('router_') ||
        lower.startsWith('verify=') ||
        lower.startsWith('model=')
      )
    })
    return filtered.join('; ')
  }

  const normalizeInternalTokens = (text) => {
    if (!text || typeof text !== 'string') return ''
    let out = text
    out = out.replace(/\bLAST_(\d+)_DAYS\b/g, (_, n) => `last ${n} days`)
    out = out.replace(/\bLAST_WEEK\b/g, 'last week')
    out = out.replace(/\bLAST_MONTH\b/g, 'last month')
    out = out.replace(/\bTHIS_MONTH\b/g, 'this month')
    out = out.replace(/No date specified; defaulting to\s*last\s*\d+\s*days\.?/gi, '')
    out = out.replace(/No date specified; defaulting to\s*LAST_\d+_DAYS\.?/gi, '')
    return out.replace(/\s+/g, ' ').trim()
  }

  const buildPlannerContext = (planData) => {
    const compactPlan = {
      executed: planData?.executed,
      notes: planData?.notes,
      error: planData?.error,
      account: planData?.plan?.account_name,
      date_range: planData?.plan?.date_range,
      customer_ids: planData?.plan?.customer_ids,
      file: planData?.result?.file_name,
      analysis_note: planData?.analysis?.note || planData?.analysis?.summary,
      summary: planData?.enhanced_summary || planData?.summary,
    }
    const cleanedNotes = stripInternalNotes(compactPlan.notes)
    const seedParts = []
    if (compactPlan.summary) seedParts.push(compactPlan.summary)
    if (compactPlan.analysis_note) seedParts.push(compactPlan.analysis_note)
    if (compactPlan.file) seedParts.push(`Report: ${compactPlan.file}`)
    if (cleanedNotes) seedParts.push(cleanedNotes)
    const summarySeed = seedParts.filter(Boolean).join(' ').trim()

    return {
      plan: compactPlan,
      result: planData?.result,
      analysis: planData?.analysis,
      summary_seed: summarySeed,
    }
  }

  const plannerSummaryViaLLM = async (plannerContext, question) => {
    if (!plannerContext) return null
    const msg = String(question || '').trim() || 'Summarize the planner output.'
    try {
      const llmResp = await axios.post(api.chat.send, {
        message: msg,
        ai_enabled: true,
        session_id: sessionId || undefined,
        context: { tool: 'performance', tool_output: plannerContext, prompt_kind: 'planner_summary' },
      })
      const raw = String(llmResp.data?.reply || '').trim()
      return normalizeInternalTokens(raw)
    } catch (err) {
      return null
    }
  }

  const routeMessage = async (text, ids) => {
    try {
      const routeResp = await axios.post(api.chat.route, {
        message: text,
        customer_ids: Array.isArray(ids) ? ids : [],
        account_name: accountName || undefined,
        session_id: sessionId || undefined,
      })
      return routeResp.data || null
    } catch (err) {
      return null
    }
  }

  const formatAnalysis = (analysis) => {
    if (!analysis) return null
    if (analysis.type === 'top_movers') {
      const items = (analysis.items || []).slice(0, 3).map((i) => {
        const pct = i.pct_change === null || i.pct_change === undefined ? '' : ` (${i.pct_change.toFixed(1)}%)`
        return `- ${i.name}: ${i.metric} ${i.current} vs ${i.previous}${pct}`
      })
      return [
        `Top movers (${analysis.entity_type}, ${analysis.metric_focus})`,
        ...items,
      ].join('\n')
    }
    if (analysis.deltas) {
      const d = analysis.deltas
      const parts = []
      const add = (k, label) => {
        const node = d[k]
        if (!node) return
        const pct = node.pct_change === null || node.pct_change === undefined ? '' : ` (${node.pct_change.toFixed(1)}%)`
        parts.push(`${label}: ${node.current} vs ${node.previous}${pct}`)
      }
      add('impressions', 'Impr')
      add('clicks', 'Clicks')
      add('conversions', 'Conv')
      add('ctr', 'CTR')
      add('cpc', 'CPC')
      return [
        `Entity: ${analysis.entity?.identifier || analysis.entity?.entity_type || 'entity'}`,
        ...parts,
        analysis.drivers && analysis.drivers.length ? `Drivers: ${analysis.drivers.join('; ')}` : null,
      ].filter(Boolean).join('\n')
    }
    if (analysis.note) return analysis.note
    return null
  }

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

  const pollJob = async (jobId, originalQuestion) => {
    const deadline = Date.now() + 12 * 60 * 1000
    while (Date.now() < deadline) {
      try {
        const statusResp = await axios.get(api.jobs.status(jobId))
        const state = statusResp?.data?.job?.status
        if (state === 'succeeded' || state === 'failed') {
          if (state === 'succeeded') {
            const resultResp = await axios.get(api.jobs.result(jobId))
            const payload = resultResp?.data?.result || resultResp?.data
            const planData = payload?.result ? payload.result : payload
            if (planData?.analysis) {
              onAnalysisComplete?.(planData.analysis)
            }
            const ctx = buildPlannerContext(planData || {})
            setLastPlannerContext(ctx)
            const summary = await plannerSummaryViaLLM(ctx, originalQuestion || 'Summarize the results.')
            const fallback = ctx?.summary_seed ? normalizeInternalTokens(ctx.summary_seed) : 'Queued job finished.'
            setMessages((prev) => [...prev, { role: 'assistant', content: summary || fallback }])
          } else {
            setMessages((prev) => [...prev, { role: 'assistant', content: 'Queued job failed. Please retry.' }])
          }
          return
        }
      } catch (err) {
        // keep polling on transient errors
      }
      await sleep(5000)
    }
    setMessages((prev) => [...prev, { role: 'assistant', content: 'Queued job is still running. Please check back shortly.' }])
  }

  // Handle sending a message
  const sendMessage = async () => {
    if (!input.trim() || loading) return

    const userMessage = input.trim()
    setInput('')
    setMessages(prev => [...prev, { role: 'user', content: userMessage }])
    setLoading(true)
    setError(null)
    setShowCapabilities(false)

    try {
      const explicitIds = mergeCustomerIds(extractCustomerIds(userMessage))
      const timeframeHint = hasTimeframeHint(userMessage)
      const canReusePlanner = !!(lastPlannerContext && !explicitIds.length && !timeframeHint && isFollowupPrompt(userMessage))
      if (canReusePlanner) {
        const followupReply = await plannerSummaryViaLLM(lastPlannerContext, userMessage)
        const fallback = lastPlannerContext?.summary_seed ? normalizeInternalTokens(lastPlannerContext.summary_seed) : null
        setMessages(prev => [...prev, { role: 'assistant', content: followupReply || fallback || 'I can walk through the results and recommend next steps. What outcome are you optimizing for (CPA, ROAS, volume, stability)?' }])
        return
      }

      // Metric/account intents -> use router and planner when it is actually a performance request.
      if (isMetricIntent(userMessage)) {
        const routing = await routeMessage(userMessage, explicitIds)
        const wantsPerformance =
          routing?.intent === 'performance' ||
          routing?.tool === 'performance' ||
          routing?.run_planner === true ||
          (!routing && isMetricIntent(userMessage))
        if (wantsPerformance) {
          const body = {
            message: userMessage,
            customer_ids: explicitIds,
            account_name: (routing?.account_name || accountName) || null,
            session_id: sessionId || undefined,
          }
          const planResp = await axios.post(api.chat.planAndRun, body)
          if (planResp.data?.status === 'queued' && planResp.data?.job_id) {
            setMessages(prev => [...prev, { role: 'assistant', content: 'Queued the analysis job. I will post results as soon as it finishes.' }])
            pollJob(planResp.data.job_id, userMessage)
            return
          }
          if (planResp.data?.analysis) {
            onAnalysisComplete?.(planResp.data.analysis)
          }
          if (planResp.data?.executed === false && (planResp.data?.summary || planResp.data?.error)) {
            const blockerText = normalizeInternalTokens(String(planResp.data.summary || planResp.data.error || 'The planner did not run.').trim())
            setMessages(prev => [...prev, { role: 'assistant', content: blockerText }])
            return
          }
          const ctx = buildPlannerContext(planResp.data || {})
          setLastPlannerContext(ctx)
          const summary = await plannerSummaryViaLLM(ctx, userMessage)
          const fallback = ctx?.summary_seed ? normalizeInternalTokens(ctx.summary_seed) : 'Analysis completed.'
          setMessages(prev => [...prev, { role: 'assistant', content: summary || fallback }])
          return
        }
      }

      // Send to AI chat endpoint with tool context
      const response = await axios.post(api.chat.send, {
        message: userMessage,
        ai_enabled: true,
        account_name: accountName || null,
        top_k: topK || undefined,
        context: { tool: toolId, mode: mode, customer_ids: explicitIds },
        session_id: sessionId || undefined,
      })

      setMessages(prev => [...prev, {
        role: 'assistant',
        content: response.data.reply,
      }])

      // Check if response contains extracted data for the tool
      if (response.data.extracted_data) {
        onDataExtracted?.(response.data.extracted_data)
      }

      // Check if analysis was performed
      if (response.data.analysis_result) {
        onAnalysisComplete?.(response.data.analysis_result)
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to process request')
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: "I encountered an error processing your request. Please try again or use the advanced mode below.",
      }])
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

  const handlePromptClick = (prompt) => {
    setInput(prompt)
  }

  const handleCapabilityClick = (capability) => {
    if (capability.examples?.[0]) {
      setInput(capability.examples[0])
    }
  }

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Capability Cards - Show when empty */}
      <Collapse in={showCapabilities && messages.length === 0}>
        <Box sx={{ p: 2 }}>
          <Typography variant="overline" sx={{ color: '#64748b', mb: 1.5, display: 'block', letterSpacing: 1.5 }}>
            I CAN HELP YOU WITH
          </Typography>
          <Grid container spacing={1.5}>
            {capabilities?.map((cap, idx) => (
              <Grid item xs={12} sm={6} key={cap.id || idx}>
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, delay: idx * 0.1 }}
                >
                  <Card
                    sx={{
                      background: 'var(--kai-surface)',
                      border: `1px solid ${cap.color || themeColor}33`,
                      borderRadius: 2,
                      transition: 'all 0.2s ease',
                      '&:hover': {
                        borderColor: cap.color || themeColor,
                        boxShadow: `0 4px 20px ${cap.color || themeColor}22`,
                        transform: 'translateY(-2px)',
                      },
                    }}
                  >
                    <CardActionArea onClick={() => handleCapabilityClick(cap)} sx={{ p: 1.5 }}>
                      <CardContent sx={{ p: 0 }}>
                        <Box display="flex" alignItems="center" gap={1.5} mb={1}>
                          <Avatar
                            sx={{
                              width: 32,
                              height: 32,
                              background: `${cap.color || themeColor}22`,
                              color: cap.color || themeColor,
                            }}
                          >
                            {cap.icon && <cap.icon sx={{ fontSize: 18 }} />}
                          </Avatar>
                          <Typography variant="subtitle2" sx={{ color: 'var(--kai-text)', fontWeight: 600 }}>
                            {cap.name}
                          </Typography>
                        </Box>
                        <Typography variant="caption" sx={{ color: 'var(--kai-text-soft)' }}>
                          {cap.description}
                        </Typography>
                      </CardContent>
                    </CardActionArea>
                  </Card>
                </motion.div>
              </Grid>
            ))}
          </Grid>
        </Box>
      </Collapse>

      {/* Messages Area */}
      <Box
        sx={{
          flex: 1,
          overflowY: 'auto',
          p: 2,
          display: 'flex',
          flexDirection: 'column',
          gap: 2,
        }}
      >
        {/* Empty state with suggested prompts */}
        {messages.length === 0 && !showCapabilities && (
          <Box
            display="flex"
            flexDirection="column"
            alignItems="center"
            justifyContent="center"
            height="100%"
            color="var(--kai-border)"
          >
            <AutoAwesome sx={{ fontSize: 40, mb: 2, opacity: 0.5 }} />
            <Typography variant="body2" sx={{ mb: 2 }}>
              Ask me anything about {toolName.toLowerCase()}
            </Typography>
          </Box>
        )}

        {/* Message Bubbles */}
        <AnimatePresence>
          {messages.map((msg, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.2 }}
            >
              <Box
                display="flex"
                gap={1.5}
                alignItems="flex-start"
                justifyContent={msg.role === 'user' ? 'flex-end' : 'flex-start'}
              >
                {msg.role === 'assistant' && (
                  <Avatar
                    sx={{
                      width: 32,
                      height: 32,
                      background: `linear-gradient(135deg, ${themeColor}, ${themeColor}cc)`,
                    }}
                  >
                    <AutoAwesome sx={{ fontSize: 16 }} />
                  </Avatar>
                )}

                <Paper
                  elevation={0}
                  data-testid={msg.role === 'assistant' ? 'chat-assistant' : 'chat-user'}
                  sx={{
                    p: 1.5,
                    maxWidth: '80%',
                    borderRadius: 2,
                    background: msg.role === 'user'
                      ? `linear-gradient(135deg, ${themeColor}, ${themeColor}cc)`
                      : 'var(--kai-surface-muted)',
                    color: 'var(--kai-text)',
                    border: msg.role === 'user'
                      ? `1px solid ${themeColor}`
                      : '1px solid var(--kai-border-strong)',
                  }}
                >
                  <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                    {msg.content}
                  </Typography>
                </Paper>

                {msg.role === 'user' && (
                  <Avatar sx={{ width: 32, height: 32, bgcolor: '#3730a3' }}>
                    <Person sx={{ fontSize: 16 }} />
                  </Avatar>
                )}
              </Box>
            </motion.div>
          ))}
        </AnimatePresence>

        {/* Loading Animation */}
        {loading && (
          <Box display="flex" gap={1.5} alignItems="center">
            <Avatar
              sx={{
                width: 32,
                height: 32,
                background: `linear-gradient(135deg, ${themeColor}, ${themeColor}cc)`,
              }}
            >
              <AutoAwesome sx={{ fontSize: 16 }} />
            </Avatar>
            <Paper sx={{ p: 1.5, borderRadius: 2, background: 'var(--kai-surface-muted)', border: '1px solid var(--kai-border-strong)' }}>
              <Box display="flex" gap={0.5}>
                {[0, 1, 2].map((i) => (
                  <motion.div
                    key={i}
                    animate={{ opacity: [0.3, 1, 0.3] }}
                    transition={{ duration: 1, repeat: Infinity, delay: i * 0.2 }}
                  >
                    <Box
                      sx={{
                        width: 6,
                        height: 6,
                        borderRadius: '50%',
                        background: themeColor,
                      }}
                    />
                  </motion.div>
                ))}
              </Box>
            </Paper>
          </Box>
        )}

        {/* Error Alert */}
        {error && (
          <Alert
            severity="error"
            onClose={() => setError(null)}
            sx={{
              borderRadius: 2,
              background: '#1c1917',
              color: '#fca5a5',
              border: '1px solid #dc2626',
            }}
          >
            {error}
          </Alert>
        )}

        <div ref={messagesEndRef} />
      </Box>

      {/* Suggested Prompts - Show when few messages */}
      <Collapse in={messages.length < 2 && examplePrompts?.length > 0}>
        <Box sx={{ px: 2, pb: 1 }}>
          <Typography variant="caption" sx={{ color: '#64748b', mb: 1, display: 'block' }}>
            Try asking:
          </Typography>
          <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
            {examplePrompts?.slice(0, 3).map((prompt, idx) => (
              <Chip
                key={idx}
                label={prompt}
                size="small"
                onClick={() => handlePromptClick(prompt)}
                sx={{
                  mb: 0.5,
                  background: `${themeColor}15`,
                  color: '#cbd5e1',
                  border: `1px solid ${themeColor}33`,
                  '&:hover': {
                    background: `${themeColor}25`,
                    borderColor: themeColor,
                  },
                }}
              />
            ))}
          </Stack>
        </Box>
      </Collapse>

      {/* Input Area */}
      <Box sx={{ p: 2, borderTop: '1px solid var(--kai-surface-muted)' }}>
        <Box display="flex" gap={1} alignItems="center">
          <TextField
            fullWidth
            multiline
            maxRows={3}
            placeholder={`Describe what you want to ${toolId === 'pmax' ? 'analyze' : toolId === 'creative' ? 'create' : 'check'}...`}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyPress}
            disabled={loading}
            variant="outlined"
            size="small"
            sx={{
              '& .MuiOutlinedInput-root': {
                borderRadius: 2,
                color: 'var(--kai-text)',
                background: 'var(--kai-surface)',
                '& fieldset': { borderColor: 'var(--kai-border-strong)' },
                '&:hover fieldset': { borderColor: themeColor },
                '&.Mui-focused fieldset': { borderColor: themeColor },
              },
              '& .MuiOutlinedInput-input::placeholder': {
                color: '#64748b',
                opacity: 1,
              },
            }}
          />
          <IconButton
            onClick={sendMessage}
            disabled={!input.trim() || loading}
            sx={{
              width: 40,
              height: 40,
              background: `linear-gradient(135deg, ${themeColor}, ${themeColor}cc)`,
              color: '#fff',
              '&:hover': {
                background: `linear-gradient(135deg, ${themeColor}ee, ${themeColor})`,
              },
              '&:disabled': {
                background: 'var(--kai-border-strong)',
                color: '#64748b',
              },
            }}
          >
            <Send sx={{ fontSize: 18 }} />
          </IconButton>
        </Box>

        {/* Advanced Mode Toggle */}
        {advancedModeContent && (
          <Box sx={{ mt: 1.5 }}>
            <Button
              size="small"
              startIcon={<Settings sx={{ fontSize: 14 }} />}
              endIcon={showAdvanced ? <ExpandLess /> : <ExpandMore />}
              onClick={() => setShowAdvanced(!showAdvanced)}
              sx={{
                color: '#64748b',
                textTransform: 'none',
                fontSize: '0.75rem',
                '&:hover': { color: 'var(--kai-text-soft)', background: 'transparent' },
              }}
            >
              {showAdvanced ? 'Hide' : 'Show'} Advanced Mode
            </Button>
            <Collapse in={showAdvanced}>
              <Paper
                sx={{
                  mt: 1,
                  p: 2,
                  background: 'var(--kai-surface)',
                  border: '1px solid var(--kai-border-strong)',
                  borderRadius: 2,
                }}
              >
                {advancedModeContent}
              </Paper>
            </Collapse>
          </Box>
        )}
      </Box>
    </Box>
  )
}
