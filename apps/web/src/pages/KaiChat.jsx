import { useState, useEffect, useRef } from 'react'
import {
  Box,
  Container,
  Typography,
  Paper,
  TextField,
  IconButton,
  Avatar,
  Chip,
  Grid,
  Card,
  CardContent,
  CardActionArea,
  Alert,
  Button,
  Fade,
  Collapse,
  Switch,
  FormControlLabel,
  CircularProgress,
  Autocomplete,
  Snackbar,
} from '@mui/material'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Send,
  AutoAwesome,
  Person,
  TrendingUp,
  Link as LinkIcon,
  Create,
  Assessment,
  Hub,
  Psychology,
  Bolt,
  Autorenew,
} from '@mui/icons-material'
import axios from 'axios'
import { api, API_BASE_URL, getOrCreateSessionId } from '../config'

// Kai Chat - Master AI Interface & Orchestrator
export default function KaiChat() {
  const defaultCustomerIds = [] // no hardcoded IDs; rely on user/router
  const defaultAccountName = '' // no hardcoded account; rely on user/router

  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [activeSystem, setActiveSystem] = useState(null)
  const [showCapabilities, setShowCapabilities] = useState(true)
  const [availableAccounts, setAvailableAccounts] = useState([])
  const [accountsLoading, setAccountsLoading] = useState(false)
  const [accountsError, setAccountsError] = useState(null)
  const [availableManagers, setAvailableManagers] = useState([])
  const [managersLoading, setManagersLoading] = useState(false)
  const [managersError, setManagersError] = useState(null)
  const [needsIdsPrompt, setNeedsIdsPrompt] = useState(false)
  const [pendingAccounts, setPendingAccounts] = useState([])
  const [activeAccount, setActiveAccount] = useState(null)
  const [accountSearch, setAccountSearch] = useState('')
  const [useAccountForSession, setUseAccountForSession] = useState(true)
  const [lastRouting, setLastRouting] = useState(null)
  const [trendsQueue, setTrendsQueue] = useState(null)
  const [processingState, setProcessingState] = useState(null)
  const [lastPlannerSnapshot, setLastPlannerSnapshot] = useState(null)
  const [pendingMessage, setPendingMessage] = useState('')
  const [sa360Status, setSa360Status] = useState({ connected: false, login_customer_id: null })
  const [sa360StatusLoading, setSa360StatusLoading] = useState(false)
  const [sa360StatusError, setSa360StatusError] = useState(null)
  const [sa360StatusSuccess, setSa360StatusSuccess] = useState(null)
  const [sa360SaveLoading, setSa360SaveLoading] = useState(false)
  const [loginCustomerIdInput, setLoginCustomerIdInput] = useState('')
  const [toast, setToast] = useState({ open: false, message: '', severity: 'success' })
  const attemptedAutoManagerRef = useRef(false)
  const lastDefaultAccountSavedRef = useRef(null)
  const messagesEndRef = useRef(null)
  const lastAccountIdRef = useRef(null)
  const [sessionId] = useState(() => {
    return getOrCreateSessionId()
  })

  const activeAccountStorageKey = sessionId ? `kai_sa360_active_account:${sessionId}` : null
  const plannerSnapshotStorageKey = (customerId) => {
    const cid = normalizeCid(customerId)
    if (!sessionId || !cid) return null
    return `kai_last_planner_snapshot_v1:${sessionId}:${cid}`
  }
  const persistPlannerSnapshot = (snapshot) => {
    try {
      if (typeof window === 'undefined') return
      const cid = snapshot?.plan?.customer_ids?.[0] || activeAccount?.customer_id
      const key = plannerSnapshotStorageKey(cid)
      if (!key) return
      window.sessionStorage.setItem(key, JSON.stringify(snapshot))
    } catch {
      // ignore
    }
  }
  const loadPlannerSnapshot = (customerId) => {
    try {
      if (typeof window === 'undefined') return null
      const key = plannerSnapshotStorageKey(customerId)
      if (!key) return null
      const raw = window.sessionStorage.getItem(key)
      if (!raw) return null
      const parsed = JSON.parse(raw)
      return parsed && typeof parsed === 'object' ? parsed : null
    } catch {
      return null
    }
  }
  const chatPrefillKey = 'kai_chat_prefill_input_v1'

  useEffect(() => {
    // If the user clicked "Use in chat" from SA360 Columns, prefill the chat box once.
    try {
      if (typeof window === 'undefined') return
      const raw = window.localStorage.getItem(chatPrefillKey)
      if (!raw) return
      window.localStorage.removeItem(chatPrefillKey)
      if (String(raw || '').trim()) {
        setInput(String(raw))
      }
    } catch {
      // ignore
    }
  }, [])

  const normalizeAccounts = (payload) => {
    if (Array.isArray(payload)) return payload
    if (payload && Array.isArray(payload.value)) return payload.value
    if (payload && Array.isArray(payload.accounts)) return payload.accounts
    return []
  }

  const normalizeCid = (raw) => {
    try {
      return String(raw || '')
        .replace(/[^\d]/g, '')
        .trim()
    } catch {
      return ''
    }
  }

  const selectableAccounts = (() => {
    try {
      const list = Array.isArray(availableAccounts) ? availableAccounts : []
      return list
        .filter((a) => a && a.customer_id && !a.manager)
        .map((a) => ({ customer_id: String(a.customer_id), name: a.name || '', manager: !!a.manager }))
        .sort((a, b) => (a.name || a.customer_id).localeCompare(b.name || b.customer_id))
    } catch {
      return []
    }
  })()

  const selectableManagers = (() => {
    try {
      const list = Array.isArray(availableManagers) ? availableManagers : []
      return list
        .filter((a) => a && a.customer_id)
        .map((a) => ({ customer_id: String(a.customer_id), name: a.name || '', manager: true }))
        .sort((a, b) => (a.name || a.customer_id).localeCompare(b.name || b.customer_id))
    } catch {
      return []
    }
  })()

  const selectedManager = (() => {
    const cid = normalizeCid(loginCustomerIdInput || sa360Status?.login_customer_id)
    if (!cid) return null
    return selectableManagers.find((m) => normalizeCid(m.customer_id) === cid) || null
  })()

  const metricTerms = [
    'impression',
    'impressions',
    'click',
    'clicks',
    'cpc',
    'ctr',
    'cpm',
    'cvr',
    'conversion rate',
    'roas',
    'cpa',
    'cost',
    'spend',
    'conversion',
    'conversions',
    'conv',
    'performance',
    'budget',
    'auction',
    'impression share',
  ]
  const coreMetricTerms = [
    'cpc',
    'cpa',
    'ctr',
    'cvr',
    'roas',
    'conversion',
    'conversions',
    'click',
    'clicks',
    'impression',
    'impressions',
    'cost',
    'spend',
    'revenue',
    'value',
  ]

  const seasonalityTerms = [
    'trend',
    'trends',
    'seasonality',
    'seasonal',
    'google trends',
    'search interest',
    'interest over time',
    'demand',
    'q1', 'q2', 'q3', 'q4',
    'quarter', 'monthly', 'weekly', 'year over year', 'yoy',
  ]
  const relationalCues = [
    'why',
    'explain',
    'because',
    'due to',
    'compare',
    'versus',
    'vs',
    'driver',
    'drivers',
    'root cause',
    'efficiency',
    'increase',
    'increased',
    'decrease',
    'decreased',
    'spike',
    'drop',
  ]
  const timeframeHints = [
    'last week',
    'last weekend',
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

  const isSeasonalityIntent = (text) => {
    const t = (text || '').toLowerCase()
    return seasonalityTerms.some((k) => t.includes(k))
  }

  const isAccountIntent = (text) => {
    const t = (text || '').toLowerCase()
    const accountKeywords = ['account', 'performance', 'how did', 'how was', 'over the weekend', 'last weekend', 'this weekend']
    return accountKeywords.some((k) => t.includes(k))
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
      'slice',
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
      'implication',
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

  const isSummaryShortcutPrompt = (text) => {
    const t = (text || '').toLowerCase()
    return [
      'summary',
      'summarize',
      'recap',
      'two-sentence',
      'two sentence',
      '2 sentence',
      'next action',
      'next step',
      'next steps',
      'next actions',
    ].some((cue) => t.includes(cue))
  }

  const hasExplicitTrendsCue = (text) => {
    const t = (text || '').toLowerCase()
    return seasonalityTerms.some((k) => t.includes(k))
  }

  const hasTimeframeHint = (text) => {
    const t = (text || '').toLowerCase()
    if (!t) return false
    if (/\b\d{4}-\d{2}-\d{2}\b/.test(t)) return true
    if (/\b(last|past|previous)\s+\d+\s+(day|week|month|quarter|year)s?\b/.test(t)) return true
    if (/\bLAST_(7|14|30|90)_DAYS\b/.test(t.toUpperCase())) return true
    return timeframeHints.some((k) => t.includes(k))
  }

  const isRelationalMetricPrompt = (text) => {
    const t = (text || '').toLowerCase()
    if (!t) return false
    const hasMetric = coreMetricTerms.some((k) => t.includes(k))
    const hasRelational = relationalCues.some((k) => t.includes(k)) || /\bvs\b/.test(t)
    return hasMetric && hasRelational
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
        lower.startsWith('defaulting to')
      )
    })
    return filtered.join('; ')
  }

  const normalizeLooseText = (text) => {
    return String(text || '')
      .toLowerCase()
      .replace(/[_-]+/g, ' ')
      .replace(/\s+/g, ' ')
      .trim()
  }

  const humanizeDateRangeToken = (token) => {
    const t = String(token || '').toUpperCase().trim()
    const m = t.match(/^LAST_(\d+)_DAYS$/)
    if (m && m[1]) return `last ${m[1]} days`
    return token
  }

  const isCandidateDateRangeOk = (candidateText, dateRange) => {
    const candidate = String(candidateText || '')
    if (!dateRange) return true
    if (!candidate) return false

    const raw = String(dateRange || '').trim()
    if (!raw) return true
    if (candidate.includes(raw)) return true

    const lower = candidate.toLowerCase()
    const token = raw.toUpperCase()
    const m = token.match(/^LAST_(\d+)_DAYS$/)
    if (m && m[1]) {
      const n = m[1]
      const phrases = [
        `last ${n} days`,
        `past ${n} days`,
        `${n} days`,
      ]
      if (phrases.some((p) => lower.includes(p))) return true
      if (n === '7' && (lower.includes('last week') || lower.includes('past week'))) return true
      if (n === '14' && (lower.includes('last two weeks') || lower.includes('past two weeks') || lower.includes('last 2 weeks') || lower.includes('past 2 weeks'))) return true
      if (n === '30' && (lower.includes('last month') || lower.includes('past month'))) return true
      if (n === '90' && (lower.includes('last quarter') || lower.includes('past quarter') || lower.includes('last 3 months') || lower.includes('past 3 months'))) return true
      return false
    }

    // Unknown date-range formats should not block a good paraphrase.
    return true
  }

  const cleanPlannerSummarySeed = (seed) => {
    if (!seed || typeof seed !== 'string') return seed
    let text = seed

    // Remove internal notes that leak planner defaults to end users.
    text = text.replace(/\bNo date specified; defaulting to\s+LAST_(7|14|30|90)_DAYS\.\s*/gi, '')

    // Humanize common internal timeframe tokens.
    text = text.replace(/\bLAST_(7|14|30|90)_DAYS\b/g, (_, n) => `last ${n} days`)

    // Prefer readable punctuation over pipe/semicolon "log" formatting.
    text = text.replace(/\s*\|\s*/g, '. ')
    text = text.replace(/\s*;\s*/g, '. ')
    text = text.replace(/\s+/g, ' ').trim()

    return text
  }

  const limitToTwoSentences = (text) => {
    if (!text) return text
    const matches = text.match(/[^.!?]+[.!?]+/g)
    if (!matches || matches.length === 0) return text
    return matches.slice(0, 2).join(' ').trim()
  }

  const buildFollowupReply = (message, snapshot) => {
    if (!snapshot) return null
    const lower = (message || '').toLowerCase()
    const wantsSummary = ['summary', 'summarize', 'recap'].some((k) => lower.includes(k))
    const wantsTwoSentence = lower.includes('two-sentence') || lower.includes('2 sentence')
    const wantsDriver = ['which driver', 'which slice', 'only look at one', 'one slice', 'first slice']
      .some((k) => lower.includes(k))
    if (wantsDriver) {
      return 'Start with campaign for that window; it usually surfaces the biggest variance first. If campaign is flat, check device next, then query or geo.'
    }
    const analysisSummary = snapshot?.analysis?.summary
    const fallbackSummary = snapshot?.enhanced_summary || snapshot?.summary || analysisSummary
    if (wantsSummary && fallbackSummary) {
      if (!wantsTwoSentence) return fallbackSummary
      const summaryBase = fallbackSummary.replace(/Next steps?:.*$/i, '').trim()
      const cleanSummary = summaryBase.replace(/[.?!]+$/, '')
      let nextAction = 'Next action: check campaign first, then device.'
      if (analysisSummary) {
        const match = analysisSummary.match(/Next step:\s*([^.]*)/i)
        if (match && match[1]) {
          nextAction = `Next action: ${match[1].trim()}`
        }
      }
      return `${cleanSummary}. ${nextAction}.`
    }
    if (analysisSummary) return analysisSummary
    return fallbackSummary || null
  }

  const extractCustomerIds = (text) => {
    const t = String(text || '')
    // Accept common copy/paste formats like "790-231-9748" or "790 231 9748".
    // Normalize to digits-only and keep 8-12 digit IDs.
    const candidates = t.match(/\b[\d][\d\s-]{6,20}\b/g) || []
    const ids = candidates
      .map((m) => String(m).replace(/[^\d]/g, ''))
      .filter((m) => m.length >= 8 && m.length <= 12)
    return Array.from(new Set(ids))
  }

  const normalizeAccountTokens = (text) => {
    const raw = String(text || '').toLowerCase()
    const stop = new Set([
      'account',
      'acct',
      'sa360',
      'google',
      'ads',
      // common chat filler (avoid accidental matches like "in" -> "bing")
      'its',
      "it's",
      'it',
      'is',
      'in',
      'on',
      'at',
      'the',
      'a',
      'an',
      'for',
      'to',
      'of',
      'my',
      'our',
      'your',
      'this',
      'that',
      'these',
      'those',
      'please',
    ])
    return raw
      .replace(/[^a-z0-9]+/g, ' ')
      .split(' ')
      .map((t) => t.trim())
      .filter(Boolean)
      .filter((t) => !stop.has(t))
  }

  const normalizeAccountQuery = (text) => {
    return normalizeAccountTokens(text).join(' ')
  }

  const resolveAccountFromText = (text, accounts) => {
    const list = Array.isArray(accounts) ? accounts : []
    if (!text || list.length === 0) return null

    // 1) Explicit ID (including hyphen/space formats)
    const ids = extractCustomerIds(text)
    if (ids.length === 1) {
      const id = String(ids[0])
      const match = list.find((a) => String(a.customer_id) === id)
      return match || { customer_id: id, name: '' }
    }

    // 2) "Name (1234567890)" pattern
    const parsed = parseCandidateAccount(String(text || '').trim())
    if (parsed?.customer_id) {
      const id = String(parsed.customer_id)
      const match = list.find((a) => String(a.customer_id) === id)
      return match || { customer_id: id, name: parsed.name || '' }
    }

    // 3) Name match (token/substring)
     const qTokens = normalizeAccountTokens(text)
     if (!qTokens.length) return null

     const scored = list
       .filter((a) => a && a.customer_id && !a.manager)
       .map((a) => {
         const nameTokens = normalizeAccountTokens(String(a.name || ''))
         const hits = qTokens.reduce((acc, tok) => {
           if (nameTokens.includes(tok)) return acc + 1
           // Allow partial tokens for longer words (e.g. "penn" -> "pennzoil").
           if (tok.length >= 4 && nameTokens.some((nt) => nt.startsWith(tok))) return acc + 1
           return acc
         }, 0)
         return { account: a, hits }
       })
       .filter((x) => x.hits > 0)
       .sort((a, b) => b.hits - a.hits)

    if (!scored.length) return null
    const bestHits = scored[0].hits
    const best = scored.filter((s) => s.hits === bestHits).map((s) => s.account)
    if (best.length === 1) return best[0]
    // Ambiguous: require user to click a chip.
    return null
  }

  const parseCandidateAccount = (candidate) => {
    if (!candidate || typeof candidate !== 'string') return null
    const match = candidate.match(/^(.+?)\s*\((\d{8,12})\)$/)
    if (!match) return null
    return { name: match[1].trim(), customer_id: match[2] }
  }

  const buildAccountSuggestions = (candidates, accounts, limit = 8) => {
    const list = []
    const seen = new Set()
    const ensureAdd = (account) => {
      if (!account?.customer_id) return
      const key = String(account.customer_id)
      if (seen.has(key)) return
      seen.add(key)
      list.push({
        name: account.name || '',
        customer_id: key,
        manager: !!account.manager,
      })
    }
    if (Array.isArray(candidates)) {
      candidates.forEach((candidate) => {
        const parsed = parseCandidateAccount(candidate)
        if (parsed) {
          const match = accounts.find((a) => String(a.customer_id) === String(parsed.customer_id))
          ensureAdd({ ...parsed, manager: match?.manager, name: match?.name || parsed.name })
        }
      })
    }
    if (list.length < limit && Array.isArray(accounts)) {
      accounts
        .filter((a) => !a.manager)
        .forEach((account) => {
          if (list.length < limit) ensureAdd(account)
        })
    }
    return list
  }

  const parseRoutingMeta = (notes) => {
    if (!notes || typeof notes !== 'string') return { model: null, verify: null }
    const modelMatch = notes.match(/model=(local|azure)/i)
    const verifyMatch = notes.match(/verify=(local|azure)/i)
    return {
      model: modelMatch ? modelMatch[1].toLowerCase() : null,
      verify: verifyMatch ? verifyMatch[1].toLowerCase() : null,
    }
  }

  const truncateText = (text, max = 160) => {
    if (!text || typeof text !== 'string') return ''
    if (text.length <= max) return text
    return `${text.slice(0, max - 3)}...`
  }

  const summarizePlanError = (errorText) => {
    if (!errorText || typeof errorText !== 'string') return null
    const lower = errorText.toLowerCase()
    if (lower.includes('user_permission_denied') || lower.includes('permission denied')) {
      return 'Access denied for this customer. Check SA360 permissions and the login-customer-id header.'
    }
    if (lower.includes('sa360 search failed')) {
      return 'SA360 search failed. Verify the customer ID and SA360 credentials.'
    }
    return truncateText(errorText.replace(/\s+/g, ' '), 180)
  }

  const buildPlannerSummary = async (planData, userMessage, detectedSystem) => {
    const compactPlan = {
      executed: planData?.executed,
      notes: planData?.notes,
      error: planData?.error,
      account: planData?.plan?.account_name,
      date_range: planData?.plan?.date_range,
      file: planData?.result?.file_name,
      analysis_note: planData?.analysis?.note || planData?.analysis?.summary,
      summary: planData?.enhanced_summary || planData?.summary,
    }
    const cleanedNotes = stripInternalNotes(compactPlan.notes)
    if (compactPlan.error) {
      const summaryError = summarizePlanError(compactPlan.error)
      return `I couldn't finish the analysis: ${summaryError}`
    }
    const seedParts = []
    if (compactPlan.summary) seedParts.push(compactPlan.summary)
    if (compactPlan.analysis_note) seedParts.push(compactPlan.analysis_note)
    if (compactPlan.file) seedParts.push(`Report: ${compactPlan.file}`)
    if (cleanedNotes) seedParts.push(cleanedNotes)
    const summarySeed = seedParts.filter(Boolean).join(' ').trim()
    // IMPORTANT:
    // Send the *real* planner output payload to the backend follow-up prompt so it can:
    // - detect performance context (plan.customer_ids, result.deltas, etc.)
    // - produce advisor-grade responses for action asks (Option A/Option B/...)
    // The backend already applies numeric grounding guardrails. We can still add a stable summary seed.
    const plannerToolOutput = (() => {
      if (planData && typeof planData === 'object' && !Array.isArray(planData)) {
        const out = { ...planData }
        if (summarySeed) out.summary_seed = summarySeed
        return out
      }
      return planData
    })()

    const question = String(userMessage || '').trim() || 'Summarize the planner output.'
    // Ask the LLM to respond to the user's question using only the planner tool output.
    try {
      const llmResp = await axios.post(api.chat.send, {
        message: question,
        ai_enabled: true,
        session_id: sessionId || undefined,
        context: { tool: detectedSystem || 'performance', tool_output: plannerToolOutput, prompt_kind: 'planner_summary' },
      })
      if (llmResp.data?.reply) {
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

        const candidateRaw = String(llmResp.data.reply || '').trim()
        const candidate = normalizeInternalTokens(candidateRaw)
        const lowered = candidate.toLowerCase()
        const banned = ['current account is not set', 'account is not set']
        const hasBanned = banned.some((phrase) => lowered.includes(phrase))

        // Broad beta UX: accept a useful paraphrase even if it doesn't repeat the full SA360 account name.
        // We can prepend a stable header (account + timeframe) instead of rejecting and falling back to a metric dump.
        // Do not reject solely because the *raw* LLM text included internal tokens; we sanitize them above.
        // Reject only if the *sanitized* candidate still leaks internal planner tokens.
        const leaksAfterSanitization = /\bLAST_\d+_DAYS\b/.test(candidate) || /\bLAST_WEEK\b/.test(candidate) || /\bLAST_MONTH\b/.test(candidate) || /\bTHIS_MONTH\b/.test(candidate)
        if (candidate && !hasBanned && !leaksAfterSanitization) {
          const prefixBits = []
          if (compactPlan.account && !normalizeLooseText(candidate).includes(normalizeLooseText(compactPlan.account))) {
            prefixBits.push(compactPlan.account)
          }
          if (compactPlan.date_range && !isCandidateDateRangeOk(candidate, compactPlan.date_range)) {
            prefixBits.push(humanizeDateRangeToken(compactPlan.date_range))
          }
          const prefix = prefixBits.length ? `${prefixBits.join(' • ')}\n` : ''
          return `${prefix}${candidate}`.trim()
        }
      }
    } catch (err) {
      // fall through to fallback template
    }

    if (summarySeed) return cleanPlannerSummarySeed(summarySeed)

    // Fallback templated response
    const parts = []
    if (compactPlan.executed) {
      parts.push(`I ran the account analysis${compactPlan.account ? ` for ${compactPlan.account}` : ''}.`)
      if (compactPlan.date_range) parts.push(`Timeframe: ${humanizeDateRangeToken(compactPlan.date_range)}.`)
      if (cleanedNotes) parts.push(cleanedNotes)
      if (compactPlan.analysis_note) parts.push(compactPlan.analysis_note)
      if (compactPlan.file) parts.push(`Report ready: ${compactPlan.file}`)
    } else if (compactPlan.error) {
      parts.push(`I couldn't finish the analysis: ${compactPlan.error}`)
    } else {
      parts.push('Planner completed.')
      if (cleanedNotes) parts.push(cleanedNotes)
    }
    return parts.filter(Boolean).join(' ')
  }

  const buildTrendsSummary = (trendsData) => {
    if (!trendsData) return null
    const allocs = trendsData?.allocations || []
    const lines = allocs.slice(0, 3).map((a) => {
      const pct = a.weight_pct ? `${a.weight_pct.toFixed(1)}%` : ''
      const mult = a.trend_multiplier ? `mult x${a.trend_multiplier.toFixed(2)}` : ''
      return `${a.theme}: ${pct} ${mult}`.trim()
    })
    const seasonalitySummary = trendsData?.seasonality_summary
    const parts = []
    if (seasonalitySummary) parts.push(`Seasonality: ${seasonalitySummary}`)
    if (lines.length) parts.push(`Allocations: ${lines.join(' | ')}`)
    if (trendsData?.trends_status && trendsData.trends_status !== 'trends_ok') {
      parts.push(`Trends status: ${trendsData.trends_status}`)
    }
    return parts.length ? parts.join('\n') : null
  }

  const pollPlanJob = async (jobId, userMessage, systemLabel, summaryText) => {
    if (!jobId) return
    const attempts = 20
    const intervalMs = 6000
    for (let i = 0; i < attempts; i += 1) {
      try {
        const statusResp = await axios.get(api.jobs.status(jobId))
        const job = statusResp.data?.job
        const status = job?.status
        if (status === 'succeeded') {
          const resultResp = await axios.get(api.jobs.result(jobId))
          const planData = resultResp.data?.result || resultResp.data
          const plannerSummary = await buildPlannerSummary(planData || {}, userMessage, systemLabel)
          const plannerSnapshot = {
            plan: planData?.plan,
            result: planData?.result,
            analysis: planData?.analysis,
            summary: planData?.summary,
            enhanced_summary: planData?.enhanced_summary,
            notes: planData?.notes,
          }
          setLastPlannerSnapshot(plannerSnapshot)
          persistPlannerSnapshot(plannerSnapshot)
          const combined = [summaryText, plannerSummary].filter(Boolean).join('\n') || 'Analysis completed.'
          setMessages((prev) => [...prev, {
            role: 'assistant',
            content: combined,
            system: systemLabel || 'performance',
          }])
          return
        }
        if (status === 'failed') {
          const errMsg = job?.error ? `Planner job failed: ${job.error}` : 'Planner job failed.'
          setMessages((prev) => [...prev, { role: 'assistant', content: errMsg, system: systemLabel || 'performance' }])
          return
        }
      } catch (err) {
        const status = err?.response?.status
        if (status && status !== 404) {
          return
        }
      }
      await new Promise((resolve) => setTimeout(resolve, intervalMs))
    }
  }

  const pollTrendsJob = async (jobId, systemLabel) => {
    if (!jobId) return
    setTrendsQueue({ jobId, systemLabel })
    const attempts = 10
    const intervalMs = 6000
    for (let i = 0; i < attempts; i += 1) {
      try {
        const resultResp = await axios.get(api.jobs.result(jobId))
        const summary = buildTrendsSummary(resultResp.data)
        if (summary) {
          setTrendsQueue((prev) => (prev?.jobId === jobId ? null : prev))
          setMessages((prev) => [...prev, {
            role: 'assistant',
            content: `Update: ${summary}`,
            system: systemLabel || 'general',
          }])
          return
        }
      } catch (err) {
        const status = err?.response?.status
        if (status && status !== 404) {
          setTrendsQueue((prev) => (prev?.jobId === jobId ? null : prev))
          return
        }
      }
      await new Promise((resolve) => setTimeout(resolve, intervalMs))
    }
    setTrendsQueue((prev) => (prev?.jobId === jobId ? null : prev))
  }

  // System capabilities that Kai can orchestrate
  // Colors follow brand temperature spectrum: cool (blue) -> warm (orange)
  const systems = [
    {
      id: 'audit',
      name: 'PPC Audit',
      icon: Assessment,
      description: 'Full account audit with 100+ checkpoints',
      color: '#3b82f6',  // Blue - analytical, cool
      examples: ['Run a Brand audit', 'Audit my NonBrand campaigns', 'Generate audit with demo data'],
    },
    {
      id: 'pmax',
      name: 'PMax Analysis',
      icon: TrendingUp,
      description: 'Channel breakout & placement analysis',
      color: '#f472b6',  // Pink - performance
      examples: ['Analyze my PMax placements', 'Where is PMax spending?', 'Show channel allocation'],
    },
    {
      id: 'serp',
      name: 'SERP Monitor',
      icon: LinkIcon,
      description: 'URL health checks & landing page status',
      color: '#a78bfa',  // Purple - technical
      examples: ['Check URL health', 'Monitor my landing pages', 'Are my URLs working?'],
    },
    {
      id: 'competitor',
      name: 'Competitor Intel',
      icon: TrendingUp,
      description: 'Analyze competitor investment signals',
      color: '#ef4444',  // Red - competitive/urgent
      examples: ['Is my competitor ramping up?', 'Analyze HomeDepot investment', 'Who is spending more?'],
    },
    {
      id: 'creative',
      name: 'Creative Studio',
      icon: Create,
      description: 'AI-generated headlines & ad copy',
      color: '#f59e0b',  // Amber - creative energy, warm
      examples: ['Write headlines for my SaaS', 'Generate RSA copy', 'Create ad variations'],
    },
  ]
  const systemDisplay = [
    {
      id: 'performance',
      name: 'Performance',
      icon: Autorenew,
      description: 'Account-level performance metrics and comparisons',
      color: '#22c55e',
      examples: ['How did Mobility perform last week?', '30-day performance for Recharge', 'Why did CPC spike?'],
    },
    ...systems,
  ]
  const getSystemMeta = (systemId) => systemDisplay.find((s) => s.id === systemId)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  const isManagerId = (cid) => {
    if (!cid) return false
    return availableAccounts.some((a) => String(a.customer_id) === String(cid) && a.manager)
  }

  const fetchSa360Status = async () => {
    if (!sessionId) return
    setSa360StatusLoading(true)
    setSa360StatusError(null)
    try {
      const resp = await axios.get(`${API_BASE_URL}/api/sa360/oauth/status`, {
        params: { session_id: sessionId },
      })
      const next = resp.data || { connected: false, login_customer_id: null }
      setSa360Status(next)
      if (next.login_customer_id && !loginCustomerIdInput) {
        setLoginCustomerIdInput(String(next.login_customer_id))
      }
      // If the server has a stored default account (cross-device), hydrate it once.
      if (!activeAccount?.customer_id && next.default_customer_id) {
        setActiveAccount({
          customer_id: String(next.default_customer_id),
          name: next.default_account_name ? String(next.default_account_name) : '',
        })
        setUseAccountForSession(true)
      }
    } catch (err) {
      setSa360StatusError(err?.response?.data?.detail || err?.message || 'SA360 status check failed')
    } finally {
      setSa360StatusLoading(false)
    }
  }

  const fetchSa360Accounts = async () => {
    if (!sessionId) return
    if (!sa360Status.connected) return
    setAccountsLoading(true)
    setAccountsError(null)
    try {
      const acctResp = await axios.get(`${API_BASE_URL}/api/sa360/accounts`, {
        params: { session_id: sessionId },
      })
      const list = normalizeAccounts(acctResp.data)
      setAvailableAccounts(list)
      return list
    } catch (err) {
      setAccountsError(err?.response?.data?.detail || err?.message || 'Failed to load SA360 accounts')
      return []
    } finally {
      setAccountsLoading(false)
    }
  }

  const fetchSa360Managers = async () => {
    if (!sessionId) return []
    if (!sa360Status.connected) return []
    setManagersLoading(true)
    setManagersError(null)
    try {
      const resp = await axios.get(`${API_BASE_URL}/api/sa360/managers`, {
        params: { session_id: sessionId },
      })
      const list = normalizeAccounts(resp.data)
        .filter((m) => m && m.customer_id)
        .map((m) => ({ customer_id: String(m.customer_id), name: m.name || '', manager: true }))
      setAvailableManagers(list)
      return list
    } catch (err) {
      setManagersError(err?.response?.data?.detail || err?.message || 'Failed to discover MCCs for this user')
      return []
    } finally {
      setManagersLoading(false)
    }
  }

  const startSa360Connect = () => {
    if (!sessionId) return
    setSa360StatusError(null)
    setSa360StatusSuccess('Opening Google sign-in…')
    // Open popup synchronously (prevents popup blockers), then navigate once we fetch the OAuth URL.
    // NOTE: we intentionally fetch the URL via XHR so Authorization headers (Entra SSO) can scope tokens per-user.
    const w = window.open('about:blank', 'kai_sa360_oauth', 'popup,width=520,height=720,noopener,noreferrer')
    if (!w) {
      setSa360StatusSuccess(null)
      setSa360StatusError('Popup blocked. Please allow popups for this site and try again.')
      return
    }

    axios
      .get(`${API_BASE_URL}/api/sa360/oauth/start-url`, {
        params: { session_id: sessionId },
      })
      .then((resp) => {
        const nextUrl = resp?.data?.url ? String(resp.data.url) : ''
        if (!nextUrl) {
          throw new Error('No OAuth URL returned.')
        }
        try {
          w.location.href = nextUrl
        } catch {
          // best-effort fallback; popup may have been blocked after open
          window.location.assign(nextUrl)
        }
        // Fallback polling in case postMessage is blocked by the browser.
        setTimeout(fetchSa360Status, 2500)
      })
      .catch((err) => {
        try {
          w.close()
        } catch {
          // ignore
        }
        setSa360StatusSuccess(null)
        setSa360StatusError(
          err?.response?.data?.detail || err?.message || 'Failed to start SA360 OAuth. Please retry.'
        )
      })
  }

  const saveLoginCustomerId = async (overrideLoginCustomerId) => {
    const candidate = normalizeCid(overrideLoginCustomerId ?? loginCustomerIdInput)
    if (!sessionId || !candidate) {
      setSa360StatusError('Enter a Manager ID to save.')
      setToast({ open: true, message: 'Enter a Manager (MCC) ID to save.', severity: 'error' })
      return
    }
    setSa360SaveLoading(true)
    setSa360StatusError(null)
    setSa360StatusSuccess(null)
    try {
      setLoginCustomerIdInput(candidate)
      await axios.post(`${API_BASE_URL}/api/sa360/login-customer`, {
        session_id: sessionId,
        login_customer_id: candidate,
      })
      await fetchSa360Status()
      // Account list depends on MCC; refresh after save for UX.
      const refreshedAccounts = await fetchSa360Accounts()
      const accountCount = Array.isArray(refreshedAccounts) ? refreshedAccounts.length : null
      setSa360StatusSuccess(
        accountCount === null ? 'Manager ID saved' : `Manager ID saved (${accountCount} accounts loaded)`
      )
      setToast({
        open: true,
        message:
          accountCount === null
            ? 'Manager (MCC) saved. Accounts refreshed.'
            : `Manager (MCC) saved. ${accountCount} accounts loaded.`,
        severity: 'success',
      })
      setTimeout(() => setSa360StatusSuccess(null), 8000)
    } catch (err) {
      const msg = err?.response?.data?.detail || err?.message || 'Failed to save Manager ID'
      setSa360StatusError(msg)
      setToast({ open: true, message: msg, severity: 'error' })
    } finally {
      setSa360SaveLoading(false)
    }
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  useEffect(() => {
    fetchSa360Status()
  }, [sessionId])

  useEffect(() => {
    // Preload accounts once SA360 is connected and MCC is available.
    if (!sessionId) return
    if (!sa360Status.connected) return
    const hasMcc = !!String(sa360Status?.login_customer_id || '').trim()
    if (!hasMcc) return
    if (availableAccounts.length) return
    fetchSa360Accounts()
  }, [sessionId, sa360Status.connected, sa360Status?.login_customer_id])

  useEffect(() => {
    // Broad beta onboarding: if SA360 is connected but MCC isn't set, try to discover manager accounts.
    if (!sessionId) return
    if (!sa360Status.connected) return
    const hasMcc = !!String(sa360Status?.login_customer_id || '').trim()
    if (hasMcc) return
    if (attemptedAutoManagerRef.current) return
    attemptedAutoManagerRef.current = true

    fetchSa360Managers().then((managers) => {
      if (Array.isArray(managers) && managers.length === 1 && managers[0]?.customer_id) {
        // Safe auto-select if the user only has one manager account.
        saveLoginCustomerId(managers[0].customer_id)
      }
    })
  }, [sessionId, sa360Status.connected, sa360Status?.login_customer_id])

  useEffect(() => {
    // Persist and restore the active account for broad beta UX.
    if (!sessionId || !activeAccountStorageKey) return
    if (activeAccount) return
    try {
      const raw = window.localStorage.getItem(activeAccountStorageKey)
      if (!raw) return
      const parsed = JSON.parse(raw)
      if (parsed && parsed.customer_id) {
        setActiveAccount({ customer_id: String(parsed.customer_id), name: parsed.name || '' })
      }
    } catch {
      // ignore
    }
  }, [sessionId])

  useEffect(() => {
    if (!sessionId || !activeAccountStorageKey) return
    try {
      if (activeAccount?.customer_id) {
        window.localStorage.setItem(activeAccountStorageKey, JSON.stringify({ customer_id: activeAccount.customer_id, name: activeAccount.name || '' }))
      } else {
        window.localStorage.removeItem(activeAccountStorageKey)
      }
    } catch {
      // ignore
    }
  }, [activeAccount?.customer_id, activeAccount?.name, sessionId])

  useEffect(() => {
    // Persist the default account server-side (keeps the selection stable across devices and future sessions).
    if (!sessionId) return
    if (!useAccountForSession) return
    if (!sa360Status.connected) return
    if (!String(sa360Status?.login_customer_id || '').trim()) return

    const cid = activeAccount?.customer_id ? String(activeAccount.customer_id) : ''
    // Never clear the server-side default implicitly. Only persist when we have a real selection.
    // (Clearing defaults on page load causes broad-beta UX regressions where the router/planner
    // repeatedly asks users to paste IDs.)
    if (!cid) return
    if (lastDefaultAccountSavedRef.current === cid) return
    lastDefaultAccountSavedRef.current = cid

    const payload = {
      session_id: sessionId,
      customer_id: cid || null,
      account_name: activeAccount?.name || null,
    }

    axios
      .post(`${API_BASE_URL}/api/sa360/default-account`, payload)
      .then(() => {
        if (cid) {
          setSa360StatusSuccess('Account saved')
          setTimeout(() => setSa360StatusSuccess(null), 3500)
        }
      })
      .catch(() => {
        // non-blocking; keep UX responsive
      })
  }, [activeAccount?.customer_id, activeAccount?.name, sessionId, sa360Status.connected, sa360Status?.login_customer_id, useAccountForSession])

  useEffect(() => {
    if (!sessionId) return
    const expectedOrigin = (() => {
      try {
        // API_BASE_URL may be empty in dev; fall back to current origin.
        const base = API_BASE_URL || window.location.origin
        return new URL(base, window.location.origin).origin
      } catch {
        return null
      }
    })()

    const handler = (event) => {
      const data = event?.data
      if (!data || typeof data !== 'object') return
      if (expectedOrigin && event.origin !== expectedOrigin) return
      if (data.type !== 'KAI_SA360_OAUTH') return

      if (data.status === 'connected') {
        setSa360StatusSuccess('SA360 connected')
        setTimeout(() => setSa360StatusSuccess(null), 5000)
        fetchSa360Status()
      } else {
        setSa360StatusSuccess(null)
        setSa360StatusError('SA360 connection failed. Please retry Connect SA360.')
      }
    }

    window.addEventListener('message', handler)
    return () => window.removeEventListener('message', handler)
  }, [sessionId])

  useEffect(() => {
    const currentId = activeAccount?.customer_id || null
    if (lastAccountIdRef.current && currentId !== lastAccountIdRef.current) {
      setLastPlannerSnapshot(null)
      setLastRouting(null)
    }
    if (!currentId && lastAccountIdRef.current) {
      setLastPlannerSnapshot(null)
      setLastRouting(null)
    }
    lastAccountIdRef.current = currentId
  }, [activeAccount?.customer_id])

  // Restore planner context after refresh/navigation so follow-up prompts stay conversational.
  // Keyed by (session_id, customer_id) to avoid cross-account bleed.
  useEffect(() => {
    const cid = activeAccount?.customer_id
    if (!cid) return
    const restored = loadPlannerSnapshot(cid)
    if (restored) {
      setLastPlannerSnapshot(restored)
    }
  }, [activeAccount?.customer_id, sessionId])

  // Detect intent and route to appropriate system
  const detectIntent = (message) => {
    const lower = message.toLowerCase()

    // Audit intents
    if (lower.includes('audit') || lower.includes('checkup') || lower.includes('health check')) {
      return 'audit'
    }

    // PMax intents
    if (lower.includes('pmax') || lower.includes('performance max') || lower.includes('channel') ||
        lower.includes('placement') || lower.includes('allocation')) {
      return 'pmax'
    }

    // Competitor intents (check BEFORE general SERP to prioritize competitor analysis)
    if (lower.includes('competitor') || lower.includes('ramping') || lower.includes('investment') ||
        lower.includes('impression share') || lower.includes('outranking') || lower.includes('auction') ||
        lower.match(/is\s+\w+\s+(increasing|decreasing|spending|investing)/) ||
        lower.includes('market pressure') || lower.includes('competitive')) {
      return 'competitor'
    }

    // SERP intents (URL health)
    if (lower.includes('serp') || lower.includes('url') || lower.includes('landing page') ||
        lower.includes('health check') || lower.includes('monitor') || lower.includes('soft 404') ||
        lower.includes('broken page')) {
      return 'serp'
    }

    // Creative intents
    if (lower.includes('headline') || lower.includes('ad copy') || lower.includes('creative') ||
        lower.includes('write') || lower.includes('generate copy') || lower.includes('rsa')) {
      return 'creative'
    }

    return null // General chat
  }

  const sendMessage = async (overrideMessage) => {
    const rawMessage = overrideMessage !== undefined ? String(overrideMessage) : input
    const trimmed = rawMessage.trim()
    if (!trimmed || loading) return
    // If we are currently prompting for an account, interpret this submission as an account selection attempt
    // (by name or by ID) instead of sending it to the LLM and risking a mis-route.
    if (needsIdsPrompt) {
      // IMPORTANT: allow users to type *any* account name, not just the limited suggestion chips.
      // Otherwise phrases like "loyalty account" can fail to match and end up being routed as "loyalty rewards".
      const pool = selectableAccounts.length
        ? selectableAccounts
        : (filteredAccountSuggestions.length ? filteredAccountSuggestions : visibleAccountSuggestions)
      const picked = resolveAccountFromText(trimmed, pool)
      if (picked?.customer_id) {
        handleAccountPick(picked)
        return
      }
      // Broad beta UX: if the user is already working in an active account and they type a name fragment
      // like "loyalty", prefer that active account rather than returning an ambiguous null that
      // can misroute into unrelated "loyalty rewards" conversations.
      try {
        const qTokens = normalizeAccountTokens(trimmed)
        const nameTokens = normalizeAccountTokens(String(activeAccount?.name || ''))
        const tokenHit = qTokens.some((t) => t.length >= 4 && nameTokens.includes(t))
        if (tokenHit && activeAccount?.customer_id) {
          handleAccountPick(activeAccount)
          return
        }
      } catch {
        // ignore
      }
      // Populate the account search box to help users who type the name into the main input.
      setAccountSearch(trimmed)
      return
    }

    const idOnlyDigits = trimmed.match(/^[\d\s-]{8,20}$/) ? trimmed.replace(/[^\d]/g, '') : null
    const idOnlyMatch = idOnlyDigits && idOnlyDigits.length >= 8 && idOnlyDigits.length <= 12
    const consumedPendingMessage = Boolean(pendingMessage && idOnlyMatch)
    if (consumedPendingMessage) {
      const match = availableAccounts.find((a) => normalizeCid(a?.customer_id) === idOnlyDigits)
      setActiveAccount({ customer_id: String(idOnlyDigits), name: match?.name || '' })
      setUseAccountForSession(true)
    }
    const userMessage = consumedPendingMessage ? pendingMessage : trimmed
    const lowerMsg = userMessage.toLowerCase()
    setInput('')
    setMessages((prev) => [...prev, { role: 'user', content: userMessage }])
    setLoading(true)
    setError(null)
    setShowCapabilities(false)
    setNeedsIdsPrompt(false)
    setPendingAccounts([])
    setAccountSearch('')
    setLastRouting(null)
    setPendingMessage('')

    // Inline health check handling (no terminal needed)
    if (lowerMsg.includes('health check') || lowerMsg.includes('healthcheck')) {
      try {
        const resp = await axios.get(`${API_BASE_URL}/api/diagnostics/health`)
        const data = resp.data || {}
        const status = data.status || 'unknown'
        const acctCount = data.accounts?.count
        const qa = Array.isArray(data.qa) ? data.qa : []
        const qaSummary = qa.length
          ? qa.map((q) => {
              if (q.error) return `Account ${q.customer_id}: ERROR ${q.error}`
              return `Account ${q.customer_id}: matches=${q.matches}`
            }).join(' | ')
          : 'No QA sample run.'
        const mgr = data.manager_guard?.guard_active ? 'Manager guard active' : 'Manager guard unknown'
        const queueEnabled = data.queue?.enabled
        const queueDepth = data.queue?.depth
        const queueText = queueEnabled
          ? `Queue depth: ${queueDepth ?? 'n/a'}.`
          : 'Queue disabled.'
        const msg = `System health: ${status}. Accounts discovered: ${acctCount ?? 'n/a'}. ${mgr}. ${queueText} QA samples: ${qaSummary}.`
        setMessages((prev) => [...prev, { role: 'assistant', content: msg, system: 'diagnostics' }])
      } catch (err) {
        setMessages((prev) => [...prev, { role: 'assistant', content: `Health check failed: ${err?.message || err}`, system: 'diagnostics' }])
      }
      setLoading(false)
      return
    }

    const preRouteExplicitIds = Array.from(
      new Set([
        ...extractCustomerIds(userMessage),
        ...(consumedPendingMessage && idOnlyDigits ? [idOnlyDigits] : []),
      ].filter(Boolean))
    )
    const selectedAccountIds =
      activeAccount?.customer_id
        ? [normalizeCid(activeAccount.customer_id)]
        : []
    const preRouteIds = Array.from(
      new Set((preRouteExplicitIds.length ? preRouteExplicitIds : selectedAccountIds).filter(Boolean))
    )

    // Ask backend router (LLM) to classify intent/tool
    let routing = null
    try {
      const routeResp = await axios.post(api.chat.route, {
        message: userMessage,
        customer_ids: preRouteIds,
        account_name: activeAccount?.name || undefined,
        session_id: sessionId || undefined,
      })
      routing = routeResp.data
      setLastRouting(routing)

      // If routing needs IDs, try to fetch SA360 account list once
      if (routing?.needs_ids && availableAccounts.length === 0) {
        try {
          const acctResp = await axios.get(`${API_BASE_URL}/api/sa360/accounts`, {
            params: { session_id: sessionId || undefined },
          })
          const list = normalizeAccounts(acctResp.data)
          if (list.length) {
            setAvailableAccounts(list)
          }
        } catch (acctErr) {
          // ignore; fallback prompt will still show
        }
      }
    } catch (routeErr) {
      routing = null // fallback to heuristics
    }

    const detectedSystem = routing?.tool || detectIntent(userMessage)
    const routedSystem = routing?.tool || (routing?.intent === 'performance' ? 'performance' : null) || detectedSystem

    const hasExplicitIds = extractCustomerIds(userMessage).length > 0
    const hasFollowupContext = !!(
      lastPlannerSnapshot?.analysis?.summary ||
      lastPlannerSnapshot?.summary ||
      lastPlannerSnapshot?.enhanced_summary
    )
    const timeframeHint = hasTimeframeHint(userMessage)
    const explicitTrendsCue = hasExplicitTrendsCue(userMessage)
    const relationalMetricPrompt = isRelationalMetricPrompt(userMessage)
    let shouldUsePlanner =
      routing?.run_planner !== undefined && routing?.run_planner !== null
        ? routing.run_planner
        : (isMetricIntent(userMessage) || isAccountIntent(userMessage) || detectedSystem === 'audit')
    let shouldUseTrends =
      routing?.run_trends !== undefined && routing?.run_trends !== null
        ? routing.run_trends
        : isSeasonalityIntent(userMessage)

    if (relationalMetricPrompt && !explicitTrendsCue) {
      shouldUseTrends = false
    }

    const canReusePlanner = (
      hasFollowupContext &&
      !hasExplicitIds &&
      !timeframeHint &&
      (isFollowupPrompt(userMessage) || relationalMetricPrompt)
    )
    if (canReusePlanner) {
      shouldUsePlanner = false
    }

    const usePlannerContext = !shouldUsePlanner && !shouldUseTrends && canReusePlanner
    const effectiveSystem = usePlannerContext ? 'performance' : routedSystem
    if (effectiveSystem) {
      setActiveSystem(effectiveSystem)
    }

    const explicitIds = [
      ...(routing?.customer_ids || []),
      ...extractCustomerIds(userMessage),
    ]
      .map((cid) => normalizeCid(cid))
      .filter(Boolean)
    const mergedIds = Array.from(
      new Set((explicitIds.length ? explicitIds : selectedAccountIds).filter(Boolean))
    )
    const followupContext = usePlannerContext
      ? {
          tool: 'performance',
          tool_output: lastPlannerSnapshot,
          customer_ids: lastPlannerSnapshot?.plan?.customer_ids || mergedIds,
          date_range: lastPlannerSnapshot?.plan?.date_range || null,
        }
      : null

    // If we have IDs but haven't loaded accounts, fetch once to enable manager detection.
    let accountsForCheck = availableAccounts
    if (mergedIds.length > 0 && availableAccounts.length === 0) {
      try {
        const acctResp = await axios.get(`${API_BASE_URL}/api/sa360/accounts`, {
          params: { session_id: sessionId || undefined },
        })
          const list = normalizeAccounts(acctResp.data)
          if (list.length) {
            accountsForCheck = list
            setAvailableAccounts(list)
          }
      } catch (acctErr) {
        // continue without blocking
      }
    }

    // Block manager IDs early and prompt for a child account
    const managerSelected = mergedIds.find((id) =>
      (accountsForCheck.length ? accountsForCheck : availableAccounts).some(
        (a) => String(a.customer_id) === String(id) && a.manager
      )
    )
    if (managerSelected) {
      const suggestions = buildAccountSuggestions([], accountsForCheck.length ? accountsForCheck : availableAccounts)
      setPendingAccounts(suggestions)
      setAccountSearch('')
      setPendingMessage(userMessage)
      setMessages((prev) => [...prev, {
        role: 'assistant',
        content: `That ID (${managerSelected}) is a manager account. Pick a child account to continue.`,
        system: routedSystem || routing?.intent || null,
      }])
      setNeedsIdsPrompt(true)
      setLoading(false)
      return
    }

    if (!usePlannerContext) {
      // Only performance/audit planners should prompt for SA360 account selection.
      // Tools like PMax/SERP/Competitor/Creative can operate without a SA360 customer id.
      const needsAccountClarification =
        !['pmax', 'serp', 'competitor', 'creative'].includes(routedSystem) &&
        (routing?.needs_ids || (routing?.needs_clarification && routing?.run_planner))
      if (needsAccountClarification && mergedIds.length === 0) {
        // If SA360 isn't connected, an account picker is a dead-end. Prompt to connect first.
        if (!sa360Status.connected) {
          setMessages((prev) => [...prev, {
            role: 'assistant',
            content: "SA360 isn't connected for this session. Click Connect SA360 at the top of Kai, then retry.",
            system: routedSystem || routing?.intent || null,
          }])
          setLoading(false)
          return
        }
        // If SA360 is connected but MCC isn't saved, we can't discover accounts yet.
        if (sa360Status.connected && !String(sa360Status?.login_customer_id || '').trim()) {
          setMessages((prev) => [...prev, {
            role: 'assistant',
            content: "You're connected to SA360, but I can't list accounts yet. Enter your Manager ID (MCC) above and click Save MCC, then retry.",
            system: routedSystem || routing?.intent || null,
          }])
          setLoading(false)
          return
        }
        const suggestions = buildAccountSuggestions(routing?.candidates || [], accountsForCheck.length ? accountsForCheck : availableAccounts)
        setPendingAccounts(suggestions)
        setAccountSearch('')
        setPendingMessage(userMessage)
        const prompt = routing?.clarification || 'Which account should I use?'
        const candidateText = suggestions.length
          ? ` Suggested accounts: ${suggestions.map((a) => `${a.name || 'Account'} (${a.customer_id})`).join(' | ')}.`
          : ''
        setMessages((prev) => [...prev, {
          role: 'assistant',
          content: `${prompt} Pick an account below (by name) or paste an ID.${candidateText}`,
          system: routedSystem || routing?.intent || null,
        }])
        setNeedsIdsPrompt(true)
        setLoading(false)
        return
      }
      if (!routing?.run_planner && routing?.needs_clarification && mergedIds.length === 0) {
        const prompt = routing?.clarification || 'Can you clarify what you need?'
        setMessages((prev) => [...prev, {
          role: 'assistant',
          content: prompt,
          system: routedSystem || routing?.intent || null,
        }])
        setLoading(false)
        return
      }
    }

    if (routing?.customer_ids && routing.customer_ids.length > 0) {
      const resolvedId = String(routing.customer_ids[0])
      let resolvedName = ''
      if (availableAccounts.length === 0) {
        try {
          const acctResp = await axios.get(`${API_BASE_URL}/api/sa360/accounts`, {
            params: { session_id: sessionId || undefined },
          })
          const list = normalizeAccounts(acctResp.data)
          if (list.length) {
            setAvailableAccounts(list)
            const match = list.find((a) => String(a.customer_id) === resolvedId)
            resolvedName = match?.name || ''
          }
        } catch (acctErr) {
          // ignore
        }
      } else {
        const match = availableAccounts.find((a) => String(a.customer_id) === resolvedId)
        resolvedName = match?.name || ''
      }
      setActiveAccount({ customer_id: resolvedId, name: resolvedName })
    }
    if (shouldUsePlanner || shouldUseTrends) {
      const routingSystem =
        routing?.intent === 'performance'
          ? 'performance'
          : routedSystem || 'general'
      const routingName = shouldUseTrends
        ? (shouldUsePlanner ? 'seasonality + performance check' : 'seasonality analysis')
        : routing?.intent === 'performance'
          ? 'account performance check'
          : systems.find((s) => s.id === routingSystem)?.name || 'analysis'
      setProcessingState({ label: routingName, eta: '30-60s' })
    }

    try {
      let handled = false
      const responseSystem = usePlannerContext
        ? 'performance'
        : (routing?.intent === 'performance' ? 'performance' : (routedSystem || 'general'))

      // If performance/metric/account intent, attempt planner (SA360) first
      if (usePlannerContext) {
        const followupReply = (isSummaryShortcutPrompt(userMessage) && isFollowupPrompt(userMessage) && !relationalMetricPrompt)
          ? buildFollowupReply(userMessage, lastPlannerSnapshot)
          : null
        if (followupReply) {
          setMessages((prev) => [...prev, {
            role: 'assistant',
            content: followupReply,
            system: responseSystem,
          }])
          handled = true
        } else {
          try {
            const response = await axios.post(api.chat.send, {
              message: userMessage,
              ai_enabled: true,
              account_name: activeAccount?.name || defaultAccountName,
              session_id: sessionId || undefined,
              context: followupContext,
            })
            if (response.data?.reply) {
              setMessages((prev) => [...prev, {
                role: 'assistant',
                content: response.data.reply,
                system: responseSystem,
              }])
              handled = true
            }
          } catch (err) {
            const fallbackReply = buildFollowupReply(userMessage, lastPlannerSnapshot)
            if (fallbackReply) {
              setMessages((prev) => [...prev, {
                role: 'assistant',
                content: fallbackReply,
                system: responseSystem,
              }])
              handled = true
            }
          }
        }
      }
      if (!handled && (shouldUsePlanner || shouldUseTrends)) {
        try {
          let summaryText = null
          const accountNameFromIds = mergedIds.length === 1
            ? (accountsForCheck.length ? accountsForCheck : availableAccounts).find(
                (a) => String(a.customer_id) === String(mergedIds[0])
              )?.name
            : ''
          const resolvedAccountName = (routing?.account_name || activeAccount?.name || accountNameFromIds || '').trim()

          if (shouldUseTrends) {
            try {
              const routingThemes = Array.isArray(routing?.themes)
                ? routing.themes.filter((t) => typeof t === 'string' && t.trim()).slice(0, 5)
                : []
              const body = {
                account_name: resolvedAccountName || null,
                customer_ids: mergedIds,
                themes: routingThemes,
                timeframe: 'now 12-m',
                geo: 'US',
                budget: null,
                use_performance: true,
                session_id: sessionId || undefined,
              }
              const trendsResp = await axios.post(api.trends.seasonality, body)
              summaryText = buildTrendsSummary(trendsResp.data)
              if (trendsResp.data?.job_id) {
                const note = 'Trends are still processing in the background; I will post a refined update shortly.'
                summaryText = summaryText ? `${summaryText}\n${note}` : note
                pollTrendsJob(trendsResp.data.job_id, responseSystem)
              }
            } catch (trendErr) {
              // ignore and fall back to planner-only
            }
          }

          // Always attempt planner to keep performance context
          const planResp = await axios.post(api.chat.planAndRun, {
            message: userMessage,
            customer_ids: mergedIds,
            account_name: resolvedAccountName || null,
            intent_hint: routing?.intent || (routing?.run_planner ? 'performance' : null),
            session_id: sessionId || undefined,
          })
          if (planResp.data?.status === 'queued' && planResp.data?.job_id) {
            const queuedNote = 'Queued the analysis job. I will post results as soon as it finishes.'
            const combinedQueued = [summaryText, queuedNote].filter(Boolean).join('\n')
            setMessages((prev) => [...prev, {
              role: 'assistant',
              content: combinedQueued,
              system: responseSystem,
            }])
            pollPlanJob(planResp.data.job_id, userMessage, routedSystem, summaryText)
            handled = true
            return
          }

          if (planResp.data?.executed === false) {
            // Planner didn't execute. If it's an explicit block (e.g., SA360 not connected),
            // surface the blocker instead of falling back to generic chat "analysis".
            if (planResp.data?.status === 'blocked') {
              const blockerText = planResp.data?.summary || planResp.data?.error || 'This request is blocked.'
              const combinedBlocked = [summaryText, blockerText].filter(Boolean).join('\n')
              setMessages((prev) => [...prev, {
                role: 'assistant',
                content: combinedBlocked,
                system: responseSystem,
              }])
              handled = true
            } else {
              // Treat any non-executed planner response as handled; otherwise we fall back to chat/send
              // and often produce low-quality "missing data" replies that confuse beta users.
              const fallbackText =
                planResp.data?.summary ||
                planResp.data?.error ||
                'The planner did not run. Please confirm SA360 is connected and an account is selected, then retry.'
              const combinedFallback = [summaryText, fallbackText].filter(Boolean).join('\n')
              setMessages((prev) => [...prev, {
                role: 'assistant',
                content: combinedFallback,
                system: responseSystem,
              }])
              handled = true
            }
          } else {
            const plannerSnapshot = {
              plan: planResp.data?.plan,
              result: planResp.data?.result,
              analysis: planResp.data?.analysis,
              summary: planResp.data?.summary,
              enhanced_summary: planResp.data?.enhanced_summary,
              notes: planResp.data?.notes,
            }
            setLastPlannerSnapshot(plannerSnapshot)
            persistPlannerSnapshot(plannerSnapshot)
            const plannerSummary = await buildPlannerSummary(planResp.data || {}, userMessage, routedSystem)
            const combined = [summaryText, plannerSummary].filter(Boolean).join('\n') || 'Analysis completed.'
            setMessages((prev) => [...prev, {
              role: 'assistant',
              content: combined,
              system: responseSystem,
            }])
            handled = true
          }
        } catch (plannerErr) {
          const detail = plannerErr?.response?.data?.detail || plannerErr?.message || 'Planner failed; switching to chat.'
          setMessages((prev) => [...prev, {
            role: 'assistant',
            content: `The performance planner hit an issue (${detail}). Please retry. If this persists, run a health check (type: \"health check\") and confirm SA360 is connected + an account is selected.`,
            system: responseSystem,
          }])
          // Do not fall back to generic chat here — it produces low-quality "missing data" replies and hides the real problem.
          handled = true
        }
      }

      if (!handled) {
        const toolContext = followupContext || {
          tool: routing?.tool || routedSystem,
          customer_ids: mergedIds,
          account_name: activeAccount?.name || undefined,
        }
        const response = await axios.post(api.chat.send, {
          message: userMessage,
          ai_enabled: true,
          account_name: activeAccount?.name || defaultAccountName,
          session_id: sessionId || undefined,
          context: toolContext,
        })

        setMessages((prev) => [...prev, {
          role: 'assistant',
          content: response.data.reply,
          system: routedSystem,
        }])
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to process request')
      setMessages((prev) => [...prev, {
        role: 'assistant',
        content: 'I encountered an error processing your request. Please try again.',
        system: null,
      }])
    } finally {
      setProcessingState(null)
      setLoading(false)
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const handleSystemClick = (system) => {
    setInput(system.examples[0])
    setActiveSystem(system.id)
  }

  const handleExampleClick = (example) => {
    setInput(example)
  }

  const formatAccountLabel = (account) => {
    if (!account) return ''
    if (account.name) return `${account.name} (${account.customer_id})`
    return String(account.customer_id || '')
  }

  const handleAccountPick = (account) => {
    if (!account?.customer_id) return
    setNeedsIdsPrompt(false)
    setPendingAccounts([])
    setActiveAccount({ customer_id: String(account.customer_id), name: account.name || '' })
    setUseAccountForSession(true)
    const followUp = pendingMessage ? String(pendingMessage) : ''
    setPendingMessage('')
    if (followUp) sendMessage(followUp)
  }

  const accountSuggestions = needsIdsPrompt
    ? (pendingAccounts.length ? pendingAccounts : buildAccountSuggestions([], availableAccounts))
    : []
  const visibleAccountSuggestions = accountSuggestions.filter((account) => !account.manager)
  const normalizedSearch = accountSearch.trim().toLowerCase()
  const filteredAccountSuggestions = normalizedSearch
    ? visibleAccountSuggestions.filter((account) => {
        const name = String(account.name || '').toLowerCase()
        const id = String(account.customer_id || '')
        return name.includes(normalizedSearch) || id.includes(normalizedSearch)
      })
    : visibleAccountSuggestions
  const showAccountPicker = needsIdsPrompt

  const debugRouting = (() => {
    try {
      if (typeof window === 'undefined') return false
      const qs = new URLSearchParams(window.location.search || '')
      // Only enable routing debug with an explicit query param (avoid accidental leaks in broad beta).
      if (qs.get('debug_routing') === '1') return true
      return false
    } catch {
      return false
    }
  })()

  const routingMeta = lastRouting ? parseRoutingMeta(lastRouting.notes) : { model: null, verify: null }
  const routingBadges = []
  if (lastRouting?.intent) routingBadges.push(`Intent: ${lastRouting.intent}`)
  if (lastRouting?.tool) routingBadges.push(`Tool: ${lastRouting.tool}`)
  if (typeof lastRouting?.confidence === 'number') routingBadges.push(`Confidence: ${lastRouting.confidence.toFixed(2)}`)
  if (routingMeta.model) routingBadges.push(`Model: ${routingMeta.model}`)
  if (routingMeta.verify) routingBadges.push(`Verify: ${routingMeta.verify}`)
  const routingNotes = truncateText(lastRouting?.notes || '')

  return (
    <Container
      maxWidth="lg"
      sx={{
        minHeight: 'calc(100vh - 64px)',
        display: 'flex',
        flexDirection: 'column',
        py: 3,
      }}
    >
      {/* Master Header - Distinct from other pages */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Paper
          elevation={0}
          sx={{
            p: 4,
            mb: 3,
            borderRadius: 4,
            background: 'linear-gradient(135deg, #ffffff 0%, #f8fafc 60%, #eef2ff 100%)',
            border: '1px solid var(--kai-border)',
            boxShadow: '0 25px 50px -12px rgba(15, 23, 42, 0.18)',
            position: 'relative',
            overflow: 'hidden',
          }}
        >
          {/* Animated background elements */}
          <Box
            sx={{
              position: 'absolute',
              top: -50,
              right: -50,
              width: 200,
              height: 200,
              borderRadius: '50%',
              background: 'radial-gradient(circle, rgba(139,92,246,0.2) 0%, transparent 70%)',
            }}
          />
          <Box
            sx={{
              position: 'absolute',
              bottom: -30,
              left: '30%',
              width: 150,
              height: 150,
              borderRadius: '50%',
              background: 'radial-gradient(circle, rgba(59,130,246,0.15) 0%, transparent 70%)',
            }}
          />

          <Box display="flex" alignItems="center" gap={2} position="relative" zIndex={1}>
            <Avatar
              sx={{
                width: 56,
                height: 56,
                background: 'linear-gradient(135deg, #8b5cf6, #6366f1)',
                boxShadow: '0 8px 32px rgba(139, 92, 246, 0.4)',
              }}
            >
              <Hub sx={{ fontSize: 32 }} />
            </Avatar>
            <Box flex={1}>
              <Typography
                variant="h3"
                sx={{
                  fontWeight: 800,
                  background: 'linear-gradient(135deg, #0f172a, #4f46e5)',
                  backgroundClip: 'text',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  letterSpacing: '-0.02em',
                }}
              >
                Kai
              </Typography>
              <Typography variant="body1" sx={{ color: 'var(--kai-text-soft)', mt: 0.5 }}>
                Your AI command center. Ask anything, route anywhere.
              </Typography>
            </Box>
            <Box display="flex" gap={1}>
              <Chip
                icon={<Psychology sx={{ color: '#4f46e5 !important' }} />}
                label="AI Orchestrator"
                sx={{
                  background: 'rgba(99, 102, 241, 0.12)',
                  color: '#4f46e5',
                  border: '1px solid rgba(99, 102, 241, 0.25)',
                }}
              />
              <Chip
                icon={<Bolt sx={{ color: '#fbbf24 !important' }} />}
                label="Live"
                sx={{
                  background: 'rgba(245, 158, 11, 0.18)',
                  color: '#b45309',
                  border: '1px solid rgba(245, 158, 11, 0.35)',
                }}
              />
            </Box>
          </Box>
          {sessionId && (
            <Box
              mt={2}
              display="flex"
              alignItems="center"
              flexWrap="wrap"
              gap={1.5}
              position="relative"
              zIndex={1}
            >
              <Chip
                label={sa360Status.connected ? 'SA360 connected' : 'SA360 not connected'}
                sx={{
                  borderColor: sa360Status.connected ? 'rgba(34,197,94,0.6)' : 'rgba(248,113,113,0.6)',
                  color: sa360Status.connected ? '#22c55e' : '#f87171',
                  backgroundColor: sa360Status.connected ? 'rgba(34,197,94,0.15)' : 'rgba(248,113,113,0.15)',
                }}
              />
              {!sa360Status.connected && (
                <Button
                  variant="contained"
                  size="small"
                  startIcon={<LinkIcon />}
                  onClick={startSa360Connect}
                >
                  Connect SA360
                </Button>
              )}
              {sa360Status.connected && (
                <Button
                  variant="outlined"
                  size="small"
                  onClick={fetchSa360Status}
                  disabled={sa360StatusLoading}
                >
                  Refresh status
                </Button>
              )}
              {sa360Status.connected && (
                <Box display="flex" alignItems="center" gap={1} flexWrap="wrap">
                  {!String(sa360Status?.login_customer_id || '').trim() && (
                    <Box display="flex" alignItems="center" gap={1} flexWrap="wrap">
                      <Autocomplete
                        size="small"
                        value={selectedManager}
                        onChange={(_e, next) => {
                          if (next?.customer_id) {
                            saveLoginCustomerId(next.customer_id)
                          }
                        }}
                        options={selectableManagers}
                        loading={managersLoading}
                        getOptionLabel={(opt) =>
                          opt?.name ? `${opt.name} (${opt.customer_id})` : String(opt?.customer_id || '')
                        }
                        isOptionEqualToValue={(a, b) => String(a?.customer_id) === String(b?.customer_id)}
                        sx={{ minWidth: 320, maxWidth: 520 }}
                        renderInput={(params) => (
                          <TextField
                            {...params}
                            label="Manager (MCC) by name"
                            placeholder={managersLoading ? 'Discovering MCCs…' : 'Optional: pick your manager'}
                            InputProps={{
                              ...params.InputProps,
                              endAdornment: (
                                <>
                                  {managersLoading ? <CircularProgress color="inherit" size={16} /> : null}
                                  {params.InputProps.endAdornment}
                                </>
                              ),
                            }}
                          />
                        )}
                      />
                      <Button
                        variant="outlined"
                        size="small"
                        onClick={fetchSa360Managers}
                        disabled={managersLoading}
                        sx={{ whiteSpace: 'nowrap' }}
                      >
                        Discover MCCs
                      </Button>
                      {managersError && (
                        <Typography variant="caption" color="error">
                          {managersError}
                        </Typography>
                      )}
                    </Box>
                  )}
                  <TextField
                    size="small"
                    label="Manager ID (MCC)"
                    value={loginCustomerIdInput}
                    onChange={(e) => setLoginCustomerIdInput(e.target.value)}
                    sx={{ minWidth: 220 }}
                  />
                  <Button
                    variant="outlined"
                    size="small"
                    onClick={() => saveLoginCustomerId()}
                    disabled={sa360SaveLoading || !loginCustomerIdInput}
                  >
                    {sa360SaveLoading ? (
                      <>
                        <CircularProgress size={14} sx={{ mr: 1 }} />
                        Saving…
                      </>
                    ) : (
                      'Save MCC'
                    )}
                  </Button>
                  {sa360StatusSuccess && (
                    <Chip
                      label={sa360StatusSuccess}
                      size="small"
                      sx={{
                        borderColor: 'rgba(34,197,94,0.6)',
                        color: '#22c55e',
                        backgroundColor: 'rgba(34,197,94,0.15)',
                      }}
                      variant="outlined"
                    />
                  )}
                </Box>
              )}
              {sa360Status.connected && String(sa360Status?.login_customer_id || '').trim() && (
                <Box display="flex" alignItems="center" gap={1} flexWrap="wrap">
                  <Autocomplete
                    size="small"
                    value={activeAccount}
                    onChange={(_e, next) => {
                      setActiveAccount(next)
                      if (next?.customer_id) {
                        setUseAccountForSession(true)
                      }
                    }}
                    options={selectableAccounts}
                    loading={accountsLoading}
                    getOptionLabel={(opt) =>
                      opt?.name ? `${opt.name} (${opt.customer_id})` : String(opt?.customer_id || '')
                    }
                    isOptionEqualToValue={(a, b) => String(a?.customer_id) === String(b?.customer_id)}
                    sx={{ minWidth: 360, maxWidth: 520 }}
                    renderInput={(params) => (
                      <TextField
                        {...params}
                        label="Account (by name)"
                        placeholder={accountsLoading ? 'Loading accounts…' : 'Select an account'}
                        InputProps={{
                          ...params.InputProps,
                          endAdornment: (
                            <>
                              {accountsLoading ? <CircularProgress color="inherit" size={16} /> : null}
                              {params.InputProps.endAdornment}
                            </>
                          ),
                        }}
                      />
                    )}
                  />
                  <Button
                    variant="outlined"
                    size="small"
                    onClick={fetchSa360Accounts}
                    disabled={accountsLoading}
                  >
                    Reload accounts
                  </Button>
                  <Button
                    variant="outlined"
                    size="small"
                    onClick={() => window.location.assign('/sa360-columns')}
                    sx={{ whiteSpace: 'nowrap' }}
                  >
                    Browse columns
                  </Button>
                  {accountsError && (
                    <Typography variant="caption" color="error">
                      {accountsError}
                    </Typography>
                  )}
                </Box>
              )}
              {sa360StatusError && (
                <Typography variant="caption" color="error">
                  {sa360StatusError}
                </Typography>
              )}
            </Box>
          )}
        </Paper>
      </motion.div>

      {/* System Capabilities Grid - Shows when no conversation */}
      <Collapse in={showCapabilities && messages.length === 0}>
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <Typography
            variant="overline"
            sx={{ color: '#64748b', mb: 2, display: 'block', letterSpacing: 2 }}
          >
            AVAILABLE SYSTEMS
          </Typography>
          <Grid container spacing={2} sx={{ mb: 3 }}>
            {systems.map((system, idx) => (
              <Grid item xs={12} sm={6} md={3} key={system.id}>
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.4, delay: 0.1 * idx }}
                >
                  <Card
                    sx={{
                      background: 'var(--kai-bg)',
                      border: `1px solid ${system.color}33`,
                      borderRadius: 3,
                      transition: 'all 0.3s ease',
                      '&:hover': {
                        borderColor: system.color,
                        boxShadow: `0 8px 32px ${system.color}22`,
                        transform: 'translateY(-4px)',
                      },
                    }}
                  >
                    <CardActionArea onClick={() => handleSystemClick(system)} sx={{ p: 2 }}>
                      <CardContent sx={{ p: 0 }}>
                        <Box display="flex" alignItems="center" gap={1.5} mb={1.5}>
                          <Avatar
                            sx={{
                              width: 40,
                              height: 40,
                              background: `${system.color}22`,
                              color: system.color,
                            }}
                          >
                            <system.icon />
                          </Avatar>
                          <Typography variant="h6" sx={{ color: 'var(--kai-text)', fontWeight: 600 }}>
                            {system.name}
                          </Typography>
                        </Box>
                        <Typography variant="body2" sx={{ color: 'var(--kai-text-soft)', mb: 2 }}>
                          {system.description}
                        </Typography>
                        <Box display="flex" flexWrap="wrap" gap={0.5}>
                          {system.examples.slice(0, 2).map((ex, i) => (
                            <Chip
                              key={i}
                              label={ex}
                              size="small"
                              onClick={(e) => {
                                e.stopPropagation()
                                handleExampleClick(ex)
                              }}
                              sx={{
                                fontSize: '0.7rem',
                                height: 24,
                                background: `${system.color}15`,
                                color: system.color,
                                border: `1px solid ${system.color}33`,
                                '&:hover': {
                                  background: `${system.color}25`,
                                },
                              }}
                            />
                          ))}
                        </Box>
                      </CardContent>
                    </CardActionArea>
                  </Card>
                </motion.div>
              </Grid>
            ))}
          </Grid>
        </motion.div>
      </Collapse>

      {/* Chat Messages Area */}
      <Paper
        elevation={0}
        sx={{
          flex: 1,
          p: 3,
          mb: 2,
          borderRadius: 3,
          background: 'var(--kai-surface)',
          border: '1px solid var(--kai-surface-muted)',
          minHeight: messages.length === 0 ? 200 : 400,
          maxHeight: 'calc(100vh - 450px)',
          overflowY: 'auto',
          display: 'flex',
          flexDirection: 'column',
          gap: 2,
        }}
      >
        {messages.length === 0 && !showCapabilities && (
          <Box
            display="flex"
            flexDirection="column"
            alignItems="center"
            justifyContent="center"
            height="100%"
            color="var(--kai-border)"
          >
            <AutoAwesome sx={{ fontSize: 48, mb: 2, opacity: 0.5 }} />
            <Typography variant="body1">Start a conversation with Kai</Typography>
          </Box>
        )}

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
                gap={2}
                alignItems="flex-start"
                justifyContent={msg.role === 'user' ? 'flex-end' : 'flex-start'}
              >
                {msg.role === 'assistant' && (
                  <Avatar
                    sx={{
                      width: 36,
                      height: 36,
                      background: 'linear-gradient(135deg, #8b5cf6, #6366f1)',
                    }}
                  >
                    <AutoAwesome fontSize="small" />
                  </Avatar>
                )}
                <Paper
                  elevation={0}
                  data-testid={msg.role === 'assistant' ? 'chat-assistant' : 'chat-user'}
                  sx={{
                    p: 2,
                    maxWidth: '70%',
                    borderRadius: 3,
                    background:
                      msg.role === 'user'
                        ? 'linear-gradient(135deg, #8b5cf6, #6366f1)'
                        : 'var(--kai-surface)',
                    color: 'var(--kai-text)',
                    border:
                      msg.role === 'user'
                        ? '1px solid #8b5cf6'
                        : '1px solid var(--kai-surface-muted)',
                  }}
                >
                  {msg.system && msg.role === 'assistant' && (
                    <Chip
                      size="small"
                      label={getSystemMeta(msg.system)?.name || 'General'}
                      sx={{
                        mb: 1,
                        height: 20,
                        fontSize: '0.65rem',
                        background: `${getSystemMeta(msg.system)?.color || '#8b5cf6'}22`,
                        color: getSystemMeta(msg.system)?.color || '#8b5cf6',
                      }}
                    />
                  )}
                  <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap' }} data-testid="chat-message-text">
                    {msg.content}
                  </Typography>
                </Paper>
                {msg.role === 'user' && (
                  <Avatar sx={{ width: 36, height: 36, bgcolor: '#3730a3' }}>
                    <Person fontSize="small" />
                  </Avatar>
                )}
              </Box>
            </motion.div>
          ))}
        </AnimatePresence>

        {loading && (
          <Box display="flex" gap={2} alignItems="center">
            <Avatar
              sx={{
                width: 36,
                height: 36,
                background: 'linear-gradient(135deg, #8b5cf6, #6366f1)',
              }}
            >
              <AutoAwesome fontSize="small" />
            </Avatar>
            <Paper
              sx={{
                p: 2,
                borderRadius: 3,
                background: 'var(--kai-surface)',
                border: '1px solid var(--kai-surface-muted)',
              }}
            >
              <Box display="flex" gap={1}>
                {[0, 1, 2].map((i) => (
                  <motion.div
                    key={i}
                    animate={{ opacity: [0.3, 1, 0.3] }}
                    transition={{ duration: 1, repeat: Infinity, delay: i * 0.2 }}
                  >
                    <Box
                      sx={{
                        width: 8,
                        height: 8,
                        borderRadius: '50%',
                        background: '#8b5cf6',
                      }}
                    />
                  </motion.div>
                ))}
              </Box>
            </Paper>
          </Box>
        )}

        {error && (
          <Alert
            severity="error"
            onClose={() => setError(null)}
            sx={{
              borderRadius: 2,
              background: '#fff1f2',
              color: '#b91c1c',
              border: '1px solid #fca5a5',
            }}
          >
            {error}
          </Alert>
        )}

        <div ref={messagesEndRef} />
      </Paper>

      {showAccountPicker && (
        <Collapse in>
          <Paper
            elevation={0}
            sx={{
              p: 2,
              mb: 2,
              borderRadius: 3,
              background: 'var(--kai-bg)',
              border: '1px dashed var(--kai-border-strong)',
            }}
          >
            <Typography variant="subtitle2" sx={{ color: 'var(--kai-text)' }}>
              Pick an account to continue
            </Typography>
            <Autocomplete
              size="small"
              value={activeAccount}
              onChange={(_e, next) => {
                if (next?.customer_id) {
                  handleAccountPick(next)
                }
              }}
              inputValue={accountSearch}
              onInputChange={(_e, next) => setAccountSearch(next)}
              options={selectableAccounts}
              loading={accountsLoading}
              getOptionLabel={(opt) => formatAccountLabel(opt)}
              isOptionEqualToValue={(a, b) => String(a?.customer_id) === String(b?.customer_id)}
              filterOptions={(options, state) => {
                const q = String(state?.inputValue || '').toLowerCase().trim()
                const qDigits = q.replace(/[^\d]/g, '')
                if (!q) return options.slice(0, 50)
                return options
                  .filter((opt) => {
                    const name = String(opt?.name || '').toLowerCase()
                    const id = String(opt?.customer_id || '')
                    return (name && name.includes(q)) || (qDigits && id.includes(qDigits))
                  })
                  .slice(0, 50)
              }}
              renderInput={(params) => (
                <TextField
                  {...params}
                  placeholder="Search accounts by name or paste an ID"
                  onKeyDown={(e) => {
                    if (e.key !== 'Enter') return
                    const cid = normalizeCid(accountSearch)
                    if (!cid || cid.length < 8 || cid.length > 12) return
                    e.preventDefault()
                    handleAccountPick({ customer_id: cid, name: '' })
                  }}
                  fullWidth
                  sx={{
                    mt: 1,
                    '& .MuiOutlinedInput-root': {
                      borderRadius: 2,
                      color: 'var(--kai-text)',
                      background: 'var(--kai-surface)',
                      '& fieldset': { borderColor: 'var(--kai-border-strong)' },
                      '&:hover fieldset': { borderColor: '#8b5cf6' },
                      '&.Mui-focused fieldset': { borderColor: '#a78bfa' },
                    },
                    '& .MuiOutlinedInput-input::placeholder': {
                      color: '#64748b',
                      opacity: 1,
                    },
                  }}
                />
              )}
            />
            {visibleAccountSuggestions.length > 0 && (
              <Box display="flex" flexWrap="wrap" gap={1} mt={1}>
                {visibleAccountSuggestions.map((account) => (
                  <Chip
                    key={String(account.customer_id)}
                    label={formatAccountLabel(account)}
                    onClick={() => handleAccountPick(account)}
                    clickable
                    size="small"
                    sx={{
                      color: 'var(--kai-text)',
                      borderColor: 'var(--kai-border-strong)',
                      background: 'var(--kai-surface)',
                    }}
                  />
                ))}
              </Box>
            )}
            {needsIdsPrompt && selectableAccounts.length === 0 && (
              <Typography variant="caption" sx={{ color: 'var(--kai-text-soft)', mt: 1, display: 'block' }}>
                No accounts loaded yet. Paste a customer ID to continue.
              </Typography>
            )}
          </Paper>
        </Collapse>
      )}

      {/* Input Area */}
      <Paper
        elevation={0}
        sx={{
          p: 2,
          borderRadius: 3,
          background: 'var(--kai-surface)',
          border: '1px solid var(--kai-surface-muted)',
          boxShadow: '0 -10px 40px rgba(15, 23, 42, 0.08)',
        }}
      >
        <Box display="flex" gap={2} alignItems="center">
          <TextField
            fullWidth
            multiline
            maxRows={3}
            placeholder={needsIdsPrompt ? 'Select an account above to continue…' : 'Ask Kai anything... audit, analyze, create, or explore'}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyPress}
            disabled={loading || needsIdsPrompt}
            variant="outlined"
            sx={{
              '& .MuiOutlinedInput-root': {
                borderRadius: 2,
                color: 'var(--kai-text)',
                background: 'var(--kai-surface)',
                '& fieldset': { borderColor: 'var(--kai-border-strong)' },
                '&:hover fieldset': { borderColor: '#8b5cf6' },
                '&.Mui-focused fieldset': { borderColor: '#a78bfa' },
              },
              '& .MuiOutlinedInput-input::placeholder': {
                color: '#64748b',
                opacity: 1,
              },
            }}
          />
          <IconButton
            onClick={() => sendMessage()}
            disabled={needsIdsPrompt || !input.trim() || loading}
            sx={{
              width: 52,
              height: 52,
              background: 'linear-gradient(135deg, #8b5cf6, #6366f1)',
              color: '#fff',
              '&:hover': {
                background: 'linear-gradient(135deg, #a78bfa, #818cf8)',
              },
              '&:disabled': {
                background: 'var(--kai-border-strong)',
                color: '#64748b',
              },
            }}
          >
            <Send />
          </IconButton>
        </Box>

        {/* Active System Indicator */}
          {activeSystem && (
            <Fade in>
              <Box display="flex" alignItems="center" gap={1} mt={1.5}>
                <Typography variant="caption" sx={{ color: '#64748b' }}>
                  Routing to:
              </Typography>
              <Chip
                size="small"
                icon={
                  (() => {
                    const SystemIcon = getSystemMeta(activeSystem)?.icon
                    return SystemIcon ? <SystemIcon sx={{ fontSize: 14 }} /> : null
                  })()
                }
                label={getSystemMeta(activeSystem)?.name}
                onDelete={() => setActiveSystem(null)}
                sx={{
                  height: 24,
                  background: `${getSystemMeta(activeSystem)?.color}22`,
                  color: getSystemMeta(activeSystem)?.color,
                  '& .MuiChip-deleteIcon': {
                    color: getSystemMeta(activeSystem)?.color,
                  },
                }}
              />
              </Box>
            </Fade>
          )}
          {activeAccount && (
            <Fade in>
              <Box display="flex" alignItems="center" gap={1} mt={1}>
                <Typography variant="caption" sx={{ color: '#64748b' }}>
                  Active account:
                </Typography>
                <Chip
                  size="small"
                  label={formatAccountLabel(activeAccount)}
                  onDelete={() => {
                    setActiveAccount(null)
                    setUseAccountForSession(false)
                    // Explicitly clear the server-side default only when the user asks to.
                    if (
                      sessionId &&
                      sa360Status.connected &&
                      String(sa360Status?.login_customer_id || '').trim()
                    ) {
                      axios
                        .post(`${API_BASE_URL}/api/sa360/default-account`, {
                          session_id: sessionId,
                          customer_id: null,
                          account_name: null,
                        })
                        .then(() => {
                          setToast({
                            open: true,
                            message: 'Default account cleared for this user.',
                            severity: 'info',
                          })
                        })
                        .catch(() => {
                          // non-blocking
                        })
                    }
                  }}
                  sx={{
                    height: 24,
                    background: 'var(--kai-surface-alt)',
                    color: 'var(--kai-text)',
                    '& .MuiChip-deleteIcon': {
                      color: 'var(--kai-text-soft)',
                    },
                  }}
                />
              </Box>
            </Fade>
          )}
          {activeAccount && (
            <Box mt={0.5}>
              <FormControlLabel
                control={
                  <Switch
                    size="small"
                    checked={useAccountForSession}
                    onChange={(e) => setUseAccountForSession(e.target.checked)}
                  />
                }
                label="Save as default account"
                sx={{
                  '.MuiFormControlLabel-label': {
                    color: 'var(--kai-text-soft)',
                    fontSize: '0.75rem',
                  },
                }}
              />
            </Box>
          )}
          {debugRouting && lastRouting && routingBadges.length > 0 && (
            <Fade in>
              <Box display="flex" flexWrap="wrap" gap={1} mt={1}>
                {routingBadges.map((badge) => (
                  <Chip
                    key={badge}
                    size="small"
                    label={badge}
                    sx={{
                      height: 22,
                      background: 'var(--kai-surface)',
                      color: '#cbd5e1',
                      border: '1px solid var(--kai-surface-alt)',
                    }}
                  />
                ))}
              </Box>
            </Fade>
          )}
          {processingState && (
            <Fade in>
              <Box display="flex" alignItems="center" gap={1} mt={1}>
                <Chip
                  size="small"
                  icon={<Autorenew sx={{ fontSize: 14 }} />}
                  label={`Processing: ${processingState.label}`}
                  sx={{
                    height: 22,
                    background: 'var(--kai-bg)',
                    color: '#fde68a',
                    border: '1px solid rgba(251, 191, 36, 0.35)',
                    '& .MuiChip-icon': {
                      color: '#f59e0b',
                    },
                  }}
                />
                <Typography variant="caption" sx={{ color: '#64748b' }}>
                  ETA {processingState.eta}
                </Typography>
              </Box>
            </Fade>
          )}
          {trendsQueue && (
            <Fade in>
              <Box display="flex" alignItems="center" gap={1} mt={1}>
                <Chip
                  size="small"
                  icon={<Autorenew sx={{ fontSize: 14 }} />}
                  label={`Trends queued${trendsQueue.systemLabel ? ` - ${trendsQueue.systemLabel}` : ''}`}
                  sx={{
                    height: 22,
                    background: 'var(--kai-bg)',
                    color: '#a7f3d0',
                    border: '1px solid rgba(34, 197, 94, 0.35)',
                    '& .MuiChip-icon': {
                      color: '#34d399',
                    },
                  }}
                />
                <Typography variant="caption" sx={{ color: '#64748b' }}>
                  Updating...
                </Typography>
              </Box>
            </Fade>
          )}
          {debugRouting && lastRouting && routingNotes && (
            <Typography variant="caption" sx={{ color: '#64748b', display: 'block', mt: 0.5 }}>
              {routingNotes}
            </Typography>
          )}
        </Paper>

        <Snackbar
          open={!!toast?.open}
          autoHideDuration={4000}
          onClose={() => setToast((prev) => ({ ...(prev || {}), open: false }))}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
        >
          <Alert
            onClose={() => setToast((prev) => ({ ...(prev || {}), open: false }))}
            severity={toast?.severity || 'success'}
            variant="filled"
            sx={{ width: '100%' }}
          >
            {toast?.message || ''}
          </Alert>
        </Snackbar>
    </Container>
  )
}
