import { motion } from 'framer-motion'
import {
  Box,
  Container,
  Typography,
  Paper,
  Stack,
  Chip,
  Grid,
  Divider,
} from '@mui/material'
import {
  InfoOutlined,
  Hub,
  Storage as StorageIcon,
  CloudQueue,
  Lan,
  Code,
  Verified,
  HelpOutline,
  AltRoute,
  LightbulbOutlined,
  SettingsEthernet,
  Schedule,
  ArrowForward,
  AccountTree,
} from '@mui/icons-material'

const apiSurface = [
  {
    title: 'Core Health + Auth',
    items: ['/api/health', '/api/version', '/api/diagnostics/health', '/api/auth/verify'],
  },
  {
    title: 'Chat + Router',
    items: [
      '/api/chat/route',
      '/api/chat/send',
      '/api/chat/history',
      '/api/chat/clear',
      '/api/chat/plan-and-run',
    ],
  },
  {
    title: 'Settings + Env',
    items: ['/api/settings', '/api/settings/env', '/api/settings/env/update'],
  },
  {
    title: 'Audit + Data',
    items: [
      '/api/audit/generate',
      '/api/audit/upload (deprecated; use /api/data/upload)',
      '/api/audit/download/{filename} (local fallback)',
      'download_url (SAS link in /api/audit/generate)',
      '/api/audit/business-units',
      '/api/data/upload (preferred)',
    ],
  },
  {
    title: 'Tools',
    items: [
      '/api/creative/generate',
      '/api/pmax/analyze',
      '/api/serp/check',
      '/api/serp/competitor-signal',
      '/api/intel/diagnose',
      '/api/trends/seasonality',
      '/api/search/web',
    ],
  },
  {
    title: 'SA360 + Ads',
    items: [
      '/api/sa360/accounts',
      '/api/integrations/sa360/fetch',
      '/api/integrations/sa360/fetch-and-audit',
      '/api/integrations/ads/connect',
      '/api/integrations/ads/fetch',
      '/api/integrations/ads/fetch-and-audit',
    ],
  },
  {
    title: 'Jobs + Queue',
    items: [
      '/api/jobs/health',
      '/api/jobs/{job_id}',
      '/api/jobs/{job_id}/result',
      '/api/jobs/{job_id}/stream',
    ],
  },
  {
    title: 'QA + Diagnostics',
    items: ['/api/qa/accuracy', '/api/diagnostics/sa360/perf-window'],
  },
]

const dependencyGroups = [
  {
    title: 'Cloud Runtime',
    icon: <CloudQueue fontSize="small" />,
    items: [
      'Azure Static Web Apps (kai-platform-react, www.kelvinale.com)',
      'Azure Container Apps (kai-platform-backend, kai-platform-worker, kai-llm)',
      'Azure Container Registry (cafbdd83c623acr)',
    ],
  },
  {
    title: 'AI + Reasoning',
    icon: <Hub fontSize="small" />,
    items: [
      'Azure OpenAI (gpt-4-turbo deployment)',
      'Local LLM (llama.cpp via kai-llm shim, /api/chat)',
      'Router verify: local first, Azure verification',
    ],
  },
  {
    title: 'Knowledge Base + Vector Search',
    icon: <StorageIcon fontSize="small" />,
    items: [
      'Azure AI Search (vector index: knowledge_base)',
      'Knowledge base docs (md/docx) indexed via scripts/index_knowledge_base.py',
      'Embeddings: Azure OpenAI text-embedding-3-small',
    ],
  },
  {
    title: 'Data + Storage',
    icon: <StorageIcon fontSize="small" />,
    items: [
      'Azure Blob Storage (ppcauditstoragekl / mock-kelvin-co)',
      'SA360 cache in blob (sa360_cache/*)',
      'Azure Storage Queue (job orchestration)',
      'Azure Table Storage (chat history + tool context)',
      'Local sqlite fallback (settings + local dev)',
    ],
  },
  {
    title: 'External APIs',
    icon: <Lan fontSize="small" />,
    items: [
      'Search Ads 360 API (OAuth + developer token)',
      'SerpAPI (bing engine for web search + SERP checks)',
      'Google Trends (pytrends) for seasonality + SerpAPI fallback',
      'Optional Ads APIs (feature-flagged; not configured)',
    ],
  },
]

const configGroups = [
  {
    title: 'LLM Routing',
    items: [
      'ENABLE_LOCAL_LLM_WRAPPER',
      'LOCAL_LLM_ENDPOINT',
      'LOCAL_LLM_MODEL',
      'LLM_BLEND_MODE',
      'LOCAL_LLM_TIMEOUT_SECONDS',
      'LOCAL_LLM_HEALTH_TIMEOUT_SECONDS',
      'AZURE_OPENAI_ENDPOINT',
      'AZURE_OPENAI_API_KEY',
      'AZURE_OPENAI_DEPLOYMENT',
    ],
  },
  {
    title: 'Vector Search + Knowledge Base',
    items: [
      'ENABLE_VECTOR_INDEXING',
      'CONCIERGE_SEARCH_ENDPOINT',
      'CONCIERGE_SEARCH_KEY',
      'CONCIERGE_SEARCH_INDEX',
      'CONCIERGE_SEARCH_VECTOR_FIELD',
      'CONCIERGE_SEARCH_VECTOR_K',
      'CONCIERGE_SEARCH_HYBRID',
      'CONCIERGE_SEARCH_VECTOR_FILTER_MODE',
      'AZURE_OPENAI_EMBEDDING_DEPLOYMENT',
      'AZURE_OPENAI_EMBEDDING_API_VERSION',
      'AZURE_OPENAI_EMBEDDING_TIMEOUT_SECONDS',
      'KNOWLEDGE_CHUNK_CHARS',
      'KNOWLEDGE_MD_CHUNK_MODE',
      'KNOWLEDGE_EMBEDDING_DELAY_SECONDS',
    ],
  },
  {
    title: 'SA360 + Cache',
    items: [
      'SA360_FETCH_ENABLED',
      'SA360_CLIENT_ID',
      'SA360_CLIENT_SECRET',
      'SA360_REFRESH_TOKEN',
      'SA360_LOGIN_CUSTOMER_ID',
      'SA360_OAUTH_REDIRECT_URI',
      'SA360_PLAN_CONCURRENCY',
      'SA360_PLAN_CHUNK_SIZE',
      'SA360_DIAGNOSTICS_CONCURRENCY',
      'SA360_DIAGNOSTICS_CHUNK_SIZE',
      'SA360_PERF_REPORTS',
      'SA360_DIAGNOSTICS_REPORTS',
      'SA360_CACHE_ENABLED',
      'SA360_CACHE_FRESHNESS_DAYS',
      'SA360_CACHE_PREFIX',
    ],
  },
  {
    title: 'Jobs + Queue',
    items: [
      'JOB_QUEUE_ENABLED',
      'JOB_QUEUE_FORCE',
      'JOB_QUEUE_NAME',
      'JOB_TABLE_NAME',
      'JOB_RESULT_CONTAINER',
      'JOB_QUEUE_POLL_SECONDS',
      'JOB_QUEUE_VISIBILITY_TIMEOUT',
      'JOB_MAX_ATTEMPTS',
    ],
  },
  {
    title: 'Storage + Security',
    items: [
      'AZURE_STORAGE_CONNECTION_STRING',
      'PPC_DATA_BLOB_CONTAINER',
      'PPC_DATA_BLOB_PREFIX',
      'AUDIT_BLOB_CONTAINER',
      'AUDIT_BLOB_PREFIX',
      'AUDIT_SAS_EXPIRY_HOURS',
      'KAI_CHAT_HISTORY_TABLE',
      'KAI_CHAT_HISTORY_TABLE_ENABLED',
      'KAI_ACCESS_PASSWORD',
    ],
  },
  {
    title: 'Feature Flags',
    items: [
      'ADS_FETCH_ENABLED',
      'ENABLE_TRENDS',
      'ENABLE_ML_REASONING',
      'ENABLE_VALIDATOR',
    ],
  },
]

const flowSteps = [
  {
    title: 'Chat + Router (General)',
    steps: [
      'UI to /api/chat/route (intent + tool + confidence)',
      'Session context (session_id + active account) included for follow-ups',
      'Router: local LLM to Azure verify',
      'General chat to /api/chat/send (LLM response)',
      'Optional web search to /api/search/web via SerpAPI',
      'Optional vector retrieval (Azure AI Search) when configured',
    ],
  },
  {
    title: 'Performance / Audit',
    steps: [
      'UI to /api/chat/plan-and-run (resolved account + date range)',
      'SA360 fetch to cache read/write in blob',
      'Long-running tasks can queue; UI polls /api/jobs for status',
      'Audit engine reads CSVs to report XLSX',
      'Report uploaded to blob (SAS download_url) with /api/audit/download fallback',
      'Optional vector indexing of audit outputs when ENABLE_VECTOR_INDEXING is set',
    ],
  },
  {
    title: 'Tools (PMax / SERP / Competitor / Creative)',
    steps: [
      'UI to /api/pmax/analyze for placements + channel split',
      'UI to /api/serp/check for URL health',
      'UI to /api/serp/competitor-signal for competitor analysis',
      'UI to /api/creative/generate for ad copy',
      'UI to /api/trends/seasonality for seasonality signal',
    ],
  },
]

const logicFlow = [
  {
    key: 'what',
    label: 'What',
    question: 'What is being asked?',
    summary: 'Intent and entities are extracted from the message and context.',
    details: [
      'Intent: general, performance, audit, or tool',
      'Entities: account name, customer IDs, date range',
      'UI context: active module and session scope',
    ],
    icon: <HelpOutline sx={{ color: '#38bdf8' }} />,
  },
  {
    key: 'where',
    label: 'Where',
    question: 'Where does it route?',
    summary: 'Router selects the tool path and API surface.',
    details: [
      'Router picks tool and planner flags',
      'Routes map to FastAPI endpoints',
      'UI highlights the active module',
    ],
    icon: <AltRoute sx={{ color: '#a78bfa' }} />,
  },
  {
    key: 'why',
    label: 'Why',
    question: 'Why that path?',
    summary: 'Guardrails enforce safe, grounded decisions.',
    details: [
      'Confidence gates and clarifications',
      'No tool on general chat',
      'Account requirement checks',
    ],
    icon: <LightbulbOutlined sx={{ color: '#f59e0b' }} />,
  },
  {
    key: 'how',
    label: 'How',
    question: 'How is it executed?',
    summary: 'Execution uses the right engine and data sources.',
    details: [
      'Local LLM first, Azure verify or fallback',
      'SA360 fetch and blob cache when needed',
      'Tool plugins run scoped analysis',
      'Seasonality uses Google Trends (pytrends) + SerpAPI fallback',
      'Vector retrieval uses Azure AI Search when configured',
    ],
    icon: <SettingsEthernet sx={{ color: '#22d3ee' }} />,
  },
  {
    key: 'when',
    label: 'When',
    question: 'When does it run?',
    summary: 'Timing, cache, and fallback conditions decide flow.',
    details: [
      'Fresh ranges fetch live SA360 data',
      'Cache used for older windows',
      'Web search only on request or need',
      'Trends only run for seasonality or forecasting requests',
      'Queue used for long-running tasks; UI polls job status',
    ],
    icon: <Schedule sx={{ color: '#f97316' }} />,
  },
]

const stackModules = [
  {
    title: 'Frontend',
    items: ['React 18', 'Vite', 'Material UI', 'Framer Motion', 'Axios', 'React Router'],
  },
  {
    title: 'Backend',
    items: ['FastAPI', 'Uvicorn', 'Pydantic', 'Pandas', 'Async job queue'],
  },
  {
    title: 'AI + Reasoning',
    items: ['Azure OpenAI (gpt-4-turbo)', 'Local LLM (kai-llm)', 'Router verify + guardrails'],
  },
  {
    title: 'Data + Storage',
    items: [
      'Azure Blob Storage',
      'Azure Storage Queue',
      'Azure Table Storage',
      'Azure AI Search (vector)',
      'Local sqlite fallback',
    ],
  },
  {
    title: 'Infra + Delivery',
    items: ['Azure Static Web Apps', 'Azure Container Apps', 'Azure Container Registry', 'Playwright tests'],
  },
]

const stackHow = [
  'Static Web App serves the React UI and assets.',
  'UI calls FastAPI endpoints for routing, tools, and data.',
  'Frontend stamps window.__BUILD__; backend exposes /api/version for provenance.',
  'Shared AI persona + guardrails (ai_sync) align chat, tool follow-ups, and audits.',
  'Heavy tasks enqueue to the worker; UI polls /api/jobs for completion.',
  'SA360 data flows to cache in blob, then into planner/audit outputs.',
  'Trends and SERP calls use external APIs with fallbacks.',
  'Knowledge base docs are chunked + embedded and indexed for retrieval grounding.',
]

const codeModules = [
  {
    title: 'Frontend Code Map',
    items: [
      'src/App.jsx (routes)',
      'src/components/Shell.jsx (layout + sidebar)',
      'src/components/Sidebar.jsx (nav + tooltips)',
      'src/pages/KaiChat.jsx (router, planner, chat send)',
      'src/pages/KlauditAudit.jsx (audit UI)',
      'src/pages/CreativeStudio.jsx (creative UI)',
      'src/pages/PMaxDeepDive.jsx (pmax UI)',
      'src/pages/SerpMonitor.jsx (serp + competitor UI)',
      'src/pages/EnvManager.jsx (env GUI)',
      'src/pages/ArchitectureInfo.jsx (this map)',
    ],
  },
  {
    title: 'Backend Code Map',
    items: [
      'main.py (FastAPI routes + orchestration)',
      'utils/db_manager.py (sqlite + Azure Table state)',
      'ml_reasoning_engine.py (local/azure reasoning)',
      'kai_core/shared/ai_sync.py (shared persona + guardrails)',
      'kai_core/core_logic/UnifiedAuditEngine (audit scoring)',
      'kai_core/core_logic/intelligent_data_mapper.py (schema alignment)',
      'kai_core/core_logic/insight_injector.py (insight enrichment)',
      'kai_core/shared/vector_search.py (retrieval + embeddings)',
      'kai_core/shared/vector_index.py (audit + KB indexing)',
      'scripts/index_knowledge_base.py (bulk KB ingest)',
      'scripts/setup_vector_index.py (index schema)',
      'kai_core/plugins (creative, pmax, serp, competitor)',
      'trends_service.py (seasonality)',
      'kai-llm-shim/app.py (local LLM wrapper)',
    ],
  },
]

const chipStyles = {
  borderColor: 'var(--kai-border)',
  color: 'var(--kai-text)',
  backgroundColor: 'var(--kai-surface-alt)',
  fontSize: '0.78rem',
}

const ArchitectureDiagram = () => (
  <Box
    sx={{
      overflowX: 'auto',
      borderRadius: 3,
      border: '1px solid rgba(148, 163, 184, 0.2)',
      background: 'linear-gradient(180deg, var(--kai-bg) 0%, var(--kai-bg) 100%)',
      p: 2,
    }}
  >
    <Box
      component="svg"
      viewBox="0 0 1600 980"
      sx={{ display: 'block', minWidth: 1600, width: '100%', height: 'auto' }}
    >
      <defs>
        <marker
          id="arrow"
          markerWidth="10"
          markerHeight="10"
          refX="9"
          refY="5"
          orient="auto"
        >
          <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--kai-text-soft)" />
        </marker>
      </defs>

      <rect x="16" y="20" width="300" height="860" rx="18" fill="var(--kai-surface)" stroke="var(--kai-border-strong)" />
      <text x="36" y="55" fill="var(--kai-text)" fontSize="16" fontWeight="700">Entry + Frontend</text>

      <rect x="36" y="85" width="260" height="90" rx="14" fill="var(--kai-surface-muted)" stroke="var(--kai-border)" />
      <text x="52" y="120" fill="var(--kai-text)" fontSize="14" fontWeight="600">User Browser</text>
      <text x="52" y="140" fill="var(--kai-text-soft)" fontSize="12">KAI UI session</text>

      <rect x="36" y="195" width="260" height="140" rx="14" fill="var(--kai-bg)" stroke="var(--kai-border)" />
      <text x="52" y="225" fill="var(--kai-text)" fontSize="14" fontWeight="600">Static Web App</text>
      <text x="52" y="245" fill="var(--kai-text-soft)" fontSize="12">kai-platform-react</text>
      <text x="52" y="265" fill="var(--kai-text-soft)" fontSize="12">React + Vite</text>
      <text x="52" y="285" fill="var(--kai-text-soft)" fontSize="12">PasswordGate</text>
      <text x="52" y="305" fill="var(--kai-text-soft)" fontSize="12">Routes + Sidebar</text>

      <rect x="36" y="355" width="260" height="110" rx="14" fill="var(--kai-bg)" stroke="var(--kai-border)" />
      <text x="52" y="385" fill="var(--kai-text)" fontSize="14" fontWeight="600">UI Modules</text>
      <text x="52" y="405" fill="var(--kai-text-soft)" fontSize="12">Kai Chat</text>
      <text x="52" y="425" fill="var(--kai-text-soft)" fontSize="12">Klaudit Audit</text>
      <text x="52" y="445" fill="var(--kai-text-soft)" fontSize="12">Creative / PMax / SERP</text>

      <rect x="36" y="485" width="260" height="110" rx="14" fill="var(--kai-bg)" stroke="var(--kai-border)" />
      <text x="52" y="515" fill="var(--kai-text)" fontSize="14" fontWeight="600">Frontend Config</text>
      <text x="52" y="535" fill="var(--kai-text-soft)" fontSize="12">API_BASE_URL</text>
      <text x="52" y="555" fill="var(--kai-text-soft)" fontSize="12">Session scoped chat</text>
      <text x="52" y="575" fill="var(--kai-text-soft)" fontSize="12">Playwright tests</text>

      <rect x="340" y="20" width="620" height="860" rx="18" fill="var(--kai-surface)" stroke="var(--kai-border-strong)" />
      <text x="360" y="55" fill="var(--kai-text)" fontSize="16" fontWeight="700">Backend (FastAPI)</text>
      <text x="360" y="75" fill="var(--kai-text-soft)" fontSize="12">kai-platform-backend container app</text>

      <rect x="360" y="95" width="280" height="150" rx="12" fill="var(--kai-bg)" stroke="var(--kai-border)" />
      <text x="376" y="125" fill="var(--kai-text)" fontSize="13" fontWeight="600">Chat + Router</text>
      <text x="376" y="145" fill="var(--kai-text-soft)" fontSize="11">/api/chat/route</text>
      <text x="376" y="162" fill="var(--kai-text-soft)" fontSize="11">/api/chat/send</text>
      <text x="376" y="179" fill="var(--kai-text-soft)" fontSize="11">/api/chat/history</text>
      <text x="376" y="196" fill="var(--kai-text-soft)" fontSize="11">/api/chat/clear</text>
      <text x="376" y="213" fill="var(--kai-text-soft)" fontSize="11">Local to Azure verify</text>

      <rect x="660" y="95" width="280" height="150" rx="12" fill="var(--kai-bg)" stroke="var(--kai-border)" />
      <text x="676" y="125" fill="var(--kai-text)" fontSize="13" fontWeight="600">Planner + Audit</text>
      <text x="676" y="145" fill="var(--kai-text-soft)" fontSize="11">/api/chat/plan-and-run</text>
      <text x="676" y="162" fill="var(--kai-text-soft)" fontSize="11">/api/audit/generate</text>
      <text x="676" y="179" fill="var(--kai-text-soft)" fontSize="11">/api/audit/upload (deprecated)</text>
      <text x="676" y="196" fill="var(--kai-text-soft)" fontSize="11">blob SAS download</text>
      <text x="676" y="213" fill="var(--kai-text-soft)" fontSize="11">/api/data/upload</text>

      <rect x="360" y="265" width="280" height="140" rx="12" fill="var(--kai-bg)" stroke="var(--kai-border)" />
      <text x="376" y="295" fill="var(--kai-text)" fontSize="13" fontWeight="600">Tools</text>
      <text x="376" y="315" fill="var(--kai-text-soft)" fontSize="11">/api/pmax/analyze</text>
      <text x="376" y="332" fill="var(--kai-text-soft)" fontSize="11">/api/serp/check</text>
      <text x="376" y="349" fill="var(--kai-text-soft)" fontSize="11">/api/serp/competitor-signal</text>
      <text x="376" y="366" fill="var(--kai-text-soft)" fontSize="11">/api/creative/generate</text>
      <text x="376" y="383" fill="var(--kai-text-soft)" fontSize="11">/api/intel/diagnose</text>

      <rect x="660" y="265" width="280" height="140" rx="12" fill="var(--kai-bg)" stroke="var(--kai-border)" />
      <text x="676" y="295" fill="var(--kai-text)" fontSize="13" fontWeight="600">Diagnostics + QA</text>
      <text x="676" y="315" fill="var(--kai-text-soft)" fontSize="11">/api/diagnostics/health</text>
      <text x="676" y="332" fill="var(--kai-text-soft)" fontSize="11">/api/diagnostics/sa360/perf-window</text>
      <text x="676" y="349" fill="var(--kai-text-soft)" fontSize="11">/api/qa/accuracy</text>
      <text x="676" y="366" fill="var(--kai-text-soft)" fontSize="11">/api/trends/seasonality</text>
      <text x="676" y="383" fill="var(--kai-text-soft)" fontSize="11">/api/search/web</text>

      <rect x="360" y="425" width="280" height="110" rx="12" fill="var(--kai-bg)" stroke="var(--kai-border)" />
      <text x="376" y="455" fill="var(--kai-text)" fontSize="13" fontWeight="600">Auth + Settings</text>
      <text x="376" y="475" fill="var(--kai-text-soft)" fontSize="11">/api/auth/verify</text>
      <text x="376" y="492" fill="var(--kai-text-soft)" fontSize="11">/api/settings</text>
      <text x="376" y="509" fill="var(--kai-text-soft)" fontSize="11">/api/settings/env</text>
      <text x="376" y="526" fill="var(--kai-text-soft)" fontSize="11">/api/version</text>

      <rect x="660" y="425" width="280" height="110" rx="12" fill="var(--kai-bg)" stroke="var(--kai-border)" />
      <text x="676" y="455" fill="var(--kai-text)" fontSize="13" fontWeight="600">Integrations</text>
      <text x="676" y="475" fill="var(--kai-text-soft)" fontSize="11">/api/sa360/accounts</text>
      <text x="676" y="492" fill="var(--kai-text-soft)" fontSize="11">/api/integrations/sa360/*</text>
      <text x="676" y="509" fill="var(--kai-text-soft)" fontSize="11">/api/integrations/ads/*</text>

      <rect x="360" y="555" width="280" height="120" rx="12" fill="var(--kai-bg)" stroke="var(--kai-border)" />
      <text x="376" y="585" fill="var(--kai-text)" fontSize="13" fontWeight="600">Data + Cache</text>
      <text x="376" y="605" fill="var(--kai-text-soft)" fontSize="11">Blob read/write</text>
      <text x="376" y="622" fill="var(--kai-text-soft)" fontSize="11">SA360 cache</text>
      <text x="376" y="639" fill="var(--kai-text-soft)" fontSize="11">CSV ingest + XLSX</text>
      <text x="376" y="656" fill="var(--kai-text-soft)" fontSize="11">table + sqlite</text>
      <text x="376" y="673" fill="var(--kai-text-soft)" fontSize="11">vector KB index</text>

      <rect x="660" y="555" width="280" height="120" rx="12" fill="var(--kai-bg)" stroke="var(--kai-border)" />
      <text x="676" y="585" fill="var(--kai-text)" fontSize="13" fontWeight="600">LLM Guardrails</text>
      <text x="676" y="605" fill="var(--kai-text-soft)" fontSize="11">JSON router + validate</text>
      <text x="676" y="622" fill="var(--kai-text-soft)" fontSize="11">Low confidence clarifies</text>
      <text x="676" y="639" fill="var(--kai-text-soft)" fontSize="11">No repetition guard</text>
      <text x="676" y="656" fill="var(--kai-text-soft)" fontSize="11">Session history in table</text>

      <rect x="980" y="20" width="600" height="860" rx="18" fill="var(--kai-surface)" stroke="var(--kai-border-strong)" />
      <text x="1000" y="55" fill="var(--kai-text)" fontSize="16" fontWeight="700">External Dependencies</text>

      <rect x="1000" y="80" width="560" height="100" rx="12" fill="var(--kai-bg)" stroke="var(--kai-border)" />
      <text x="1016" y="110" fill="var(--kai-text)" fontSize="13" fontWeight="600">Azure OpenAI</text>
      <text x="1016" y="130" fill="var(--kai-text-soft)" fontSize="11">gpt-4-turbo deployment</text>
      <text x="1016" y="147" fill="var(--kai-text-soft)" fontSize="11">Azure OpenAI endpoint</text>
      <text x="1016" y="164" fill="var(--kai-text-soft)" fontSize="11">Used for verification + fallback</text>

      <rect x="1000" y="192" width="560" height="100" rx="12" fill="var(--kai-bg)" stroke="var(--kai-border)" />
      <text x="1016" y="222" fill="var(--kai-text)" fontSize="13" fontWeight="600">Local LLM (kai-llm)</text>
      <text x="1016" y="242" fill="var(--kai-text-soft)" fontSize="11">llama.cpp + gguf model</text>
      <text x="1016" y="259" fill="var(--kai-text-soft)" fontSize="11">/api/chat + /api/tags</text>
      <text x="1016" y="276" fill="var(--kai-text-soft)" fontSize="11">Internal Container App</text>

      <rect x="1000" y="304" width="560" height="100" rx="12" fill="var(--kai-bg)" stroke="var(--kai-border)" />
      <text x="1016" y="334" fill="var(--kai-text)" fontSize="13" fontWeight="600">Azure AI Search (Vector KB)</text>
      <text x="1016" y="354" fill="var(--kai-text-soft)" fontSize="11">knowledge_base index + embeddings</text>
      <text x="1016" y="371" fill="var(--kai-text-soft)" fontSize="11">Retrieval grounding for concierge</text>
      <text x="1016" y="388" fill="var(--kai-text-soft)" fontSize="11">Hybrid + vector queries</text>

      <rect x="1000" y="416" width="560" height="110" rx="12" fill="var(--kai-bg)" stroke="var(--kai-border)" />
      <text x="1016" y="446" fill="var(--kai-text)" fontSize="13" fontWeight="600">Search Ads 360</text>
      <text x="1016" y="466" fill="var(--kai-text-soft)" fontSize="11">OAuth + developer token</text>
      <text x="1016" y="483" fill="var(--kai-text-soft)" fontSize="11">Account list + report fetch</text>
      <text x="1016" y="500" fill="var(--kai-text-soft)" fontSize="11">Used for planner + diagnostics</text>
      <text x="1016" y="517" fill="var(--kai-text-soft)" fontSize="11">Cache to blob for reuse</text>

      <rect x="1000" y="538" width="560" height="100" rx="12" fill="var(--kai-bg)" stroke="var(--kai-border)" />
      <text x="1016" y="568" fill="var(--kai-text)" fontSize="13" fontWeight="600">Storage + Registry</text>
      <text x="1016" y="588" fill="var(--kai-text-soft)" fontSize="11">Blob: uploads, SA360 cache, reports</text>
      <text x="1016" y="605" fill="var(--kai-text-soft)" fontSize="11">ACR: backend + llm images</text>
      <text x="1016" y="622" fill="var(--kai-text-soft)" fontSize="11">Static Web Apps hosting</text>
      <text x="1016" y="636" fill="var(--kai-text-soft)" fontSize="11">Container Apps runtime</text>

      <rect x="1000" y="650" width="560" height="100" rx="12" fill="var(--kai-bg)" stroke="var(--kai-border)" />
      <text x="1016" y="680" fill="var(--kai-text)" fontSize="13" fontWeight="600">SerpAPI + Bing Engine</text>
      <text x="1016" y="700" fill="var(--kai-text-soft)" fontSize="11">/api/search/web for grounded lookup</text>
      <text x="1016" y="717" fill="var(--kai-text-soft)" fontSize="11">/api/serp/check for URL health</text>
      <text x="1016" y="734" fill="var(--kai-text-soft)" fontSize="11">/api/serp/competitor-signal</text>

      <rect x="1000" y="762" width="560" height="90" rx="12" fill="var(--kai-bg)" stroke="var(--kai-border)" />
      <text x="1016" y="792" fill="var(--kai-text)" fontSize="13" fontWeight="600">Google Trends (pytrends)</text>
      <text x="1016" y="812" fill="var(--kai-text-soft)" fontSize="11">/api/trends/seasonality for demand signal</text>
      <text x="1016" y="829" fill="var(--kai-text-soft)" fontSize="11">SerpAPI fallback if pytrends fails</text>

      <line x1="296" y1="240" x2="340" y2="240" stroke="var(--kai-text-soft)" strokeWidth="2" markerEnd="url(#arrow)" />
      <line x1="296" y1="410" x2="340" y2="410" stroke="var(--kai-text-soft)" strokeWidth="2" markerEnd="url(#arrow)" />
      <line x1="940" y1="130" x2="1000" y2="130" stroke="var(--kai-text-soft)" strokeWidth="2" markerEnd="url(#arrow)" />
      <line x1="940" y1="242" x2="1000" y2="242" stroke="var(--kai-text-soft)" strokeWidth="2" markerEnd="url(#arrow)" />
      <line x1="940" y1="354" x2="1000" y2="354" stroke="var(--kai-text-soft)" strokeWidth="2" markerEnd="url(#arrow)" />
      <line x1="940" y1="471" x2="1000" y2="471" stroke="var(--kai-text-soft)" strokeWidth="2" markerEnd="url(#arrow)" />
      <line x1="940" y1="588" x2="1000" y2="588" stroke="var(--kai-text-soft)" strokeWidth="2" markerEnd="url(#arrow)" />
      <line x1="940" y1="700" x2="1000" y2="700" stroke="var(--kai-text-soft)" strokeWidth="2" markerEnd="url(#arrow)" />
      <line x1="940" y1="807" x2="1000" y2="807" stroke="var(--kai-text-soft)" strokeWidth="2" markerEnd="url(#arrow)" />
    </Box>
  </Box>
)

const SoftwareMapDiagram = () => (
  <Box
    sx={{
      overflowX: 'auto',
      overflowY: 'hidden',
      borderRadius: 3,
      border: '1px solid rgba(148, 163, 184, 0.2)',
      background: 'linear-gradient(180deg, rgba(248, 250, 252, 0.98) 0%, rgba(241, 245, 249, 0.96) 100%)',
      p: 2,
    }}
  >
    <Stack spacing={3}>
      <Box>
        <Typography variant="subtitle1" fontWeight={700} sx={{ color: '#0f172a', mb: 1 }}>
          AI + Decision Flow (Routing)
        </Typography>
        <Box sx={{ minWidth: 2200, width: 'fit-content' }}>
          <Box
            component="svg"
            data-testid="software-map-routing"
            viewBox="0 0 1700 880"
            width={2200}
            height={1140}
            preserveAspectRatio="xMinYMin meet"
            sx={{ display: 'block', maxWidth: 'none' }}
          >
          <defs>
            <marker id="arrow-ui" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto">
              <path d="M 0 0 L 10 5 L 0 10 z" fill="#3b82f6" />
            </marker>
            <marker id="arrow-guard" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto">
              <path d="M 0 0 L 10 5 L 0 10 z" fill="#6366f1" />
            </marker>
            <marker id="arrow-ai" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto">
              <path d="M 0 0 L 10 5 L 0 10 z" fill="#f59e0b" />
            </marker>
          </defs>

          <rect x="20" y="20" width="1660" height="820" rx="22" fill="#f8fafc" stroke="#e2e8f0" />

          <rect x="40" y="100" width="300" height="660" rx="18" fill="#eff6ff" stroke="#e2e8f0" />
          <text x="70" y="135" fill="#0f172a" fontSize="16" fontWeight="700">Frontend + UI</text>

          <rect x="380" y="100" width="420" height="660" rx="18" fill="#ecfdf5" stroke="#e2e8f0" />
          <text x="410" y="135" fill="#0f172a" fontSize="16" fontWeight="700">Backend Core</text>

          <rect x="820" y="100" width="240" height="660" rx="18" fill="#eef2ff" stroke="#e2e8f0" />
          <text x="850" y="135" fill="#0f172a" fontSize="16" fontWeight="700">Guardrails + QA</text>

          <rect x="1100" y="100" width="240" height="660" rx="18" fill="#fff7ed" stroke="#e2e8f0" />
          <text x="1130" y="135" fill="#0f172a" fontSize="16" fontWeight="700">LLM + Vector</text>

          <g>
            <circle cx="190" cy="160" r="30" fill="#ffffff" stroke="#3b82f6" strokeWidth="3" />
            <text x="190" y="165" textAnchor="middle" fill="#1e3a8a" fontSize="12" fontWeight="700">Browser</text>

            <circle cx="190" cy="230" r="30" fill="#ffffff" stroke="#38bdf8" strokeWidth="3" />
            <text x="190" y="235" textAnchor="middle" fill="#0c4a6e" fontSize="12" fontWeight="700">Kai Chat</text>

            <circle cx="190" cy="300" r="30" fill="#ffffff" stroke="#38bdf8" strokeWidth="3" />
            <text x="190" y="305" textAnchor="middle" fill="#0c4a6e" fontSize="12" fontWeight="700">Klaudit</text>

            <circle cx="190" cy="370" r="30" fill="#ffffff" stroke="#38bdf8" strokeWidth="3" />
            <text x="190" y="375" textAnchor="middle" fill="#0c4a6e" fontSize="12" fontWeight="700">Creative</text>

            <circle cx="190" cy="440" r="30" fill="#ffffff" stroke="#38bdf8" strokeWidth="3" />
            <text x="190" y="445" textAnchor="middle" fill="#0c4a6e" fontSize="12" fontWeight="700">PMax</text>

            <circle cx="190" cy="510" r="30" fill="#ffffff" stroke="#38bdf8" strokeWidth="3" />
            <text x="190" y="515" textAnchor="middle" fill="#0c4a6e" fontSize="12" fontWeight="700">SERP</text>

            <circle cx="190" cy="580" r="30" fill="#ffffff" stroke="#3b82f6" strokeWidth="3" />
            <text x="190" y="585" textAnchor="middle" fill="#1e3a8a" fontSize="12" fontWeight="700">Settings</text>

            <circle cx="190" cy="650" r="30" fill="#ffffff" stroke="#3b82f6" strokeWidth="3" />
            <text x="190" y="655" textAnchor="middle" fill="#1e3a8a" fontSize="12" fontWeight="700">Env</text>

            <circle cx="190" cy="720" r="30" fill="#ffffff" stroke="#3b82f6" strokeWidth="3" />
            <text x="190" y="725" textAnchor="middle" fill="#1e3a8a" fontSize="12" fontWeight="700">Info</text>
          </g>

          <g stroke="#3b82f6" strokeWidth="3" fill="none">
            <line x1="300" y1="150" x2="300" y2="730" />
            <line x1="214" y1="160" x2="300" y2="160" />
            <line x1="214" y1="230" x2="300" y2="230" />
            <line x1="214" y1="300" x2="300" y2="300" />
            <line x1="214" y1="370" x2="300" y2="370" />
            <line x1="214" y1="440" x2="300" y2="440" />
            <line x1="214" y1="510" x2="300" y2="510" />
            <line x1="214" y1="580" x2="300" y2="580" />
            <line x1="214" y1="650" x2="300" y2="650" />
            <line x1="214" y1="720" x2="300" y2="720" />
          </g>

          <g>
            <circle cx="590" cy="200" r="34" fill="#ffffff" stroke="#22c55e" strokeWidth="3" />
            <text x="590" y="205" textAnchor="middle" fill="#14532d" fontSize="12" fontWeight="700">API Gateway</text>

            <circle cx="590" cy="320" r="34" fill="#ffffff" stroke="#22c55e" strokeWidth="3" />
            <text x="590" y="325" textAnchor="middle" fill="#14532d" fontSize="12" fontWeight="700">Router</text>
          </g>

          <g stroke="#22c55e" strokeWidth="2.5" fill="none">
            <line x1="590" y1="228" x2="590" y2="292" />
          </g>

          <g stroke="#94a3b8" strokeWidth="2" fill="none">
            <line x1="680" y1="220" x2="680" y2="420" />
            <text x="640" y="210" fill="#334155" fontSize="12" fontWeight="600">Routing bus</text>
          </g>

          <g stroke="#3b82f6" strokeWidth="2.5" fill="none" markerEnd="url(#arrow-ui)">
            <path d="M 300 200 L 562 200" />
            <path d="M 562 320 L 680 320" />
          </g>

          <g>
            <circle cx="940" cy="240" r="32" fill="#ffffff" stroke="#6366f1" strokeWidth="3" />
            <text x="940" y="245" textAnchor="middle" fill="#312e81" fontSize="12" fontWeight="700">Guardrails</text>

            <circle cx="940" cy="360" r="32" fill="#ffffff" stroke="#6366f1" strokeWidth="3" />
            <text x="940" y="365" textAnchor="middle" fill="#312e81" fontSize="12" fontWeight="700">QA</text>
          </g>

          <g stroke="#6366f1" strokeWidth="2.5" fill="none" markerEnd="url(#arrow-guard)">
            <path d="M 680 240 L 914 240" />
            <path d="M 680 360 L 914 360" />
          </g>

          <g>
            <circle cx="1220" cy="220" r="34" fill="#ffffff" stroke="#f59e0b" strokeWidth="3" />
            <text x="1220" y="225" textAnchor="middle" fill="#78350f" fontSize="12" fontWeight="700">Local LLM</text>

            <circle cx="1220" cy="320" r="34" fill="#ffffff" stroke="#f59e0b" strokeWidth="3" />
            <text x="1220" y="325" textAnchor="middle" fill="#78350f" fontSize="12" fontWeight="700">Azure OpenAI</text>

            <circle cx="1220" cy="420" r="34" fill="#ffffff" stroke="#0ea5e9" strokeWidth="3" />
            <text x="1220" y="425" textAnchor="middle" fill="#0c4a6e" fontSize="12" fontWeight="700">Vector KB</text>
          </g>

          <g stroke="#f59e0b" strokeWidth="2.5" fill="none" markerEnd="url(#arrow-ai)">
            <path d="M 680 220 L 1192 220" />
            <path d="M 680 320 L 1192 320" />
            <path d="M 680 420 L 1192 420" />
          </g>

          <g>
            <rect x="60" y="770" width="560" height="60" rx="16" fill="#ffffff" stroke="#e2e8f0" />
            <text x="80" y="795" fill="#0f172a" fontSize="13" fontWeight="700">Legend</text>
            <line x1="90" y1="815" x2="140" y2="815" stroke="#3b82f6" strokeWidth="3" markerEnd="url(#arrow-ui)" />
            <text x="150" y="819" fill="#334155" fontSize="12">UI to API</text>
            <line x1="260" y1="815" x2="310" y2="815" stroke="#6366f1" strokeWidth="3" markerEnd="url(#arrow-guard)" />
            <text x="320" y="819" fill="#334155" fontSize="12">Guardrails + QA</text>
            <line x1="460" y1="815" x2="510" y2="815" stroke="#f59e0b" strokeWidth="3" markerEnd="url(#arrow-ai)" />
            <text x="520" y="819" fill="#334155" fontSize="12">LLM routing</text>
          </g>
          </Box>
        </Box>
      </Box>

      <Box>
        <Typography variant="subtitle1" fontWeight={700} sx={{ color: '#0f172a', mb: 1 }}>
          Data + Integration Flow
        </Typography>
        <Box sx={{ minWidth: 2200, width: 'fit-content' }}>
          <Box
            component="svg"
            data-testid="software-map-data"
            viewBox="0 0 1700 880"
            width={2200}
            height={1140}
            preserveAspectRatio="xMinYMin meet"
            sx={{ display: 'block', maxWidth: 'none' }}
          >
          <defs>
            <marker id="arrow-data" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto">
              <path d="M 0 0 L 10 5 L 0 10 z" fill="#0ea5e9" />
            </marker>
            <marker id="arrow-external" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto">
              <path d="M 0 0 L 10 5 L 0 10 z" fill="#ef4444" />
            </marker>
          </defs>

          <rect x="20" y="20" width="1660" height="820" rx="22" fill="#f8fafc" stroke="#e2e8f0" />

          <rect x="40" y="100" width="520" height="660" rx="18" fill="#ecfdf5" stroke="#e2e8f0" />
          <text x="70" y="135" fill="#0f172a" fontSize="16" fontWeight="700">Backend Core</text>

          <rect x="600" y="100" width="360" height="660" rx="18" fill="#f0f9ff" stroke="#e2e8f0" />
          <text x="630" y="135" fill="#0f172a" fontSize="16" fontWeight="700">Data Stores</text>

          <rect x="1020" y="100" width="620" height="660" rx="18" fill="#fef2f2" stroke="#e2e8f0" />
          <text x="1050" y="135" fill="#0f172a" fontSize="16" fontWeight="700">External Systems</text>

          <g>
            <circle cx="300" cy="220" r="34" fill="#ffffff" stroke="#22c55e" strokeWidth="3" />
            <text x="300" y="225" textAnchor="middle" fill="#14532d" fontSize="12" fontWeight="700">Planner</text>

            <circle cx="300" cy="340" r="34" fill="#ffffff" stroke="#22c55e" strokeWidth="3" />
            <text x="300" y="345" textAnchor="middle" fill="#14532d" fontSize="12" fontWeight="700">Audit Engine</text>

            <circle cx="300" cy="460" r="34" fill="#ffffff" stroke="#22c55e" strokeWidth="3" />
            <text x="300" y="465" textAnchor="middle" fill="#14532d" fontSize="12" fontWeight="700">Tools</text>

            <circle cx="300" cy="600" r="34" fill="#ffffff" stroke="#22c55e" strokeWidth="3" />
            <text x="300" y="605" textAnchor="middle" fill="#14532d" fontSize="12" fontWeight="700">Jobs</text>
          </g>

          <g>
            <circle cx="780" cy="280" r="34" fill="#ffffff" stroke="#0ea5e9" strokeWidth="3" />
            <text x="780" y="285" textAnchor="middle" fill="#0c4a6e" fontSize="12" fontWeight="700">Blob Storage</text>

            <circle cx="780" cy="600" r="34" fill="#ffffff" stroke="#0ea5e9" strokeWidth="3" />
            <text x="780" y="605" textAnchor="middle" fill="#0c4a6e" fontSize="12" fontWeight="700">Queue + Table</text>

            <circle cx="780" cy="720" r="34" fill="#ffffff" stroke="#0ea5e9" strokeWidth="3" />
            <text x="780" y="725" textAnchor="middle" fill="#0c4a6e" fontSize="12" fontWeight="700">Cache</text>
          </g>

          <g stroke="#0ea5e9" strokeWidth="2.5" fill="none" markerEnd="url(#arrow-data)">
            <path d="M 328 220 L 520 220 L 520 280 L 752 280" />
            <path d="M 328 340 L 500 340 L 500 280 L 752 280" />
            <path d="M 328 600 L 752 600" />
            <path d="M 780 628 L 780 692" />
          </g>

          <g stroke="#ef4444" strokeWidth="2.5" fill="none" markerEnd="url(#arrow-external)">
            <path d="M 808 280 L 1180 280" />
            <path d="M 328 460 L 1180 460" />
          </g>

          <g stroke="#ef4444" strokeWidth="3" fill="none">
            <line x1="1180" y1="260" x2="1180" y2="740" />
            <line x1="1180" y1="280" x2="1322" y2="280" />
            <line x1="1180" y1="420" x2="1322" y2="420" />
            <line x1="1180" y1="500" x2="1322" y2="500" />
            <line x1="1180" y1="640" x2="1322" y2="640" />
            <line x1="1180" y1="740" x2="1322" y2="740" />
          </g>
          <g>
            <rect x="1204" y="230" width="120" height="26" rx="6" fill="#ffffff" stroke="#fecaca" />
            <text x="1264" y="249" textAnchor="middle" fill="#dc2626" fontSize="14" fontWeight="700">
              External Bus
            </text>
          </g>

          <g>
            <circle cx="1350" cy="280" r="34" fill="#ffffff" stroke="#ef4444" strokeWidth="3" />
            <text x="1350" y="285" textAnchor="middle" fill="#7f1d1d" fontSize="12" fontWeight="700">SA360 API</text>

            <circle cx="1350" cy="420" r="34" fill="#ffffff" stroke="#ef4444" strokeWidth="3" />
            <text x="1350" y="425" textAnchor="middle" fill="#7f1d1d" fontSize="12" fontWeight="700">SerpAPI</text>

            <circle cx="1350" cy="500" r="34" fill="#ffffff" stroke="#ef4444" strokeWidth="3" />
            <text x="1350" y="505" textAnchor="middle" fill="#7f1d1d" fontSize="12" fontWeight="700">Trends</text>

            <circle cx="1350" cy="640" r="34" fill="#ffffff" stroke="#ef4444" strokeWidth="3" />
            <text x="1350" y="645" textAnchor="middle" fill="#7f1d1d" fontSize="12" fontWeight="700">Ads APIs</text>

            <circle cx="1350" cy="740" r="34" fill="#ffffff" stroke="#ef4444" strokeWidth="3" />
            <text x="1350" y="745" textAnchor="middle" fill="#7f1d1d" fontSize="12" fontWeight="700">Web</text>
          </g>

          <g>
            <rect x="60" y="770" width="520" height="60" rx="16" fill="#ffffff" stroke="#e2e8f0" />
            <text x="80" y="795" fill="#0f172a" fontSize="13" fontWeight="700">Legend</text>
            <line x1="90" y1="815" x2="140" y2="815" stroke="#0ea5e9" strokeWidth="3" markerEnd="url(#arrow-data)" />
            <text x="150" y="819" fill="#334155" fontSize="12">Data flow</text>
            <line x1="260" y1="815" x2="310" y2="815" stroke="#ef4444" strokeWidth="3" markerEnd="url(#arrow-external)" />
            <text x="320" y="819" fill="#334155" fontSize="12">External integrations</text>
          </g>
          </Box>
        </Box>
      </Box>
    </Stack>
  </Box>
)

export default function ArchitectureInfo() {
  return (
    <Container maxWidth="lg">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Box mb={4} display="flex" alignItems="center" gap={2}>
          <InfoOutlined sx={{ fontSize: 36, color: '#22d3ee' }} />
          <Box>
            <Typography variant="h3" fontWeight={800} gutterBottom>
              Technical Architecture
            </Typography>
            <Typography variant="body1" color="text.secondary">
              Full system map of UI, backend logic, integrations, and data paths.
            </Typography>
          </Box>
        </Box>

        <Paper elevation={3} sx={{ p: 3, borderRadius: 3, mb: 4 }}>
          <Stack direction="row" alignItems="center" gap={1} mb={2}>
            <Hub sx={{ color: '#38bdf8' }} />
            <Typography variant="h5" fontWeight={700}>
              System Topology
            </Typography>
          </Stack>
          <Typography variant="body2" color="text.secondary" mb={2}>
            End-to-end flow from browser to services, with external dependencies and data stores.
          </Typography>
          <ArchitectureDiagram />
        </Paper>

        <Paper elevation={3} sx={{ p: 3, borderRadius: 3, mb: 4 }}>
          <Stack direction="row" alignItems="center" gap={1} mb={2}>
            <AccountTree sx={{ color: '#2563eb' }} />
            <Typography variant="h5" fontWeight={700}>
              Software Mapping (System Graph)
            </Typography>
          </Stack>
          <Typography variant="body2" color="text.secondary" mb={2}>
            A graph-style map of Kai's end-to-end system: UI modules, backend services, AI engines,
            data stores, and external integrations with their primary paths.
          </Typography>
          <SoftwareMapDiagram />
        </Paper>

        <Paper elevation={3} sx={{ p: 3, borderRadius: 3, mb: 4 }}>
          <Stack direction="row" alignItems="center" gap={1} mb={2}>
            <Code sx={{ color: '#f97316' }} />
            <Typography variant="h5" fontWeight={700}>
              Code + Logic Map
            </Typography>
          </Stack>
          <Grid container spacing={2}>
            {codeModules.map((group) => (
              <Grid item xs={12} md={6} key={group.title}>
                <Paper variant="outlined" sx={{ p: 2, borderRadius: 2, borderColor: 'rgba(148,163,184,0.3)' }}>
                  <Typography variant="subtitle1" fontWeight={700} mb={1}>
                    {group.title}
                  </Typography>
                  <Stack spacing={0.5}>
                    {group.items.map((item) => (
                      <Typography key={item} variant="body2" color="text.secondary">
                        - {item}
                      </Typography>
                    ))}
                  </Stack>
                </Paper>
              </Grid>
            ))}
          </Grid>
        </Paper>

        <Paper elevation={3} sx={{ p: 3, borderRadius: 3, mb: 4 }}>
          <Stack direction="row" alignItems="center" gap={1} mb={2}>
            <StorageIcon sx={{ color: '#38bdf8' }} />
            <Typography variant="h5" fontWeight={700}>
              Stack + Execution Model
            </Typography>
          </Stack>
          <Typography variant="body2" color="text.secondary" mb={2}>
            Stack used and how requests flow end to end.
          </Typography>
          <Grid container spacing={2} mb={2}>
            {stackModules.map((group) => (
              <Grid item xs={12} md={6} key={group.title}>
                <Paper variant="outlined" sx={{ p: 2, borderRadius: 2, borderColor: 'rgba(148,163,184,0.3)' }}>
                  <Typography variant="subtitle1" fontWeight={700} mb={1}>
                    {group.title}
                  </Typography>
                  <Stack spacing={0.5}>
                    {group.items.map((item) => (
                      <Typography key={item} variant="body2" color="text.secondary">
                        - {item}
                      </Typography>
                    ))}
                  </Stack>
                </Paper>
              </Grid>
            ))}
          </Grid>
          <Divider sx={{ mb: 2 }} />
          <Typography variant="subtitle1" fontWeight={700} mb={1}>
            How It Works
          </Typography>
          <Stack spacing={1}>
            {stackHow.map((item) => (
              <Typography key={item} variant="body2" color="text.secondary">
                - {item}
              </Typography>
            ))}
          </Stack>
        </Paper>

        <Grid container spacing={3} mb={4}>
          {flowSteps.map((flow) => (
            <Grid item xs={12} md={4} key={flow.title}>
              <Paper elevation={2} sx={{ p: 2.5, borderRadius: 3, height: '100%' }}>
                <Typography variant="h6" fontWeight={700} gutterBottom>
                  {flow.title}
                </Typography>
                <Stack spacing={1}>
                  {flow.steps.map((step) => (
                    <Chip key={step} label={step} sx={chipStyles} />
                  ))}
                </Stack>
              </Paper>
            </Grid>
          ))}
        </Grid>

        <Paper elevation={3} sx={{ p: 3, borderRadius: 3, mb: 4 }}>
          <Stack direction="row" alignItems="center" gap={1} mb={2}>
            <Verified sx={{ color: '#34d399' }} />
            <Typography variant="h5" fontWeight={700}>
              System Flow Map (What, Where, Why, How, When)
            </Typography>
          </Stack>
          <Typography variant="body2" color="text.secondary" mb={3}>
            A quick logic flow that explains why the platform routes and executes the way it does.
          </Typography>
          <Stack direction="row" flexWrap="wrap" alignItems="center" gap={1} mb={3}>
            {logicFlow.map((step, index) => (
              <Box key={`${step.key}-strip`} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Chip label={step.label} size="small" sx={chipStyles} />
                {index < logicFlow.length - 1 && (
                  <ArrowForward sx={{ color: 'var(--kai-border-strong)', fontSize: 18 }} />
                )}
              </Box>
            ))}
          </Stack>
          <Grid container spacing={2}>
            {logicFlow.map((step) => (
              <Grid item xs={12} sm={6} lg={4} key={step.key}>
                <Paper
                  elevation={2}
                  sx={{
                    p: 2,
                    borderRadius: 3,
                    height: '100%',
                    background: 'linear-gradient(180deg, rgba(248,250,252,0.98) 0%, rgba(241,245,249,0.96) 100%)',
                    border: '1px solid var(--kai-border)',
                  }}
                >
                  <Stack direction="row" alignItems="center" gap={1} mb={1}>
                    {step.icon}
                    <Typography variant="subtitle1" fontWeight={700}>
                      {step.label}
                    </Typography>
                  </Stack>
                  <Typography variant="body2" color="text.secondary" mb={1}>
                    {step.question}
                  </Typography>
                  <Typography variant="caption" color="text.secondary" display="block" mb={1.5}>
                    {step.summary}
                  </Typography>
                  <Stack spacing={0.5}>
                    {step.details.map((detail) => (
                      <Typography key={detail} variant="caption" color="text.secondary">
                        - {detail}
                      </Typography>
                    ))}
                  </Stack>
                </Paper>
              </Grid>
            ))}
          </Grid>
        </Paper>

        <Paper elevation={3} sx={{ p: 3, borderRadius: 3, mb: 4 }}>
          <Stack direction="row" alignItems="center" gap={1} mb={2}>
            <Code sx={{ color: '#f59e0b' }} />
            <Typography variant="h5" fontWeight={700}>
              API Surface (All Routes)
            </Typography>
          </Stack>
          <Grid container spacing={2}>
            {apiSurface.map((group) => (
              <Grid item xs={12} md={6} key={group.title}>
                <Paper variant="outlined" sx={{ p: 2, borderRadius: 2, borderColor: 'rgba(148,163,184,0.3)' }}>
                  <Typography variant="subtitle1" fontWeight={700} mb={1}>
                    {group.title}
                  </Typography>
                  <Stack direction="row" flexWrap="wrap" gap={1}>
                    {group.items.map((item) => (
                      <Chip key={item} label={item} size="small" sx={chipStyles} />
                    ))}
                  </Stack>
                </Paper>
              </Grid>
            ))}
          </Grid>
        </Paper>

        <Grid container spacing={3} mb={4}>
          <Grid item xs={12} md={6}>
            <Paper elevation={3} sx={{ p: 3, borderRadius: 3, height: '100%' }}>
              <Stack direction="row" alignItems="center" gap={1} mb={2}>
                <Lan sx={{ color: '#a78bfa' }} />
                <Typography variant="h5" fontWeight={700}>
                  External Dependencies
                </Typography>
              </Stack>
              <Stack spacing={2}>
                {dependencyGroups.map((group) => (
                  <Box key={group.title}>
                    <Stack direction="row" alignItems="center" gap={1} mb={1}>
                      {group.icon}
                      <Typography variant="subtitle1" fontWeight={700}>
                        {group.title}
                      </Typography>
                    </Stack>
                    <Stack spacing={1}>
                      {group.items.map((item) => (
                        <Typography key={item} variant="body2" color="text.secondary">
                          - {item}
                        </Typography>
                      ))}
                    </Stack>
                  </Box>
                ))}
              </Stack>
            </Paper>
          </Grid>

          <Grid item xs={12} md={6}>
            <Paper elevation={3} sx={{ p: 3, borderRadius: 3, height: '100%' }}>
              <Stack direction="row" alignItems="center" gap={1} mb={2}>
                <Verified sx={{ color: '#22d3ee' }} />
                <Typography variant="h5" fontWeight={700}>
                  Configuration Sources
                </Typography>
              </Stack>
              <Typography variant="body2" color="text.secondary" mb={2}>
                Feature flags and env bindings that control routing, data, and integrations.
              </Typography>
              <Stack spacing={2}>
                {configGroups.map((group) => (
                  <Box key={group.title}>
                    <Typography variant="subtitle1" fontWeight={700} mb={1}>
                      {group.title}
                    </Typography>
                    <Stack direction="row" flexWrap="wrap" gap={1}>
                      {group.items.map((item) => (
                        <Chip key={item} label={item} size="small" sx={chipStyles} />
                      ))}
                    </Stack>
                  </Box>
                ))}
              </Stack>
            </Paper>
          </Grid>
        </Grid>

        <Paper elevation={3} sx={{ p: 3, borderRadius: 3 }}>
          <Typography variant="h6" fontWeight={700} gutterBottom>
            Data Paths and Sources
          </Typography>
          <Divider sx={{ mb: 2 }} />
          <Stack spacing={1.5}>
            <Typography variant="body2" color="text.secondary">
              - SA360 data to cached blob to planner comparisons and audit generation.
            </Typography>
            <Typography variant="body2" color="text.secondary">
              - QA accuracy compares raw SA360 totals vs aggregated performance (single + multi-account).
            </Typography>
            <Typography variant="body2" color="text.secondary">
              - CSV uploads to blob storage to audit engine reads to XLSX report output.
            </Typography>
            <Typography variant="body2" color="text.secondary">
              - Chat history in Azure Table (session scoped); sqlite fallback for settings.
            </Typography>
            <Typography variant="body2" color="text.secondary">
              - Web search to SerpAPI (bing engine) to grounded snippets for chat responses.
            </Typography>
            <Typography variant="body2" color="text.secondary">
              - Seasonality to Google Trends (pytrends), with SerpAPI fallback when needed.
            </Typography>
            <Typography variant="body2" color="text.secondary">
              - Long-running tasks enqueue to the worker via the job queue; results land in job storage.
            </Typography>
            <Typography variant="body2" color="text.secondary">
              - LLM routing to local first to Azure verification to response guardrails.
            </Typography>
          </Stack>
        </Paper>
      </motion.div>
    </Container>
  )
}
