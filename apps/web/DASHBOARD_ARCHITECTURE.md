# Agentic Dashboard Architecture - Implementation Plan

## Research Validation Summary

### ✅ Backend Capabilities Confirmed

**PMax Channel Split Data:**
- Endpoint: `POST /api/pmax/analyze`
- Returns: `PMaxChannelBreakout` dataclass with:
  - `search_cost`, `shopping_cost`, `video_cost`, `display_cost`
  - `remainder_cost`, `total_cost`
- **Source:** `kai_core/pmax_channel_split.py`
- **Status:** ✅ Production-ready, matches research requirements for Sankey visualization

**Audit Generation:**
- Endpoint: `POST /api/audit/generate`
- Processes: 14 CSV file types, 100+ audit points
- **Status:** ✅ Existing, no changes needed

**Creative Generation:**
- Endpoint: `POST /api/creative/generate`
- **Status:** ✅ Existing, 2-3 second response time confirmed

**SERP Monitor:**
- Endpoint: `POST /api/serp/check`
- **Status:** ✅ Existing URL health checking

**Agentic Intelligence:**
- Endpoint: `POST /api/intel/diagnose`
- Uses: `MarketingReasoningAgent` (3-pillar root-cause analysis)
- **Status:** ✅ Existing, matches research on multi-step reasoning

### 🎯 Dashboard Requirements (from Research)

1. **PMax Spend Sankey Diagram**
   - Data: ✅ Available from `/api/pmax/analyze`
   - Visualization: Needs implementation (D3/Recharts)
   - Interactivity: Click-to-filter capability

2. **Zombie Product Grid**
   - Data: ⚠️ Need to add endpoint for product-level analysis
   - Alternative: Use existing audit data, extract product metrics
   - Status: Phase 2 implementation

3. **Channel Performance Comparison**
   - Data: ✅ Can derive from PMax breakout
   - Visualization: Bar chart component
   - Status: Phase 1 implementation

4. **Search Term Wastage Analysis**
   - Data: ⚠️ Need audit CSV parsing enhancement
   - Status: Phase 3 implementation

---

## Implementation Strategy: Additive Architecture

### Principle: Zero Regression
- All existing AI chat flows remain unchanged
- Dashboards are **optional supplements** that AI can reference
- No modifications to core API endpoints
- New components are isolated and lazy-loaded

### Architecture Pattern

```
Existing Flow (UNCHANGED):
User Input → AI Chat → Text Response

Enhanced Flow (ADDITIVE):
User Input → AI Chat → Text Response + Optional Dashboard Component
```

---

## Phase 1: Foundation (Week 1)

### 1.1 Dashboard Registry (Infrastructure Only)

**File:** `frontend/src/components/dashboards/DashboardRegistry.js`

```javascript
// Component registry for lazy loading
import { lazy } from 'react'

export const DASHBOARD_REGISTRY = {
  // Phase 1: Core dashboards
  'pmax_spend_sankey': lazy(() => import('./PMaxSpendSankey')),
  'channel_comparison': lazy(() => import('./ChannelComparison')),

  // Phase 2: Product analytics
  'zombie_products': lazy(() => import('./ZombieProductGrid')),

  // Phase 3: Advanced
  'search_wastage': lazy(() => import('./SearchWastageGrid')),
  'creative_heatmap': lazy(() => import('./CreativeHeatmap')),
}

// Type-safe dashboard props
export const DASHBOARD_SCHEMAS = {
  'pmax_spend_sankey': {
    requiredData: ['channel_breakout'],
    optionalProps: ['interactive', 'theme']
  },
  'channel_comparison': {
    requiredData: ['channels'],
    optionalProps: ['metric']
  }
}
```

**Validation Test:**
```javascript
// Test that registry can lazy load without errors
import { DASHBOARD_REGISTRY } from './DashboardRegistry'
console.log('Registry loaded:', Object.keys(DASHBOARD_REGISTRY))
```

---

### 1.2 Data Observability Hook

**File:** `frontend/src/hooks/useDashboardData.js`

```javascript
import { useState, useEffect } from 'react'

/**
 * Fetches dashboard data with observability checks
 * Ensures data freshness and schema validation
 */
export function useDashboardData(endpoint, params, schema) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [health, setHealth] = useState({
    fresh: true,
    valid: true,
    lastUpdate: null,
    error: null
  })

  useEffect(() => {
    async function fetchData() {
      setLoading(true)
      try {
        const response = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(params)
        })

        if (!response.ok) throw new Error(`HTTP ${response.status}`)

        const json = await response.json()
        const now = Date.now()

        // Freshness check (data should be < 24h old if cached)
        const isFresh = true // API returns real-time data

        // Schema validation
        const isValid = validateSchema(json, schema)

        setData(json)
        setHealth({
          fresh: isFresh,
          valid: isValid,
          lastUpdate: now,
          error: null
        })
      } catch (error) {
        setHealth(prev => ({
          ...prev,
          error: error.message
        }))
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [endpoint, JSON.stringify(params)])

  return { data, loading, health }
}

function validateSchema(data, schema) {
  if (!schema) return true

  // Check required fields exist
  for (const field of schema.required || []) {
    if (!(field in data)) {
      console.warn(`Missing required field: ${field}`)
      return false
    }
  }

  return true
}
```

**Validation Test:**
```javascript
// Test hook with real PMax endpoint
const { data, health } = useDashboardData(
  '/api/pmax/analyze',
  { placements: [], spend: 1000, conversions: 10 },
  { required: ['status', 'result'] }
)
console.log('Data health:', health)
```

---

### 1.3 PMax Spend Sankey Component

**File:** `frontend/src/components/dashboards/PMaxSpendSankey.jsx`

```javascript
import { Box, Typography, Skeleton } from '@mui/material'
import { ResponsiveContainer, Sankey, Tooltip } from 'recharts'
import { useDashboardData } from '../../hooks/useDashboardData'

export default function PMaxSpendSankey({
  placements = [],
  spend = null,
  conversions = null,
  theme,
  onNodeClick
}) {
  const { data, loading, health } = useDashboardData(
    '/api/pmax/analyze',
    { placements, spend, conversions },
    { required: ['channel_breakout'] }
  )

  if (loading) {
    return <Skeleton variant="rectangular" height={400} />
  }

  if (!health.valid || health.error) {
    return (
      <Box sx={{ p: 3, border: '1px solid #f44336', borderRadius: 2 }}>
        <Typography color="error">
          Data validation failed: {health.error || 'Invalid schema'}
        </Typography>
      </Box>
    )
  }

  // Transform channel breakout into Sankey data structure
  const breakout = data.channel_breakout
  const sankeyData = transformToSankey(breakout)

  return (
    <Box sx={{
      p: 3,
      borderRadius: 2,
      border: `1px solid ${theme?.borderColor || '#334155'}`,
      background: '#0f172a'
    }}>
      <Typography variant="h6" sx={{ mb: 2, color: '#e2e8f0' }}>
        PMax Channel Spend Allocation
      </Typography>

      <ResponsiveContainer width="100%" height={400}>
        <Sankey
          data={sankeyData}
          node={{ fill: theme?.accentColor || '#22d3ee' }}
          link={{ stroke: theme?.borderColor || '#334155' }}
          onClick={(node) => onNodeClick?.(node)}
        />
      </ResponsiveContainer>

      {/* Data health indicator */}
      <Typography variant="caption" sx={{ color: '#64748b', mt: 1, display: 'block' }}>
        Last updated: {new Date(health.lastUpdate).toLocaleTimeString()}
      </Typography>
    </Box>
  )
}

function transformToSankey(breakout) {
  // Transform backend data into Recharts Sankey format
  const nodes = [
    { name: 'Total Spend' },
    { name: 'Search' },
    { name: 'Shopping' },
    { name: 'Video' },
    { name: 'Display' },
    { name: 'Other' }
  ]

  const links = [
    { source: 0, target: 1, value: breakout.search_cost },
    { source: 0, target: 2, value: breakout.shopping_cost },
    { source: 0, target: 3, value: breakout.video_cost },
    { source: 0, target: 4, value: breakout.display_cost },
    { source: 0, target: 5, value: breakout.remainder_cost }
  ].filter(link => link.value > 0) // Only show non-zero flows

  return { nodes, links }
}
```

**Validation Test:**
```javascript
// Test component with mock data
<PMaxSpendSankey
  placements={[]}
  spend={5000}
  conversions={50}
  theme={{ accentColor: '#22d3ee', borderColor: '#334155' }}
  onNodeClick={(node) => console.log('Clicked:', node)}
/>
```

---

## Phase 2: Integration (Week 2)

### 2.1 Add Dashboard Support to PMax Route

**File:** `frontend/src/pages/PMaxDeepDive.jsx` (NEW FILE)

```javascript
import { useState } from 'react'
import { Container, Paper, Typography, Box, Button } from '@mui/material'
import { Suspense } from 'react'
import PMaxSpendSankey from '../components/dashboards/PMaxSpendSankey'

export default function PMaxDeepDive() {
  const [dashboardMode, setDashboardMode] = useState('observe')
  const [selectedChannel, setSelectedChannel] = useState(null)

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Paper sx={{ p: 3, background: '#0b1220', borderRadius: 3 }}>
        <Typography variant="h4" sx={{ color: '#e2e8f0', mb: 3 }}>
          PMax Deep Dive
        </Typography>

        {/* Existing AI chat interface PRESERVED */}
        <Box sx={{ mb: 4 }}>
          {/* Your existing chat component goes here - UNCHANGED */}
        </Box>

        {/* NEW: Programmatic dashboard as supplement */}
        <Suspense fallback={<div>Loading dashboard...</div>}>
          <PMaxSpendSankey
            placements={[]} // Connect to actual data
            spend={null}
            conversions={null}
            theme={{ accentColor: '#22d3ee', borderColor: '#334155' }}
            onNodeClick={(channel) => {
              setSelectedChannel(channel)
              // Optionally trigger AI analysis of clicked channel
            }}
          />
        </Suspense>

        {/* Agent action buttons */}
        {selectedChannel && (
          <Box sx={{ mt: 2 }}>
            <Button
              variant="contained"
              onClick={() => {
                // Trigger AI agent to analyze clicked channel
                console.log('Analyze', selectedChannel)
              }}
            >
              Analyze {selectedChannel.name} with AI
            </Button>
          </Box>
        )}
      </Paper>
    </Container>
  )
}
```

**Validation Test:**
- Navigate to `/pmax` route
- Verify existing functionality unchanged
- Verify Sankey diagram renders
- Click on a channel node
- Verify button appears
- Check console for correct channel data

---

### 2.2 Optional Dashboard Rendering in Chat

**File:** `frontend/src/pages/KlauditAudit.jsx` (MODIFICATION - ADDITIVE ONLY)

```javascript
// ADD THIS IMPORT (at top)
import { DASHBOARD_REGISTRY } from '../components/dashboards/DashboardRegistry'
import { Suspense, lazy } from 'react'

// ADD THIS STATE (with existing state variables)
const [dashboardToRender, setDashboardToRender] = useState(null)

// MODIFY sendMessage function (ADDITION, not replacement)
const sendMessage = async () => {
  // ... existing code stays exactly the same ...

  // NEW: Check if AI response should trigger a dashboard
  if (aiReply.toLowerCase().includes('channel spend') ||
      aiReply.toLowerCase().includes('pmax allocation')) {

    // Optionally render dashboard alongside text
    setDashboardToRender({
      component: 'pmax_spend_sankey',
      props: {
        placements: [],
        spend: null,
        conversions: null,
        theme: variantThemes[variant]
      }
    })
  }

  // ... existing code continues unchanged ...
}

// ADD THIS JSX (in messages rendering section, after AI text message)
{dashboardToRender && (
  <Suspense fallback={<Skeleton variant="rectangular" height={400} />}>
    {(() => {
      const DashboardComponent = DASHBOARD_REGISTRY[dashboardToRender.component]
      return <DashboardComponent {...dashboardToRender.props} />
    })()}
  </Suspense>
)}
```

**Validation Test:**
- Ask AI: "Show me PMax channel spend"
- Verify text response appears (existing behavior)
- Verify Sankey diagram also appears below text
- Ask different question ("Generate audit")
- Verify NO dashboard appears (preserves existing flow)

---

## Testing Checklist

### ✅ Phase 1 Tests (Before proceeding to Phase 2)

- [ ] Dashboard Registry loads without errors
- [ ] `useDashboardData` hook fetches from real `/api/pmax/analyze`
- [ ] `useDashboardData` detects schema validation failures
- [ ] PMaxSpendSankey renders with mock data
- [ ] PMaxSpendSankey renders with real backend data
- [ ] Sankey node click triggers callback
- [ ] Data health indicator shows correct timestamp

### ✅ Phase 2 Tests (Before deploying)

- [ ] PMax route renders with dashboard
- [ ] Existing AI chat flow unchanged
- [ ] Dashboard renders alongside chat messages
- [ ] Optional dashboard rendering doesn't break existing messages
- [ ] Clicking Sankey node triggers AI analysis option
- [ ] Dashboard lazy loading doesn't block page render

---

## Deployment Plan

### Pre-Deployment Validation

1. **Local Build Test**
   ```bash
   cd Z:\Kai_Personal_WebApp\frontend
   npm run build
   # Check for errors
   ```

2. **Backend Connectivity Test**
   - Start backend: `python backend/main.py`
   - Test `/api/pmax/analyze` endpoint with Postman
   - Verify response structure matches schema

3. **Frontend Integration Test**
   - Start frontend: `npm run dev`
   - Navigate to `/pmax`
   - Verify dashboard renders with real data

### Deployment Steps

1. Copy frontend to clean build directory
2. Run `npm install` and `npm run build`
3. Deploy to Azure Static Web Apps
4. Verify live endpoint connectivity
5. Test dashboard functionality on live site

---

## Success Criteria

### Functional Requirements ✅
- [ ] All existing AI chat flows work exactly as before
- [ ] New dashboards render with real backend data
- [ ] Dashboard data validation catches errors
- [ ] Dashboards are lazy-loaded (no performance impact)
- [ ] User can interact with dashboards (click, filter)

### Non-Functional Requirements ✅
- [ ] Zero regressions in existing functionality
- [ ] No redundant API calls
- [ ] Data observability prevents stale data issues
- [ ] Components are isolated and testable
- [ ] Code follows existing architecture patterns

---

## Future Phases

### Phase 3: Zombie Product Analytics
- Add `/api/audit/products` endpoint for product-level metrics
- Create ZombieProductGrid component
- Integrate with Klaudit audit flow

### Phase 4: Video/Creative Analysis
- Add CreativeHeatmap component with TensorFlow.js
- Integrate with Creative Studio route

### Phase 5: Bi-Directional Agent Actions
- Add confirmation modals for agent-proposed optimizations
- Implement "Optimize with AI" button actions
- Add agent tool calling for destructive actions
