/**
 * Dashboard Infrastructure Test Page
 * Tests DashboardRegistry, useDashboardData, and dashboard components in isolation
 *
 * Usage: Navigate to /dashboard-test to verify:
 * - Registry lazy loading works
 * - Schema validation catches bad data
 * - Health checks detect errors
 * - Components render with mock data
 */
import { Suspense, useState } from 'react'
import { Box, Typography, Button, Paper, Divider, Alert, CircularProgress } from '@mui/material'
import { CheckCircle, Error, Warning } from '@mui/icons-material'
import { getDashboard, getDashboardSchema, validateDashboardData } from '../DashboardRegistry'
import { useDashboardData, isDataUsable } from '../../../hooks/useDashboardData'

export default function DashboardTest() {
  const [testResults, setTestResults] = useState([])

  // Test 1: Registry can load dashboard components
  const testRegistryLoading = () => {
    const results = []

    try {
      const SankeyComponent = getDashboard('pmax_spend_sankey')
      results.push({ test: 'Load Sankey Component', status: 'pass', message: 'Component loaded successfully' })
    } catch (err) {
      results.push({ test: 'Load Sankey Component', status: 'fail', message: err.message })
    }

    try {
      const ComparisonComponent = getDashboard('channel_comparison')
      results.push({ test: 'Load Comparison Component', status: 'pass', message: 'Component loaded successfully' })
    } catch (err) {
      results.push({ test: 'Load Comparison Component', status: 'fail', message: err.message })
    }

    try {
      getDashboard('nonexistent_dashboard')
      results.push({ test: 'Invalid Dashboard Key', status: 'fail', message: 'Should have thrown error' })
    } catch (err) {
      results.push({ test: 'Invalid Dashboard Key', status: 'pass', message: 'Correctly rejected invalid key' })
    }

    setTestResults(prev => [...prev, ...results])
  }

  // Test 2: Schema validation works correctly
  const testSchemaValidation = () => {
    const results = []

    // Valid data for Sankey
    const validSankeyData = {
      channel_breakout: {
        search_cost: 100,
        shopping_cost: 200,
        video_cost: 50,
        display_cost: 75,
        remainder_cost: 25,
        total_cost: 450
      }
    }

    const validation1 = validateDashboardData('pmax_spend_sankey', validSankeyData)
    results.push({
      test: 'Valid Sankey Data',
      status: validation1.valid ? 'pass' : 'fail',
      message: validation1.valid ? 'Validation passed' : `Errors: ${validation1.errors.join(', ')}`
    })

    // Invalid data (missing required fields)
    const invalidSankeyData = {
      channel_breakout: {
        search_cost: 100
        // Missing other required fields
      }
    }

    const validation2 = validateDashboardData('pmax_spend_sankey', invalidSankeyData)
    results.push({
      test: 'Invalid Sankey Data (Missing Fields)',
      status: !validation2.valid ? 'pass' : 'fail',
      message: !validation2.valid ? `Correctly caught errors: ${validation2.errors.join(', ')}` : 'Should have failed validation'
    })

    // Missing channel_breakout entirely
    const missingData = {}

    const validation3 = validateDashboardData('pmax_spend_sankey', missingData)
    results.push({
      test: 'Missing Required Data Key',
      status: !validation3.valid ? 'pass' : 'fail',
      message: !validation3.valid ? 'Correctly detected missing data' : 'Should have failed validation'
    })

    setTestResults(prev => [...prev, ...results])
  }

  // Test 3: useDashboardData hook integration
  const TestHookComponent = ({ endpoint, params, dashboardKey }) => {
    const { data, loading, health } = useDashboardData(endpoint, params, dashboardKey)

    if (loading) {
      return <Alert severity="info">Hook is fetching data...</Alert>
    }

    if (!isDataUsable(health)) {
      return (
        <Alert severity="error">
          Health check failed: {health.error || health.warnings.join(', ')}
        </Alert>
      )
    }

    return (
      <Alert severity="success">
        Hook fetched data successfully. Health: {health.fresh ? 'Fresh' : 'Stale'}, Valid: {health.valid ? 'Yes' : 'No'}
      </Alert>
    )
  }

  // Test 4: Actual component rendering
  const [showComponents, setShowComponents] = useState(false)

  const mockPMaxData = {
    placements: [],
    spend: null,
    conversions: null
  }

  const mockTheme = {
    accentColor: '#22d3ee',
    borderColor: 'var(--kai-border-strong)'
  }

  return (
    <Box sx={{ p: 4, maxWidth: 1200, mx: 'auto' }}>
      <Typography variant="h4" sx={{ mb: 3, color: 'var(--kai-text)', fontWeight: 700 }}>
        Dashboard Infrastructure Test Suite
      </Typography>
      <Typography variant="body1" sx={{ mb: 4, color: 'var(--kai-text-soft)' }}>
        This page tests the dashboard registry, schema validation, and data hooks in isolation
        before integrating into production pages.
      </Typography>

      {/* Test Controls */}
      <Paper sx={{ p: 3, mb: 3, background: 'var(--kai-surface-alt)', borderRadius: 2 }}>
        <Typography variant="h6" sx={{ mb: 2, color: '#cbd5e1' }}>
          Run Tests
        </Typography>
        <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
          <Button
            variant="contained"
            onClick={testRegistryLoading}
            sx={{ background: '#3b82f6', '&:hover': { background: '#2563eb' } }}
          >
            Test Registry Loading
          </Button>
          <Button
            variant="contained"
            onClick={testSchemaValidation}
            sx={{ background: '#10b981', '&:hover': { background: '#059669' } }}
          >
            Test Schema Validation
          </Button>
          <Button
            variant="contained"
            onClick={() => setShowComponents(!showComponents)}
            sx={{ background: '#8b5cf6', '&:hover': { background: '#7c3aed' } }}
          >
            {showComponents ? 'Hide' : 'Show'} Component Rendering
          </Button>
          <Button
            variant="outlined"
            onClick={() => setTestResults([])}
            sx={{ borderColor: '#64748b', color: '#64748b' }}
          >
            Clear Results
          </Button>
        </Box>
      </Paper>

      {/* Test Results */}
      {testResults.length > 0 && (
        <Paper sx={{ p: 3, mb: 3, background: 'var(--kai-surface-alt)', borderRadius: 2 }}>
          <Typography variant="h6" sx={{ mb: 2, color: '#cbd5e1' }}>
            Test Results ({testResults.length})
          </Typography>
          {testResults.map((result, idx) => (
            <Box key={idx} sx={{ mb: 2 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                {result.status === 'pass' ? (
                  <CheckCircle sx={{ color: '#10b981' }} />
                ) : result.status === 'fail' ? (
                  <Error sx={{ color: '#f44336' }} />
                ) : (
                  <Warning sx={{ color: '#f59e0b' }} />
                )}
                <Typography variant="body2" sx={{ color: 'var(--kai-text)', fontWeight: 600 }}>
                  {result.test}
                </Typography>
              </Box>
              <Typography variant="caption" sx={{ color: 'var(--kai-text-soft)', ml: 4 }}>
                {result.message}
              </Typography>
              {idx < testResults.length - 1 && <Divider sx={{ mt: 2, borderColor: 'var(--kai-border-strong)' }} />}
            </Box>
          ))}
        </Paper>
      )}

      {/* Hook Integration Test */}
      <Paper sx={{ p: 3, mb: 3, background: 'var(--kai-surface-alt)', borderRadius: 2 }}>
        <Typography variant="h6" sx={{ mb: 2, color: '#cbd5e1' }}>
          useDashboardData Hook Test
        </Typography>
        <Typography variant="body2" sx={{ mb: 2, color: 'var(--kai-text-soft)' }}>
          Testing live API call to /api/pmax/analyze with schema validation
        </Typography>
        <Suspense fallback={<CircularProgress />}>
          <TestHookComponent
            endpoint="/api/pmax/analyze"
            params={mockPMaxData}
            dashboardKey="pmax_spend_sankey"
          />
        </Suspense>
      </Paper>

      {/* Component Rendering Test */}
      {showComponents && (
        <>
          <Paper sx={{ p: 3, mb: 3, background: 'var(--kai-surface-alt)', borderRadius: 2 }}>
            <Typography variant="h6" sx={{ mb: 2, color: '#cbd5e1' }}>
              PMax Spend Sankey Component
            </Typography>
            <Suspense fallback={<CircularProgress />}>
              <DashboardRenderer
                dashboardKey="pmax_spend_sankey"
                props={{
                  placements: mockPMaxData.placements,
                  spend: mockPMaxData.spend,
                  conversions: mockPMaxData.conversions,
                  theme: mockTheme,
                  interactive: true
                }}
              />
            </Suspense>
          </Paper>

          <Paper sx={{ p: 3, mb: 3, background: 'var(--kai-surface-alt)', borderRadius: 2 }}>
            <Typography variant="h6" sx={{ mb: 2, color: '#cbd5e1' }}>
              Channel Comparison Component
            </Typography>
            <Suspense fallback={<CircularProgress />}>
              <DashboardRenderer
                dashboardKey="channel_comparison"
                props={{
                  placements: mockPMaxData.placements,
                  spend: mockPMaxData.spend,
                  conversions: mockPMaxData.conversions,
                  theme: mockTheme,
                  defaultMetric: 'cost'
                }}
              />
            </Suspense>
          </Paper>
        </>
      )}

      {/* Schema Reference */}
      <Paper sx={{ p: 3, background: 'var(--kai-surface-alt)', borderRadius: 2 }}>
        <Typography variant="h6" sx={{ mb: 2, color: '#cbd5e1' }}>
          Dashboard Schemas
        </Typography>

        <Typography variant="subtitle2" sx={{ color: '#cbd5e1', mb: 1 }}>
          pmax_spend_sankey
        </Typography>
        <Box sx={{ mb: 2, p: 2, background: 'var(--kai-bg)', borderRadius: 1, fontFamily: 'monospace', fontSize: '0.875rem' }}>
          <pre style={{ margin: 0, color: 'var(--kai-text-soft)' }}>
            {JSON.stringify(getDashboardSchema('pmax_spend_sankey'), null, 2)}
          </pre>
        </Box>

        <Typography variant="subtitle2" sx={{ color: '#cbd5e1', mb: 1 }}>
          channel_comparison
        </Typography>
        <Box sx={{ p: 2, background: 'var(--kai-bg)', borderRadius: 1, fontFamily: 'monospace', fontSize: '0.875rem' }}>
          <pre style={{ margin: 0, color: 'var(--kai-text-soft)' }}>
            {JSON.stringify(getDashboardSchema('channel_comparison'), null, 2)}
          </pre>
        </Box>
      </Paper>
    </Box>
  )
}

/**
 * Helper component to render dashboard from registry
 */
function DashboardRenderer({ dashboardKey, props }) {
  const DashboardComponent = getDashboard(dashboardKey)

  return (
    <Suspense fallback={<CircularProgress />}>
      <DashboardComponent {...props} />
    </Suspense>
  )
}

