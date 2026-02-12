/**
 * Dashboard Component Registry
 * Lazy-loaded components for programmatic dashboards that supplement AI interfaces
 *
 * Architecture: Additive enhancement to existing AI chat flows
 * - Does NOT replace existing functionality
 * - Provides structured data visualizations AI can reference
 * - Supports bi-directional agent workflows
 */
import { lazy } from 'react'

/**
 * Component Registry
 * Maps semantic keys to lazy-loaded dashboard components
 */
export const DASHBOARD_REGISTRY = {
  // Phase 1: Core PMax Analytics
  'pmax_spend_sankey': lazy(() => import('./PMaxSpendSankey')),
  'channel_comparison': lazy(() => import('./ChannelComparison')),

  // Phase 2: Product Analytics (Future)
  // 'zombie_products': lazy(() => import('./ZombieProductGrid')),

  // Phase 3: Advanced Analytics (Future)
  // 'search_wastage': lazy(() => import('./SearchWastageGrid')),
  // 'creative_heatmap': lazy(() => import('./CreativeHeatmap')),
}

/**
 * Dashboard Schemas
 * Type-safe data requirements for each dashboard
 * Used for validation and observability
 */
export const DASHBOARD_SCHEMAS = {
  'pmax_spend_sankey': {
    requiredData: ['channel_breakout'],
    requiredFields: ['search_cost', 'shopping_cost', 'video_cost', 'display_cost', 'total_cost'],
    optionalProps: ['interactive', 'theme', 'onNodeClick']
  },
  'channel_comparison': {
    requiredData: ['channels'],
    requiredFields: ['name', 'cost', 'conversions'],
    optionalProps: ['metric', 'theme']
  }
}

/**
 * Helper: Get a dashboard component by key
 */
export function getDashboard(key) {
  if (!isDashboardAvailable(key)) {
    throw new Error(`Dashboard not found: ${key}`)
  }
  return DASHBOARD_REGISTRY[key]
}

/**
 * Helper: Check if a dashboard is registered
 */
export function isDashboardAvailable(key) {
  return key in DASHBOARD_REGISTRY
}

/**
 * Helper: Get schema for validation
 */
export function getDashboardSchema(key) {
  return DASHBOARD_SCHEMAS[key] || null
}

/**
 * Helper: Validate dashboard data against schema
 */
export function validateDashboardData(key, data) {
  const schema = getDashboardSchema(key)
  if (!schema) {
    console.warn(`No schema defined for dashboard: ${key}`)
    return { valid: true, errors: [] } // Allow unknown dashboards
  }

  const errors = []

  // Check required data keys exist
  for (const requiredKey of schema.requiredData || []) {
    if (!(requiredKey in data)) {
      errors.push(`Missing required data key: ${requiredKey}`)
    }
  }

  // Check required fields within data
  if (schema.requiredData && schema.requiredData.length > 0) {
    const dataKey = schema.requiredData[0]
    const dataObject = data[dataKey]

    if (dataObject && typeof dataObject === 'object') {
      for (const field of schema.requiredFields || []) {
        if (!(field in dataObject)) {
          errors.push(`Missing required field in ${dataKey}: ${field}`)
        }
      }
    }
  }

  return {
    valid: errors.length === 0,
    errors
  }
}

export default DASHBOARD_REGISTRY
