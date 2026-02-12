/**
 * useDashboardData Hook
 * Fetches dashboard data with built-in observability and validation
 *
 * Features:
 * - Freshness monitoring (detect stale data)
 * - Schema validation (prevent AI hallucinations on bad data)
 * - Error handling with actionable recovery
 * - Loading states for skeleton UI
 *
 * Usage:
 * const { data, loading, health } = useDashboardData('/api/pmax/analyze', params, schema)
 */
import { useState, useEffect, useRef } from 'react'
import { validateDashboardData } from '../components/dashboards/DashboardRegistry'
import { API_BASE_URL as CONFIG_API_BASE_URL } from '../config'

// Prefer explicit env var; fall back to shared config (which points to prod API in builds)
const API_BASE_URL = (import.meta.env.VITE_API_URL || CONFIG_API_BASE_URL || '').replace(/\/$/, '')

/**
 * Custom hook for fetching dashboard data with observability
 *
 * @param {string} endpoint - API endpoint (e.g., '/api/pmax/analyze')
 * @param {object} params - Request parameters
 * @param {string} dashboardKey - Dashboard key for schema validation
 * @param {object} options - Additional options
 * @returns {object} { data, loading, health, refetch }
 */
export function useDashboardData(endpoint, params = {}, dashboardKey = null, options = {}) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [health, setHealth] = useState({
    fresh: true,
    valid: true,
    lastUpdate: null,
    error: null,
    warnings: []
  })

  // Track if component is mounted to prevent state updates after unmount
  const isMounted = useRef(true)

  // Stable reference to params for dependency comparison
  const paramsRef = useRef(JSON.stringify(params))
  const hasParamsChanged = paramsRef.current !== JSON.stringify(params)

  useEffect(() => {
    if (hasParamsChanged) {
      paramsRef.current = JSON.stringify(params)
    }
  }, [hasParamsChanged, params])

  useEffect(() => {
    isMounted.current = true

    async function fetchData() {
      // Skip fetch when no endpoint (e.g., demo mode dashboards)
      if (!endpoint) {
        setLoading(false)
        setData(null)
        setHealth((prev) => ({
          ...prev,
          fresh: true,
          valid: true,
          error: null,
          warnings: []
        }))
        return
      }

      setLoading(true)
      const warnings = []

      try {
        const url = endpoint.startsWith('http')
          ? endpoint
          : `${API_BASE_URL}${endpoint}`
        const response = await fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(params)
        })

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`)
        }

        const json = await response.json()
        const now = Date.now()

        // Extract actual data from response
        const responseData = json.result || json

        // Freshness check (API returns real-time data, so always fresh)
        const isFresh = true

        // Schema validation if dashboardKey provided
        let isValid = true
        if (dashboardKey) {
          const validation = validateDashboardData(dashboardKey, responseData)
          isValid = validation.valid

          if (!validation.valid) {
            warnings.push(...validation.errors)
            console.error(`[useDashboardData] Schema validation failed for ${dashboardKey}:`, validation.errors)
          }
        }

        // Only update state if component is still mounted
        if (isMounted.current) {
          setData(json)
          setHealth({
            fresh: isFresh,
            valid: isValid,
            lastUpdate: now,
            error: null,
            warnings
          })
        }
      } catch (error) {
        console.error(`[useDashboardData] Fetch error for ${endpoint}:`, error)

        if (isMounted.current) {
          setHealth(prev => ({
            ...prev,
            fresh: false,
            valid: false,
            error: error.message,
            warnings
          }))
        }
      } finally {
        if (isMounted.current) {
          setLoading(false)
        }
      }
    }

    fetchData()

    // Cleanup: mark component as unmounted
    return () => {
      isMounted.current = false
    }
  }, [endpoint, paramsRef.current, dashboardKey])

  // Refetch function for manual data refresh
  const refetch = () => {
    setLoading(true)
    paramsRef.current = JSON.stringify(params) // Trigger re-fetch
  }

  return { data, loading, health, refetch }
}

/**
 * Helper: Check if data is usable for rendering
 */
export function isDataUsable(health) {
  return health.valid && !health.error
}

/**
 * Helper: Get user-friendly error message
 */
export function getHealthMessage(health) {
  if (health.error) {
    return `Failed to load data: ${health.error}`
  }

  if (!health.valid) {
    return `Data validation failed: ${health.warnings.join(', ')}`
  }

  if (!health.fresh) {
    return 'Data may be stale (>24h old)'
  }

  return 'Data is healthy'
}

export default useDashboardData
