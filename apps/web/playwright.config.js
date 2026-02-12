// Basic Playwright config
// - Uses env FRONTEND_URL for UI smoke (skips UI test if not set)
// - Backend API base defaults to localhost (avoid accidental prod hits)
// - HTML reports/artifacts are only written when PLAYWRIGHT_REPORT_DIR / PLAYWRIGHT_OUTPUT_DIR are set

import { defineConfig } from '@playwright/test'

const backendBase = process.env.BACKEND_URL || 'http://localhost:8000'
const reportDir = (process.env.PLAYWRIGHT_REPORT_DIR || '').trim()
const outputDir = (process.env.PLAYWRIGHT_OUTPUT_DIR || '').trim() || 'test-results'

const reporters = [['list']]
if (reportDir) {
  reporters.push(['html', { outputFolder: reportDir, open: 'never' }])
}

export default defineConfig({
  timeout: 120000,
  outputDir,
  use: {
    baseURL: process.env.FRONTEND_URL || 'http://localhost:5173',
    extraHTTPHeaders: {
      // Allow tests to reach the backend if CORS blocks UI context; API tests use request context directly.
    },
  },
  expect: {
    timeout: 30000,
  },
  reporter: reporters,
  metadata: {
    backendBase,
  },
})
