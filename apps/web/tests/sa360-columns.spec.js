import { test, expect } from '@playwright/test'

// Broad beta guardrail: SA360 column browser should load for a connected session.
// This spec is intentionally light to avoid brittle UI coupling.

test('SA360 Columns page loads and shows conversion actions', async ({ page }) => {
  const frontendUrl = (process.env.FRONTEND_URL || '').trim()
  const sessionId = (process.env.KAI_SESSION_ID || '').trim()
  test.skip(!frontendUrl, 'FRONTEND_URL not set; skipping SA360 columns UI test')
  test.skip(!sessionId, 'KAI_SESSION_ID not set; skipping SA360 columns UI test')

  // Bypass the password gate for e2e by priming the session flag + session id.
  await page.addInitScript(({ accessKey, sessionKey, sessionIdValue, activeKey, activeAccount }) => {
    sessionStorage.setItem(accessKey, 'true')
    localStorage.setItem(sessionKey, sessionIdValue)
    sessionStorage.setItem(sessionKey, sessionIdValue)
    localStorage.setItem(activeKey, JSON.stringify(activeAccount))
  }, {
    accessKey: 'kai_access_granted_v2',
    sessionKey: 'kai_chat_session_id',
    sessionIdValue: sessionId,
    activeKey: `kai_sa360_active_account:${sessionId}`,
    activeAccount: { customer_id: '7902313748', name: 'Havas_Shell_GoogleAds_US_Mobility Loyalty' },
  })

  const url = `${frontendUrl.replace(/\/$/, '')}/sa360-columns`
  await page.goto(url, { waitUntil: 'domcontentloaded' })

  // Expect the page + at least one row of conversion actions to render.
  await expect(page.getByRole('heading', { name: /SA360 Columns/i })).toBeVisible()

  const tableRows = page.locator('table[aria-label="SA360 conversion columns"] tbody tr')
  await expect(tableRows.first()).toBeVisible({ timeout: 120000 })
  const count = await tableRows.count()
  expect(count).toBeGreaterThan(0)
})
