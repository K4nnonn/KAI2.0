import { test, expect } from '@playwright/test'

const frontendUrl = (process.env.FRONTEND_URL || '').trim()

const requireFrontend = () => {
  test.skip(!frontendUrl, 'FRONTEND_URL not set; skipping SA360 OAuth popup tests')
}

test.describe('SA360 OAuth Popup Robustness', () => {
  test('opens the direct /oauth/start endpoint in a popup (no start-url indirection)', async ({ page }) => {
    requireFrontend()

    const sid = `pw-sa360-oauth-${Date.now()}`
    let startUrlCalled = false

    // If the UI regresses to fetching /start-url, we want a hard signal.
    await page.route('**/api/sa360/oauth/start-url**', async (route) => {
      startUrlCalled = true
      await route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ detail: 'start-url should not be called by the UI' }),
      })
    })
    await page.route('**/api/sa360/oauth/status**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ connected: false, login_customer_id: null, default_customer_id: null }),
      })
    })

    await page.addInitScript(({ sid }) => {
      // Bypass the password wall for UI tests.
      const grantKey = 'kai_access_granted_v2'
      sessionStorage.setItem(grantKey, 'true')
      localStorage.setItem(grantKey, 'true')

      sessionStorage.setItem('kai_chat_session_id', sid)
      localStorage.setItem('kai_chat_session_id', sid)

      window.__pwOpenUrl = null
      window.__pwOpenIsStub = true
      window.open = (url) => {
        window.__pwOpenUrl = String(url || '')
        return { focus: () => {}, close: () => {} }
      }
    }, { sid })

    await page.goto(frontendUrl, { waitUntil: 'domcontentloaded' })
    await expect.poll(() => page.evaluate(() => window.__pwOpenIsStub === true)).toBe(true)

    const connect = page.getByRole('button', { name: /connect sa360/i })
    await expect(connect).toBeVisible()
    await connect.click()

    await expect.poll(() => page.evaluate(() => window.__pwOpenUrl)).toBeTruthy()
    const openedUrl = await page.evaluate(() => window.__pwOpenUrl)

    expect(openedUrl).toMatch(/\/api\/sa360\/oauth\/start/i)
    expect(openedUrl).toContain(`session_id=${encodeURIComponent(sid)}`)
    expect(startUrlCalled).toBe(false)
  })

  test('falls back to same-tab when popup handle is unavailable (window.open returns null)', async ({ page }) => {
    requireFrontend()

    const sid = `pw-sa360-oauth-${Date.now()}`

    await page.route('**/api/sa360/oauth/status**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ connected: false, login_customer_id: null, default_customer_id: null }),
      })
    })
    await page.route('**/api/sa360/oauth/start**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'text/html',
        body: '<!doctype html><title>oauth</title><p>oauth-fallback-ok</p>',
      })
    })

    await page.addInitScript(({ sid }) => {
      const grantKey = 'kai_access_granted_v2'
      sessionStorage.setItem(grantKey, 'true')
      localStorage.setItem(grantKey, 'true')

      sessionStorage.setItem('kai_chat_session_id', sid)
      localStorage.setItem('kai_chat_session_id', sid)

      window.__pwOpenIsStub = true
      window.open = () => null
    }, { sid })

    await page.goto(frontendUrl, { waitUntil: 'domcontentloaded' })
    await expect.poll(() => page.evaluate(() => window.__pwOpenIsStub === true)).toBe(true)

    const connect = page.getByRole('button', { name: /connect sa360/i })
    await expect(connect).toBeVisible()
    await connect.click()

    await expect(page).toHaveURL(/\/api\/sa360\/oauth\/start/i, { timeout: 15000 })
    await expect(page.getByText('oauth-fallback-ok')).toBeVisible({ timeout: 15000 })
  })
})
