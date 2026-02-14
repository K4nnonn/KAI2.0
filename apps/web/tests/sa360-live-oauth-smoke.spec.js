import { test, expect, request } from '@playwright/test'

const frontendUrl = (process.env.FRONTEND_URL || '').trim()
const backendUrl = (process.env.BACKEND_URL || '').trim()
const liveSmoke =
  String(process.env.KAI_LIVE_OAUTH_SMOKE || '')
    .trim()
    .toLowerCase() === 'true' || String(process.env.KAI_LIVE_OAUTH_SMOKE || '').trim() === '1'

const requireFrontend = () => {
  test.skip(!frontendUrl, 'FRONTEND_URL not set; skipping UI smoke tests')
}

const seedSession = async (page, sid) => {
  await page.addInitScript(({ key, value, accessKey }) => {
    sessionStorage.setItem(accessKey, 'true')
    if (value) sessionStorage.setItem(key, value)
  }, { key: 'kai_chat_session_id', value: sid || '', accessKey: 'kai_access_granted_v2' })
}

test.describe('SA360 OAuth Live Smoke', () => {
  test('Connect SA360 navigates popup to Google consent URL (no blank about:blank)', async ({ page }) => {
    requireFrontend()
    test.skip(!liveSmoke, 'KAI_LIVE_OAUTH_SMOKE not enabled; skipping live OAuth smoke')
    test.skip(!backendUrl, 'BACKEND_URL not set')

    // Use a fresh session id to ensure the UI is in a "not connected" state (connect button visible).
    const sid = `ui-oauth-${Date.now()}`

    // Fetch the expected consent URL directly (validates server-side OAuth config and avoids guessing).
    const ctx = await request.newContext()
    const startResp = await ctx.get(`${backendUrl}/api/sa360/oauth/start-url?session_id=${encodeURIComponent(sid)}`)
    expect(startResp.ok()).toBeTruthy()
    const startBody = await startResp.json()
    const expectedUrl = String(startBody?.url || '')
    expect(expectedUrl.startsWith('https://accounts.google.com/')).toBe(true)

    const expected = new URL(expectedUrl)
    const expectedClientId = expected.searchParams.get('client_id')
    const expectedRedirect = expected.searchParams.get('redirect_uri')
    expect(expectedClientId).toBeTruthy()
    expect(expectedRedirect).toBeTruthy()

    await seedSession(page, sid)
    await page.goto(frontendUrl, { waitUntil: 'domcontentloaded' })

    const connectBtn = page.getByRole('button', { name: /connect sa360/i })
    await expect(connectBtn).toBeVisible({ timeout: 60000 })

    // Avoid flaky external network dependencies while still validating that the UI:
    // - opens a popup (not null)
    // - navigates it away from about:blank to the expected Google OAuth URL (with critical params).
    await page.context().route('https://accounts.google.com/**', async (route) => {
      await route.fulfill({ status: 200, contentType: 'text/html', body: '<html><body>oauth-ok</body></html>' })
    })

    const popupPromise = page.waitForEvent('popup', { timeout: 15000 })
    await connectBtn.click()
    const popup = await popupPromise

    // The popup starts at about:blank and should be navigated by the UI once the start-url is fetched.
    await popup.waitForURL('https://accounts.google.com/**', { timeout: 20000 })
    const finalUrl = popup.url()
    expect(finalUrl.startsWith('https://accounts.google.com/')).toBe(true)

    // Validate critical query params survived navigation (guards against silent empty/invalid URL bugs).
    const final = new URL(finalUrl)
    expect(final.searchParams.get('client_id')).toBe(expectedClientId)
    expect(final.searchParams.get('redirect_uri')).toBe(expectedRedirect)
    expect(String(final.searchParams.get('scope') || '')).toMatch(/doubleclicksearch/i)

    try {
      await popup.close()
    } catch {
      // ignore
    }
  })

  test('Connect SA360 falls back to same-tab navigation when popup is blocked', async ({ page }) => {
    requireFrontend()
    test.skip(!liveSmoke, 'KAI_LIVE_OAUTH_SMOKE not enabled; skipping live OAuth smoke')
    test.skip(!backendUrl, 'BACKEND_URL not set')

    const sid = `ui-oauth-blocked-${Date.now()}`

    const ctx = await request.newContext()
    const startResp = await ctx.get(`${backendUrl}/api/sa360/oauth/start-url?session_id=${encodeURIComponent(sid)}`)
    expect(startResp.ok()).toBeTruthy()
    const startBody = await startResp.json()
    const expectedUrl = String(startBody?.url || '')
    expect(expectedUrl.startsWith('https://accounts.google.com/')).toBe(true)

    const expected = new URL(expectedUrl)
    const expectedClientId = expected.searchParams.get('client_id')
    const expectedRedirect = expected.searchParams.get('redirect_uri')
    expect(expectedClientId).toBeTruthy()
    expect(expectedRedirect).toBeTruthy()

    // Simulate a popup blocker: return null from window.open so the UI must navigate in the same tab.
    await page.addInitScript(() => {
      window.open = () => null
    })
    await seedSession(page, sid)

    // Prevent flaky external navigation by fulfilling the OAuth URL request locally while preserving the URL.
    await page.route('https://accounts.google.com/**', async (route) => {
      await route.fulfill({ status: 200, contentType: 'text/html', body: '<html><body>oauth-ok</body></html>' })
    })

    await page.goto(frontendUrl, { waitUntil: 'domcontentloaded' })

    const connectBtn = page.getByRole('button', { name: /connect sa360/i })
    await expect(connectBtn).toBeVisible({ timeout: 60000 })

    await connectBtn.click()

    await page.waitForURL('https://accounts.google.com/**', { timeout: 20000 })
    const finalUrl = page.url()
    expect(finalUrl.startsWith('https://accounts.google.com/')).toBe(true)

    const final = new URL(finalUrl)
    expect(final.searchParams.get('client_id')).toBe(expectedClientId)
    expect(final.searchParams.get('redirect_uri')).toBe(expectedRedirect)
    expect(String(final.searchParams.get('scope') || '')).toMatch(/doubleclicksearch/i)
  })
})
