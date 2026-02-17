import { test, expect } from '@playwright/test'

const frontendUrl = (process.env.FRONTEND_URL || '').trim()
const backendUrl = (process.env.BACKEND_URL || '').trim()

const requireFrontend = () => {
  test.skip(!frontendUrl, 'FRONTEND_URL not set; skipping SA360 OAuth popup tests')
}

test.describe('SA360 OAuth Popup Robustness', () => {
  test('falls back to same-tab when popup navigation is blocked (blank popup)', async ({ page }, testInfo) => {
    requireFrontend()
    test.skip(!backendUrl, 'BACKEND_URL not set')

    const fakeSessionId = `pw-sa360-oauth-${Date.now()}`

    const consoleLines = []
    page.on('console', (msg) => {
      const type = msg.type()
      if (type === 'error' || type === 'warning') {
        consoleLines.push(`[console.${type}] ${msg.text()}`)
      }
    })
    page.on('pageerror', (err) => {
      consoleLines.push(`[pageerror] ${err?.message || String(err)}`)
    })

    await page.addInitScript(({ sid }) => {
      // Bypass the password wall for UI tests.
      const grantKey = 'kai_access_granted_v2'
      sessionStorage.setItem(grantKey, 'true')
      localStorage.setItem(grantKey, 'true')

      // Some code paths prefer localStorage, others sessionStorage.
      sessionStorage.setItem('kai_chat_session_id', sid)
      localStorage.setItem('kai_chat_session_id', sid)

      // Persist the counter across navigations so the test can assert window.open was invoked
      // even if the app immediately falls back to a same-tab navigation (which destroys the
      // current execution context). We use window.name because it persists across cross-origin
      // navigations (localStorage/sessionStorage do not).
      const openCalledPrefix = 'pw-openCalled:'
      const existing = (() => {
        try {
          const name = String(window.name || '')
          if (name.startsWith(openCalledPrefix)) {
            const raw = name.slice(openCalledPrefix.length)
            const n = Number.parseInt(raw || '0', 10)
            return Number.isFinite(n) ? n : 0
          }
        } catch {
          // ignore
        }
        return 0
      })()
      window.name = `${openCalledPrefix}${existing}`
      window.__pwOpenCalled = existing
      window.__pwOpenIsStub = true

      // Simulate a popup where setting location.href is silently ignored (COOP/COEP-style failure),
      // leaving the window stuck at about:blank without throwing.
      const locationStub = {}
      Object.defineProperty(locationStub, 'href', {
        configurable: true,
        enumerable: true,
        get: () => 'about:blank',
        set: () => {},
      })

      window.open = () => {
        const prev = (() => {
          try {
            const name = String(window.name || '')
            if (name.startsWith(openCalledPrefix)) {
              const raw = name.slice(openCalledPrefix.length)
              const n = Number.parseInt(raw || '0', 10)
              return Number.isFinite(n) ? n : 0
            }
          } catch {
            // ignore
          }
          return 0
        })()
        const next = prev + 1
        window.name = `${openCalledPrefix}${next}`
        window.__pwOpenCalled = next
        return {
          location: locationStub,
          focus: () => {},
          close: () => {},
        }
      }
    }, { sid: fakeSessionId })

    // If the fallback triggers, the page will navigate to Google OAuth.
    // Fulfill it so the test can assert the navigation without hitting Google.
    await page.route('https://accounts.google.com/**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'text/html',
        body: '<!doctype html><title>oauth</title><p>ok</p>',
      })
    })

    await page.goto(frontendUrl, { waitUntil: 'domcontentloaded' })

    // Ensure our init-script overrides actually applied.
    await expect.poll(() => page.evaluate(() => window.__pwOpenIsStub === true)).toBe(true)

    const connect = page.getByRole('button', { name: /connect sa360/i })
    await expect(connect).toBeVisible()

    const startReq = page.waitForRequest(/\/api\/sa360\/oauth\/start-url/i)
    const oauthReqPromise = page.waitForRequest(/accounts\.google\.com/i, { timeout: 15_000 }).catch(() => null)

    // If a real popup is created, this test isn't exercising the blank-popup fallback.
    const popupPromise = page.waitForEvent('popup', { timeout: 2000 }).catch(() => null)

    await connect.click()
    await startReq

    const popup = await popupPromise

    // Wait briefly so any same-tab navigation (fallback) can complete and the document is stable.
    await page.waitForTimeout(50)
    const openCalled = await page.evaluate(() => {
      const prefix = 'pw-openCalled:'
      const name = String(window.name || '')
      if (!name.startsWith(prefix)) return 0
      const raw = name.slice(prefix.length)
      return Number.parseInt(raw || '0', 10) || 0
    })

    if (popup) {
      await testInfo.attach('console.txt', {
        body: consoleLines.join('\n') || '(no console warnings/errors captured)',
        contentType: 'text/plain',
      })
      throw new Error(`Unexpected real popup was created (openCalled=${openCalled}). This test must simulate a blank popup.`)
    }

    // Sanity: ensure our stub window.open was invoked.
    expect(openCalled).toBeGreaterThan(0)

    // The app should detect the popup stayed blank and fall back to same-tab OAuth,
    // issuing a request to Google OAuth in the main page (not via the popup stub).
    const oauthReq = await oauthReqPromise
    expect(oauthReq, 'Expected same-tab OAuth request to accounts.google.com').not.toBeNull()
  })
})
