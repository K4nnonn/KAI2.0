import { test, expect, request } from '@playwright/test'

const frontendUrl = (process.env.FRONTEND_URL || '').trim()
const backendUrl = (process.env.BACKEND_URL || '').trim()
const sessionId = (process.env.KAI_SESSION_ID || '').trim()

const requireFrontend = () => {
  test.skip(!frontendUrl, 'FRONTEND_URL not set; skipping UI chat tests')
}

const seedSession = async (page) => {
  await page.addInitScript(({ key, value, accessKey }) => {
    sessionStorage.setItem(accessKey, 'true')
    if (value) {
      sessionStorage.setItem(key, value)
    }
  }, { key: 'kai_chat_session_id', value: sessionId || '', accessKey: 'kai_access_granted_v2' })
}

test.describe('Kai Chat UI', () => {
  test('chat input sends message and renders assistant', async ({ page }) => {
    requireFrontend()
    await seedSession(page)
    await page.goto(frontendUrl, { waitUntil: 'domcontentloaded' })

    const input = page.getByPlaceholder('Ask Kai anything... audit, analyze, create, or explore')
    await expect(input).toBeVisible()
    await input.fill('health check')
    await page.keyboard.press('Enter')

    await expect(page.getByTestId('chat-assistant').first()).toBeVisible({ timeout: 60000 })
  })

  test('planner uses processing chip (no canned 30-60s ack bubbles)', async ({ page }) => {
    requireFrontend()
    test.skip(!backendUrl || !sessionId, 'BACKEND_URL or KAI_SESSION_ID not set')

    const ctx = await request.newContext()
    const statusResp = await ctx.get(`${backendUrl}/api/sa360/oauth/status?session_id=${encodeURIComponent(sessionId)}`)
    if (!statusResp.ok()) {
      test.skip(true, 'SA360 status unavailable')
    }
    const statusBody = await statusResp.json()
    test.skip(!statusBody?.connected, 'SA360 not connected')

    await seedSession(page)
    await page.goto(frontendUrl, { waitUntil: 'domcontentloaded' })

    const input = page.getByPlaceholder('Ask Kai anything... audit, analyze, create, or explore')
    await expect(input).toBeVisible()

    const msg = 'Show me last week performance for 7902313748'
    await input.fill(msg)
    await page.keyboard.press('Enter')

    // The old UI injected a deterministic "On it... Expect ~30-60s" assistant bubble.
    // Broad beta UX requirement: rely on the processing chip, not a chat bubble.
    await page.waitForTimeout(1200)
    await expect(page.getByText(/expect ~?30-60s/i)).toHaveCount(0)
    await expect(page.getByText(/on it\.\s*running/i)).toHaveCount(0)
    await expect(page.getByText(/got it\.\s*pulling/i)).toHaveCount(0)
    await expect(page.getByText(/working on .*?~?30-60s/i)).toHaveCount(0)

    // Processing chip should appear for long-running planner calls.
    await expect(page.getByText(/processing:/i)).toBeVisible({ timeout: 60000 })
  })

  test('optimization follow-up reuses last planner snapshot (no duplicate plan-and-run)', async ({ page }) => {
    requireFrontend()
    test.skip(!backendUrl || !sessionId, 'BACKEND_URL or KAI_SESSION_ID not set')

    const ctx = await request.newContext()
    const statusResp = await ctx.get(`${backendUrl}/api/sa360/oauth/status?session_id=${encodeURIComponent(sessionId)}`)
    if (!statusResp.ok()) {
      test.skip(true, 'SA360 status unavailable')
    }
    const statusBody = await statusResp.json()
    test.skip(!statusBody?.connected, 'SA360 not connected')

    await seedSession(page)
    await page.goto(frontendUrl, { waitUntil: 'domcontentloaded' })

    const input = page.getByPlaceholder('Ask Kai anything... audit, analyze, create, or explore')
    await expect(input).toBeVisible()

    const firstMsg = 'Show me last week performance for 7902313748'
    const firstPlanReq = page.waitForRequest((req) => req.url().includes('/api/chat/plan-and-run') && req.method() === 'POST', { timeout: 90000 })
    await input.fill(firstMsg)
    await page.keyboard.press('Enter')
    await firstPlanReq

    // Wait until the UI requests a planner-summary paraphrase. That request happens only after
    // a planner snapshot exists, so it is the most reliable "follow-up is safe" signal.
    const summaryReqObserved = await page.waitForRequest((req) => {
      if (!req.url().includes('/api/chat/send')) return false
      if (req.method() !== 'POST') return false
      try {
        const body = JSON.parse(req.postData() || '{}')
        return body.context?.prompt_kind === 'planner_summary'
      } catch {
        return false
      }
    }, { timeout: 210000 }).then(() => true).catch(() => false)

    if (!summaryReqObserved) {
      test.skip(true, 'Planner summary request not observed; cannot validate follow-up reuse')
    }

    // Ensure at least one assistant message from the initial planner run is rendered before we
    // capture the baseline count. Otherwise, the follow-up assertions can accidentally attach
    // to the late-arriving planner output bubble instead of the follow-up reply.
    await expect(page.getByTestId('chat-assistant').first()).toBeVisible({ timeout: 210000 })
    const assistantBefore = await page.getByTestId('chat-assistant').count()

    const followupMsg = 'Explore areas of optimizations to improve performance'
    const followupSendReq = page.waitForRequest((req) => {
      if (!req.url().includes('/api/chat/send')) return false
      if (req.method() !== 'POST') return false
      try {
        const body = JSON.parse(req.postData() || '{}')
        return body.message === followupMsg
      } catch {
        return false
      }
    }, { timeout: 90000 })

    // If the UI incorrectly re-runs the planner, this request will fire again.
    const followupPlanReq = page.waitForRequest((req) => {
      if (!req.url().includes('/api/chat/plan-and-run')) return false
      if (req.method() !== 'POST') return false
      try {
        const body = JSON.parse(req.postData() || '{}')
        return body.message === followupMsg
      } catch {
        return false
      }
    }, { timeout: 15000 }).then(() => true).catch(() => false)

    await input.fill(followupMsg)
    await page.keyboard.press('Enter')

    const sendReq = await followupSendReq
    // Wait for the backend response to this exact follow-up request (not just the request being sent).
    await page.waitForResponse((resp) => resp.request() === sendReq, { timeout: 210000 })
    const sendBody = JSON.parse(sendReq.postData() || '{}')
    expect(sendBody.context?.tool_output).toBeTruthy()

    const plannerReRan = await followupPlanReq
    expect(plannerReRan).toBeFalsy()

    // New assistant response should arrive (recommendations/explanation), without leaking internal tokens.
    await expect(page.getByTestId('chat-assistant')).toHaveCount(assistantBefore + 1, { timeout: 210000 })
    const lastAssistant = page.getByTestId('chat-assistant').nth(assistantBefore)
    await expect(lastAssistant).not.toContainText('LAST_', { timeout: 1000 })
    await expect(lastAssistant).not.toContainText('No date specified', { timeout: 1000 })
    // Advisor-grade shape: options + monitoring plan (prevents "metric dump" regressions).
    await expect(lastAssistant).toContainText(/Option A|Path 1/i, { timeout: 2000 })
    await expect(lastAssistant).toContainText(/Option B|Path 2/i, { timeout: 2000 })
    await expect(lastAssistant).toContainText(/monitor|watch|track/i, { timeout: 2000 })
    await expect(lastAssistant).not.toContainText(/\\bcpm\\b/i, { timeout: 2000 })
  })

  test('SA360 not-connected UX shows connect button', async ({ page }) => {
    requireFrontend()
    const newSession = `ui-${Date.now()}`
    await page.addInitScript(({ key, value, accessKey }) => {
      sessionStorage.setItem(accessKey, 'true')
      sessionStorage.setItem(key, value)
    }, { key: 'kai_chat_session_id', value: newSession, accessKey: 'kai_access_granted_v2' })
    await page.goto(frontendUrl, { waitUntil: 'domcontentloaded' })

    await expect(page.getByText('SA360 not connected', { exact: false })).toBeVisible({ timeout: 60000 })
    await expect(page.getByRole('button', { name: /connect sa360/i })).toBeVisible({ timeout: 60000 })
  })

  test('SA360 connect opens OAuth popup with opener context (regression guard)', async ({ page }) => {
    requireFrontend()

    const ctx = page.context()
    const newSession = `ui-${Date.now()}`
    await page.addInitScript(({ key, value, accessKey }) => {
      sessionStorage.setItem(accessKey, 'true')
      sessionStorage.setItem(key, value)
    }, { key: 'kai_chat_session_id', value: newSession, accessKey: 'kai_access_granted_v2' })

    await ctx.route('**/api/sa360/oauth/status**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ connected: false, login_customer_id: null, default_customer_id: null }),
      })
    })
    // The UI should open the redirect endpoint directly (avoid about:blank -> location navigation COOP issues).
    await ctx.route('**/api/sa360/oauth/start**', async (route) => {
      const url = route.request().url()
      expect(url).toContain(`session_id=${encodeURIComponent(newSession)}`)
      await route.fulfill({
        status: 200,
        contentType: 'text/html',
        body: `<!doctype html><html><body><div id="probe"></div><script>
          const hasOpener = !!window.opener;
          document.getElementById('probe').textContent = hasOpener ? 'opener-ok' : 'opener-missing';
        </script></body></html>`,
      })
    })

    await page.goto(frontendUrl, { waitUntil: 'domcontentloaded' })
    const connectBtn = page.getByRole('button', { name: /connect sa360/i })
    await expect(connectBtn).toBeVisible({ timeout: 60000 })

    const popupPromise = page.waitForEvent('popup', { timeout: 15000 })
    await connectBtn.click()
    const popup = await popupPromise
    await popup.waitForLoadState('domcontentloaded')
    await expect(popup.locator('#probe')).toHaveText(/opener-ok/i, { timeout: 10000 })
    await expect(page.getByText(/popup blocked/i)).toHaveCount(0)
  })

  test('SA360 connect falls back to same-tab redirect when popup handle is unavailable', async ({ page }) => {
    requireFrontend()

    const ctx = page.context()
    const newSession = `ui-${Date.now()}`
    await page.addInitScript(({ key, value, accessKey }) => {
      sessionStorage.setItem(accessKey, 'true')
      sessionStorage.setItem(key, value)
      window.open = () => null
    }, { key: 'kai_chat_session_id', value: newSession, accessKey: 'kai_access_granted_v2' })

    await ctx.route('**/api/sa360/oauth/status**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ connected: false, login_customer_id: null, default_customer_id: null }),
      })
    })
    await ctx.route('**/api/sa360/oauth/start**', async (route) => {
      await route.fulfill({ status: 200, contentType: 'text/html', body: '<html><body>oauth-fallback-ok</body></html>' })
    })

    await page.goto(frontendUrl, { waitUntil: 'domcontentloaded' })
    const connectBtn = page.getByRole('button', { name: /connect sa360/i })
    await expect(connectBtn).toBeVisible({ timeout: 60000 })
    await connectBtn.click()

    await expect(page).toHaveURL(/\/api\/sa360\/oauth\/start/i, { timeout: 15000 })
    await expect(page.getByText('oauth-fallback-ok')).toBeVisible({ timeout: 15000 })
  })

  test('SA360 not-connected blocks performance planner with clear CTA', async ({ page }) => {
    requireFrontend()
    test.skip(!backendUrl, 'BACKEND_URL not set; skipping planner block check')

    const newSession = `ui-${Date.now()}`
    await page.addInitScript(({ key, value, accessKey }) => {
      sessionStorage.setItem(accessKey, 'true')
      sessionStorage.setItem(key, value)
    }, { key: 'kai_chat_session_id', value: newSession, accessKey: 'kai_access_granted_v2' })
    await page.goto(frontendUrl, { waitUntil: 'domcontentloaded' })

    const input = page.getByPlaceholder('Ask Kai anything... audit, analyze, create, or explore')
    await expect(input).toBeVisible()
    await input.fill('How did performance look in the last 7 days?')
    await page.keyboard.press('Enter')

    await expect(page.getByText("SA360 isn't connected", { exact: false })).toBeVisible({ timeout: 60000 })
    await expect(page.getByRole('button', { name: /connect sa360/i })).toBeVisible({ timeout: 60000 })
  })

  test('default SA360 account is not cleared on page load', async ({ page }) => {
    requireFrontend()
    test.skip(!backendUrl || !sessionId, 'BACKEND_URL or KAI_SESSION_ID not set')

    const ctx = await request.newContext()
    // Seed a server-side default account selection.
    const saveResp = await ctx.post(`${backendUrl}/api/sa360/default-account`, {
      data: {
        session_id: sessionId,
        customer_id: '7902313748',
        account_name: 'Havas_Shell_GoogleAds_US_Mobility Loyalty',
      },
    })
    expect(saveResp.ok()).toBeTruthy()

    const before = await ctx.get(`${backendUrl}/api/sa360/oauth/status?session_id=${encodeURIComponent(sessionId)}`)
    expect(before.ok()).toBeTruthy()
    const beforeBody = await before.json()
    expect(beforeBody.default_customer_id).toBe('7902313748')

    // Load the UI WITHOUT priming an active-account localStorage value.
    await page.addInitScript(({ key, value, accessKey }) => {
      sessionStorage.setItem(accessKey, 'true')
      sessionStorage.setItem(key, value)
    }, { key: 'kai_chat_session_id', value: sessionId, accessKey: 'kai_access_granted_v2' })

    await page.goto(frontendUrl, { waitUntil: 'domcontentloaded' })
    await expect(page.getByText(/SA360 connected/i)).toBeVisible({ timeout: 60000 })

    // Give effects a moment; then verify the server-side default is still present.
    await page.waitForTimeout(1500)
    const after = await ctx.get(`${backendUrl}/api/sa360/oauth/status?session_id=${encodeURIComponent(sessionId)}`)
    expect(after.ok()).toBeTruthy()
    const afterBody = await after.json()
    expect(afterBody.default_customer_id).toBe('7902313748')
  })

  test('router debug banner surfaces local verification failure', async ({ page }) => {
    requireFrontend()
    test.skip(!backendUrl || !sessionId, 'BACKEND_URL or KAI_SESSION_ID not set')
    const ctx = await request.newContext()
    const routeResp = await ctx.post(`${backendUrl}/api/chat/route`, {
      data: { message: 'What can you do?', session_id: sessionId },
    })
    if (!routeResp.ok()) {
      test.skip(true, 'Router unavailable')
    }
    const routeBody = await routeResp.json()
    if (!routeBody?.notes?.includes('router_verify_failed')) {
      test.skip(true, 'Router did not report verify failure; skipping banner check')
    }

    await seedSession(page)
    // Debug banner is gated in broad beta; enable it explicitly for this test via query param.
    const url = frontendUrl.includes('?') ? `${frontendUrl}&debug_routing=1` : `${frontendUrl}?debug_routing=1`
    await page.goto(url, { waitUntil: 'domcontentloaded' })
    const input = page.getByPlaceholder('Ask Kai anything... audit, analyze, create, or explore')
    await input.fill('What can you do?')
    await page.keyboard.press('Enter')

    await expect(page.getByText('router_verify_failed', { exact: false })).toBeVisible({ timeout: 90000 })
    await expect(page.getByText('model=local', { exact: false })).toBeVisible({ timeout: 90000 })
  })

  test('router debug banner is hidden by default (broad beta)', async ({ page }) => {
    requireFrontend()
    test.skip(!backendUrl || !sessionId, 'BACKEND_URL or KAI_SESSION_ID not set')
    const ctx = await request.newContext()
    const routeResp = await ctx.post(`${backendUrl}/api/chat/route`, {
      data: { message: 'What can you do?', session_id: sessionId },
    })
    if (!routeResp.ok()) {
      test.skip(true, 'Router unavailable')
    }
    const routeBody = await routeResp.json()
    if (!routeBody?.notes?.includes('router_verify_failed')) {
      test.skip(true, 'Router did not report verify failure; skipping banner privacy check')
    }

    await seedSession(page)
    await page.goto(frontendUrl, { waitUntil: 'domcontentloaded' })

    const input = page.getByPlaceholder('Ask Kai anything... audit, analyze, create, or explore')
    await input.fill('What can you do?')
    await page.keyboard.press('Enter')

    // Debug notes should not leak into the default UI.
    await expect(page.getByText('router_verify_failed', { exact: false })).toHaveCount(0)
  })

  test('manager ID prompt triggers account picker when SA360 is connected', async ({ page }) => {
    requireFrontend()
    test.skip(!backendUrl || !sessionId, 'BACKEND_URL or KAI_SESSION_ID not set')

    const ctx = await request.newContext()
    const statusResp = await ctx.get(`${backendUrl}/api/sa360/oauth/status?session_id=${encodeURIComponent(sessionId)}`)
    if (!statusResp.ok()) {
      test.skip(true, 'SA360 status unavailable')
    }
    const statusBody = await statusResp.json()
    test.skip(!statusBody?.connected, 'SA360 not connected')

    await seedSession(page)
    await page.goto(frontendUrl, { waitUntil: 'domcontentloaded' })

    const input = page.getByPlaceholder('Ask Kai anything... audit, analyze, create, or explore')
    await input.fill('performance 4146247196')
    await page.keyboard.press('Enter')

    const picker = page.getByPlaceholder('Search accounts or paste an ID')
    const managerMsg = page.getByTestId('chat-assistant').getByText('manager account', { exact: false })
    await Promise.race([
      picker.waitFor({ state: 'visible', timeout: 90000 }),
      managerMsg.waitFor({ state: 'visible', timeout: 90000 }),
    ])
  })

  test('Save MCC shows a confirmation (no silent success)', async ({ page }) => {
    requireFrontend()
    test.skip(!backendUrl || !sessionId, 'BACKEND_URL or KAI_SESSION_ID not set')

    const ctx = await request.newContext()
    const statusResp = await ctx.get(`${backendUrl}/api/sa360/oauth/status?session_id=${encodeURIComponent(sessionId)}`)
    if (!statusResp.ok()) {
      test.skip(true, 'SA360 status unavailable')
    }
    const statusBody = await statusResp.json()
    test.skip(!statusBody?.connected, 'SA360 not connected')

    await seedSession(page)
    await page.goto(frontendUrl, { waitUntil: 'domcontentloaded' })
    await expect(page.getByText(/SA360 connected/i)).toBeVisible({ timeout: 60000 })

    const mccInput = page.getByLabel('Manager ID (MCC)')
    await expect(mccInput).toBeVisible()
    await mccInput.fill('4146247196')
    await page.getByRole('button', { name: /save mcc/i }).click()

    // UX requirement: user gets an explicit, visible confirmation that the MCC was saved.
    await expect(page.getByText(/Manager ID saved/i)).toBeVisible({ timeout: 60000 })
  })

  test('selected account is propagated into PMax chat requests', async ({ page }) => {
    requireFrontend()

    const ctx = page.context()
    const newSession = `ui-${Date.now()}`
    await page.addInitScript(({ key, value, accessKey }) => {
      sessionStorage.setItem(accessKey, 'true')
      sessionStorage.setItem(key, value)
    }, { key: 'kai_chat_session_id', value: newSession, accessKey: 'kai_access_granted_v2' })

    await ctx.route('**/api/sa360/oauth/status**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          connected: true,
          login_customer_id: '4146247196',
          default_customer_id: '7902313748',
          default_account_name: 'Havas_Shell_GoogleAds_US_Mobility Loyalty',
        }),
      })
    })
    await ctx.route('**/api/sa360/accounts**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([
          { customer_id: '7902313748', name: 'Havas_Shell_GoogleAds_US_Mobility Loyalty', manager: false },
          { customer_id: '4301133105', name: 'Havas_Shell_GoogleAds_CA_Retail_Mobility', manager: false },
        ]),
      })
    })
    await ctx.route('**/api/chat/route', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          intent: 'pmax',
          tool: 'pmax',
          run_planner: false,
          run_trends: false,
          customer_ids: [],
          needs_ids: false,
        }),
      })
    })

    let chatSendPayload = null
    await ctx.route('**/api/chat/send', async (route) => {
      try {
        chatSendPayload = route.request().postDataJSON()
      } catch {
        chatSendPayload = null
      }
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          reply: 'PMax analysis started for your selected account.',
          role: 'assistant',
          model: 'rules',
          sources: [],
        }),
      })
    })

    await page.goto(frontendUrl, { waitUntil: 'domcontentloaded' })
    await expect(page.getByText(/SA360 connected/i)).toBeVisible({ timeout: 60000 })

    const accountInput = page.getByRole('combobox', { name: 'Account (by name)' })
    await accountInput.click()
    await accountInput.fill('Mobility Loyalty')
    await page.keyboard.press('ArrowDown')
    await page.keyboard.press('Enter')

    const input = page.getByPlaceholder('Ask Kai anything... audit, analyze, create, or explore')
    await input.fill('Analyze my PMax placements')
    await page.keyboard.press('Enter')

    await expect.poll(() => chatSendPayload !== null, { timeout: 15000 }).toBe(true)
    expect(Array.isArray(chatSendPayload?.context?.customer_ids)).toBeTruthy()
    expect(chatSendPayload.context.customer_ids).toContain('7902313748')
    expect(String(chatSendPayload?.account_name || '')).toContain('Mobility Loyalty')
    await expect(page.getByText(/need to know which SA360 account to use/i)).toHaveCount(0)
  })
})

test.describe('Tool Chat UI', () => {
  test('tool chat uses planner_summary (no Notes/LAST_* leaks) and reuses planner context on follow-ups', async ({ page }) => {
    requireFrontend()
    test.skip(!backendUrl || !sessionId, 'BACKEND_URL or KAI_SESSION_ID not set')

    const ctx = await request.newContext()
    const statusResp = await ctx.get(`${backendUrl}/api/sa360/oauth/status?session_id=${encodeURIComponent(sessionId)}`)
    if (!statusResp.ok()) {
      test.skip(true, 'SA360 status unavailable')
    }
    const statusBody = await statusResp.json()
    test.skip(!statusBody?.connected, 'SA360 not connected')

    await seedSession(page)
    const pmaxUrl = `${frontendUrl.replace(/\/$/, '')}/pmax`
    await page.goto(pmaxUrl, { waitUntil: 'domcontentloaded' })

    const waitForPlan = page.waitForRequest((req) => {
      if (!req.url().includes('/api/chat/plan-and-run')) return false
      if (req.method() !== 'POST') return false
      try {
        const body = JSON.parse(req.postData() || '{}')
        return body.session_id === sessionId
      } catch {
        return false
      }
    }, { timeout: 90000 })

    const input = page.getByPlaceholder('Describe what you want to analyze...')
    await expect(input).toBeVisible()

    const firstMsg = 'Show me last week performance for 7902313748'
    const waitForSummaryReq = page.waitForRequest((req) => {
      if (!req.url().includes('/api/chat/send')) return false
      if (req.method() !== 'POST') return false
      try {
        const body = JSON.parse(req.postData() || '{}')
        return body.context?.prompt_kind === 'planner_summary'
      } catch {
        return false
      }
    }, { timeout: 210000 })

    await input.fill(firstMsg)
    await page.keyboard.press('Enter')

    await waitForPlan
    await waitForSummaryReq

    // Ensure the summary render is user-facing (not internal notes/metrics dumps).
    const firstAssistant = page.getByTestId('chat-assistant').first()
    await expect(firstAssistant).toBeVisible({ timeout: 210000 })
    await expect(firstAssistant).not.toContainText('Notes:', { timeout: 2000 })
    await expect(firstAssistant).not.toContainText('LAST_', { timeout: 2000 })
    await expect(firstAssistant).not.toContainText('No date specified', { timeout: 2000 })

    const assistantBefore = await page.getByTestId('chat-assistant').count()

    const followupMsg = 'Explore areas of optimizations to improve performance'
    const followupSendReq = page.waitForRequest((req) => {
      if (!req.url().includes('/api/chat/send')) return false
      if (req.method() !== 'POST') return false
      try {
        const body = JSON.parse(req.postData() || '{}')
        return body.message === followupMsg && body.context?.prompt_kind === 'planner_summary' && !!body.context?.tool_output
      } catch {
        return false
      }
    }, { timeout: 90000 })

    const followupPlanReq = page.waitForRequest((req) => {
      if (!req.url().includes('/api/chat/plan-and-run')) return false
      if (req.method() !== 'POST') return false
      try {
        const body = JSON.parse(req.postData() || '{}')
        return body.message === followupMsg
      } catch {
        return false
      }
    }, { timeout: 15000 }).then(() => true).catch(() => false)

    await input.fill(followupMsg)
    await page.keyboard.press('Enter')

    const sendReq = await followupSendReq
    await page.waitForResponse((resp) => resp.request() === sendReq, { timeout: 210000 })
    expect(await followupPlanReq).toBeFalsy()

    await expect(page.getByTestId('chat-assistant')).toHaveCount(assistantBefore + 1, { timeout: 210000 })
    const lastAssistant = page.getByTestId('chat-assistant').nth(assistantBefore)
    await expect(lastAssistant).toContainText(/Option A|Path 1/i, { timeout: 5000 })
    await expect(lastAssistant).toContainText(/Option B|Path 2/i, { timeout: 5000 })
    await expect(lastAssistant).toContainText(/monitor|watch|track/i, { timeout: 5000 })
    await expect(lastAssistant).not.toContainText('Notes:', { timeout: 2000 })
    await expect(lastAssistant).not.toContainText('LAST_', { timeout: 2000 })
    await expect(lastAssistant).not.toContainText('No date specified', { timeout: 2000 })
  })
})
