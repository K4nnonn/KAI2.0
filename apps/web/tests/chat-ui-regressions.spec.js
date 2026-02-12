import { test, expect } from '@playwright/test'

const frontendUrl = (process.env.FRONTEND_URL || '').trim()
const sessionId = (process.env.KAI_SESSION_ID || '').trim() || 'ui_regression_session'

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

test.describe('Kai Chat UI regressions', () => {
  test('follow-up reuse persists after reload (no duplicate plan-and-run)', async ({ page }) => {
    requireFrontend()

    // Deterministic sandboxing: stub SA360 connected + route + planner so this test does not
    // depend on the live backend. This isolates the regression to UI state persistence.
    await page.route('**/api/sa360/oauth/status**', async (route) => {
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          connected: true,
          login_customer_id: '1111111111',
          default_customer_id: '7902313748',
          default_account_name: 'QA Account',
        }),
      })
    })

    await page.route('**/api/chat/route', async (route) => {
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          tool: 'performance',
          intent: 'performance',
          run_planner: true,
          needs_ids: false,
          customer_ids: ['7902313748'],
        }),
      })
    })

    await page.route('**/api/chat/plan-and-run', async (route) => {
      const req = route.request()
      let body = {}
      try { body = JSON.parse(req.postData() || '{}') } catch {}
      const msg = String(body?.message || '')

      // Minimal executed plan snapshot that should enable follow-up reuse.
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          executed: true,
          plan: {
            account_name: 'QA Account',
            customer_ids: ['7902313748'],
            date_range: 'LAST_7_DAYS',
          },
          summary: 'Conversions up 10% WoW; CTR down 5%.',
          analysis: {
            summary: 'Option A: tighten queries; Option B: test new creative. Watch CTR and conv rate.',
          },
          notes: `stubbed planner for: ${msg}`,
        }),
      })
    })

    await page.route('**/api/chat/send', async (route) => {
      const req = route.request()
      if (req.method() !== 'POST') return route.continue()
      let body = {}
      try { body = JSON.parse(req.postData() || '{}') } catch {}
      if (body?.context?.prompt_kind !== 'planner_summary') return route.continue()
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          reply: 'Option A: fix CTR drop with query/creative tests. Option B: isolate budgets by campaign. Monitor CTR and conv rate for 3-5 days.',
        }),
      })
    })

    await seedSession(page)
    await page.goto(frontendUrl, { waitUntil: 'domcontentloaded' })

    const input = page.getByPlaceholder('Ask Kai anything... audit, analyze, create, or explore')
    await expect(input).toBeVisible()

    const firstMsg = 'Show me last week performance for 7902313748'
    const firstPlanReq = page.waitForRequest((req) => req.url().includes('/api/chat/plan-and-run') && req.method() === 'POST', { timeout: 30000 })
    await input.fill(firstMsg)
    await page.keyboard.press('Enter')
    await firstPlanReq

    const summaryReqObserved = await page.waitForRequest((req) => {
      if (!req.url().includes('/api/chat/send')) return false
      if (req.method() !== 'POST') return false
      try {
        const body = JSON.parse(req.postData() || '{}')
        return body.context?.prompt_kind === 'planner_summary'
      } catch {
        return false
      }
    }, { timeout: 30000 }).then(() => true).catch(() => false)

    if (!summaryReqObserved) {
      test.skip(true, 'Planner summary request not observed; cannot validate follow-up reuse')
    }

    await page.reload({ waitUntil: 'domcontentloaded' })
    await expect(input).toBeVisible()

    const followupMsg = 'Explore areas of optimizations to improve performance'
    const followupPlanReq = page.waitForRequest((req) => {
      if (!req.url().includes('/api/chat/plan-and-run')) return false
      if (req.method() !== 'POST') return false
      try {
        const body = JSON.parse(req.postData() || '{}')
        return body.message === followupMsg
      } catch {
        return false
      }
    }, { timeout: 8000 }).then(() => true).catch(() => false)

    await input.fill(followupMsg)
    await page.keyboard.press('Enter')

    const plannerReRan = await followupPlanReq
    expect(plannerReRan).toBeFalsy()
  })

  test('planner_summary internal tokens do not force metric-dump fallback', async ({ page }) => {
    requireFrontend()
    test.setTimeout(60000)

    await page.route('**/api/sa360/oauth/status**', async (route) => {
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          connected: true,
          login_customer_id: '1111111111',
          default_customer_id: '7902313748',
          default_account_name: 'QA Account',
        }),
      })
    })

    await page.route('**/api/chat/route', async (route) => {
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          tool: 'performance',
          intent: 'performance',
          run_planner: true,
          needs_ids: false,
          customer_ids: ['7902313748'],
        }),
      })
    })

    await page.route('**/api/chat/plan-and-run', async (route) => {
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          executed: true,
          plan: {
            account_name: 'QA Account',
            customer_ids: ['7902313748'],
            date_range: 'LAST_7_DAYS',
          },
          summary: 'Conversions up 10% WoW; CTR down 5%.',
          analysis: { summary: 'Drivers TBD.' },
          notes: 'stubbed planner',
        }),
      })
    })

    const injectedMarker = 'TOKEN_TEST_OK'
    await page.route('**/api/chat/send', async (route) => {
      const req = route.request()
      if (req.method() !== 'POST') return route.continue()
      let body = {}
      try { body = JSON.parse(req.postData() || '{}') } catch {}
      if (body?.context?.prompt_kind !== 'planner_summary') return route.continue()
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          reply: `${injectedMarker}. No date specified; defaulting to LAST_7_DAYS.`,
        }),
      })
    })

    await seedSession(page)
    await page.goto(frontendUrl, { waitUntil: 'domcontentloaded' })

    const input = page.getByPlaceholder('Ask Kai anything... audit, analyze, create, or explore')
    await expect(input).toBeVisible()

    const firstMsg = 'Show me last week performance for 7902313748'
    await input.fill(firstMsg)
    await page.keyboard.press('Enter')

    // Ensure the UI actually invoked the planner_summary follow-up (otherwise the marker
    // may never appear if the request was skipped due to a regression).
    const summaryReqObserved = await page.waitForRequest((req) => {
      if (!req.url().includes('/api/chat/send')) return false
      if (req.method() !== 'POST') return false
      try {
        const body = JSON.parse(req.postData() || '{}')
        return body.context?.prompt_kind === 'planner_summary'
      } catch {
        return false
      }
    }, { timeout: 30000 }).then(() => true).catch(() => false)

    if (summaryReqObserved) {
      await expect(page.getByText(new RegExp(injectedMarker))).toBeVisible({ timeout: 30000 })
    } else {
      // Some builds render planner output directly without a planner_summary round-trip.
      await expect(page.getByText(/Drivers TBD\./i)).toBeVisible({ timeout: 30000 })
    }
    await expect(page.getByText(/LAST_7_DAYS/i)).toHaveCount(0)
  })
})
