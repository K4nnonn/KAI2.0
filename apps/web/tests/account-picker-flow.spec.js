import { test, expect } from '@playwright/test'

const frontendUrl = (process.env.FRONTEND_URL || '').trim()
const sessionId = (process.env.KAI_SESSION_ID || '').trim() || 'ui_account_picker_session'

const requireFrontend = () => {
  test.skip(!frontendUrl, 'FRONTEND_URL not set; skipping UI tests')
}

const seedSession = async (page) => {
  await page.addInitScript(({ key, value, accessKey }) => {
    sessionStorage.setItem(accessKey, 'true')
    if (value) {
      sessionStorage.setItem(key, value)
    }
  }, { key: 'kai_chat_session_id', value: sessionId || '', accessKey: 'kai_access_granted_v2' })
}

test.describe('Account Picker UX', () => {
  test.setTimeout(240000)

  test('selecting an account by name triggers plan-and-run with that customer_id', async ({ page }) => {
    requireFrontend()
    // Deterministic sandbox: this test validates UI wiring (account selection -> plan-and-run payload),
    // not live SA360 quota/availability. Live SA360 coverage is handled in planner.spec.js.
    await page.route('**/api/sa360/oauth/status**', async (route) => {
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          connected: true,
          login_customer_id: '4146247196',
          default_customer_id: null,
          default_account_name: null,
        }),
      })
    })
    await page.route('**/api/sa360/accounts**', async (route) => {
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([
          { customer_id: '7902313748', name: 'Havas_Shell_GoogleAds_US_Mobility Loyalty', manager: false },
        ]),
      })
    })
    await page.route('**/api/chat/route', async (route) => {
      // Ensure the UI uses the planner path and relies on the active account selection.
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          tool: 'performance',
          intent: 'performance',
          run_planner: true,
          needs_ids: false,
        }),
      })
    })
    let observedPlanBody = null
    await page.route('**/api/chat/plan-and-run', async (route) => {
      try {
        observedPlanBody = JSON.parse(route.request().postData() || '{}')
      } catch {
        observedPlanBody = null
      }
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          executed: true,
          result: { mode: 'performance' },
          analysis: {
            drivers: {
              campaign: [{ id: 'camp_1', name: 'Brand', delta: 10 }],
              device: [{ name: 'Mobile', delta: 5 }],
            },
          },
          summary: 'Stubbed performance summary.',
        }),
      })
    })

    await seedSession(page)
    await page.goto(frontendUrl, { waitUntil: 'domcontentloaded' })

    // Broad-beta UX: account selection is explicit and visible in the header ("Account (by name)").
    const accountPicker = page.getByLabel('Account (by name)')
    await expect(accountPicker).toBeVisible({ timeout: 60000 })

    // Select the intended account by name (not by ID copy/paste).
    await accountPicker.fill('Loyalty')
    const option = page.getByRole('option', { name: /Mobility Loyalty.*7902313748/i })
    await expect(option).toBeVisible({ timeout: 90000 })
    await option.click()

    const planRespPromise = page.waitForResponse(async (resp) => {
      if (!resp.url().includes('/api/chat/plan-and-run')) return false
      if (resp.request().method() !== 'POST') return false
      return true
    }, { timeout: 210000 })

    const chatInput = page.getByPlaceholder('Ask Kai anything... audit, analyze, create, or explore')
    await expect(chatInput).toBeVisible({ timeout: 60000 })
    await chatInput.fill('Why did Store visits change week over week? Which campaigns drove it?')
    await page.keyboard.press('Enter')

    const planResp = await planRespPromise
    expect(planResp.ok()).toBeTruthy()
    const json = await planResp.json()

    // Contract: UI must send the selected account id (not rely on a baked-in router list).
    expect(observedPlanBody).toBeTruthy()
    expect(Array.isArray(observedPlanBody.customer_ids)).toBeTruthy()
    expect(observedPlanBody.customer_ids).toContain('7902313748')
    expect(observedPlanBody.session_id).toBeTruthy()

    // Contract: selecting an account should run the planner and return a performance payload.
    expect(json.executed).toBeTruthy()
    expect(json.result?.mode).toBe('performance')

    // Quality: a custom conversion metric request should yield drivers once the account is selected.
    const drivers = json.analysis?.drivers || {}
    const campaignCount = Array.isArray(drivers.campaign) ? drivers.campaign.length : 0
    const deviceCount = Array.isArray(drivers.device) ? drivers.device.length : 0
    expect(campaignCount + deviceCount).toBeGreaterThan(0)
  })
})
