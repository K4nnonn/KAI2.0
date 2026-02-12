import { test, expect, request } from '@playwright/test'

const frontendUrl = (process.env.FRONTEND_URL || '').trim()
const backendUrl = (process.env.BACKEND_URL || '').trim()
const sessionId = (process.env.KAI_SESSION_ID || '').trim()

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
    test.skip(!backendUrl || !sessionId, 'BACKEND_URL or KAI_SESSION_ID not set')

    const ctx = await request.newContext()
    const statusResp = await ctx.get(`${backendUrl}/api/sa360/oauth/status?session_id=${encodeURIComponent(sessionId)}`)
    test.skip(!statusResp.ok(), 'SA360 status unavailable')
    const statusBody = await statusResp.json()
    test.skip(!statusBody?.connected, 'SA360 not connected')

    await seedSession(page)
    await page.goto(frontendUrl, { waitUntil: 'domcontentloaded' })

    // Broad-beta UX: account selection is explicit and visible in the header ("Account (by name)").
    // If the MCC isn't set yet, set it so the account picker can load accounts.
    const ensureAccountPicker = async () => {
      const picker = page.getByLabel('Account (by name)')
      try {
        await expect(picker).toBeVisible({ timeout: 5000 })
        return picker
      } catch {
        // MCC not set; set it and wait for the picker to appear.
        const mccInput = page.getByLabel('Manager ID (MCC)')
        await expect(mccInput).toBeVisible({ timeout: 60000 })
        await mccInput.fill('4146247196')
        await page.getByRole('button', { name: /save mcc/i }).click()
        // UX copy can vary; contract is that save succeeds and account picker becomes usable.
        const saveToast = page.getByText(/Manager ID saved|Manager \(MCC\) saved/i)
        await Promise.race([
          saveToast.waitFor({ state: 'visible', timeout: 20000 }),
          picker.waitFor({ state: 'visible', timeout: 60000 }),
        ])
        await expect(picker).toBeVisible({ timeout: 60000 })
        return picker
      }
    }

    const accountPicker = await ensureAccountPicker()

    // Select the intended account by name (not by ID copy/paste).
    await accountPicker.fill('Loyalty')
    const option = page.getByRole('option', { name: /Mobility Loyalty.*7902313748/i })
    await expect(option).toBeVisible({ timeout: 90000 })
    await option.click()

    const planRespPromise = page.waitForResponse(async (resp) => {
      if (!resp.url().includes('/api/chat/plan-and-run')) return false
      if (resp.request().method() !== 'POST') return false
      try {
        const body = JSON.parse(resp.request().postData() || '{}')
        return body.session_id === sessionId && Array.isArray(body.customer_ids) && body.customer_ids.includes('7902313748')
      } catch {
        return false
      }
    }, { timeout: 210000 })

    const chatInput = page.getByPlaceholder('Ask Kai anything... audit, analyze, create, or explore')
    await expect(chatInput).toBeVisible({ timeout: 60000 })
    await chatInput.fill('Why did Store visits change week over week? Which campaigns drove it?')
    await page.keyboard.press('Enter')

    const planResp = await planRespPromise
    expect(planResp.ok()).toBeTruthy()
    const json = await planResp.json()

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
