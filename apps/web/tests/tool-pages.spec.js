import { test, expect } from '@playwright/test'

const frontendUrl = (process.env.FRONTEND_URL || '').trim()

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization',
}

const fulfillJson = async (route, status, bodyObj) => {
  await route.fulfill({
    status,
    contentType: 'application/json',
    headers: corsHeaders,
    body: bodyObj === null ? '' : JSON.stringify(bodyObj),
  })
}

const requireFrontend = () => {
  test.skip(!frontendUrl, 'FRONTEND_URL not set; skipping tool page UI tests')
}

const seedSession = async (page) => {
  // PasswordGate expects this flag; KAI session id is not required for these page-level tests.
  await page.addInitScript(({ accessKey }) => {
    sessionStorage.setItem(accessKey, 'true')
  }, { accessKey: 'kai_access_granted_v2' })
}

const urlFor = (path) => `${frontendUrl.replace(/\/$/, '')}${path}`

test.describe('Tool Pages (UI)', () => {
  test('Creative Studio: advanced mode triggers /api/creative/generate and renders output', async ({ page }) => {
    requireFrontend()
    await seedSession(page)

    let called = false
    await page.route('**/api/creative/generate', async (route) => {
      if (route.request().method() === 'OPTIONS') {
        return route.fulfill({ status: 204, headers: corsHeaders, body: '' })
      }
      called = true
      const body = await route.request().postDataJSON()
      expect(body).toHaveProperty('business_name')
      expect(body.business_name).toBe('Acme Co')
      await fulfillJson(route, 200, {
        result: {
          headlines: ['Acme Headline - UI Test'],
          descriptions: ['Acme Description - UI Test'],
        },
      })
    })

    await page.goto(urlFor('/creative-studio'), { waitUntil: 'domcontentloaded' })
    await expect(page.getByRole('heading', { name: 'Creative Studio', exact: true })).toBeVisible()

    await page.getByRole('button', { name: 'Show Advanced Mode' }).click()

    await page.getByLabel('Business name').fill('Acme Co')
    await page.getByRole('button', { name: 'Generate Ad Copy' }).click()

    await expect(page.getByRole('heading', { name: 'Acme Headline - UI Test' }).first()).toBeVisible({ timeout: 60000 })
    expect(called).toBeTruthy()
  })

  test('PMax Deep Dive: advanced mode triggers /api/pmax/analyze and renders AI Findings', async ({ page }) => {
    requireFrontend()
    await seedSession(page)

    let called = false
    await page.route('**/api/pmax/analyze', async (route) => {
      if (route.request().method() === 'OPTIONS') {
        return route.fulfill({ status: 204, headers: corsHeaders, body: '' })
      }
      called = true
      const body = await route.request().postDataJSON()
      expect(body).toHaveProperty('placements')
      await fulfillJson(route, 200, {
        result: {
          channel_breakout: { search: 50, shopping: 50 },
          findings: ['PMax Finding - UI Test'],
        },
      })
    })

    await page.goto(urlFor('/pmax'), { waitUntil: 'domcontentloaded' })
    await expect(page.getByRole('heading', { name: 'PMax Deep Dive', exact: true })).toBeVisible()

    await page.getByRole('button', { name: 'Show Advanced Mode' }).click()

    await page.getByLabel('Placements JSON (array of objects)').fill('[]')
    await page.getByRole('button', { name: 'Run Analysis' }).click()

    await expect(page.getByText('AI Findings', { exact: false })).toBeVisible({ timeout: 60000 })
    await expect(page.getByText('PMax Finding - UI Test', { exact: false })).toBeVisible({ timeout: 60000 })
    expect(called).toBeTruthy()
  })

  test('SERP Monitor: URL Health advanced mode triggers /api/serp/check and renders URL Status', async ({ page }) => {
    requireFrontend()
    await seedSession(page)

    let called = false
    await page.route('**/api/serp/check', async (route) => {
      if (route.request().method() === 'OPTIONS') {
        return route.fulfill({ status: 204, headers: corsHeaders, body: '' })
      }
      called = true
      const body = await route.request().postDataJSON()
      expect(body).toHaveProperty('urls')
      expect(Array.isArray(body.urls)).toBeTruthy()
      await fulfillJson(route, 200, {
        results: [
          { url: 'https://example.com/ui-test', status: 200, soft_404: false, responseTime: 123 },
        ],
      })
    })

    await page.goto(urlFor('/serp'), { waitUntil: 'domcontentloaded' })
    await expect(page.getByRole('heading', { name: 'SERP Monitor', exact: true })).toBeVisible()

    await page.getByRole('button', { name: 'Show Advanced Mode' }).click()

    await page.getByLabel('URLs (one per line)').fill('https://example.com/ui-test')
    await page.getByRole('button', { name: 'Run SERP Check' }).click()

    await expect(page.getByText('URL Status (1)', { exact: false })).toBeVisible({ timeout: 60000 })
    // Framer-motion can briefly leave hidden nodes in the DOM; assert that *some* matching node is visible.
    const urlText = page.getByText('https://example.com/ui-test', { exact: true })
    await expect.poll(async () => {
      const count = await urlText.count()
      for (let i = 0; i < count; i++) {
        if (await urlText.nth(i).isVisible()) return true
      }
      return false
    }, { timeout: 60000 }).toBeTruthy()
    expect(called).toBeTruthy()
  })

  test('SERP Monitor: Competitor Intelligence advanced mode triggers /api/serp/competitor-signal and renders competitor', async ({ page }) => {
    requireFrontend()
    await seedSession(page)

    let called = false
    await page.route('**/api/serp/competitor-signal', async (route) => {
      if (route.request().method() === 'OPTIONS') {
        return route.fulfill({ status: 204, headers: corsHeaders, body: '' })
      }
      called = true
      const body = await route.request().postDataJSON()
      expect(body).toHaveProperty('competitor_domain')
      expect(body.competitor_domain).toBe('example.com')
      await fulfillJson(route, 200, {
        result: {
          competitor: 'example.com',
          signal: 'stable',
          confidence: 0.77,
          impression_share_current: 30,
          impression_share_previous: 29,
          outranking_rate: 45,
          interpretation: 'UI test stub interpretation.',
        },
      })
    })

    await page.goto(urlFor('/serp'), { waitUntil: 'domcontentloaded' })
    await expect(page.getByRole('heading', { name: 'SERP Monitor', exact: true })).toBeVisible()

    await page.getByRole('tab', { name: 'Competitor Intelligence' }).click()
    await page.getByRole('button', { name: 'Show Advanced Mode' }).click()
    await page.getByTestId('competitor-domain').fill('example.com')
    await page.getByRole('button', { name: 'Run Competitor Signal' }).click()

    await expect(page.getByText('example.com', { exact: false })).toBeVisible({ timeout: 60000 })
    expect(called).toBeTruthy()
  })

  test('Settings page loads and save action triggers a dialog', async ({ page }) => {
    requireFrontend()
    await seedSession(page)

    let dialogText = ''
    page.on('dialog', async (dialog) => {
      dialogText = dialog.message()
      await dialog.accept()
    })

    await page.goto(urlFor('/settings'), { waitUntil: 'domcontentloaded' })
    await expect(page.getByRole('heading', { name: 'Settings', exact: true })).toBeVisible()

    await page.getByRole('button', { name: 'Save Settings' }).click()
    expect(dialogText).toContain('Settings saved')
  })

  test('Info page loads (Technical Architecture visible)', async ({ page }) => {
    requireFrontend()
    await seedSession(page)

    await page.goto(urlFor('/info'), { waitUntil: 'domcontentloaded' })
    await expect(page.getByText('Technical Architecture', { exact: false })).toBeVisible()
  })

  test('Env & Keys page requires passphrase and renders env list after unlock', async ({ page }) => {
    requireFrontend()
    await seedSession(page)

    await page.route('**/api/settings/env**', async (route) => {
      if (route.request().method() === 'OPTIONS') {
        return route.fulfill({ status: 204, headers: corsHeaders, body: '' })
      }
      await fulfillJson(route, 200, {
        env: [
          { key: 'SA360_CLIENT_ID', value: '***' },
          { key: 'SA360_CLIENT_SECRET', value: '***' },
        ],
      })
    })

    await page.goto(urlFor('/env'), { waitUntil: 'domcontentloaded' })
    await expect(page.getByText('Env & Keys Access', { exact: false })).toBeVisible()

    await page.getByLabel('Passphrase').fill('test-passphrase')
    await page.getByRole('button', { name: 'Unlock' }).click()

    await expect(page.getByRole('cell', { name: 'SA360_CLIENT_ID', exact: true })).toBeVisible({ timeout: 60000 })
  })
})
