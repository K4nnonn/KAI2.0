import { test, expect } from '@playwright/test'

const frontendUrl = (process.env.FRONTEND_URL || '').trim()

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization',
}

const requireFrontend = () => {
  test.skip(!frontendUrl, 'FRONTEND_URL not set; skipping Klaudit upload flow test')
}

const seedSession = async (page) => {
  await page.addInitScript(({ accessKey }) => {
    sessionStorage.setItem(accessKey, 'true')
  }, { accessKey: 'kai_access_granted_v2' })
}

const urlFor = (path) => `${frontendUrl.replace(/\/$/, '')}${path}`

test.describe('Klaudit Audit (UI)', () => {
  test('upload -> generate audit -> renders Audit Ready + download enabled', async ({ page }) => {
    requireFrontend()
    await seedSession(page)

    let uploadCalled = false
    let generateCalled = false

    await page.route('**/api/data/upload', async (route) => {
      if (route.request().method() === 'OPTIONS') {
        return route.fulfill({ status: 204, headers: corsHeaders, body: '' })
      }
      uploadCalled = true
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        headers: corsHeaders,
        body: JSON.stringify({ ok: true }),
      })
    })

    await page.route('**/api/audit/generate', async (route) => {
      if (route.request().method() === 'OPTIONS') {
        return route.fulfill({ status: 204, headers: corsHeaders, body: '' })
      }
      generateCalled = true
      const body = await route.request().postDataJSON()
      expect(body).toHaveProperty('account_name')
      expect(body).toHaveProperty('async_mode')
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        headers: corsHeaders,
        body: JSON.stringify({
          file_name: 'klaudit_ui_test.pdf',
          download_url: 'https://example.com/klaudit_ui_test.pdf',
          result: { overall_score: 91 },
        }),
      })
    })

    await page.goto(urlFor('/klaudit'), { waitUntil: 'domcontentloaded' })
    await expect(page.getByRole('heading', { name: 'Klaudit Audit', exact: true })).toBeVisible()

    const skipTour = page.getByRole('button', { name: 'Skip tour' })
    if ((await skipTour.count()) > 0) {
      await skipTour.click()
    }

    // Upload a tiny CSV
    await page.getByTestId('audit-file-input').setInputFiles({
      name: 'sample.csv',
      mimeType: 'text/csv',
      buffer: Buffer.from('col_a,col_b\\n1,2\\n'),
    })

    await page.getByTestId('audit-run').click()

    await expect(page.getByText('Audit Ready', { exact: false })).toBeVisible({ timeout: 60000 })
    await expect(page.getByText('File: klaudit_ui_test.pdf', { exact: false })).toBeVisible({ timeout: 60000 })

    // Download icon should be enabled when download_url exists.
    await expect(page.getByText('Click to download your audit report', { exact: false })).toBeVisible()
    await expect(page.locator('a[href="https://example.com/klaudit_ui_test.pdf"]')).toBeVisible()

    expect(uploadCalled).toBeTruthy()
    expect(generateCalled).toBeTruthy()
  })
})
