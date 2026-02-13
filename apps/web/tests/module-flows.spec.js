import { test, expect } from '@playwright/test'
import { fileURLToPath } from 'url'

const frontendUrl = (process.env.FRONTEND_URL || '').trim()

const requireFrontend = () => {
  test.skip(!frontendUrl, 'FRONTEND_URL not set; skipping module flow tests')
}

const seedSession = async (page) => {
  const sid = (process.env.KAI_SESSION_ID || '').trim()
  await page.addInitScript(({ key, value, accessKey }) => {
    sessionStorage.setItem(accessKey, 'true')
    if (value) sessionStorage.setItem(key, value)
    // Disable onboarding overlays that can legitimately intercept clicks in E2E.
    localStorage.setItem('kai-audit-visited', 'true')
    localStorage.setItem('kai-chat-visited', 'true')
  }, { key: 'kai_chat_session_id', value: sid, accessKey: 'kai_access_granted_v2' })
}

test.describe('Module Functional Flows (UI wiring)', () => {
  test('Klaudit: multisheet XLSX upload triggers audit job path', async ({ page }) => {
    requireFrontend()
    await seedSession(page)

    // Intercept heavy backend work; we assert UI wiring without waiting minutes.
    await page.route('**/api/data/upload', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ status: 'success', uploaded: ['qa/qa-multisheet.xlsx'], account: 'QA MultiSheet', prefix: 'qa/' }),
      })
    })
    await page.route('**/api/audit/generate', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ status: 'queued', job_id: 'job-klaudit-1' }),
      })
    })
    await page.route('**/api/jobs/job-klaudit-1', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ job: { id: 'job-klaudit-1', status: 'succeeded' } }),
      })
    })
    await page.route('**/api/jobs/job-klaudit-1/result', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ file_name: 'audit.xlsx', download_url: '/api/audit/download/audit.xlsx' }),
      })
    })

    await page.goto(`${frontendUrl.replace(/\/$/, '')}/klaudit`, { waitUntil: 'domcontentloaded' })
    await expect(page.getByRole('heading', { name: 'Klaudit Audit', exact: true, level: 4 })).toBeVisible()

    const fixture = fileURLToPath(new URL('./fixtures/multisheet.xlsx', import.meta.url))
    await page.getByTestId('audit-file-input').setInputFiles(fixture)
    await expect(page.getByText('multisheet.xlsx')).toBeVisible()

    await page.getByTestId('audit-run').click()

    // The UI may show both a chat bubble and a progress-stage banner; assert the chat bubble text.
    await expect(page.getByText(/Audit queued \(job/i)).toBeVisible({ timeout: 30_000 })
    await expect(page.getByText(/Audit complete!/i)).toBeVisible({ timeout: 30_000 })
  })

  test('PMax: advanced analyze posts to /api/pmax/analyze and renders result', async ({ page }) => {
    requireFrontend()
    await seedSession(page)

    await page.route('**/api/pmax/analyze', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          status: 'success',
          result: {
            placements: [{ placement: 'YouTube', cost: 1200, conversions: 24 }],
            insights: ['YouTube is your top cost driver.'],
            recommendations: ['Exclude low-quality placements and add audience signals.'],
          },
        }),
      })
    })

    await page.goto(`${frontendUrl.replace(/\/$/, '')}/pmax`, { waitUntil: 'domcontentloaded' })
    await expect(page.getByPlaceholder('Describe what you want to analyze...')).toBeVisible()

    // Switch to advanced mode via the ChatLedLayout "Advanced" toggle if present.
    const showAdvanced = page.getByRole('button', { name: /show advanced mode/i })
    if (await showAdvanced.count()) {
      await showAdvanced.click()
    }

    await page.getByLabel(/Placements JSON/i).fill('[{\"placement\":\"YouTube\",\"cost\":1200,\"conversions\":24}]')
    await page.getByRole('button', { name: 'Run Analysis', exact: true }).click()

    await expect(page.getByText(/top cost driver/i)).toBeVisible({ timeout: 30_000 })
  })

  test('SERP Monitor: manual URL health check renders results', async ({ page }) => {
    requireFrontend()
    await seedSession(page)

    await page.route('**/api/serp/check', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          results: [
            { url: 'https://example.com/', status: 200, ok: true, notes: 'OK' },
            { url: 'https://example.com/missing', status: 404, ok: false, notes: 'Not found' },
          ],
        }),
      })
    })

    await page.goto(`${frontendUrl.replace(/\/$/, '')}/serp`, { waitUntil: 'domcontentloaded' })
    await expect(page.getByPlaceholder('Describe what you want to check...')).toBeVisible()

    const showAdvanced = page.getByRole('button', { name: /show advanced mode/i })
    if (await showAdvanced.count()) {
      await showAdvanced.click()
    }

    await page.getByLabel(/URLs/i).fill('https://example.com/\nhttps://example.com/missing')
    await page.getByRole('button', { name: 'Run SERP Check', exact: true }).click()

    await expect(page.getByText('https://example.com/missing')).toBeVisible({ timeout: 30_000 })
  })

  test('SERP Monitor: competitor signal posts to /api/serp/competitor-signal and renders summary', async ({ page }) => {
    requireFrontend()
    await seedSession(page)

    await page.route('**/api/serp/competitor-signal', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          status: 'success',
          result: {
            competitor: 'example.com',
            signal: 'ramping_up',
            confidence: 0.85,
            impression_share_current: 20,
            impression_share_previous: 10,
            outranking_rate: 50,
            interpretation: 'QA summary: Competitor appears to be ramping up coverage.',
            metrics_used: ['impression_share_current', 'impression_share_previous', 'outranking_rate'],
          },
        }),
      })
    })

    await page.goto(`${frontendUrl.replace(/\/$/, '')}/serp`, { waitUntil: 'domcontentloaded' })
    await expect(page.getByPlaceholder('Describe what you want to check...')).toBeVisible()

    await page.getByRole('tab', { name: /competitor/i }).click()

    const showAdvanced = page.getByRole('button', { name: /show advanced mode/i })
    if (await showAdvanced.count()) {
      await showAdvanced.click()
    }

    await page.getByLabel(/Competitor domain/i).fill('example.com')
    await page.getByRole('button', { name: 'Run Competitor Signal', exact: true }).click()

    await expect(page.getByText('QA summary: Competitor appears to be ramping up coverage.', { exact: true })).toBeVisible({ timeout: 30_000 })
  })
})
