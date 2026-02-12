import { test, expect } from '@playwright/test'

// UI smoke: only runs if FRONTEND_URL is provided; otherwise it skips.

test('UI smoke: audit page renders inputs', async ({ page }) => {
  const frontendUrl = (process.env.FRONTEND_URL || '').trim()
  test.skip(!frontendUrl, 'FRONTEND_URL not set; skipping UI smoke')

  // Bypass the password gate for smoke by priming the session flag
  await page.addInitScript(({ key }) => {
    sessionStorage.setItem(key, 'true')
  }, { key: 'kai_access_granted_v2' })

  await page.goto(frontendUrl, { waitUntil: 'domcontentloaded' })

  // Go directly to audit experience to assert audit controls
  const auditUrl = `${frontendUrl.replace(/\/$/, '')}/klaudit`
  await page.goto(auditUrl, { waitUntil: 'domcontentloaded' })
  await expect(page.getByTestId('audit-account')).toBeVisible()
  await expect(page.getByTestId('audit-file-input')).toBeVisible()
  await expect(page.getByTestId('audit-browse')).toBeVisible()
  await expect(page.getByTestId('audit-run')).toBeVisible()
})
