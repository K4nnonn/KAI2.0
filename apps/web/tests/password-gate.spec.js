import { test, expect } from '@playwright/test'

const frontendUrl = (process.env.FRONTEND_URL || '').trim()

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization',
}

const requireFrontend = () => {
  test.skip(!frontendUrl, 'FRONTEND_URL not set; skipping password gate tests')
}

const urlFor = (path) => `${frontendUrl.replace(/\/$/, '')}${path}`

test.describe('Password Gate (UI)', () => {
  test('wrong password shows Incorrect password; correct password unlocks app (stubbed)', async ({ page }) => {
    requireFrontend()

    // Stub server-side auth verification to keep this deterministic and secret-free.
    await page.route('**/api/auth/verify', async (route) => {
      if (route.request().method() === 'OPTIONS') {
        return route.fulfill({ status: 204, headers: corsHeaders, body: '' })
      }
      const body = await route.request().postDataJSON()
      const pw = String(body?.password || '')
      if (pw === 'letmein') {
        return route.fulfill({
          status: 200,
          contentType: 'application/json',
          headers: corsHeaders,
          body: JSON.stringify({ authenticated: true }),
        })
      }
      return route.fulfill({
        status: 401,
        contentType: 'application/json',
        headers: corsHeaders,
        body: JSON.stringify({ authenticated: false }),
      })
    })

    await page.goto(urlFor('/'), { waitUntil: 'domcontentloaded' })

    // If SSO is required in this deployment, the password form is hidden; we cannot automate SSO here.
    const pwInput = page.getByPlaceholder('Password')
    if (!(await pwInput.isVisible().catch(() => false))) {
      test.skip(true, 'Password form not present (SSO required or enabled-only).')
    }

    await expect(page.getByText('KAI Platform', { exact: false })).toBeVisible()

    // Wrong password -> inline error.
    await pwInput.fill('wrong-password')
    await page.getByRole('button', { name: 'Access KAI' }).click()
    await expect(page.getByText('Incorrect password', { exact: false })).toBeVisible()

    // Correct password -> app loads (sidebar visible).
    await pwInput.fill('letmein')
    await page.getByRole('button', { name: 'Access KAI' }).click()
    await expect(page.getByText('Kai Chat', { exact: true })).toBeVisible({ timeout: 60000 })
  })
})

