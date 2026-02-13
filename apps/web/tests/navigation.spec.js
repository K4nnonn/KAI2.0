import { test, expect } from '@playwright/test'

const frontendUrl = (process.env.FRONTEND_URL || '').trim()

const requireFrontend = () => {
  test.skip(!frontendUrl, 'FRONTEND_URL not set; skipping navigation tests')
}

const seedSession = async (page) => {
  const sid = (process.env.KAI_SESSION_ID || '').trim()
  await page.addInitScript(({ key, value, accessKey }) => {
    // Bypass PasswordGate for E2E (same mechanism used by existing suites).
    sessionStorage.setItem(accessKey, 'true')
    if (value) sessionStorage.setItem(key, value)
  }, { key: 'kai_chat_session_id', value: sid, accessKey: 'kai_access_granted_v2' })
}

test.describe('Navigation Smoke', () => {
  const cases = [
    {
      name: 'Kai Chat',
      path: '/',
      expect: async (page) => {
        await expect(page.getByPlaceholder('Ask Kai anything... audit, analyze, create, or explore')).toBeVisible()
      },
    },
    {
      name: 'Klaudit Audit',
      path: '/klaudit',
      expect: async (page) => {
        await expect(page.getByRole('heading', { name: 'Klaudit Audit', exact: true, level: 4 })).toBeVisible()
        await expect(page.getByText(/Upload exports/i)).toHaveCount(1)
      },
    },
    {
      name: 'Creative Studio',
      path: '/creative-studio',
      expect: async (page) => {
        await expect(page.getByPlaceholder('Describe what you want to create...')).toBeVisible()
      },
    },
    {
      name: 'PMax Deep Dive',
      path: '/pmax',
      expect: async (page) => {
        await expect(page.getByPlaceholder('Describe what you want to analyze...')).toBeVisible()
      },
    },
    {
      name: 'SERP Monitor',
      path: '/serp',
      expect: async (page) => {
        await expect(page.getByPlaceholder('Describe what you want to check...')).toBeVisible()
      },
    },
    {
      name: 'Env & Keys',
      path: '/env',
      expect: async (page) => {
        // Default state is an access gate unless the tester provides the GUI passphrase.
        await expect(page.getByRole('heading', { name: 'Env & Keys Access', exact: true, level: 5 })).toBeVisible()
      },
    },
    {
      name: 'SA360 Columns',
      path: '/sa360-columns',
      expect: async (page) => {
        await expect(page.getByRole('heading', { name: 'SA360 Columns (Conversion Actions)', exact: true, level: 4 })).toBeVisible()
      },
    },
    {
      name: 'Info',
      path: '/info',
      expect: async (page) => {
        await expect(page.getByText('Technical Architecture', { exact: true })).toBeVisible()
      },
    },
    {
      name: 'Settings',
      path: '/settings',
      expect: async (page) => {
        await expect(page.getByRole('heading', { name: 'Settings', exact: true, level: 3 })).toBeVisible()
      },
    },
  ]

  for (const c of cases) {
    test(`loads ${c.name}`, async ({ page }) => {
      requireFrontend()
      await seedSession(page)
      const url = `${frontendUrl.replace(/\/$/, '')}${c.path}`
      await page.goto(url, { waitUntil: 'domcontentloaded' })
      await c.expect(page)
    })
  }
})
