import { test, expect, request } from '@playwright/test'

const backendBase = (process.env.BACKEND_URL || '').trim()
const sessionId = (process.env.KAI_SESSION_ID || '').trim()
const routeBudget = Number(process.env.KAI_ROUTE_BUDGET_SEC || '')
const sendBudget = Number(process.env.KAI_SEND_BUDGET_SEC || '')
const routeSamples = Number(process.env.KAI_ROUTE_SAMPLES || '7')
const sendSamples = Number(process.env.KAI_SEND_SAMPLES || '5')

const requireBackend = () => {
  test.skip(!backendBase, 'BACKEND_URL not set; skipping latency tests')
  test.skip(!sessionId, 'KAI_SESSION_ID not set; skipping latency tests')
}

const timeRequest = async (fn) => {
  const start = Date.now()
  const result = await fn()
  const elapsed = (Date.now() - start) / 1000
  return { result, elapsed }
}

const pctl = (samples, p) => {
  const sorted = [...samples].sort((a, b) => a - b)
  if (!sorted.length) return null
  const idx = Math.floor(p * (sorted.length - 1))
  return sorted[Math.max(0, Math.min(sorted.length - 1, idx))]
}

test.describe('Latency budgets', () => {
  test('chat route meets budget', async () => {
    requireBackend()
    test.skip(!routeBudget, 'KAI_ROUTE_BUDGET_SEC not set')
    const ctx = await request.newContext()
    // Warm-up once (cold starts can dominate a single-sample test).
    await ctx.post(`${backendBase}/api/chat/route`, {
      data: { message: 'What can you do?', session_id: sessionId },
    })

    const samples = []
    for (let i = 0; i < routeSamples; i++) {
      const { result, elapsed } = await timeRequest(() =>
        ctx.post(`${backendBase}/api/chat/route`, {
          data: { message: 'What can you do?', session_id: sessionId },
        })
      )
      expect(result.ok()).toBeTruthy()
      samples.push(elapsed)
    }

    // Gate on reliability, not best-case: p95 <= budget.
    const p95 = pctl(samples, 0.95)
    expect(p95).toBeLessThanOrEqual(routeBudget)
  })

  test('chat send meets budget', async () => {
    requireBackend()
    test.skip(!sendBudget, 'KAI_SEND_BUDGET_SEC not set')
    const ctx = await request.newContext()
    // Warm-up once to reduce one-off cold-start noise.
    await ctx.post(`${backendBase}/api/chat/send`, {
      data: { message: 'What can you do?', ai_enabled: true, session_id: sessionId },
    })

    const samples = []
    for (let i = 0; i < sendSamples; i++) {
      const { result, elapsed } = await timeRequest(() =>
        ctx.post(`${backendBase}/api/chat/send`, {
          data: { message: 'What can you do?', ai_enabled: true, session_id: sessionId },
        })
      )
      expect(result.ok()).toBeTruthy()
      samples.push(elapsed)
    }

    const p95 = pctl(samples, 0.95)
    expect(p95).toBeLessThanOrEqual(sendBudget)
  })
})
