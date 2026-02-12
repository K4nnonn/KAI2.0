import { test, expect, request } from '@playwright/test'

const requireEnv = (name) => {
  const value = (process.env[name] || '').trim()
  if (!value) throw new Error(`Missing required env var: ${name}`)
  return value
}

// Backend base is provided via env (see playwright.config.js)
const backendBase = requireEnv('BACKEND_URL')
const sessionId = (process.env.KAI_SESSION_ID || '').trim()

const customerId = '7902313748' // known leaf used in prior validation

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

const waitForJobResult = async (ctx, jobId, timeoutMs) => {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    const stResp = await ctx.get(`${backendBase}/api/jobs/${jobId}`, { timeout: 30000 })
    if (stResp.ok()) {
      const stBody = await stResp.json()
      const job = stBody?.job || {}
      const status = String(job.status || '').toLowerCase()
      if (status === 'succeeded') {
        const resResp = await ctx.get(`${backendBase}/api/jobs/${jobId}/result`, { timeout: 30000 })
        expect(resResp.ok()).toBeTruthy()
        const resBody = await resResp.json()
        return resBody?.result || null
      }
      if (status === 'failed') {
        throw new Error(`Job failed: ${job.error || 'unknown error'}`)
      }
    }
    await sleep(2000)
  }
  throw new Error(`Job did not complete within ${timeoutMs}ms (job_id=${jobId})`)
}

const runPlanAndRun = async (ctx, data, timeoutMs) => {
  const resp = await ctx.post(`${backendBase}/api/chat/plan-and-run`, {
    data,
    timeout: timeoutMs,
  })
  expect(resp.ok()).toBeTruthy()
  const body = await resp.json()
  if (body.executed) return body
  if (body.job_id) {
    const result = await waitForJobResult(ctx, body.job_id, timeoutMs)
    if (result) return result
    throw new Error(`Job succeeded but returned empty result (job_id=${body.job_id})`)
  }
  // Fail with the body payload for debugging (prefer surfacing real backend errors over flakey retries).
  throw new Error(`plan-and-run did not execute and did not return a job_id: ${JSON.stringify(body).slice(0, 2000)}`)
}

test.describe('Chat planner -> SA360 fetch-and-audit', () => {
  test.setTimeout(240000)

  test('last week plan-and-run returns audit', async ({}) => {
    test.skip(!sessionId, 'KAI_SESSION_ID not set; skipping planner tests')
    const ctx = await request.newContext()
    const body = await runPlanAndRun(ctx, {
      message: 'give me last week performance',
      customer_ids: [customerId],
      session_id: sessionId,
    }, 210000)
    // Performance intent returns chat summary (no XLSX); audit intent returns file_name
    const hasFile = !!body.result?.file_name
    const isPerf = body.result?.mode === 'performance'
    expect(hasFile || isPerf).toBeTruthy()
  })

  test('yesterday plan-and-run returns audit', async ({}) => {
    test.skip(!sessionId, 'KAI_SESSION_ID not set; skipping planner tests')
    const ctx = await request.newContext()
    const body = await runPlanAndRun(ctx, {
      message: 'show me yesterday performance',
      customer_ids: [customerId],
      session_id: sessionId,
    }, 210000)
    const hasFile = !!body.result?.file_name
    const isPerf = body.result?.mode === 'performance'
    expect(hasFile || isPerf).toBeTruthy()
  })

  test('3 days ago plan-and-run returns audit', async ({}) => {
    test.skip(!sessionId, 'KAI_SESSION_ID not set; skipping planner tests')
    const ctx = await request.newContext()
    const body = await runPlanAndRun(ctx, {
      message: 'performance 3 days ago',
      customer_ids: [customerId],
      session_id: sessionId,
    }, 210000)
    const hasFile = !!body.result?.file_name
    const isPerf = body.result?.mode === 'performance'
    expect(hasFile || isPerf).toBeTruthy()
  })

  test('custom metric explicit token (FR_Intent_Clicks) is blocked with suggestions (no silent remap)', async ({}) => {
    test.skip(!sessionId, 'KAI_SESSION_ID not set; skipping planner tests')
    const ctx = await request.newContext()
    const resp = await ctx.post(`${backendBase}/api/chat/plan-and-run`, {
      data: {
        message: 'Why did FR_Intent_Clicks change week over week? Which campaigns drove it?',
        customer_ids: [customerId],
        session_id: sessionId,
      },
      timeout: 210000,
    })
    expect(resp.ok()).toBeTruthy()
    const body = await resp.json()
    expect(body.executed).toBeFalsy()
    expect(body.error).toBe('custom_metric_not_found')
    const sugs = body.analysis?.custom_metric?.suggestions || []
    expect(Array.isArray(sugs)).toBeTruthy()
    expect(sugs.length).toBeGreaterThan(0)
  })

  test('conversion action name infers a real SA360 column and returns drivers', async ({}) => {
    test.skip(!sessionId, 'KAI_SESSION_ID not set; skipping planner tests')
    const ctx = await request.newContext()

    // Pick a real SA360 conversion action from the catalog at runtime to avoid brittle hard-coding.
    const catalogResp = await ctx.get(
      `${backendBase}/api/sa360/conversion-actions?session_id=${encodeURIComponent(sessionId)}&customer_id=${customerId}`,
      { timeout: 210000 }
    )
    expect(catalogResp.ok()).toBeTruthy()
    const catalog = await catalogResp.json()
    const actions = Array.isArray(catalog.actions) ? catalog.actions : []
    expect(actions.length).toBeGreaterThan(0)

    // Prefer an action that is excluded from "Conversions" (conversions==0 but all_conversions>0),
    // because it forces the planner to use the custom metric inference path.
    const candidate =
      actions.find((a) => Number(a?.conversions || 0) === 0 && Number(a?.all_conversions || 0) > 0) ||
      actions.find((a) => Number(a?.all_conversions || 0) > 0) ||
      actions[0]

    expect(candidate?.name).toBeTruthy()
    expect(String(candidate?.metric_key || '')).toMatch(/^custom:/)

    const resp = await ctx.post(`${backendBase}/api/chat/plan-and-run`, {
      data: {
        message: `Why did ${candidate.name} change week over week? Which campaigns drove it?`,
        customer_ids: [customerId],
        session_id: sessionId,
      },
      timeout: 210000,
    })
    expect(resp.ok()).toBeTruthy()
    const body = await resp.json()
    if (!body.executed && body.job_id) {
      const polled = await waitForJobResult(ctx, body.job_id, 210000)
      expect(polled?.executed).toBeTruthy()
      // Replace body with the polled result for downstream assertions.
      Object.assign(body, polled)
    }
    expect(body.executed).toBeTruthy()
    expect(body.result?.mode).toBe('performance')
    const dq = body.result?.data_quality || {}
    expect(dq.custom_metric_inferred).toBeTruthy()
    expect(dq.custom_metric_key).toBe(candidate.metric_key)

    const drivers = body.analysis?.drivers || {}
    const campaignCount = Array.isArray(drivers.campaign) ? drivers.campaign.length : 0
    const deviceCount = Array.isArray(drivers.device) ? drivers.device.length : 0
    expect(campaignCount + deviceCount).toBeGreaterThan(0)
  })
})
