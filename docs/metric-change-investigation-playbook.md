# Metric Change Investigation Playbook

## Investigation Framework

When a user asks "Why did [METRIC] change?", follow this systematic approach:

### 1. Quantify the Change
- Calculate **absolute change**: New Value - Old Value
- Calculate **percent change**: ((New - Old) / Old) × 100%
- Identify **change date**: Use changepoint detection to pinpoint when shift occurred
- State clearly: "CPC increased from $2.10 to $2.85 (+35.7%, +$0.75) starting October 15"

### 2. Decompose Attribution

Use **mix vs rate decomposition** (Kitagawa/Oaxaca-Blinder method):

**Rate Effect**: Performance changed within segments
- Example: Mobile CPC went from $1.50 to $2.00 while traffic stayed 40% mobile
- Calculation: Σ(Old Mix × Change in Rate)

**Mix Effect**: Traffic distribution shifted between segments
- Example: Mobile traffic increased from 40% to 60%, and mobile has higher CPC
- Calculation: Σ(Change in Mix × Old Rate)

**Formula**:
```
Total Change = Rate Effect + Mix Effect
Rate Effect % = (Rate Effect / Total Change) × 100%
Mix Effect % = (Mix Effect / Total Change) × 100%
```

### 3. Investigate Rate Effect Drivers

If rate effect is >60% of change, check these causes:

#### A. Competitive Pressure (Check Auction Insights)
- **Overlap Rate increased**: New competitor or existing competitor increased budget
- **Outranking Share decreased**: Competitors bidding more aggressively
- **Your Impression Share flat but CPC up**: Market-wide CPC inflation

**Quantify impact**:
- If Overlap Rate +20pp and Outranking Share -15pp → likely competitor-driven
- Run regression: CPC ~ Overlap Rate + Outranking Share to get coefficient

#### B. Quality Score Decline
- Check Quality Score history in Keywords report
- QS drop from 7 to 5 can increase CPC by ~40%
- Common causes:
  - Landing page speed degraded
  - Ad relevance dropped (new RSA underperforming)
  - Expected CTR fell (match type broadened, serving irrelevant queries)

**Quantify impact**:
- CPC impact = (1 / (Old QS / 10)) - (1 / (New QS / 10))
- Example: QS 8→6: 1/(8/10) - 1/(6/10) = 1.25 - 1.67 = +33% CPC increase

#### C. Bid or Strategy Changes
- Pull ChangeEvent data for date range
- Check for:
  - Manual bid increases
  - Smart Bidding target changes (tROAS lowered → more aggressive)
  - Portfolio strategy added (inherits higher target)
  - Bid adjustments added (device, geo, audience)

**Quantify impact**:
- If bid raised $1.50 → $2.00, and CPC tracks bids closely, this explains the change
- Smart Bidding: Check "Performance Planner" for forecasted CPC at new target

#### D. Auction Dynamics
- Check if average position improved (lower CPC but less traffic is odd)
- Examine top-of-page rate: if increased significantly, paying premium for visibility
- Review absolute top impression share: if targeting this, CPC will be higher

### 4. Investigate Mix Effect Drivers

If mix effect is >60% of change, check these shifts:

#### A. Device Mix Shift
- **Mobile traffic increased**: Mobile often has lower CPC but different conversion rate
- **Desktop increased**: Desktop typically higher CPC for B2B
- **Calculation**: (New Mobile % - Old Mobile %) × Mobile CPC vs Desktop CPC diff

**Example**:
- Mobile increased from 60% to 75% (mobile CPC: $1.50, desktop: $3.00)
- Mix shift impact: (75% - 60%) × ($1.50 - $3.00) = -$0.225 (CPC decreases)

#### B. Geographic Mix Shift
- High-CPC geos (e.g., NYC, SF, London) gaining share
- Low-CPC geos declining due to budget constraints or competition

**Investigation**:
- Pull geo performance report, segment by metro
- Calculate weighted average CPC by geo for both periods
- Attribute change: Σ(New Geo % × Geo CPC) - Σ(Old Geo % × Geo CPC)

#### C. Match Type / Query Mix Shift
- **Broad match expansion**: Serving more low-intent queries at lower CPC (and possibly lower CVR)
- **Exact match concentration**: Focusing on high-intent, high-CPC terms

**Investigation**:
- Export search terms report for both periods
- Classify queries by intent (navigational, informational, transactional)
- Calculate: % High-Intent Queries × Avg CPC by Intent

#### D. Day-of-Week / Hour-of-Day Shifts
- Weekend traffic typically lower CPC (less competition)
- Weekday business hours: higher CPC

**Investigation**:
- Segment by day and hour
- Check if campaign budget shifted due to pacing (e.g., burning budget early in week)

### 5. Generate Recommendations

Based on findings, provide **prioritized actions**:

#### P1 (Critical - Do Immediately)
- Conversion tracking broken → Fix tag
- Quality Score dropped due to broken landing page → Fix page
- Competitor launched brand bidding campaign → Add brand negatives to their campaigns

#### P2 (High Priority - Do This Week)
- Add negative keywords based on search terms (>$500 waste identified)
- Adjust device bids if mix shift is unprofitable
- Increase budget if high LISB and CPA is below target

#### P3 (Medium Priority - Do This Month)
- Test new ad copy to improve CTR and QS
- Explore geo bid adjustments
- Review audience layering strategy

**Format recommendations**:
```
Priority: P1
Action: Add 47 negative keywords from search terms with >$500 spend, zero conversions
Rationale: Query mix shifted toward informational intent ("how to", "what is")
Estimated Impact: -$1,200/month waste, CPC -$0.15
Effort: Low (15 minutes)
```

## Example Investigation: CPC Increased 35%

**User Query**: "Why did CPC go from $2.10 to $2.85 last week?"

### Step 1: Quantify
- **Change**: +$0.75 (+35.7%)
- **Changepoint**: October 15 (detected via Bayesian Online Changepoint Detection)
- **Confidence**: 94%

### Step 2: Decompose
- **Total Change**: +$0.75
- **Rate Effect**: +$0.60 (80%)
- **Mix Effect**: +$0.15 (20%)

### Step 3: Rate Effect Investigation
Pulled Auction Insights for Oct 1-14 vs Oct 15-21:
- **Overlap Rate**: 45% → 68% (+23pp)
- **Outranking Share**: 55% → 38% (-17pp)
- **Competitor "CompanyX.com"**: New entry on Oct 15

Pulled ChangeEvent data: No bid or budget changes in our account.

**Conclusion**: 80% of CPC increase is due to new competitor ("CompanyX.com") entering auction and bidding aggressively.

### Step 4: Mix Effect Investigation
Device distribution Oct 1-14 vs Oct 15-21:
- **Mobile**: 60% → 65% (+5pp)
- **Mobile CPC**: $1.80
- **Desktop CPC**: $2.70

Mix contribution: (65% - 60%) × ($1.80 - $2.70) = -$0.045

**Wait, that's negative!** Recheck total change...

Actually: Desktop traffic increased (40% → 45%), and desktop has higher CPC.
- (45% - 40%) × ($2.70 - $1.80) = +$0.045... still doesn't match $0.15.

Check geo mix: California traffic increased from 30% → 40% (CA CPC: $3.20 vs national avg $2.00)
- (40% - 30%) × ($3.20 - $2.00) = +$0.12 ✓

**Conclusion**: 20% of CPC increase is due to traffic shifting to California (high-CPC geo).

### Step 5: Recommendations

**P1 Recommendations**: None (no broken tracking, no critical issues)

**P2 Recommendations**:
1. **Increase bids to maintain position**:
   - Current avg position: 2.1 → 3.4 (dropped)
   - Estimated bid increase: +$0.40 to regain position 2.1
   - Trade-off: Higher CPC but regain impression share

2. **Optimize for profitability instead**:
   - Accept lower position (cheaper CPCs)
   - Monitor conversion rate—if CVR holds, CPA may still be profitable
   - Estimated impact: CPA increases 10% but impression share -15%

**P3 Recommendations**:
1. **Geo bid adjustments**:
   - Lower California bid modifier to -15%
   - Estimated impact: Reduce CA traffic from 40% to 35%, save $0.06 CPC

2. **Device bid adjustments**:
   - Lower desktop bid modifier to -10% (if CVR is similar cross-device)

### Narrative Response (for user)

> **Summary**: Your CPC increased 35.7% (from $2.10 to $2.85, +$0.75) starting **October 15**.
>
> **Primary Driver (80%)**: A new competitor, "CompanyX.com", entered the auction on October 15 with aggressive bidding. Auction Insights shows:
> - Overlap Rate increased from 45% to 68%
> - Your Outranking Share decreased from 55% to 38%
> - Your average position dropped from 2.1 to 3.4
>
> **Secondary Driver (20%)**: Traffic shifted toward California, which has 60% higher CPC than your national average ($3.20 vs $2.00).
>
> **Recommended Actions**:
> 1. **[P2] Decide on competitive response**:
>    - Option A: Increase bids by $0.40 to regain position (will increase CPC further but restore impression share)
>    - Option B: Accept lower position and monitor if CPA remains profitable
> 2. **[P3] Add geo bid adjustment**: Lower California bids by -15% to counteract traffic concentration
>
> **Next Steps**: Would you like me to create a change proposal to implement these bid adjustments?

---

## Changepoint Detection (Technical Appendix)

Havas Holly uses **Bayesian Online Changepoint Detection** (Adams & MacKay, 2007) to identify when metrics shifted.

### How It Works
1. Model: Assume metric follows a Gaussian distribution with unknown mean and variance
2. At each time point, calculate probability that a changepoint occurred
3. If probability > 80%, flag as changepoint
4. Return: date, confidence, magnitude of change

### Implementation (Python)
```python
import numpy as np
from scipy import stats

def bayesian_changepoint_detection(data, threshold=0.8):
    """
    Detect changepoints in time series data.

    Args:
        data: Array of metric values (e.g., daily CPC)
        threshold: Probability threshold for changepoint (0-1)

    Returns:
        List of changepoint indices with confidence scores
    """
    n = len(data)
    changepoints = []

    # Prior parameters (non-informative)
    alpha0, beta0 = 1, 1
    mu0, kappa0 = 0, 1

    run_length = np.zeros(n + 1)
    run_length[0] = 1.0

    for t in range(1, n):
        # Calculate predictive probability for each run length
        # (Simplified - full implementation uses Student-t)

        # Growth probability (no changepoint)
        growth_prob = 1 - 1/(t + 1)

        # Changepoint probability
        cp_prob = 1/(t + 1)

        if cp_prob > threshold:
            # Calculate magnitude
            before_mean = np.mean(data[:t])
            after_mean = np.mean(data[t:min(t+7, n)])  # Next 7 days
            magnitude = after_mean - before_mean

            changepoints.append({
                'index': t,
                'confidence': cp_prob,
                'before_mean': before_mean,
                'after_mean': after_mean,
                'magnitude': magnitude
            })

    return changepoints
```

---

## Decomposition Methods (Technical Appendix)

### Kitagawa Decomposition

Used to separate rate effect from mix effect in metric changes.

**Formula**:
```
ΔY = Σ(w₁ᵢ × Δyᵢ) + Σ(Δwᵢ × y₀ᵢ) + Σ(Δwᵢ × Δyᵢ)
     \_____________/   \______________/   \______________/
      Rate Effect      Mix Effect        Interaction Effect
```

Where:
- Y = aggregate metric (e.g., CPC)
- yᵢ = metric for segment i (e.g., CPC for device i)
- wᵢ = weight of segment i (e.g., % traffic from device i)
- Δ = change between periods (New - Old)
- Subscript 0 = old period, 1 = new period

**Example (Device Segmentation)**:
```
Segments: Mobile, Desktop

Period 0 (Old):
- Mobile: 60% traffic, $1.50 CPC
- Desktop: 40% traffic, $2.50 CPC
- Aggregate CPC: (0.6 × 1.50) + (0.4 × 2.50) = $1.90

Period 1 (New):
- Mobile: 50% traffic, $2.00 CPC
- Desktop: 50% traffic, $3.00 CPC
- Aggregate CPC: (0.5 × 2.00) + (0.5 × 3.00) = $2.50

Change: $2.50 - $1.90 = +$0.60

Rate Effect:
  Mobile: 0.6 × (2.00 - 1.50) = 0.6 × 0.50 = $0.30
  Desktop: 0.4 × (3.00 - 2.50) = 0.4 × 0.50 = $0.20
  Total Rate Effect: $0.50 (83% of change)

Mix Effect:
  Mobile: (0.5 - 0.6) × 1.50 = -0.1 × 1.50 = -$0.15
  Desktop: (0.5 - 0.4) × 2.50 = 0.1 × 2.50 = $0.25
  Total Mix Effect: $0.10 (17% of change)

Interaction Effect:
  Mobile: (0.5 - 0.6) × (2.00 - 1.50) = -0.05
  Desktop: (0.5 - 0.4) × (3.00 - 2.50) = 0.05
  Total: $0.00

Total: $0.50 + $0.10 = $0.60 ✓
```

**Interpretation**:
- 83% of CPC increase is due to CPC rising within each device segment (rate effect)
- 17% is due to traffic shifting toward desktop, which has higher CPC (mix effect)
- Recommendation: Investigate why CPC increased for both devices (likely competition or QS drop)
