# Paid Search Best Practices

## Campaign Structure

### Naming Conventions
- Use consistent naming: `[Platform]_[Country]_[Campaign Type]_[Product/Service]_[Match Type]`
- Example: `GOOG_US_Brand_Shoes_Exact`
- Include dates for promotional campaigns: `GOOG_US_BlackFriday2024_Electronics`

### Account Organization
1. **Separate Brand and Non-Brand**: Always isolate brand campaigns to control messaging and bidding
2. **Geography Split**: Create separate campaigns per major market for budget control
3. **Device Strategy**:
   - Mobile-first for local intent
   - Desktop for high-value B2B
   - Consider device bid adjustments vs separate campaigns (prefer adjustments for simplicity)

### Ad Group Granularity
- **Single Keyword Ad Groups (SKAGs)**: Use for high-value keywords requiring specific ad copy
- **Theme-based Ad Groups**: Group 5-10 related keywords with shared intent
- **Avoid overly broad ad groups**: More than 20 keywords indicates poor structure

## Bidding & Budget

### Smart Bidding Readiness
To use tROAS or tCPA successfully:
- **Minimum conversions**: 30 conversions in last 30 days per campaign
- **Consistent data**: Avoid campaigns with erratic conversion patterns
- **Proper attribution**: Ensure conversion tracking is accurate and complete

### Budget Pacing Issues
When "Limited by Budget" appears:
1. Check if campaign is actually hitting targets despite the warning
2. Analyze Lost IS (Budget) - if <10%, consider leaving as-is
3. Calculate true CPL/CPA before increasing spend
4. Consider day-parting to focus budget on high-performing hours

### Bid Strategy Transitions
- Allow **14 days** for learning phase after strategy change
- Don't panic if performance dips days 3-7 (expected)
- Keep historical bid adjustments; algorithms incorporate them
- Set appropriate constraints (max CPC, portfolio target) to avoid runaway spending

## Quality Score Optimization

### QS Components (Google Ads)
1. **Expected CTR** (highest weight): Improve with relevant ad copy and match types
2. **Ad Relevance**: Ensure keywords appear in headlines
3. **Landing Page Experience**: Speed + content relevance + mobile-friendliness

### Improving QS from 3-4 to 7+
- **Pause broad match keywords** with low CTR (<2%)
- **Add negative keywords** aggressively (search term report weekly)
- **Rewrite ads** to include keyword in Headline 1
- **Check mobile landing page speed** (target <3s load time)
- **Ensure keyword in H1 tag** on landing page

## Conversion Tracking

### Enhanced Conversions (Google Ads)
- **Why critical**: Improves attribution in privacy-focused browsers
- **Implementation**: Hash first-party data (email, phone) and send to Google
- **Validation**: Check status in Conversions UI - should show "Recording enhanced conversions"
- **Common issue**: HTTPS required; gtag.js must be in page head

### Microsoft Advertising UET
- **Tag placement**: Global site tag on all pages, event snippet on conversion page
- **Testing**: Use UET Tag Helper Chrome extension
- **Event validation**: Allow 24-48 hours for events to appear in UI
- **Tip**: Send value and currency with all conversion events for ROAS bidding

### Offline Conversions
- **Use case**: Phone calls, in-store purchases, sales cycle >90 days
- **Upload frequency**: Weekly minimum, daily preferred
- **GCLID persistence**: Store GCLID in CRM for at least 90 days
- **Validation**: Check import errors in UI and fix rejected rows

## Auction Insights Interpretation

### Key Metrics
- **Impression Share**: Your share of total eligible impressions
- **Overlap Rate**: % of time competitor appeared when you did
- **Position Above Rate**: % of time competitor ranked higher
- **Top of Page Rate**: % of impressions at top of SERP
- **Outranking Share**: % of time you ranked higher than competitor (inverse of Position Above)

### When Overlap Rate Increases Suddenly
- Competitor may have increased bids or budget
- Check if your absolute IS declined (true competition) or stayed flat (market expansion)
- Examine search term report for new query patterns
- Review CPC trends - if CPC rising without IS loss, market is heating up

### Action Thresholds
- **Overlap >70% + Outranking <30%**: Competitor is dominating; consider bid increase or better ad copy
- **Top of Page <50%**: Examine rank and CPC; may need bid lift or QS improvement
- **IS <40%**: Check for budget constraints or low QS causing auction entry failures

## Keyword Match Types & Negatives

### Match Type Strategy (Post-BMM Sunset)
- **Exact match**: High-intent, proven converters only
- **Phrase match**: Use for most keywords; replaces old BMM
- **Broad match**: Only with Smart Bidding and 50+ conversions/month

### Negative Keyword Hygiene
- **Review frequency**: Weekly for new campaigns, bi-weekly for mature
- **Campaign-level negatives**: Generic exclusions (e.g., "free", "job", "salary" for B2B SaaS)
- **Ad group negatives**: Cross-pollination prevention (e.g., negative "women" in men's shoe ad group)
- **Exact match negatives**: High-spend, zero-conversion queries

### Search Term Report Analysis
Sort by:
1. **Spend, zero conversions**: Immediate negative candidates
2. **Impressions >1000, CTR <1%**: Relevance issue; add negative or pause keyword
3. **Conversions but high CPA**: May need separate campaign with lower target

## RSA (Responsive Search Ad) Best Practices

### Asset Requirements
- **Headlines**: 10-15 unique headlines (15 maximum)
- **Descriptions**: 3-4 unique descriptions (4 maximum)
- **Pinning**: Pin Headline 1 only if brand/compliance requires; avoid over-pinning

### Asset Diversity
- Include keywords in at least 3 headlines
- Vary length: short (25-30 char) and long (60+ char)
- Mix benefit-driven, feature-driven, and CTA-driven copy
- Test price/offer callouts in headlines

### Performance Review
- **Asset rating**: Review "Low" rated assets monthly; consider replacing
- **Combination report**: Identify top 3 combinations and create manual ads as backup
- **Impr. share**: Ensure >70% impression share for RSAs to get sufficient learning

## Account Audit Checklist

### Structure
- [ ] Naming conventions consistent
- [ ] Brand/non-brand separated
- [ ] Ad groups have <20 keywords each
- [ ] Campaigns aligned to business goals (not arbitrary)

### Measurement
- [ ] All conversion actions properly tagged
- [ ] Enhanced Conversions / UET implemented
- [ ] Conversion values accurate (if using ROAS bidding)
- [ ] Audience tags firing on all pages

### Bidding & Budget
- [ ] Smart Bidding campaigns have 30+ conversions/month
- [ ] Portfolio strategies used where appropriate (>5 campaigns with shared goal)
- [ ] Budget pacing shows "Eligible (learning)" or no badge
- [ ] Lost IS (Budget) <10% or campaign is hitting CPA targets

### Creative
- [ ] RSAs have 10+ headlines in all ad groups
- [ ] Asset ratings mostly "Good" or "Best"
- [ ] Sitelinks, callouts, structured snippets enabled at account level
- [ ] Image/video assets added (for responsive display expansion)

### Query Management
- [ ] Search terms reviewed in last 14 days
- [ ] Campaign-level negative list applied (common exclusions)
- [ ] Broad match only used with Smart Bidding
- [ ] Low-performing queries (high spend, zero conv.) paused

## Common Performance Issues & Fixes

### CPC Increased Without Change
**Possible causes**:
1. **Competitor activity**: Check Auction Insights for Overlap Rate increase
2. **Seasonality**: Compare YoY for same period
3. **Quality Score drop**: Check QS history in Keywords report
4. **Match type expansion**: Phrase/broad serving broader queries

**Investigation steps**:
1. Pull ChangeEvent API data (Google Ads) or Change History (UI) for bid/budget changes
2. Run Auction Insights for date range showing CPC increase
3. Export search terms and compare distribution pre/post increase
4. Check device/geo mix shifts (mobile vs desktop CPC difference)

### Conversions Dropped Suddenly
**Possible causes**:
1. **Tracking broken**: Check last pixel fire timestamp
2. **Attribution change**: Google Ads default changed to data-driven (7-day window)
3. **Landing page change**: Deployment broke form or added friction
4. **Traffic mix shift**: Lower-funnel keywords paused, broad match serving informational queries

**Investigation steps**:
1. Verify conversion tag firing (Google Tag Assistant, UET Tag Helper)
2. Compare GA4 conversions to platform conversions (should be ±10%)
3. Review landing page changes in last 7 days
4. Analyze search term report for query intent shift

### Lost Impression Share Budget (LISB) High
**Not always bad**:
- If campaign hitting CPA targets with 40%+ LISB, you're filtering out expensive impressions
- Budget constraint can act as implicit bid cap with Smart Bidding

**When to increase budget**:
- CPA well below target AND LISB >20%
- High-value keywords show "Below first page bid" due to budget spreading thin

**Alternatives to budget increase**:
1. Add negative keywords to reduce waste
2. Adjust day-parting to focus budget on peak hours
3. Reduce bids on low-performing devices/geos instead of increasing budget

## Resources & Further Reading

- **Google Ads API**: Use ChangeEvent resource to track all account modifications
- **Microsoft Advertising API**: GetAuctionInsightData for competitive data
- **Conversion tracking**: Google Tag Manager + server-side tagging for privacy compliance
- **Attribution**: Understand data-driven attribution requires 3,000+ conversions + 300+ conversion path variations
