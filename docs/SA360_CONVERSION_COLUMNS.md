# SA360 Conversion Columns (Conversion Actions Catalog)

This document describes what Kai can discover from SA360 via API for "conversion-like" columns, and how to use them in chat.

## What Kai Pulls

Kai's SA360 "conversion columns" browser is backed by:

- `GET /api/sa360/conversion-actions`

Each entry is a conversion action (often shown in SA360 as a conversion column or as a component of a custom conversion column).

### Fields Returned Per Conversion Action

- `metric_key` (example: `custom:store_visits`)
- `name` (human-readable conversion action name)
- `category`
- `status`
- `conversions` / `conversions_value`
- `all_conversions` / `all_conversions_value`
- `cross_device_conversions` / `cross_device_conversions_value`

Important:

- `metrics.conversions` is typically "Primary conversions".
- `metrics.all_conversions` includes all conversion actions (including Store Visits / Local actions that often show `conversions=0` but `all_conversions>0`).

## Snapshot (Evidence)

Snapshot generated: 2026-02-07

Inputs:

- SA360 customer/account: Havas_Shell_GoogleAds_US_Mobility Loyalty (`7902313748`)
- Date range: `LAST_30_DAYS`

Results:

- Total conversion actions discovered: 361
- Actions with `all_conversions > 0`: 55

### Top Conversion Actions by `all_conversions` (LAST_30_DAYS)

- 151,959 all_conv - `custom:local_actions_directions` - Local actions - Directions
- 111,911.89 all_conv - `custom:store_visits` - Store visits
- 109,345 all_conv - `custom:local_actions_other_engagements` - Local actions - Other engagements
- 99,380 all_conv - `custom:store_sales` - Store sales
- 26,359 all_conv - `custom:clicks_to_call` - Clicks to call
- 24,855.71 all_conv - `custom:shell_global_page_view` - Shell Global - Page View
- 13,471 all_conv - `custom:all_shell_glo_visits_allpages_std_2023_12_20` - ALL_Shell_GLO_Visits_AllPages_STD_2023-12-20
- 6,846 all_conv - `custom:all_shell_glo_visits_allpages_unq_2023_11_30` - ALL_Shell_GLO_Visits_Allpages_UNQ_2023-11-30
- 5,811 all_conv - `custom:local_actions_website_visits` - Local actions - Website visits
- 3,274 all_conv - `custom:google_analytics_engaged_30s_342943257_ga4_shell_us_shell_united_states_of_america` - Google Analytics - engaged_30s - (342943257) GA4 - shell.us - Shell (United States of America)
- 2,641 all_conv - `custom:shell_us_canada_android_session_begin_9_30` - Shell US & Canada (Android) Session Begin 9.30
- 2,127 all_conv - `custom:eng_shell_usa_onclick_fuel_rewards_sign_up_unique_2024_11_22` - ENG_Shell_USA_OnClick-Fuel-Rewards-Sign-Up_Unique_2024-11-22

### Fuel Rewards Related (simple "fuel" string match)

- 2,127 all_conv / 1,681 conv - `custom:eng_shell_usa_onclick_fuel_rewards_sign_up_unique_2024_11_22`
- 602 all_conv - `custom:google_analytics_renewable_race_fuel_app_fr_342943257_ga4_shell_us_shell_united_states_of_america`
- 1 all_conv - `custom:eng_shell_usa_wholesale_fuel_b2b_landing_page_unique_2024_11_22`

## How Beta Testers Should Use This

1. In Kai Chat, connect SA360 (Google OAuth) and select your account by name.
2. Open SA360 Columns in the left nav.
3. Search for the action you want (examples: "store visits", "directions", "fuel rewards").
4. Copy the `metric_key` and reference it explicitly in chat when needed.

Example prompts:

- "Why did `custom:store_visits` change week over week? Which campaigns drove it?"
- "Compare `custom:local_actions_directions` week over week and explain the drivers."

