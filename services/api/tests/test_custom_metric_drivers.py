import sys

import pandas as pd


def _load_main():
    # Import main.py from the services/api package path without starting the server.
    sys.path.append("services/api")
    import main  # noqa: WPS433

    return main


def test_custom_metric_driver_breakdown_allows_prev_only_when_current_zero():
    """
    Regression guard:
    When a conversion action drops to 0 in the current window, the segmented report may contain
    no rows for that action. Driver breakdowns should still surface the previous-window drivers
    (e.g., "which campaigns drove the drop").
    """
    main = _load_main()

    metric_key = "custom:store_visits"
    current = pd.DataFrame(
        [
            {
                "campaign.name": "Campaign A",
                "segments.device": "DESKTOP",
                "segments.conversion_action_name": "Purchases",
                "metrics.all_conversions": 12,
            }
        ]
    )
    previous = pd.DataFrame(
        [
            {
                "campaign.name": "Campaign A",
                "segments.device": "DESKTOP",
                "segments.conversion_action_name": "Store visits",
                "metrics.all_conversions": 100,
            },
            {
                "campaign.name": "Campaign B",
                "segments.device": "MOBILE",
                "segments.conversion_action_name": "Store visits",
                "metrics.all_conversions": 60,
            },
        ]
    )

    frames_current = {"campaign_conversion_action": current}
    frames_previous = {"campaign_conversion_action": previous}

    out = main._compute_custom_metric_breakdown(  # noqa: WPS437
        frames_current,
        frames_previous,
        ["campaign.name", "campaign.id", "campaign"],
        metric_key,
    )
    assert isinstance(out, list)
    assert out, "expected non-empty driver breakdown when previous window has the custom metric"

    # Expect Campaign A to show a drop from 100 -> 0
    a = next((row for row in out if row.get("name") == "Campaign A"), None)
    assert a is not None
    assert a.get("previous") == 100
    assert a.get("current") == 0.0
    assert a.get("change") == -100.0

