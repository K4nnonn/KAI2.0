import sys
from datetime import date, timedelta


def _load_main():
    sys.path.append("services/api")
    import main  # noqa: WPS433

    return main


def test_parse_human_date_uses_explicit_default_for_matching_last_14_days():
    """
    Deterministic QA guard:
    When the caller supplies an explicit stable window (yyyy-mm-dd,yyyy-mm-dd) as default_date_range,
    and the message asks for "last 14 days", the planner should honor the explicit stable window.

    This prevents parity/accuracy gates from drifting when SA360 data backfills for very recent days.
    """
    main = _load_main()
    default_span = "2026-02-01,2026-02-14"
    out = main._parse_human_date("give me last 14 days performance", default_span)  # noqa: WPS437
    assert out == default_span


def test_parse_human_date_keeps_dynamic_range_when_default_length_mismatch():
    main = _load_main()
    default_span = "2026-02-01,2026-02-07"  # 7 days inclusive
    out = main._parse_human_date("give me last 14 days performance", default_span)  # noqa: WPS437
    assert out != default_span
    # Must still end on today for the dynamic range behavior.
    assert out.endswith(f"{date.today():%Y-%m-%d}")


def test_parse_human_date_last_7_days_matches_expected_when_no_default():
    main = _load_main()
    end = date.today()
    start = end - timedelta(days=6)
    expected = f"{start:%Y-%m-%d},{end:%Y-%m-%d}"
    out = main._parse_human_date("past 7 days performance")  # noqa: WPS437
    assert out == expected

