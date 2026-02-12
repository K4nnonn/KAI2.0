import importlib.util
from pathlib import Path
from unittest import TestCase, main, mock


ROOT = Path(__file__).resolve().parents[2]
MAIN_PATH = ROOT / "services" / "api" / "main.py"

spec = importlib.util.spec_from_file_location("kai_main", str(MAIN_PATH))
kai_main = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(kai_main)


class AdaptiveRegressionTests(TestCase):
    def test_humanize_relative_date_tokens(self):
        raw = "No date specified; defaulting to LAST_7_DAYS."
        out = kai_main._humanize_relative_date_tokens(raw)
        self.assertIn("last 7 days", out)
        self.assertNotIn("LAST_7_DAYS", out)

    def test_normalize_reply_text_humanizes_date_ranges(self):
        raw = "Window 2026-02-02,2026-02-08 vs LAST_30_DAYS."
        out = kai_main._normalize_reply_text(raw)
        self.assertIn("2026-02-02 to 2026-02-08", out)
        self.assertIn("last 30 days", out.lower())
        self.assertNotIn("LAST_30_DAYS", out)

    def test_llm_advise_keeps_conversational_reply_without_hard_drop(self):
        payload = {
            "summary_seed": "Conversions up 11.7%, CTR down 11.7%.",
            "result": {"current": {"conversions": 383.0, "ctr": 12.66}},
        }
        with mock.patch.object(
            kai_main,
            "_call_llm",
            return_value=("Focus first on query quality and audience overlap.", {"model": "azure"}),
        ):
            reply, meta = kai_main._llm_advise_performance(
                "What are your recommendations from the data to improve performance?",
                payload,
            )
        self.assertIsNotNone(reply)
        self.assertIn("Option A", reply)
        self.assertIn("Option B", reply)
        self.assertRegex(reply.lower(), r"recommend|start with|monitor")
        self.assertEqual(meta.get("model"), "azure")

    def test_llm_advise_humanizes_internal_date_tokens(self):
        payload = {
            "summary_seed": "No date specified; defaulting to LAST_7_DAYS.",
            "result": {"current": {"conversions": 383.0}},
        }
        with mock.patch.object(
            kai_main,
            "_call_llm",
            return_value=(
                "Option A: Keep current setup for LAST_7_DAYS. Recommendation: start with bid cleanup. Monitor conversion trend.",
                {"model": "azure"},
            ),
        ):
            reply, _ = kai_main._llm_advise_performance(
                "Optimize this account performance",
                payload,
            )
        self.assertIsNotNone(reply)
        self.assertIn("last 7 days", reply.lower())
        self.assertNotIn("LAST_7_DAYS", reply)

    def test_resolve_account_context_skips_alias_load_for_explicit_id_with_name(self):
        with mock.patch.object(
            kai_main,
            "_load_account_aliases",
            side_effect=AssertionError("aliases should not be loaded"),
        ):
            ids, acct, notes, candidates = kai_main._resolve_account_context(
                message="analyze pmax placements",
                customer_ids=["7902313748"],
                account_name="Havas_Shell_GoogleAds_US_Mobility Loyalty",
                explicit_ids=True,
                session_id="qa-latency-check",
            )
        self.assertEqual(ids, ["7902313748"])
        self.assertEqual(acct, "Havas_Shell_GoogleAds_US_Mobility Loyalty")
        self.assertEqual(candidates, [])
        self.assertTrue(notes is None or isinstance(notes, str))


if __name__ == "__main__":
    main()
