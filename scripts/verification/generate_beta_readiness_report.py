from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

Z_95 = 1.96


def wilson_lower_bound_95(successes: int, trials: int) -> float:
    if trials <= 0:
        return 0.0
    successes = max(0, min(int(successes), int(trials)))
    p = successes / trials
    z = Z_95
    denom = 1.0 + (z * z) / trials
    center = p + (z * z) / (2.0 * trials)
    margin = z * ((p * (1 - p) + (z * z) / (4.0 * trials)) / trials) ** 0.5
    return max(0.0, min(1.0, (center - margin) / denom))


def read_text(path: Path) -> str:
    """
    Read text artifacts written by a mix of Python and PowerShell.

    Notes:
    - Some PowerShell-written JSON artifacts include a UTF-8 BOM.
    - Some PowerShell-written text artifacts (e.g., Set-Content default) are UTF-16LE.
    """
    raw = path.read_bytes()
    # Heuristic: UTF-16LE files contain many NUL bytes.
    if raw.count(b"\x00") > max(16, len(raw) // 8):
        try:
            return raw.decode("utf-16", errors="replace")
        except Exception:
            pass
    try:
        return raw.decode("utf-8-sig", errors="replace")
    except Exception:
        return raw.decode("utf-8", errors="replace")


def read_json(path: Path) -> Any:
    return json.loads(read_text(path))


def find_latest_dir(base: Path, prefix: str) -> Path | None:
    candidates = [p for p in base.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def git_rev(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True)
        return out.strip()
    except Exception:
        return "unknown"


@dataclass(frozen=True)
class ModuleDef:
    name: str
    route: str
    frontend_file: str
    ui_patterns: tuple[str, ...]
    backend_categories: tuple[str, ...]
    backend_flags: tuple[str, ...] = ()


def _match_any(line: str, patterns: Iterable[str]) -> bool:
    return any(p in line for p in patterns)


def parse_playwright_output(text: str) -> dict[str, Any]:
    passed: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("ok "):
            passed.append(line)
        elif line.startswith("-  "):
            skipped.append(line)
        elif line.startswith("not ok") or line.startswith("failed") or line.startswith("✘"):
            failed.append(line)
    return {"passed": passed, "skipped": skipped, "failed": failed}


def format_confidence(successes: int, trials: int) -> dict[str, Any]:
    score = wilson_lower_bound_95(successes, trials)
    return {
        "method": "wilson_lower_bound_95",
        "successes": int(successes),
        "trials": int(trials),
        "score": round(float(score), 4),
        "score_0_10": round(float(score) * 10.0, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verification-runs", default="verification_runs")
    parser.add_argument("--fullqa-dir", default="")
    parser.add_argument("--ui-dir", default="")
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    runs_dir = (repo_root / args.verification_runs).resolve()
    if not runs_dir.exists():
        raise SystemExit(f"verification_runs directory not found: {runs_dir}")

    fullqa = Path(args.fullqa_dir).resolve() if args.fullqa_dir else find_latest_dir(runs_dir, "fullqa_")
    ui = Path(args.ui_dir).resolve() if args.ui_dir else find_latest_dir(runs_dir, "ui_e2e_")
    if not fullqa or not ui:
        raise SystemExit(f"Could not locate fullqa/ui_e2e runs under {runs_dir}")

    summary_path = fullqa / "summary.json"
    ui_out_path = ui / "playwright_output.txt"
    if not summary_path.exists():
        raise SystemExit(f"Missing fullqa summary.json: {summary_path}")
    if not ui_out_path.exists():
        raise SystemExit(f"Missing UI playwright_output.txt: {ui_out_path}")

    summary = read_json(summary_path)
    ui_parsed = parse_playwright_output(read_text(ui_out_path))
    capability_conf = (
        summary.get("spec_eval_summary", {}).get("capability_confidence", {}) if isinstance(summary, dict) else {}
    )
    if not isinstance(capability_conf, dict):
        capability_conf = {}

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (runs_dir / f"beta_report_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    modules: list[ModuleDef] = [
        ModuleDef(
            name="Password Gate",
            route="(all)",
            frontend_file="apps/web/src/components/PasswordGate.jsx",
            ui_patterns=("Password Gate (UI)",),
            backend_categories=("api_contract", "health"),
            backend_flags=("auth_wrong_denied", "auth_correct_ok"),
        ),
        ModuleDef(
            name="Kai Chat",
            route="/",
            frontend_file="apps/web/src/pages/KaiChat.jsx",
            ui_patterns=("Kai Chat UI", "Account Picker UX", "Chat planner", "Latency budgets", "Tool Chat UI"),
            backend_categories=(
                "chat_send",
                "routing",
                "planner",
                "account_context",
                "custom_metric_inference",
                "custom_metric_catalog",
                "oauth",
                "sa360",
            ),
            backend_flags=("route_latency_ok", "plan_latency_ok", "chat_latency_ok"),
        ),
        ModuleDef(
            name="Klaudit Audit",
            route="/klaudit",
            frontend_file="apps/web/src/pages/KlauditAudit.jsx",
            ui_patterns=("Klaudit Audit (UI)", "UI smoke: audit page"),
            backend_categories=("audit", "jobs", "api_contract"),
            backend_flags=("audit_plan_and_run_ok", "multisheet_upload_ok"),
        ),
        ModuleDef(
            name="Creative Studio",
            route="/creative-studio",
            frontend_file="apps/web/src/pages/CreativeStudio.jsx",
            ui_patterns=("Creative Studio:",),
            backend_categories=("api_contract",),
            backend_flags=("creative_ok",),
        ),
        ModuleDef(
            name="PMax Deep Dive",
            route="/pmax",
            frontend_file="apps/web/src/pages/PMaxDeepDive.jsx",
            ui_patterns=("PMax Deep Dive:",),
            backend_categories=("pmax", "routing", "api_contract"),
            backend_flags=("pmax_ok",),
        ),
        ModuleDef(
            name="SERP Monitor / Competitor Intel",
            route="/serp",
            frontend_file="apps/web/src/pages/SerpMonitor.jsx",
            ui_patterns=("SERP Monitor:",),
            backend_categories=("serp", "competitor", "api_contract"),
            backend_flags=("serp_ok", "competitor_ok"),
        ),
        ModuleDef(
            name="SA360 Columns",
            route="/sa360-columns",
            frontend_file="apps/web/src/pages/Sa360Columns.jsx",
            ui_patterns=("SA360 Columns page", "SA360 Columns"),
            backend_categories=("sa360_conversion_actions", "sa360_accounts", "oauth"),
            backend_flags=("sa360_conversion_actions_ok",),
        ),
        ModuleDef(
            name="Env & Keys",
            route="/env",
            frontend_file="apps/web/src/pages/EnvManager.jsx",
            ui_patterns=("Env & Keys",),
            backend_categories=("api_contract",),
            backend_flags=("env_checked",),
        ),
        ModuleDef(
            name="Settings",
            route="/settings",
            frontend_file="apps/web/src/pages/Settings.jsx",
            ui_patterns=("Settings page",),
            backend_categories=("api_contract",),
            backend_flags=("diagnostics_ok",),
        ),
        ModuleDef(
            name="Info",
            route="/info",
            frontend_file="apps/web/src/pages/ArchitectureInfo.jsx",
            ui_patterns=("Info page",),
            backend_categories=("health",),
            backend_flags=(),
        ),
    ]

    report: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "git_commit": git_rev(repo_root),
        "fullqa_run": str(fullqa.name),
        "ui_run": str(ui.name),
        "backend_url": summary.get("backend") if isinstance(summary, dict) else None,
        "modules": [],
        "notes": [
            "Confidence is derived from the amount of passing evidence, not from subjective claims.",
            "Backend evidence comes from spec assertions + full_qa flags in summary.json.",
            "UI evidence comes from Playwright 'ok' lines in playwright_output.txt.",
        ],
    }

    ui_passed_lines: list[str] = ui_parsed.get("passed", [])

    for mod in modules:
        matched_ui = [ln for ln in ui_passed_lines if _match_any(ln, mod.ui_patterns)]
        ui_trials = len(matched_ui)
        ui_success = ui_trials  # passed-only list

        backend_evidence_total = 0
        backend_evidence_passed = 0
        backend_breakdown: list[dict[str, Any]] = []
        for cat in mod.backend_categories:
            cat_obj = capability_conf.get(cat)
            if not isinstance(cat_obj, dict):
                continue
            e_total = int(cat_obj.get("evidence_total") or 0)
            e_passed = int(cat_obj.get("evidence_passed") or 0)
            backend_evidence_total += e_total
            backend_evidence_passed += e_passed
            backend_breakdown.append(
                {
                    "category": cat,
                    "evidence_total": e_total,
                    "evidence_passed": e_passed,
                    "confidence": cat_obj.get("confidence"),
                }
            )

        flag_values: dict[str, Any] = {}
        for flag in mod.backend_flags:
            if isinstance(summary, dict) and flag in summary:
                flag_values[flag] = summary.get(flag)

        combined_successes = int(ui_success + backend_evidence_passed)
        combined_trials = int(ui_trials + backend_evidence_total)

        report["modules"].append(
            {
                "name": mod.name,
                "route": mod.route,
                "frontend_file": mod.frontend_file,
                "ui": {
                    "matched_tests": matched_ui,
                    "passed_count": ui_success,
                    "trials_count": ui_trials,
                    "confidence": format_confidence(ui_success, ui_trials),
                },
                "backend": {
                    "categories": backend_breakdown,
                    "flags": flag_values,
                    "evidence": {
                        "passed": backend_evidence_passed,
                        "total": backend_evidence_total,
                        "confidence": format_confidence(backend_evidence_passed, backend_evidence_total),
                    },
                },
                "combined_confidence": format_confidence(combined_successes, combined_trials),
            }
        )

    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_lines: list[str] = []
    md_lines.append("# Beta Readiness Report\n")
    md_lines.append(f"- Generated: `{report['generated_at']}`")
    md_lines.append(f"- Git commit: `{report['git_commit']}`")
    md_lines.append(f"- Full QA run: `{report['fullqa_run']}`")
    md_lines.append(f"- UI E2E run: `{report['ui_run']}`")
    md_lines.append(f"- Backend: `{report.get('backend_url')}`\n")
    md_lines.append("## Modules\n")
    md_lines.append("| Module | Route | UI Evidence (tests) | Backend Evidence (assertions) | Combined Confidence (0-10) |")
    md_lines.append("|---|---|---:|---:|---:|")
    for m in report["modules"]:
        ui_cnt = int(m["ui"]["passed_count"])
        be_total = int(m["backend"]["evidence"]["total"])
        combined_0_10 = float(m["combined_confidence"]["score_0_10"])
        md_lines.append(f"| {m['name']} | `{m['route']}` | {ui_cnt} | {be_total} | {combined_0_10:.2f} |")
    md_lines.append("\n## Notes\n")
    for n in report["notes"]:
        md_lines.append(f"- {n}")
    (out_dir / "report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
