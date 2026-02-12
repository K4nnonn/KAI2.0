from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path


FORBIDDEN_PATTERNS = (
    "KEY",
    "SECRET",
    "TOKEN",
    "PASSWORD",
    "CONNECTION_STRING",
    "CLIENT_SECRET",
    "REFRESH_TOKEN",
)


def fail(message: str) -> None:
    print(f"[guardrail-check] FAIL: {message}")
    raise SystemExit(1)


def require_contains(text: str, needle: str, label: str) -> None:
    if needle not in text:
        fail(f"missing {label}: {needle}")


def parse_allowlist(main_path: Path) -> list[str]:
    tree = ast.parse(main_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_ENV_EXPOSURE_ALLOWLIST":
                    if not isinstance(node.value, (ast.List, ast.Tuple)):
                        fail("_ENV_EXPOSURE_ALLOWLIST is not a literal list/tuple")
                    values: list[str] = []
                    for item in node.value.elts:
                        if isinstance(item, ast.Constant) and isinstance(item.value, str):
                            values.append(item.value)
                    return values
    fail("_ENV_EXPOSURE_ALLOWLIST not found")
    return []


def ensure_no_forbidden_keys(keys: list[str]) -> None:
    for key in keys:
        upper = key.upper()
        if any(pattern in upper for pattern in FORBIDDEN_PATTERNS):
            fail(f"forbidden env key pattern found in exposure allowlist: {key}")


def ensure_env_not_committed(repo_root: Path) -> None:
    proc = subprocess.run(
        ["git", "ls-files", ".env"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.stdout.strip():
        fail(".env is tracked by git")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    main_path = repo_root / "services" / "api" / "main.py"
    env_example = repo_root / ".env.example"
    obf_script = repo_root / "scripts" / "build_brain_gate_obfuscated.ps1"

    if not main_path.exists():
        fail(f"missing file: {main_path}")
    if not env_example.exists():
        fail(f"missing file: {env_example}")
    if not obf_script.exists():
        fail(f"missing file: {obf_script}")

    main_text = main_path.read_text(encoding="utf-8")
    env_text = env_example.read_text(encoding="utf-8")
    obf_text = obf_script.read_text(encoding="utf-8")

    require_contains(main_text, "from kai_core.shared import brain_gate as source_brain_gate", "brain_gate import")
    require_contains(main_text, "_brain_gate_allows_request(path)", "brain_gate middleware hook")
    require_contains(main_text, '@app.get("/api/diagnostics/license")', "diagnostics license route")
    require_contains(main_text, "_brain_gate_refresh(force=True)", "startup license refresh")
    require_contains(env_text, "LICENSE_ENFORCEMENT_MODE", ".env contract")
    require_contains(env_text, "BRAIN_GATE_USE_OBFUSCATED", ".env obfuscation toggle")
    require_contains(main_text, "_ENV_FORBIDDEN_KEY_PATTERNS", "env forbidden-key guard")
    require_contains(main_text, "def _env_exposure_allowlist()", "env exposure allowlist function")
    require_contains(obf_text, "brain_gate_build_manifest.txt", "obfuscation manifest")

    allowlist = parse_allowlist(main_path)
    ensure_no_forbidden_keys(allowlist)
    ensure_env_not_committed(repo_root)
    print("[guardrail-check] PASS")


if __name__ == "__main__":
    main()
