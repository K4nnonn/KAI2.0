# Traceability Map

Goal -> Task -> Evidence

G1 Cost-safe LLM usage
- Task: Add Azure budget guardrails in _call_llm and _call_azure
- Evidence: integrity_runs/20260120_183713 -> api.llm.usage.snapshot + /api/diagnostics/llm-usage

G2 Persona tone quality + non-repetition
- Task: Persona variety gate (5 prompts + uniqueness check)
- Evidence: integrity_runs/20260120_231215 -> persona.variation.uniqueness pass + tone checks pass

G3 Advisor grounding with numeric evidence
- Task: Enforce numeric metric in performance explanation + advisor check
- Evidence: integrity_runs/20260120_231215 -> api.chat.performance.grounded + api.diagnostics.advisor.performance pass

G4 Replicable integrity evidence
- Task: Integrity suite run + run_meta.json with script hashes
- Evidence: C:\Software Builds\repo_shadow\repo\integrity_runs\20260121_103800

G5 Production baseline on live backend
- Task: Deploy + /api/version + integrity run
- Evidence: /api/version git_sha=20260121_100618; integrity_runs/20260121_103800

G6 Info map legibility on live site
- Task: Scale software map + enforce font/contrast in UI smoke
- Evidence: integrity_runs/20260121_103800/gate_03_ui_smoke/stdout.txt (min font/contrast OK)

G7 Missing-data reply is deterministic (no Azure fallback)
- Task: Missing-data intent returns rules-based response with required exports list
- Evidence: verification_runs/missing_data_20260121_105659.json (model=rules)

G8 Presence scoring consistency (impressions vs activity)
- Task: Round impressions before scoring so 0 impressions cannot yield score 5
- Evidence: verification_runs/engine_presence_rounding_20260121_140128/engine_presence_rounding.txt

G9 Local LLM output quality guard (non-missing-data rows)
- Task: Reject low-content actions/rationale and fall back to original text
- Evidence: services/api/kai_core/core_logic/audit_verbalizer.py (_is_low_content) + verification_runs/verbalizer_check_20260121_142707/diagnostics_verbalizer.json
