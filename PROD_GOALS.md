# Production Goals

G1. Cost-safe LLM usage (mindful Azure fallback with caps)
- Status: Met
- Evidence: /api/diagnostics/llm-usage includes azure_budget snapshot; integrity_runs/20260120_183713 -> api.llm.usage.snapshot

G2. Persona tone quality + non-repetition across prompts
- Status: Met
- Evidence: integrity_runs/20260121_103800 -> persona.variation.uniqueness pass + tone checks pass

G3. Advisor grounding with numeric evidence
- Status: Met
- Evidence: integrity_runs/20260121_103800 -> api.chat.performance.grounded + api.diagnostics.advisor.performance pass

G4. Replicable integrity evidence with script hashes
- Status: Met
- Evidence: integrity_runs/20260121_103800/run_meta.json + results.json

G5. Production baseline on live backend
- Status: Met
- Evidence: /api/version git_sha=20260121_100618 + integrity_runs/20260121_103800

G6. Info map legibility + contrast gate on live site
- Status: Met
- Evidence: integrity_runs/20260121_103800/gate_03_ui_smoke/stdout.txt (min font/contrast OK)

G7. Deterministic missing-data reply (no Azure fallback)
- Status: Met
- Evidence: verification_runs/missing_data_20260121_105659.json (model=rules)

G8. Presence scoring consistency (0 impressions cannot be "active")
- Status: Met
- Evidence: verification_runs/engine_presence_rounding_20260121_140128/engine_presence_rounding.txt

G9. Local LLM low-content guard (data-present rows)
- Status: Met
- Evidence: verification_runs/verbalizer_check_20260121_142707/diagnostics_verbalizer.json
