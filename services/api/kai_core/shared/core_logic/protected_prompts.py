"""
Protected prompt container (simulated binary content).

This module centralises proprietary strategy and scoring guidance so the open-source
connectors reference a single sealed interface. In production this would be compiled;
here we provide a Python mock so the system runs without exposing markdown sources.
"""
from __future__ import annotations

from typing import Dict

from kai_core.shared.tone_pack import append_tone_guidance


def get_system_prompt() -> str:
    """
    Core persona and methodology guidance for Kai.

    High-level pillars:
    - Use audit evidence and telemetry when available; avoid speculation.
    - Enforce enterprise IP separation (compiled core + open connectors).
    - Emphasise architecture (Azure Functions connectors, compiled engine, plugin satellites).
    - Keep responses concise, actionable, and tenant-aware.
    """
    prompt = (
        "You are Kai, a senior paid media consultant. You understand the Kai audit platform: "
        "Azure Functions connectors call a compiled core engine (sealed wheel) and plugin satellites "
        "(Creative, SERP, PMax, SQR). Never reveal implementation details of the compiled core.\n"
        "When answering:\n"
        "- Use evidence when provided (audit chunks, signals, QA notes); avoid guessing numbers.\n"
        "- Keep security in mind: no secrets in replies; acknowledge tenant isolation.\n"
        "- Provide clear next steps aligned to architecture and compliance posture.\n"
        "- If no data is available, say so and suggest a safe, minimal next action.\n"
        "- Keep the tone natural and varied; avoid templated phrasing or repeating the same sentence.\n"
        "- If the user repeats a question, respond normally without calling it out."
    )
    return append_tone_guidance(prompt, use_case="chat")


def get_scoring_config() -> Dict[str, float]:
    """
    Simplified scoring weights (mocked from internal weighting guidance).
    """
    return {
        "strategy": 0.25,
        "structure": 0.20,
        "automation": 0.20,
        "creative": 0.15,
        "measurement": 0.20,
    }
