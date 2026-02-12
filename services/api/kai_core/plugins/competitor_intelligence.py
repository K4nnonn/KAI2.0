"""
Competitor Investment Intelligence Plugin

Infers competitor investment signals from conversational observations.
Uses fuzzy inference from natural language descriptions when precise
auction insights data is not available.

This plugin follows the intelligent mapping pattern - data is extracted
from conversation rather than requiring file uploads.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class InvestmentSignal(Enum):
    """Investment signal direction."""
    RAMPING_UP = "ramping_up"
    STABLE = "stable"
    DECLINING = "declining"


@dataclass
class CompetitorMetrics:
    """Metrics extracted from conversation about a competitor."""
    competitor_domain: str
    impression_share_current: Optional[float] = None
    impression_share_previous: Optional[float] = None
    outranking_rate: Optional[float] = None
    top_of_page_rate: Optional[float] = None
    position_above_rate: Optional[float] = None
    raw_description: Optional[str] = None


@dataclass
class InvestmentSignalResult:
    """Result of investment signal inference."""
    signal: InvestmentSignal
    confidence: float  # 0.0 to 1.0
    reasoning: str
    metrics_used: list[str]


class CompetitorSignalInferencer:
    """
    Infers investment signals from conversational descriptions.
    Uses the same pattern as PMaxChannelSplitter's inference logic.
    """

    # Keywords that indicate investment direction
    SIGNAL_KEYWORDS = {
        InvestmentSignal.RAMPING_UP: [
            'more often', 'increased', 'ramping', 'ramping up', 'aggressive',
            'everywhere', 'dominating', 'up from', 'higher', 'growing',
            'jumped', 'surged', 'spiked', 'doubled', 'tripled', 'above me',
            'beating', 'outranking', 'taking over', 'all over'
        ],
        InvestmentSignal.DECLINING: [
            'less often', 'decreased', 'pulling back', 'disappeared',
            'down from', 'lower', 'retreating', 'dropped', 'fell',
            'declining', 'reduced', 'cutting back', 'gone', 'missing'
        ],
        InvestmentSignal.STABLE: [
            'same', 'unchanged', 'consistent', 'steady', 'flat',
            'no change', 'similar', 'maintaining', 'holding'
        ]
    }

    # Weights for auction insight metrics when available
    METRIC_WEIGHTS = {
        'impression_share': 0.30,
        'outranking_share': 0.25,
        'top_of_page_rate': 0.25,
        'position_above_rate': 0.20
    }

    def infer_from_description(self, text: str) -> InvestmentSignalResult:
        """
        Extract signal from natural language description.
        Uses keyword matching and numeric delta detection.
        """
        text_lower = text.lower()
        metrics_used = ['keyword_analysis']

        # Score each signal type based on keyword matches
        scores = {signal: 0 for signal in InvestmentSignal}
        matched_keywords = {signal: [] for signal in InvestmentSignal}

        for signal, keywords in self.SIGNAL_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    scores[signal] += 1
                    matched_keywords[signal].append(keyword)

        # Try to extract numeric deltas (e.g., "25% to 45%", "30 -> 50")
        delta_patterns = [
            r'(\d+)%?\s*(?:to|->|→|from)\s*(\d+)%?',  # "25% to 45%"
            r'(\d+)%?\s*(?:up from|down from)\s*(\d+)%?',  # "45% up from 25%"
            r'was\s*(\d+)%?\s*(?:now|is)\s*(\d+)%?',  # "was 25% now 45%"
        ]

        numeric_delta = None
        for pattern in delta_patterns:
            match = re.search(pattern, text_lower)
            if match:
                val1, val2 = int(match.group(1)), int(match.group(2))
                # Handle "up from" vs "to" direction
                if 'up from' in text_lower or 'down from' in text_lower:
                    # "45% up from 25%" means current=45, previous=25
                    numeric_delta = val1 - val2
                else:
                    # "25% to 45%" means previous=25, current=45
                    numeric_delta = val2 - val1
                metrics_used.append('numeric_delta_extraction')
                break

        # Numeric deltas override keyword detection
        if numeric_delta is not None:
            if numeric_delta > 5:
                signal = InvestmentSignal.RAMPING_UP
                confidence = min(0.9, 0.6 + (numeric_delta / 50))
                reasoning = f"Detected +{numeric_delta} point change indicating increased investment"
            elif numeric_delta < -5:
                signal = InvestmentSignal.DECLINING
                confidence = min(0.9, 0.6 + (abs(numeric_delta) / 50))
                reasoning = f"Detected {numeric_delta} point change indicating reduced investment"
            else:
                signal = InvestmentSignal.STABLE
                confidence = 0.7
                reasoning = f"Change of {numeric_delta} points indicates stable investment"
        else:
            # Fall back to keyword-based inference
            max_score = max(scores.values()) if max(scores.values()) > 0 else 0
            if max_score == 0:
                # No clear signal detected
                signal = InvestmentSignal.STABLE
                confidence = 0.3
                reasoning = "No clear investment signal detected in description"
            else:
                # Find the signal with highest score
                signal = max(scores, key=scores.get)
                confidence = min(0.8, 0.4 + (max_score * 0.1))
                keywords_found = matched_keywords[signal]
                reasoning = f"Detected keywords: {', '.join(keywords_found[:3])}"

        return InvestmentSignalResult(
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            metrics_used=metrics_used
        )

    def calculate_from_metrics(self, metrics: CompetitorMetrics) -> InvestmentSignalResult:
        """
        Calculate investment signal from structured auction insight metrics.
        This is more precise than keyword inference.
        """
        metrics_used = []
        deltas = {}

        # Calculate deltas for available metrics
        if metrics.impression_share_current is not None and metrics.impression_share_previous is not None:
            deltas['impression_share'] = metrics.impression_share_current - metrics.impression_share_previous
            metrics_used.append('impression_share')

        if metrics.outranking_rate is not None:
            # If we have outranking rate, assume high rate = ramping up
            deltas['outranking_share'] = metrics.outranking_rate - 50  # Compare to 50% baseline
            metrics_used.append('outranking_rate')

        if metrics.top_of_page_rate is not None:
            deltas['top_of_page_rate'] = metrics.top_of_page_rate - 50  # Compare to 50% baseline
            metrics_used.append('top_of_page_rate')

        if metrics.position_above_rate is not None:
            deltas['position_above_rate'] = metrics.position_above_rate - 50
            metrics_used.append('position_above_rate')

        if not deltas:
            # Fall back to description if no metrics
            if metrics.raw_description:
                return self.infer_from_description(metrics.raw_description)
            return InvestmentSignalResult(
                signal=InvestmentSignal.STABLE,
                confidence=0.2,
                reasoning="Insufficient data to determine investment signal",
                metrics_used=[]
            )

        # Calculate weighted change
        weighted_change = 0.0
        total_weight = 0.0

        for metric_name, delta in deltas.items():
            weight_key = metric_name.replace('_rate', '_share') if '_rate' in metric_name else metric_name
            if weight_key in self.METRIC_WEIGHTS:
                weight = self.METRIC_WEIGHTS[weight_key]
            else:
                weight = 0.2  # Default weight
            weighted_change += delta * weight
            total_weight += weight

        if total_weight > 0:
            weighted_change = weighted_change / total_weight * len(deltas)

        # Determine signal based on weighted change
        if weighted_change > 5.0:
            signal = InvestmentSignal.RAMPING_UP
            confidence = min(0.95, 0.7 + (weighted_change / 50))
            reasoning = f"Weighted metric change of +{weighted_change:.1f} indicates ramping up"
        elif weighted_change < -5.0:
            signal = InvestmentSignal.DECLINING
            confidence = min(0.95, 0.7 + (abs(weighted_change) / 50))
            reasoning = f"Weighted metric change of {weighted_change:.1f} indicates declining"
        else:
            signal = InvestmentSignal.STABLE
            confidence = 0.75
            reasoning = f"Weighted metric change of {weighted_change:.1f} indicates stable investment"

        return InvestmentSignalResult(
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            metrics_used=metrics_used
        )

    def generate_interpretation(
        self,
        metrics: CompetitorMetrics,
        signal_result: InvestmentSignalResult
    ) -> str:
        """
        Generate a human-readable AI interpretation of the competitor analysis.
        """
        competitor = metrics.competitor_domain or "The competitor"
        signal = signal_result.signal

        # Build context details
        details = []
        if metrics.impression_share_current is not None:
            if metrics.impression_share_previous is not None:
                delta = metrics.impression_share_current - metrics.impression_share_previous
                details.append(f"impression share {'increased' if delta > 0 else 'decreased'} from {metrics.impression_share_previous}% to {metrics.impression_share_current}%")
            else:
                details.append(f"current impression share is {metrics.impression_share_current}%")

        if metrics.outranking_rate is not None:
            details.append(f"outranking rate of {metrics.outranking_rate}%")

        if metrics.top_of_page_rate is not None:
            details.append(f"top-of-page rate of {metrics.top_of_page_rate}%")

        detail_text = ", ".join(details) if details else "the observed competitive behavior"

        if signal == InvestmentSignal.RAMPING_UP:
            action = "significantly increasing their paid search investment"
            recommendation = "Consider defensive bid increases on your core brand terms to maintain position."
        elif signal == InvestmentSignal.DECLINING:
            action = "reducing their paid search investment"
            recommendation = "This may be an opportunity to capture additional market share at lower cost."
        else:
            action = "maintaining stable paid search investment"
            recommendation = "Continue monitoring for any shifts in competitive behavior."

        confidence_text = "high" if signal_result.confidence > 0.7 else "moderate" if signal_result.confidence > 0.4 else "low"

        return (
            f"{competitor} appears to be {action} based on {detail_text}. "
            f"Confidence level: {confidence_text} ({signal_result.confidence:.0%}). "
            f"{recommendation}"
        )


def analyze_competitor(
    competitor_domain: str,
    impression_share_current: Optional[float] = None,
    impression_share_previous: Optional[float] = None,
    outranking_rate: Optional[float] = None,
    top_of_page_rate: Optional[float] = None,
    position_above_rate: Optional[float] = None,
    raw_description: Optional[str] = None
) -> dict:
    """
    Main entry point for competitor analysis.
    Returns a dictionary with signal, confidence, and interpretation.
    """
    metrics = CompetitorMetrics(
        competitor_domain=competitor_domain,
        impression_share_current=impression_share_current,
        impression_share_previous=impression_share_previous,
        outranking_rate=outranking_rate,
        top_of_page_rate=top_of_page_rate,
        position_above_rate=position_above_rate,
        raw_description=raw_description
    )

    inferencer = CompetitorSignalInferencer()

    # Choose inference method based on available data
    if impression_share_current is not None or outranking_rate is not None:
        signal_result = inferencer.calculate_from_metrics(metrics)
    elif raw_description:
        signal_result = inferencer.infer_from_description(raw_description)
    else:
        signal_result = InvestmentSignalResult(
            signal=InvestmentSignal.STABLE,
            confidence=0.1,
            reasoning="Insufficient data provided",
            metrics_used=[]
        )

    interpretation = inferencer.generate_interpretation(metrics, signal_result)

    return {
        "competitor": competitor_domain,
        "signal": signal_result.signal.value,
        "confidence": signal_result.confidence,
        "reasoning": signal_result.reasoning,
        "interpretation": interpretation,
        "metrics_used": signal_result.metrics_used
    }
