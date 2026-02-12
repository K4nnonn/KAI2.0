"""
STRATEGIC CONTEXT KNOWLEDGE BASE
Domain expertise for contextual PPC audit scoring

This module provides campaign strategy detection and context-aware scoring adjustments.
Validated by domain expert as "precise components of domain expertise required."

Author: Claude (Sonnet 4.5)
Date: November 5, 2025
"""

import re
from typing import Dict, List, Optional, Tuple

# ============================================================================
# CAMPAIGN STRATEGY TAXONOMY
# ============================================================================

CAMPAIGN_STRATEGIES = {
    'brand_defense': {
        'description': 'Protecting brand terms from competitors',

        'detection_rules': {
            'name_patterns': [
                r'\bbrand\b',
                r'\btrademark\b',
                r'\bTM\b',
                r'\bcompetitor\b',
                r'\bcompany\b',
                # Add actual client brand names here
            ],
            'keyword_indicators': {
                'match_types_common': ['exact', 'phrase'],
                'expected_ctr_range': [15.0, 40.0],  # Very high CTR
                'expected_qs_range': [8, 10]  # High QS
            },
            'performance_indicators': {
                'high_ctr': True,
                'high_cvr': True,
                'high_qs': True
            }
        },

        'priority_metrics': {
            'impression_share': {
                'weight': 0.30,
                'target_min': 90,
                'target_excellent': 95,
                'target_poor': 75,
                'rationale': 'Must dominate brand traffic - competitors stealing IS is critical'
            },
            'position': {
                'weight': 0.20,
                'target_range': [1.0, 1.5],
                'rationale': 'Must be position 1-1.5'
            },
            'quality_score': {
                'weight': 0.15,
                'target_min': 8,
                'rationale': 'Brand should have high relevance'
            },
            'search_impression_share_lost_to_rank': {
                'weight': 0.15,
                'target_max': 5,
                'rationale': 'Low rank loss indicates strong position'
            }
        },

        'acceptable_cpa_multiplier': 2.0,

        'scoring_adjustments': {
            'impression_share': {
                'score_5_threshold': 95,
                'score_4_threshold': 90,
                'score_3_threshold': 80,
                'score_2_threshold': 70,
                'score_1_threshold': 0,
                'rationale': 'Brand must dominate'
            },
            'cpa': {
                'allow_higher_cpa': True,
                'multiplier': 2.0,
                'rationale': 'Brand protection justifies higher CPA'
            },
            'quality_score': {
                'high_expectations': True,
                'score_5_threshold': 9,
                'score_4_threshold': 8,
                'score_3_threshold': 7,
                'rationale': 'Brand should have excellent relevance'
            }
        },

        'expected_issues': [
            'Competitor ads appearing on brand terms',
            'Budget constraints limiting IS',
            'Trademark violations'
        ],

        'recommended_tactics': [
            'Maximize IS to 95%+',
            'Use exact match for core brand terms',
            'Monitor competitor ad copy',
            'Aggressive bid strategy acceptable',
            'Consider trademark enforcement'
        ]
    },

    'generic_acquisition': {
        'description': 'Non-brand keywords for new customer acquisition',

        'detection_rules': {
            'name_patterns': [
                r'\bgeneric\b',
                r'\bnonbrand\b',
                r'\bnon-brand\b',
                r'\bacquisition\b',
                r'\bprospecting\b',
                r'\bcold\b'
            ],
            'keyword_indicators': {
                'match_types_common': ['broad', 'phrase'],
                'expected_ctr_range': [3.0, 10.0],
                'expected_qs_range': [6, 8]
            },
            'performance_indicators': {
                'moderate_ctr': True,
                'variable_cvr': True,
                'focus_on_roas': True
            }
        },

        'priority_metrics': {
            'roas': {
                'weight': 0.30,
                'target_min': 300,
                'target_excellent': 500,
                'rationale': 'Profit-focused - ROAS is king'
            },
            'cpa': {
                'weight': 0.25,
                'target_multiplier': 1.0,
                'rationale': 'Must be efficient'
            },
            'impression_share': {
                'weight': 0.10,
                'target_range': [60, 75],
                'target_max_warning': 90,
                'rationale': 'Budget-constrained, profit-focused strategy'
            },
            'conversion_rate': {
                'weight': 0.20,
                'rationale': 'Quality of traffic matters'
            }
        },

        'acceptable_cpa_multiplier': 1.0,

        'scoring_adjustments': {
            'impression_share': {
                'score_5_threshold': 65,
                'score_4_threshold': 55,
                'score_3_threshold': 45,
                'score_2_threshold': 30,
                'score_1_threshold': 0,
                'flag_if_above': 90,
                'flag_rationale': 'IS above 90% may indicate overpaying for generic'
            },
            'cpa': {
                'strict_targets': True,
                'multiplier': 1.0,
                'rationale': 'Generic must be efficient'
            },
            'quality_score': {
                'moderate_expectations': True,
                'score_5_threshold': 8,
                'score_4_threshold': 7,
                'score_3_threshold': 6,
                'rationale': 'QS 6-8 is acceptable for generic'
            }
        },

        'expected_issues': [
            'High CPCs in competitive auctions',
            'Variable CVR across keywords',
            'Impression share limited by budget',
            'Search term waste'
        ],

        'recommended_tactics': [
            'Use Smart Bidding (Target CPA/ROAS)',
            'Aggressive negative keyword strategy',
            'Focus on IS in 60-75% range',
            'Test broad match with Smart Bidding',
            'Monitor search term reports weekly'
        ]
    },

    'competitor_conquesting': {
        'description': 'Bidding on competitor brand terms',

        'detection_rules': {
            'name_patterns': [
                r'\bcompetitor\b',
                r'\bconquest\b',
                r'\bcompetitive\b',
                r'\bvs\b',
                r'\balternative\b'
            ],
            'keyword_indicators': {
                'contains_competitor_names': True,
                'match_types_common': ['phrase', 'broad'],
                'expected_ctr_range': [2.0, 8.0],
                'expected_qs_range': [4, 7]
            }
        },

        'priority_metrics': {
            'ctr': {
                'weight': 0.25,
                'rationale': 'Winning clicks from competitor brand terms'
            },
            'conversion_rate': {
                'weight': 0.30,
                'rationale': 'Quality matters more than volume'
            },
            'cpa': {
                'weight': 0.25,
                'acceptable_multiplier': 1.5,
                'rationale': 'Competitive terms are expensive'
            }
        },

        'acceptable_cpa_multiplier': 1.5,

        'scoring_adjustments': {
            'quality_score': {
                'lower_expectations': True,
                'score_5_threshold': 7,
                'score_4_threshold': 6,
                'score_3_threshold': 5,
                'score_2_threshold': 4,
                'score_1_threshold': 0,
                'rationale': 'Lower relevance is expected on competitor terms'
            },
            'cpa': {
                'allow_higher_cpa': True,
                'multiplier': 1.5,
                'rationale': 'Competitive conquesting justifies premium'
            },
            'impression_share': {
                'lower_targets': True,
                'target_range': [30, 50],
                'rationale': 'Budget-efficient conquest'
            },
            'ctr': {
                'lower_expectations': True,
                'score_5_threshold': 6,
                'score_4_threshold': 4,
                'score_3_threshold': 2,
                'rationale': 'Lower CTR expected on competitor terms'
            }
        },

        'expected_issues': [
            'Low Quality Scores (expected)',
            'High CPCs',
            'Competitor trademark policies',
            'Lower CTRs than brand'
        ],

        'recommended_tactics': [
            'Focus on differentiation in ad copy',
            'Target dissatisfied competitor customers',
            'Monitor ROI closely',
            'Use phrase/broad match',
            'Test negative match on competitor brand + product combos'
        ]
    },

    'remarketing': {
        'description': 'Targeting previous site visitors with RLSA',

        'detection_rules': {
            'name_patterns': [
                r'\bremarketing\b',
                r'\bretargeting\b',
                r'\brlsa\b',
                r'\baudiences?\b',
                r'\bcustomer_?match\b'
            ],
            'audience_targeting_present': True
        },

        'priority_metrics': {
            'conversion_rate': {
                'weight': 0.35,
                'expected_multiplier': 2.0,
                'rationale': 'Already qualified audience'
            },
            'cpa': {
                'weight': 0.30,
                'expected_multiplier': 0.5,
                'rationale': 'More efficient than cold traffic'
            },
            'roas': {
                'weight': 0.25,
                'expected_multiplier': 1.5,
                'rationale': 'Higher return expected'
            }
        },

        'acceptable_cpa_multiplier': 0.5,

        'scoring_adjustments': {
            'conversion_rate': {
                'high_expectations': True,
                'benchmark_multiplier': 2.0,
                'score_5_threshold': 2.0,
                'score_3_threshold': 1.5,
                'score_1_threshold': 1.0,
                'rationale': 'Remarketing should convert significantly better'
            },
            'cpa': {
                'strict_efficiency': True,
                'benchmark_multiplier': 0.5,
                'score_5_threshold': 0.5,
                'score_3_threshold': 0.7,
                'score_1_threshold': 1.0,
                'rationale': 'More qualified audience = lower CPA'
            }
        },

        'expected_issues': [
            'Audience lists too small',
            'Insufficient lookback window',
            'Not segmented by intent',
            'Frequency fatigue'
        ],

        'recommended_tactics': [
            'Segment by page visited',
            'Adjust bids by audience value',
            'Dynamic remarketing for e-commerce',
            'Frequency caps to avoid fatigue',
            'Exclude converters from remarketing'
        ]
    }
}


# ============================================================================
# STRATEGY DETECTION FUNCTIONS
# ============================================================================

def detect_campaign_strategy(campaign_name: str, campaign_data: Optional[Dict] = None) -> Dict:
    """
    Detect campaign strategy from campaign name and optional performance data

    Args:
        campaign_name: Campaign name string
        campaign_data: Optional dict with performance metrics
                      {'is': float, 'ctr': float, 'cvr': float, 'qs': float}

    Returns:
        {
            'strategy': str,  # brand_defense | generic_acquisition | competitor_conquesting | remarketing
            'confidence': str,  # high | medium | low
            'detected_signals': list,  # What triggered detection
            'strategy_config': dict  # Full strategy configuration
        }
    """
    campaign_name_lower = campaign_name.lower()
    detected_signals = []
    best_match = None
    best_match_score = 0

    # Check each strategy's detection rules
    for strategy_key, strategy_config in CAMPAIGN_STRATEGIES.items():
        rules = strategy_config['detection_rules']
        match_score = 0

        # Check name patterns (primary signal)
        for pattern in rules.get('name_patterns', []):
            if re.search(pattern, campaign_name_lower, re.IGNORECASE):
                match_score += 2  # Name match is strong signal
                detected_signals.append(f"Name contains pattern '{pattern}'")

        # If campaign_data provided, check performance indicators
        if campaign_data and match_score > 0:  # Only check performance if name matched
            indicators = rules.get('keyword_indicators', {})

            # Check CTR range
            if 'expected_ctr_range' in indicators and 'ctr' in campaign_data:
                ctr_range = indicators['expected_ctr_range']
                actual_ctr = campaign_data['ctr']
                if actual_ctr and ctr_range[0] <= actual_ctr <= ctr_range[1]:
                    match_score += 1
                    detected_signals.append(f"CTR {actual_ctr:.1f}% in expected range")

            # Check QS range
            if 'expected_qs_range' in indicators and 'qs' in campaign_data:
                qs_range = indicators['expected_qs_range']
                actual_qs = campaign_data['qs']
                if actual_qs and qs_range[0] <= actual_qs <= qs_range[1]:
                    match_score += 1
                    detected_signals.append(f"QS {actual_qs:.1f} in expected range")

        # Track best match
        if match_score > best_match_score:
            best_match_score = match_score
            best_match = strategy_key

    # Return best match if found
    if best_match and best_match_score >= 2:
        confidence = 'high' if best_match_score >= 3 else 'medium'

        return {
            'strategy': best_match,
            'confidence': confidence,
            'detected_signals': detected_signals,
            'strategy_config': CAMPAIGN_STRATEGIES[best_match]
        }

    # Default: generic_acquisition if no clear signals
    return {
        'strategy': 'generic_acquisition',
        'confidence': 'low',
        'detected_signals': ['No clear strategy signals - defaulting to generic acquisition'],
        'strategy_config': CAMPAIGN_STRATEGIES['generic_acquisition']
    }


def get_strategy_adjusted_target(strategy: str, metric: str, base_target: float) -> float:
    """
    Get strategy-adjusted target for a metric

    Args:
        strategy: Campaign strategy (brand_defense | generic_acquisition | etc.)
        metric: Metric name (impression_share | cpa | quality_score | etc.)
        base_target: Base target value

    Returns:
        Adjusted target value
    """
    strategy_config = CAMPAIGN_STRATEGIES.get(strategy)
    if not strategy_config:
        return base_target

    # Check for metric-specific targets in priority_metrics
    priority_metrics = strategy_config.get('priority_metrics', {})
    metric_config = priority_metrics.get(metric)

    if metric_config:
        # Return strategy-specific target if available
        if 'target_min' in metric_config:
            return metric_config['target_min']
        elif 'target_range' in metric_config:
            return metric_config['target_range'][0]  # Return min of range

    # Check for multipliers in scoring_adjustments
    adjustments = strategy_config.get('scoring_adjustments', {})
    metric_adjustment = adjustments.get(metric, {})

    if 'multiplier' in metric_adjustment:
        return base_target * metric_adjustment['multiplier']

    # No adjustment found - return base target
    return base_target


def get_strategy_scoring_thresholds(strategy: str, metric: str) -> Dict:
    """
    Get strategy-specific scoring thresholds

    Args:
        strategy: Campaign strategy
        metric: Metric name

    Returns:
        {
            'score_5': float,
            'score_4': float,
            'score_3': float,
            'score_2': float,
            'score_1': float,
            'rationale': str,
            'flag_if_above': float (optional),
            'flag_rationale': str (optional)
        }
    """
    strategy_config = CAMPAIGN_STRATEGIES.get(strategy, {})
    adjustments = strategy_config.get('scoring_adjustments', {})
    metric_adjustment = adjustments.get(metric, {})

    # Return strategy-specific thresholds if available
    if 'score_5_threshold' in metric_adjustment:
        result = {
            'score_5': metric_adjustment.get('score_5_threshold'),
            'score_4': metric_adjustment.get('score_4_threshold'),
            'score_3': metric_adjustment.get('score_3_threshold'),
            'score_2': metric_adjustment.get('score_2_threshold'),
            'score_1': metric_adjustment.get('score_1_threshold', 0),
            'rationale': metric_adjustment.get('rationale', '')
        }

        # Add flag thresholds if present
        if 'flag_if_above' in metric_adjustment:
            result['flag_if_above'] = metric_adjustment['flag_if_above']
            result['flag_rationale'] = metric_adjustment.get('flag_rationale', '')

        return result

    # Return empty dict if no thresholds defined
    return {}


def get_acceptable_cpa_multiplier(strategy: str) -> float:
    """
    Get acceptable CPA multiplier for a strategy

    Args:
        strategy: Campaign strategy

    Returns:
        Multiplier (e.g., 2.0 = can pay 2x normal CPA)
    """
    strategy_config = CAMPAIGN_STRATEGIES.get(strategy, {})
    return strategy_config.get('acceptable_cpa_multiplier', 1.0)


def get_recommended_tactics(strategy: str) -> List[str]:
    """
    Get recommended tactics for a strategy

    Args:
        strategy: Campaign strategy

    Returns:
        List of recommended tactics
    """
    strategy_config = CAMPAIGN_STRATEGIES.get(strategy, {})
    return strategy_config.get('recommended_tactics', [])


def get_expected_issues(strategy: str) -> List[str]:
    """
    Get expected issues for a strategy

    Args:
        strategy: Campaign strategy

    Returns:
        List of expected issues
    """
    strategy_config = CAMPAIGN_STRATEGIES.get(strategy, {})
    return strategy_config.get('expected_issues', [])


def get_priority_metrics_config(strategy: str) -> Dict:
    """
    Get priority metrics configuration for a strategy

    Args:
        strategy: Campaign strategy

    Returns:
        Dict of priority metrics with weights and targets
    """
    strategy_config = CAMPAIGN_STRATEGIES.get(strategy, {})
    return strategy_config.get('priority_metrics', {})


# ============================================================================
# SCORING HELPERS
# ============================================================================

def score_metric_with_strategy_context(
    metric_name: str,
    actual_value: float,
    strategy: str,
    base_thresholds: Optional[Dict] = None
) -> Tuple[int, str, str]:
    """
    Score a metric with strategy-aware thresholds

    Args:
        metric_name: Name of metric (impression_share, cpa, quality_score, etc.)
        actual_value: Actual value
        strategy: Detected campaign strategy
        base_thresholds: Optional base thresholds if strategy-specific not available

    Returns:
        (score, assessment, rationale)
    """
    # Get strategy-specific thresholds
    thresholds = get_strategy_scoring_thresholds(strategy, metric_name)

    if not thresholds and base_thresholds:
        # Use base thresholds if no strategy-specific available
        thresholds = base_thresholds

    if thresholds:
        # Score against thresholds
        if actual_value >= thresholds.get('score_5', 100):
            score = 5
            assessment = "Excellent"
        elif actual_value >= thresholds.get('score_4', 80):
            score = 4
            assessment = "Good"
        elif actual_value >= thresholds.get('score_3', 60):
            score = 3
            assessment = "Acceptable"
        elif actual_value >= thresholds.get('score_2', 40):
            score = 2
            assessment = "Poor"
        else:
            score = 1
            assessment = "Critical"

        rationale = thresholds.get('rationale', '')

        # Check for flag conditions
        if 'flag_if_above' in thresholds and actual_value > thresholds['flag_if_above']:
            assessment += " (FLAGGED)"
            rationale += f" {thresholds.get('flag_rationale', '')}"

        return score, assessment, rationale
    else:
        # No thresholds available
        return None, "N/A", "No strategy-specific thresholds defined"


# ============================================================================
# TESTING / EXAMPLES
# ============================================================================

if __name__ == "__main__":
    # Example 1: Brand campaign
    print("=== Example 1: Brand Campaign ===")
    result = detect_campaign_strategy(
        "Brand - Trademark Protection",
        campaign_data={'is': 85, 'ctr': 25.0, 'qs': 9.0}
    )
    print(f"Strategy: {result['strategy']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Signals: {result['detected_signals']}")
    print()

    # Example 2: Generic campaign
    print("=== Example 2: Generic Campaign ===")
    result = detect_campaign_strategy(
        "Generic - Acquisition - IRA",
        campaign_data={'is': 65, 'ctr': 7.0, 'qs': 7.0}
    )
    print(f"Strategy: {result['strategy']}")
    print(f"Confidence: {result['confidence']}")

    # Get IS scoring thresholds
    thresholds = get_strategy_scoring_thresholds(result['strategy'], 'impression_share')
    print(f"IS Thresholds: {thresholds}")
    print()

    # Example 3: Score same IS (65%) for both strategies
    print("=== Example 3: Score IS 65% for Brand vs Generic ===")

    brand_score, brand_assessment, brand_rationale = score_metric_with_strategy_context(
        'impression_share', 65.0, 'brand_defense'
    )
    print(f"Brand: {brand_score}/5 ({brand_assessment}) - {brand_rationale}")

    generic_score, generic_assessment, generic_rationale = score_metric_with_strategy_context(
        'impression_share', 65.0, 'generic_acquisition'
    )
    print(f"Generic: {generic_score}/5 ({generic_assessment}) - {generic_rationale}")
    print()

    # Example 4: Competitor campaign
    print("=== Example 4: Competitor Campaign ===")
    result = detect_campaign_strategy(
        "Competitor - Conquest",
        campaign_data={'ctr': 5.0, 'qs': 6.0}
    )
    print(f"Strategy: {result['strategy']}")
    print(f"Acceptable CPA Multiplier: {get_acceptable_cpa_multiplier(result['strategy'])}")
    print(f"Recommended Tactics: {get_recommended_tactics(result['strategy'])[:3]}")
