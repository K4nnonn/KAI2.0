from typing import Optional

"""
INDUSTRY KNOWLEDGE BASE
Benchmarks, best practices, and scoring rules for PPC audit engine

Sources:
- WordStream 2024-2025 benchmarks
- Google Ads official guidelines
- Ad Labz industry reports
- Unbounce Conversion Benchmark Report 2024
- LocaliQ Google Ads Benchmarks 2025
"""

# ============================================================================
# INDUSTRY BENCHMARKS (2024-2025)
# ============================================================================

BENCHMARKS = {
    'financial_services': {
        'search': {
            'ctr': 7.71,  # Average CTR %
            'cvr': 2.78,  # Average CVR %
            'cpa_min': 50,   # Minimum typical CPA
            'cpa_max': 95,   # Maximum typical CPA
            # Context: Retirement products (high LTV) will be on high end
            # Free checking (low friction) will be on low end
        },
        'display': {
            'ctr': 0.5,
            'cvr': 0.8,
            'cpa_min': 60,
            'cpa_max': 120
        }
    },
    'ecommerce': {
        'search': {
            'ctr': 8.92,  # Shopping/Gifts category
            'cvr': 3.00,  # Average of 2.5-3.5% range
            'cpa_min': 25,
            'cpa_max': 45
        }
    },
    'b2b_saas': {
        'search': {
            'ctr': 3.20,
            'cvr': 3.20,  # Average of 2.9-3.5% range
            'cpa': 95.00
        }
    }
}

# ============================================================================
# PERFORMANCE DISTRIBUTIONS / DYNAMIC THRESHOLDS
# ============================================================================

PERFORMANCE_DISTRIBUTIONS = {
    'cvr': {
        # Source: Unbounce Conversion Benchmark Report 2024
        'all_industries': {
            'median': 6.6,
            'top25': 11.4,
            'top10': 18.5,
            'notes': 'Top performers convert roughly 3x higher than median.',
            'source': 'Unbounce 2024 – 41k landing pages / 464M visits'
        },
        'saas': {
            'median': 3.8,
            'top25': 11.6,
            'top10': 15.0,
            'notes': 'Achieving a 5 requires solving technical friction in funnels.',
            'source': 'Unbounce 2024'
        },
        'legal': {
            'median': 6.1,
            'top25': 14.1,
            'top10': 19.5,
            'notes': 'High-intent queries push the “Great” threshold upward.',
            'source': 'Unbounce 2024'
        },
        'finance': {
            'median': 8.3,
            'top25': 15.2,
            'top10': 21.0,
            'notes': 'Trust and compliance signals differentiate the top 10%.',
            'source': 'Unbounce 2024'
        }
    },
    'ctr': {
        # Source: WordStream + LocaliQ Google Ads Benchmarks 2025
        'finance_insurance': {
            'median': 6.18,
            'top25': 9.5,
            'top10': 12.3,
            'notes': 'Leaders dominate SERP share with double-digit CTR.',
            'source': 'WordStream & LocaliQ 2025'
        },
        'b2b_services': {
            'median': 5.11,
            'top25': 7.9,
            'top10': 9.8,
            'notes': 'Precision messaging and ICP alignment create the lift.',
            'source': 'WordStream & LocaliQ 2025'
        },
        'real_estate': {
            'median': 8.85,
            'top25': 11.8,
            'top10': 14.0,
            'notes': 'Rich assets (maps/images) skew the top end upward.',
            'source': 'WordStream & LocaliQ 2025'
        }
    }
}

# ============================================================================
# BUSINESS PHYSICS GUARDRAILS
# ============================================================================

BUSINESS_PHYSICS_GUARDRAILS = {
    'legal_cpl_quality_check': {
        'metric': 'cpl',
        'vertical': 'legal',
        'median_cpl': 111,
        'suspect_threshold': 30,
        'failure_mode': 'Spam / disqualified leads >90%',
        'action': 'Force score downgrade until lead quality verified.',
        'source': 'LocaliQ 2025 – Legal CPL benchmark',
        'notes': 'Extremely low CPL in Legal is correlated with junk leads; prevents vanity scoring.'
    }
}

# ============================================================================
# GOOGLE ADS BEST PRACTICES
# ============================================================================

BEST_PRACTICES = {
    'smart_bidding': {
        # Conversion volume requirements
        'min_conversions_30d_tcpa': 30,  # Target CPA minimum
        'min_conversions_30d_troas': 50,  # Target ROAS minimum
        'recommended_conversions_30d': 50,  # Ideal for all

        # Learning period
        'learning_period_days_min': 7,
        'learning_period_days_max': 14,

        # Status progression
        'expected_statuses': ['Learning', 'Eligible', 'Active'],

        # Performance thresholds (vs target)
        'performance_excellent': 0.10,  # Within 10% of target
        'performance_good': 0.25,       # Within 25% of target
        'performance_poor': 0.25,       # Beyond 25% = poor

        # Scoring rules
        'score_not_ready': 1,  # <30 conversions
        'score_learning': None,  # Don't score while learning
        'score_within_10pct': 5,
        'score_within_25pct': 3,
        'score_beyond_25pct': 1
    },

    'rsa': {
        # Asset counts
        'max_headlines': 15,
        'max_descriptions': 4,
        'min_headlines': 3,
        'min_descriptions': 2,

        # Best practice thresholds
        'recommended_headlines_min': 8,
        'recommended_headlines_ideal': 10,
        'recommended_descriptions_min': 3,

        # Scoring by headline count
        'score_poor_threshold': 5,    # <5 headlines = poor
        'score_average_threshold': 7,  # 5-7 headlines = average
        'score_good_threshold': 8,     # 8+ headlines = good/excellent

        # Ad Strength mapping
        'ad_strength_scores': {
            'Poor': 1,
            'Average': 3,
            'Good': 4,
            'Excellent': 5
        },

        # Pinning rules
        'max_recommended_pins': 2,  # >2 pins defeats RSA purpose
        'flag_if_pins_exceed': 2
    },

    'broad_match': {
        # Modern best practice: Broad + Smart Bidding
        'requires_smart_bidding': True,
        'requires_conversion_history': True,
        'min_conversions_for_broad': 30,

        # Recommended percentages
        'recommended_pct_with_smart_bidding': 40,  # 40%+ broad with Smart Bidding
        'recommended_pct_without_smart_bidding': 10,  # <10% broad without Smart Bidding

        # Scoring
        'score_broad_with_smart_bidding_and_volume': 5,  # Excellent
        'score_broad_without_smart_bidding': 1,  # CRITICAL - waste risk
        'score_broad_with_smart_bidding_no_volume': 2  # Poor - not ready
    },

    'quality_score': {
        # Thresholds
        'excellent_qs': 8,  # QS 8-10
        'good_qs': 7,       # QS 7
        'poor_qs': 6,       # QS <=6 (increases CPC, lowers rank)

        # Portfolio health
        'healthy_account_pct_qs7_plus': 70,  # 70%+ keywords should be QS 7+

        # Scoring
        'score_qs_8_plus': 5,
        'score_qs_7': 4,
        'score_qs_6': 3,
        'score_qs_5_or_less': 1
    },

    'impression_share': {
        # Strategy-specific targets
        'brand_campaigns': {
            'target_is': 90,      # Minimum target
            'excellent_is': 95,   # Ideal target
            'poor_is': 75,
            'budget_strategy': 'uncapped',  # Never cap brand budget
            'score_above_95': 5,
            'score_90_to_95': 4,
            'score_75_to_90': 2,
            'score_below_75': 1
        },
        'generic_campaigns': {
            'target_is_min': 40,
            'target_is_max': 75,  # 100% here might mean overpaying
            'excellent_range': (60, 75),
            'budget_strategy': 'profit_focused',
            'score_60_to_75': 5,
            'score_40_to_60': 3,
            'score_below_40': 2,
            'score_above_90': 2  # Flag: might be overpaying
        },
        'acquisition_campaigns': {
            'target_is_min': 50,
            'target_is_max': 70,
            'excellent_range': (50, 70),
            'score_in_range': 5,
            'score_below_range': 3,
            'score_above_range': 3
        }
    }
}

# ============================================================================
# PERFORMANCE GRADING SCALES
# ============================================================================

def get_benchmark(vertical: str, network: str = 'search', metric: str = 'ctr') -> float:
    """
    Get industry benchmark for contextual scoring

    Args:
        vertical: 'financial_services', 'ecommerce', 'b2b_saas'
        network: 'search' or 'display'
        metric: 'ctr', 'cvr', 'cpa', etc.

    Returns:
        Benchmark value or None if not found
    """
    try:
        return BENCHMARKS[vertical][network][metric]
    except KeyError:
        return None


def score_against_benchmark(actual: float, benchmark: float, metric_type: str = 'higher_is_better') -> dict:
    """
    Score actual performance against industry benchmark

    Args:
        actual: Actual metric value (e.g., CTR of 5.2%)
        benchmark: Industry benchmark (e.g., 7.71% for financial services)
        metric_type: 'higher_is_better' (CTR, CVR) or 'lower_is_better' (CPA)

    Returns:
        {
            'score': 1-5,
            'vs_benchmark': '+20%' or '-15%',
            'performance_level': 'Excellent' | 'Above Average' | 'On Par' | 'Below Average' | 'Poor'
        }
    """
    if benchmark == 0 or benchmark is None:
        return {'score': None, 'vs_benchmark': 'N/A', 'performance_level': 'No Benchmark'}

    # Calculate percentage difference
    pct_diff = ((actual - benchmark) / benchmark) * 100

    if metric_type == 'higher_is_better':
        # CTR, CVR - higher is better
        if pct_diff >= 20:
            return {'score': 5, 'vs_benchmark': f'+{pct_diff:.0f}%', 'performance_level': 'Excellent'}
        elif pct_diff >= 0:
            return {'score': 4, 'vs_benchmark': f'+{pct_diff:.0f}%', 'performance_level': 'Above Average'}
        elif pct_diff >= -20:
            return {'score': 3, 'vs_benchmark': f'{pct_diff:.0f}%', 'performance_level': 'On Par'}
        elif pct_diff >= -40:
            return {'score': 2, 'vs_benchmark': f'{pct_diff:.0f}%', 'performance_level': 'Below Average'}
        else:
            return {'score': 1, 'vs_benchmark': f'{pct_diff:.0f}%', 'performance_level': 'Poor'}
    else:
        # CPA - lower is better (invert scoring)
        if pct_diff <= -20:  # 20% lower CPA than benchmark
            return {'score': 5, 'vs_benchmark': f'{pct_diff:.0f}%', 'performance_level': 'Excellent'}
        elif pct_diff <= 0:
            return {'score': 4, 'vs_benchmark': f'{pct_diff:.0f}%', 'performance_level': 'Above Average'}
        elif pct_diff <= 20:
            return {'score': 3, 'vs_benchmark': f'+{pct_diff:.0f}%', 'performance_level': 'On Par'}
        elif pct_diff <= 40:
            return {'score': 2, 'vs_benchmark': f'+{pct_diff:.0f}%', 'performance_level': 'Below Average'}
        else:
            return {'score': 1, 'vs_benchmark': f'+{pct_diff:.0f}%', 'performance_level': 'Poor'}


def get_performance_distribution(metric: str, vertical: str) -> dict:
    """
    Retrieve percentile distribution data for a metric/vertical combo.
    Returns an empty dict if no distribution is available.
    """
    return PERFORMANCE_DISTRIBUTIONS.get(metric.lower(), {}).get(vertical.lower(), {})


def get_dynamic_threshold(metric: str, vertical: str, score_target: int = 5) -> Optional[float]:
    """
    Fetch the metric value associated with a rubric score for a given vertical.
    score_target: 5 -> top10, 4 -> top25, 3 -> median.
    """
    distribution = get_performance_distribution(metric, vertical)
    if not distribution:
        return None

    if score_target >= 5:
        return distribution.get('top10')
    if score_target == 4:
        return distribution.get('top25')
    if score_target == 3:
        return distribution.get('median')
    return None


def get_guardrail(rule_name: str) -> dict:
    """
    Lookup business-physics guardrail metadata (e.g., legal CPL quality check).
    """
    return BUSINESS_PHYSICS_GUARDRAILS.get(rule_name, {})


# ============================================================================
# PRIORITY & IMPACT FRAMEWORK
# ============================================================================

PRIORITY_FRAMEWORK = {
    'critical': {
        'priority_score': 10,
        'examples': [
            'Broken conversion tracking (no data)',
            'All ads disapproved',
            'Budget-capping brand campaigns',
            'Account suspended'
        ],
        'impact': 'Stops all value generation'
    },
    'high': {
        'priority_score': 8,
        'examples': [
            'High search term waste (>15% spend)',
            'Broad Match with Manual CPC',
            'Low Quality Scores (<5) across portfolio',
            'No negative keywords',
            'Brand IS <80%'
        ],
        'impact': 'Stops inefficient spending / wastes budget'
    },
    'medium': {
        'priority_score': 5,
        'examples': [
            'Low RSA Ad Strength (Average)',
            'Not using ad extensions',
            'Not testing audience segments',
            'Manual bidding when Smart Bidding ready',
            'Generic IS <50%'
        ],
        'impact': 'Follows best practices / improves efficiency'
    },
    'low': {
        'priority_score': 2,
        'examples': [
            'Inconsistent naming conventions',
            'No campaign labels',
            'Messy account structure',
            'Missing UTM parameters'
        ],
        'impact': 'Organization and reporting improvements'
    }
}


def determine_business_impact(criterion_name: str, score: float, details: str) -> dict:
    """
    Determine business impact and priority for a scored criterion

    Args:
        criterion_name: Name of criterion being scored
        score: The score (1-5)
        details: Details string (may contain indicators)

    Returns:
        {
            'business_impact': 'critical' | 'high' | 'medium' | 'low',
            'priority_score': 1-10,
            'rationale': str
        }
    """
    criterion_lower = criterion_name.lower()
    details_lower = details.lower()

    # CRITICAL impact detection
    if 'conversion tracking' in criterion_lower and score <= 2:
        return {
            'business_impact': 'critical',
            'priority_score': 10,
            'rationale': 'Broken conversion tracking stops all value measurement'
        }

    if 'brand' in criterion_lower and 'impression share' in criterion_lower and score <= 2:
        return {
            'business_impact': 'critical',
            'priority_score': 10,
            'rationale': 'Low brand IS means competitors stealing brand traffic'
        }

    # HIGH impact detection
    if 'negative keyword' in criterion_lower and score <= 2:
        return {
            'business_impact': 'high',
            'priority_score': 8,
            'rationale': 'High waste spend detected - urgent negative keyword additions needed'
        }

    if 'waste' in details_lower and score <= 2:
        return {
            'business_impact': 'high',
            'priority_score': 8,
            'rationale': 'Significant budget waste identified'
        }

    if 'quality score' in criterion_lower and score <= 2:
        return {
            'business_impact': 'high',
            'priority_score': 8,
            'rationale': 'Poor Quality Scores increase CPCs and reduce visibility'
        }

    if 'broad match' in criterion_lower and 'manual' in details_lower and score <= 2:
        return {
            'business_impact': 'high',
            'priority_score': 8,
            'rationale': 'Broad Match with Manual CPC is high waste risk'
        }

    # MEDIUM impact detection
    if 'rsa' in criterion_lower or 'ad strength' in criterion_lower:
        if score <= 3:
            return {
                'business_impact': 'medium',
                'priority_score': 5,
                'rationale': 'RSA improvements follow best practices'
            }

    if 'smart bidding' in criterion_lower and score <= 3:
        return {
            'business_impact': 'medium',
            'priority_score': 5,
            'rationale': 'Smart Bidding adoption improves efficiency'
        }

    if 'audience' in criterion_lower and score <= 3:
        return {
            'business_impact': 'medium',
            'priority_score': 5,
            'rationale': 'Audience targeting improves relevance and performance'
        }

    # LOW impact (default for organizational/structural)
    if any(keyword in criterion_lower for keyword in ['naming', 'label', 'structure', 'consolidation']):
        return {
            'business_impact': 'low',
            'priority_score': 2,
            'rationale': 'Organizational improvement for better management'
        }

    # Default: Medium if score is poor, Low if score is good
    if score <= 2:
        return {
            'business_impact': 'medium',
            'priority_score': 5,
            'rationale': 'Performance improvement opportunity'
        }
    else:
        return {
            'business_impact': 'low',
            'priority_score': 2,
            'rationale': 'Minor optimization opportunity'
        }


# ============================================================================
# WEIGHTING FRAMEWORK
# ============================================================================

CRITERION_WEIGHTS = {
    # Critical infrastructure (15-20% weight each)
    'conversion_tracking': 0.20,
    'quality_score': 0.15,

    # High impact tactics (10-12% weight each)
    'negative_keywords': 0.12,
    'audience_targeting': 0.10,
    'smart_bidding': 0.10,

    # Medium impact (5-8% weight each)
    'rsa_coverage': 0.08,
    'ad_strength': 0.07,
    'broad_match': 0.06,
    'impression_share': 0.05,

    # Lower impact (2-3% weight each)
    'sitelinks': 0.03,
    'callouts': 0.02,
    'naming_conventions': 0.02,

    # Default for unspecified
    'default': 0.01
}


def get_criterion_weight(criterion_name: str) -> float:
    """
    Get weight for a criterion in overall score calculation

    Returns weight (0-1), defaults to 0.01 if not found
    """
    criterion_lower = criterion_name.lower()

    for key, weight in CRITERION_WEIGHTS.items():
        if key in criterion_lower:
            return weight

    return CRITERION_WEIGHTS['default']


# ============================================================================
# CONFIDENCE SCORING
# ============================================================================

def calculate_confidence(data_completeness: float, criteria_scored: int, total_criteria: int) -> dict:
    """
    Calculate confidence in overall audit score

    Args:
        data_completeness: 0-1, percentage of required data fields present
        criteria_scored: Number of criteria that received scores
        total_criteria: Total number of criteria in template

    Returns:
        {
            'confidence': 'low' | 'medium' | 'high',
            'confidence_score': 0-100,
            'rationale': str
        }
    """
    coverage_pct = (criteria_scored / total_criteria * 100) if total_criteria > 0 else 0
    data_pct = data_completeness * 100

    # Combined confidence score
    confidence_score = (coverage_pct * 0.7) + (data_pct * 0.3)

    if confidence_score >= 70:
        confidence = 'high'
        rationale = f'{criteria_scored}/{total_criteria} criteria scored ({coverage_pct:.0f}%) with {data_pct:.0f}% data completeness'
    elif confidence_score >= 40:
        confidence = 'medium'
        rationale = f'{criteria_scored}/{total_criteria} criteria scored ({coverage_pct:.0f}%) - partial data coverage'
    else:
        confidence = 'low'
        rationale = f'Only {criteria_scored}/{total_criteria} criteria scored ({coverage_pct:.0f}%) - limited data available'

    return {
        'confidence': confidence,
        'confidence_score': confidence_score,
        'rationale': rationale
    }
