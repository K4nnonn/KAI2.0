"""
Kai Core Package
Local business logic and plugins for the Kai platform.

This package is intentionally disjoint from the Microsoft `azure` SDK
namespace to avoid collisions.
"""
__version__ = "1.0.0"

# Public facades for the new intelligence layer
from kai_core.market_intelligence import MarketIntelligence, MarketVolatilitySummary
from kai_core.unified_schema_manager import UnifiedSchemaManager
from kai_core.agentic_orchestrator import MarketingReasoningAgent

__all__ = [
    "MarketIntelligence",
    "MarketVolatilitySummary",
    "UnifiedSchemaManager",
    "MarketingReasoningAgent",
]
