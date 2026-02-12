"""
Lightweight dataframe validation inspired by Great Expectations.
Returns a ValidationResult instead of throwing, so existing flows remain intact.
"""
from dataclasses import dataclass, field
from typing import List
import pandas as pd


@dataclass
class ValidationResult:
    passed: bool
    issues: List[str] = field(default_factory=list)


class DataValidator:
    def __init__(self, required_columns: List[str], non_negative_cols: List[str] = None, unique_cols: List[str] = None):
        self.required_columns = required_columns
        self.non_negative_cols = non_negative_cols or []
        self.unique_cols = unique_cols or []

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        issues = []
        cols = set(df.columns)

        missing = [c for c in self.required_columns if c not in cols]
        if missing:
            issues.append(f"Missing columns: {missing}")

        for c in self.non_negative_cols:
            if c in cols and (df[c] < 0).any():
                issues.append(f"Negative values found in {c}")

        for c in self.unique_cols:
            if c in cols:
                dupes = df[c][df[c].duplicated()]
                if not dupes.empty:
                    issues.append(f"Duplicate values in {c}: {dupes.unique()[:5]}")

        return ValidationResult(passed=len(issues) == 0, issues=issues)
