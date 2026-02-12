"""
SQR Analyzer using n-gram mining with pandas explode pattern.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd


class SqrAnalyzer:
    def __init__(self, search_term_col: str = "Search term", cost_col: str = "Cost", conv_col: str = "Conversions"):
        self.search_term_col = search_term_col
        self.cost_col = cost_col
        self.conv_col = conv_col

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
        return [t for t in cleaned.split() if t]

    @staticmethod
    def _generate_ngrams(tokens: List[str], n: int) -> List[str]:
        return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    def analyze_ngrams(self, df: pd.DataFrame, n_values: Tuple[int, ...] = (1, 2, 3)) -> pd.DataFrame:
        if df.empty or self.search_term_col not in df.columns:
            return pd.DataFrame(columns=["ngram", self.cost_col, self.conv_col, "frequency", "n"])

        df = df.copy()
        df["__tokens"] = df[self.search_term_col].astype(str).apply(self._tokenize)

        frames = []
        for n in n_values:
            temp = df.copy()
            temp["ngram"] = temp["__tokens"].apply(lambda toks: self._generate_ngrams(toks, n))
            temp = temp.explode("ngram")
            grouped = (
                temp.dropna(subset=["ngram"])
                .groupby("ngram")
                .agg(
                    {
                        self.cost_col: "sum",
                        self.conv_col: "sum",
                        self.search_term_col: "count",
                    }
                )
                .rename(columns={self.search_term_col: "frequency"})
                .reset_index()
            )
            grouped["n"] = n
            frames.append(grouped)

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def negative_candidates(self, ngram_stats: pd.DataFrame, target_cpa: float) -> pd.DataFrame:
        if ngram_stats.empty:
            return ngram_stats
        threshold = target_cpa * 1.5
        waste = ngram_stats[(ngram_stats[self.conv_col] == 0) & (ngram_stats[self.cost_col] > threshold)]
        return waste.sort_values(by=self.cost_col, ascending=False)

    def analyze(self, data: Iterable[Dict[str, Any]], target_cpa: float = 50.0, n_values: Tuple[int, ...] = (1, 2, 3)) -> Dict[str, Any]:
        df = pd.DataFrame(list(data))
        ngram_stats = self.analyze_ngrams(df, n_values=n_values)
        negatives = self.negative_candidates(ngram_stats, target_cpa=target_cpa)
        return {
            "ngrams": ngram_stats.to_dict(orient="records"),
            "negative_candidates": negatives.to_dict(orient="records"),
        }
