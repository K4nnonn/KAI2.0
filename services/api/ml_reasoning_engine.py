"""
ML Reasoning Engine using llama.cpp
Enhances audit scoring with deeper pattern recognition and anomaly detection
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import requests

from kai_core.config import is_azure_openai_enabled
from kai_core.shared.azure_budget import allow_azure_usage

logger = logging.getLogger(__name__)


class MLReasoningEngine:
    """
    Uses Ollama LLM for ML-enhanced reasoning in audit assessments

    Capabilities:
    1. Pattern Recognition: Identify trends and anomalies in PPC data
    2. Comparative Analysis: Compare against industry benchmarks
    3. Scoring Validation: Cross-validate scores with ML reasoning
    4. Recommendation Generation: Suggest optimizations based on patterns
    """

    def __init__(self):
        self.enabled = os.getenv('ENABLE_ML_REASONING', 'false').lower() == 'true'
        self.model_path = Path(os.getenv('LOCAL_LLM_MODEL_PATH', '/app/models/llama-3.2-1b-instruct-q4_k_m.gguf'))
        self.timeout = float(os.getenv('LOCAL_LLM_TIMEOUT_SECONDS', '30'))

        # Fallback to Azure OpenAI if local model not available
        self.azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT', '')
        self.azure_key = os.getenv('AZURE_OPENAI_API_KEY', '')
        self.azure_deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT', 'gpt-4-turbo')

        # Local LLM instance (loaded on-demand)
        self.llm = None
        if self.enabled:
            self._load_local_model()

        logger.info(f"MLReasoningEngine initialized: enabled={self.enabled}, local_model_loaded={self.llm is not None}")

    def analyze_campaign_patterns(
        self,
        campaign_data: Dict[str, Any],
        criterion: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze campaign data for patterns and anomalies

        Returns:
            {
                'patterns': List of identified patterns,
                'anomalies': List of detected anomalies,
                'confidence': Confidence score 0-1,
                'reasoning': Detailed reasoning,
                'recommendations': List of actionable recommendations
            }
        """
        if not self.enabled:
            return self._default_analysis()

        # Prepare prompt for ML reasoning
        prompt = self._build_pattern_analysis_prompt(campaign_data, criterion, context)

        # Try local llama.cpp first, fallback to Azure
        analysis = self._call_local_llm(prompt)
        if not analysis:
            analysis = self._call_azure_reasoning(prompt)

        if not analysis:
            return self._default_analysis()

        return self._parse_analysis_response(analysis)

    def validate_score(
        self,
        criterion: str,
        initial_score: int,
        evidence: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Use ML reasoning to validate and potentially adjust audit scores

        Returns:
            {
                'validated_score': int (1-5),
                'confidence': float (0-1),
                'reasoning': str,
                'adjustments': List of suggested adjustments,
                'evidence_strength': str ('weak', 'moderate', 'strong')
            }
        """
        if not self.enabled:
            return {
                'validated_score': initial_score,
                'confidence': 0.5,
                'reasoning': 'ML reasoning disabled',
                'adjustments': [],
                'evidence_strength': 'moderate'
            }

        prompt = self._build_score_validation_prompt(
            criterion, initial_score, evidence, context
        )

        response = self._call_local_llm(prompt)
        if not response:
            response = self._call_azure_reasoning(prompt)

        if not response:
            return {
                'validated_score': initial_score,
                'confidence': 0.5,
                'reasoning': 'ML reasoning unavailable',
                'adjustments': [],
                'evidence_strength': 'moderate'
            }

        return self._parse_validation_response(response, initial_score)

    def detect_anomalies(
        self,
        metrics: Dict[str, float],
        benchmarks: Dict[str, float],
        tolerance: float = 0.2
    ) -> List[Dict[str, Any]]:
        """
        Detect anomalies by comparing metrics against benchmarks

        Returns list of anomalies with:
            {
                'metric': str,
                'value': float,
                'expected': float,
                'deviation': float,
                'severity': str ('low', 'medium', 'high'),
                'explanation': str
            }
        """
        if not self.enabled:
            return []

        anomalies = []

        for metric, value in metrics.items():
            if metric not in benchmarks:
                continue

            expected = benchmarks[metric]
            if expected == 0:
                continue

            deviation = abs(value - expected) / expected

            if deviation > tolerance:
                # Use ML to explain the anomaly
                explanation = self._explain_anomaly(metric, value, expected, deviation)

                anomalies.append({
                    'metric': metric,
                    'value': value,
                    'expected': expected,
                    'deviation': deviation,
                    'severity': self._classify_severity(deviation),
                    'explanation': explanation
                })

        return anomalies

    def generate_insights(
        self,
        audit_results: Dict[str, Any],
        historical_data: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Generate ML-enhanced insights from audit results

        Returns:
            {
                'key_findings': List of main insights,
                'trends': List of identified trends,
                'predictions': List of predictions,
                'priority_actions': List of high-priority recommendations
            }
        """
        if not self.enabled:
            return {
                'key_findings': [],
                'trends': [],
                'predictions': [],
                'priority_actions': []
            }

        prompt = self._build_insights_prompt(audit_results, historical_data)

        response = self._call_local_llm(prompt)
        if not response:
            response = self._call_azure_reasoning(prompt)

        if not response:
            return {
                'key_findings': ['ML insights unavailable'],
                'trends': [],
                'predictions': [],
                'priority_actions': []
            }

        return self._parse_insights_response(response)

    # ========== PRIVATE METHODS ==========

    def _load_local_model(self):
        """Load llama.cpp model for in-process inference"""
        try:
            from llama_cpp import Llama

            if not self.model_path.exists():
                logger.warning(f"Model file not found: {self.model_path}")
                return

            logger.info(f"Loading llama.cpp model from {self.model_path}")
            self.llm = Llama(
                model_path=str(self.model_path),
                n_ctx=2048,  # Context window
                n_threads=2,  # CPU threads for inference
                n_gpu_layers=0,  # CPU-only for Azure Container Apps
                verbose=False
            )
            logger.info("Local LLM model loaded successfully")
        except ImportError:
            logger.warning("llama-cpp-python not installed, local LLM disabled")
            self.llm = None
        except Exception as e:
            logger.warning(f"Failed to load llama.cpp model: {e}")
            self.llm = None

    def _call_local_llm(self, prompt: str) -> Optional[str]:
        """Call llama.cpp for local ML reasoning"""
        if not self.llm:
            return None
        try:
            response = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": "You are an ML reasoning engine for PPC audit analysis. Provide structured, data-driven insights."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            logger.warning(f"Local LLM inference failed: {e}")
            return None

    def _call_ollama_reasoning(self, prompt: str) -> Optional[str]:
        """Call Ollama for ML reasoning"""
        try:
            url = f"{self.ollama_endpoint}/api/generate"
            headers = {'Content-Type': 'application/json'}

            payload = {
                'model': self.ollama_model,
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': 0.3,  # Lower temp for more consistent reasoning
                    'num_predict': 1000,
                }
            }

            response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            response.raise_for_status()

            result = response.json()
            return result.get('response', '').strip()

        except Exception as e:
            logger.warning(f"Ollama reasoning failed: {e}")
            return None

    def _call_azure_reasoning(self, prompt: str) -> Optional[str]:
        """Fallback to Azure OpenAI for reasoning"""
        if os.getenv("ML_REASONING_ALLOW_AZURE", "true").lower() != "true":
            return None
        if not is_azure_openai_enabled():
            return None
        allowed, reason = allow_azure_usage(module="ml_reasoning_engine", purpose="reasoning")
        if not allowed:
            logger.warning("Azure reasoning blocked by policy: %s", reason)
            return None
        if not self.azure_endpoint or not self.azure_key:
            return None

        try:
            url = f"{self.azure_endpoint.rstrip('/')}/openai/deployments/{self.azure_deployment}/chat/completions?api-version=2024-02-15-preview"
            headers = {
                'Content-Type': 'application/json',
                'api-key': self.azure_key
            }

            payload = {
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are an ML reasoning engine for PPC audit analysis. Provide structured, data-driven insights.'
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                'temperature': 0.3,
                'max_tokens': 1000
            }

            response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            response.raise_for_status()

            result = response.json()
            return result['choices'][0]['message']['content'].strip()

        except Exception as e:
            logger.warning(f"Azure reasoning failed: {e}")
            return None

    def _build_pattern_analysis_prompt(
        self,
        campaign_data: Dict[str, Any],
        criterion: str,
        context: Dict[str, Any]
    ) -> str:
        """Build prompt for pattern analysis"""
        return f"""Analyze the following PPC campaign data for patterns and anomalies.

Criterion: {criterion}
Campaign Data: {json.dumps(campaign_data, indent=2)}
Context: {json.dumps(context, indent=2)}

Provide a structured analysis in JSON format:
{{
    "patterns": ["list of identified patterns"],
    "anomalies": ["list of detected anomalies"],
    "confidence": 0.85,
    "reasoning": "detailed reasoning",
    "recommendations": ["actionable recommendations"]
}}

Focus on data-driven insights and quantitative analysis."""

    def _build_score_validation_prompt(
        self,
        criterion: str,
        initial_score: int,
        evidence: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """Build prompt for score validation"""
        return f"""Validate the audit score for this PPC criterion using ML reasoning.

Criterion: {criterion}
Initial Score: {initial_score}/5
Evidence: {json.dumps(evidence, indent=2)}
Context: {json.dumps(context, indent=2)}

Provide validation in JSON format:
{{
    "validated_score": 3,
    "confidence": 0.9,
    "reasoning": "detailed reasoning for score",
    "adjustments": ["suggested adjustments if any"],
    "evidence_strength": "strong|moderate|weak"
}}

Consider: data completeness, metric thresholds, industry standards, and logical consistency."""

    def _build_insights_prompt(
        self,
        audit_results: Dict[str, Any],
        historical_data: Optional[List[Dict]]
    ) -> str:
        """Build prompt for insights generation"""
        hist_str = json.dumps(historical_data, indent=2) if historical_data else "None"

        return f"""Generate ML-enhanced insights from these PPC audit results.

Current Audit: {json.dumps(audit_results, indent=2)}
Historical Data: {hist_str}

Provide insights in JSON format:
{{
    "key_findings": ["3-5 main insights"],
    "trends": ["identified trends over time"],
    "predictions": ["data-driven predictions"],
    "priority_actions": ["high-priority recommendations"]
}}

Focus on actionable, quantitative insights."""

    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from analysis"""
        try:
            # Try to extract JSON from response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
        except Exception as e:
            logger.warning(f"Failed to parse analysis response: {e}")

        return self._default_analysis()

    def _parse_validation_response(
        self,
        response: str,
        initial_score: int
    ) -> Dict[str, Any]:
        """Parse JSON response from validation"""
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                parsed = json.loads(json_str)

                # Ensure validated_score is in range 1-5
                validated_score = parsed.get('validated_score', initial_score)
                validated_score = max(1, min(5, int(validated_score)))

                return {
                    'validated_score': validated_score,
                    'confidence': float(parsed.get('confidence', 0.5)),
                    'reasoning': parsed.get('reasoning', ''),
                    'adjustments': parsed.get('adjustments', []),
                    'evidence_strength': parsed.get('evidence_strength', 'moderate')
                }
        except Exception as e:
            logger.warning(f"Failed to parse validation response: {e}")

        return {
            'validated_score': initial_score,
            'confidence': 0.5,
            'reasoning': 'Parse failed',
            'adjustments': [],
            'evidence_strength': 'moderate'
        }

    def _parse_insights_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from insights"""
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
        except Exception as e:
            logger.warning(f"Failed to parse insights response: {e}")

        return {
            'key_findings': [],
            'trends': [],
            'predictions': [],
            'priority_actions': []
        }

    def _explain_anomaly(
        self,
        metric: str,
        value: float,
        expected: float,
        deviation: float
    ) -> str:
        """Use ML to explain an anomaly"""
        prompt = f"""Explain this PPC metric anomaly in one concise sentence:

Metric: {metric}
Actual Value: {value}
Expected Value: {expected}
Deviation: {deviation:.1%}

Provide a brief, actionable explanation."""

        response = self._call_local_llm(prompt)
        if not response:
            response = self._call_azure_reasoning(prompt)

        return response if response else f"{metric} deviates {deviation:.1%} from expected"

    def _classify_severity(self, deviation: float) -> str:
        """Classify anomaly severity"""
        if deviation > 0.5:
            return 'high'
        elif deviation > 0.3:
            return 'medium'
        else:
            return 'low'

    def _default_analysis(self) -> Dict[str, Any]:
        """Return default analysis when ML reasoning unavailable"""
        return {
            'patterns': [],
            'anomalies': [],
            'confidence': 0.0,
            'reasoning': 'ML reasoning unavailable',
            'recommendations': []
        }


# Global instance
_ml_engine = None

def get_ml_reasoning_engine() -> MLReasoningEngine:
    """Get singleton ML reasoning engine"""
    global _ml_engine
    if _ml_engine is None:
        _ml_engine = MLReasoningEngine()
    return _ml_engine
