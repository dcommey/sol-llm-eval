"""
Ollama-based LLM clients for efficient local inference on Apple Silicon.
"""

import json
import logging
import requests
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from src.prompt_engineering import PromptTemplate

logger = logging.getLogger(__name__)

OLLAMA_API_URL = "http://localhost:11434"


class VulnerabilityReport:
    """Structured vulnerability report."""
    
    def __init__(self, vulnerability_type: str, line_numbers: List[int],
                 severity: str, explanation: str):
        self.vulnerability_type = vulnerability_type
        self.line_numbers = line_numbers
        self.severity = severity
        self.explanation = explanation
    
    def to_dict(self) -> Dict:
        return {
            "vulnerability_type": self.vulnerability_type,
            "line_numbers": self.line_numbers,
            "severity": self.severity,
            "explanation": self.explanation
        }


class OllamaClient:
    """Client for Ollama-based inference - works great on Apple Silicon."""
    
    def __init__(self, model_name: str, display_name: str, config: Dict):
        """
        Initialize Ollama client.
        
        Args:
            model_name: Ollama model name (e.g., 'qwen2.5-coder:7b')
            display_name: Human-readable model name
            config: Configuration dictionary
        """
        self.model_name = model_name
        self.display_name = display_name
        self.config = config
        self.prompt_template = PromptTemplate(config)
        self.api_url = OLLAMA_API_URL
        self._model_loaded = False
    
    def _check_ollama_running(self) -> bool:
        """Check if Ollama service is running."""
        try:
            response = requests.get(f"{self.api_url}/api/version", timeout=5)
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False
    
    def load_model(self) -> None:
        """Ensure model is available (pull if needed)."""
        if not self._check_ollama_running():
            raise RuntimeError(
                "Ollama is not running. Start it with 'ollama serve' or open Ollama app."
            )
        
        logger.info(f"Checking if {self.model_name} is available...")
        
        # Check if model exists
        try:
            response = requests.get(f"{self.api_url}/api/tags", timeout=30)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                
                # Check for exact match or partial match
                found = any(self.model_name in name for name in model_names)
                
                if not found:
                    logger.info(f"Pulling model {self.model_name}...")
                    # Pull model
                    pull_response = requests.post(
                        f"{self.api_url}/api/pull",
                        json={"name": self.model_name},
                        stream=True,
                        timeout=3600  # 1 hour timeout for large models
                    )
                    for line in pull_response.iter_lines():
                        if line:
                            data = json.loads(line)
                            status = data.get('status', '')
                            if 'pulling' in status or 'download' in status.lower():
                                logger.info(f"  {status}")
                else:
                    logger.info(f"Model {self.model_name} already available")
        except Exception as e:
            logger.error(f"Error checking/pulling model: {e}")
            raise
        
        self._model_loaded = True
        logger.info(f"{self.display_name} ready for inference")
    
    def cleanup(self) -> None:
        """No cleanup needed for Ollama - model stays in memory."""
        logger.info(f"{self.display_name} session ended (Ollama keeps model loaded)")
        self._model_loaded = False
    
    def _generate_response(self, prompt: str) -> str:
        """Generate response using Ollama API."""
        gen_config = self.config['inference']['generation']
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": gen_config.get('temperature', 0.1),
                "top_p": gen_config.get('top_p', 0.95),
                "num_predict": gen_config.get('max_new_tokens', 2048),
            }
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/api/generate",
                json=payload,
                timeout=300  # 5 min timeout per request
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '')
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return ""
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return ""
    
    def format_prompt(self, contract_code: str) -> str:
        """Format prompt for the model."""
        messages = self.prompt_template.create_prompt(contract_code, model_type="chat")
        # Simple format that works with most Ollama models
        return f"SYSTEM: {messages['system']}\n\nUSER: {messages['user']}\n\nASSISTANT:"
    
    def parse_response(self, raw_response: str) -> List[VulnerabilityReport]:
        """Parse LLM response into structured vulnerability reports."""
        try:
            json_str = PromptTemplate.extract_json_from_response(raw_response)
            vulnerabilities_data = json.loads(json_str)
            
            # Handle case where LLM returns a non-list
            if not isinstance(vulnerabilities_data, list):
                vulnerabilities_data = [vulnerabilities_data] if isinstance(vulnerabilities_data, dict) else []
            
            reports = []
            for vuln in vulnerabilities_data:
                # Skip non-dict items (e.g., integers, strings)
                if not isinstance(vuln, dict):
                    logger.warning(f"Skipping non-dict item in response: {type(vuln)}")
                    continue
                    
                report = VulnerabilityReport(
                    vulnerability_type=vuln.get('vulnerability_type', 'unknown'),
                    line_numbers=vuln.get('line_numbers', []),
                    severity=vuln.get('severity', 'unknown'),
                    explanation=vuln.get('explanation', '')
                )
                reports.append(report)
            
            return reports
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON: {e}")
            logger.debug(f"Raw response: {raw_response[:500]}")
            return []
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return []
    
    def analyze_contract(self, contract_code: str) -> Dict[str, Any]:
        """Analyze a smart contract for vulnerabilities."""
        try:
            prompt = self.format_prompt(contract_code)
            raw_response = self._generate_response(prompt)
            vulnerabilities = self.parse_response(raw_response)
            
            return {
                "raw_response": raw_response,
                "vulnerabilities": [v.to_dict() for v in vulnerabilities],
                "num_vulnerabilities": len(vulnerabilities),
                "error": None
            }
        except Exception as e:
            logger.error(f"Error analyzing contract: {e}")
            return {
                "raw_response": "",
                "vulnerabilities": [],
                "num_vulnerabilities": 0,
                "error": str(e)
            }


# Available Ollama models for the benchmark
OLLAMA_MODELS = {
    'qwen': {
        'model_name': 'qwen2.5-coder:7b',
        'display_name': 'Qwen2.5-Coder-7B'
    },
    'deepseek': {
        'model_name': 'deepseek-coder:6.7b',
        'display_name': 'DeepSeek-Coder-6.7B'
    },
    'codellama': {
        'model_name': 'codellama:7b-instruct',
        'display_name': 'CodeLLaMA-7B'
    },
    'mistral': {
        'model_name': 'mistral:7b-instruct',
        'display_name': 'Mistral-7B'
    }
}


def create_ollama_client(model_key: str, config: Dict) -> OllamaClient:
    """
    Factory function to create Ollama client.
    
    Args:
        model_key: Key from config (e.g., 'qwen', 'deepseek')
        config: Full configuration dictionary
        
    Returns:
        Initialized OllamaClient
    """
    if model_key not in OLLAMA_MODELS:
        raise ValueError(f"Unknown model key: {model_key}. Available: {list(OLLAMA_MODELS.keys())}")
    
    model_info = OLLAMA_MODELS[model_key]
    
    # Override with config if available
    if 'models' in config and model_key in config['models']:
        model_config = config['models'][model_key]
        if 'ollama_model' in model_config:
            model_info['model_name'] = model_config['ollama_model']
        if 'display_name' in model_config:
            model_info['display_name'] = model_config['display_name']
    
    return OllamaClient(
        model_name=model_info['model_name'],
        display_name=model_info['display_name'],
        config=config
    )
