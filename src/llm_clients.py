"""
LLM clients for local model inference with memory-efficient loading.
"""

import json
import logging
import torch
import platform
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from src.prompt_engineering import PromptTemplate

logger = logging.getLogger(__name__)

# Check if we're on Apple Silicon - BitsAndBytes doesn't work on MPS
IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"


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


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(self, model_id: str, display_name: str, config: Dict):
        """
        Initialize LLM client.
        
        Args:
            model_id: HuggingFace model identifier
            display_name: Human-readable model name
            config: Configuration dictionary
        """
        self.model_id = model_id
        self.display_name = display_name
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = self._get_device()
        self.prompt_template = PromptTemplate(config)
        
    def _get_device(self) -> str:
        """Determine best available device."""
        device_config = self.config['inference']['device']
        
        if device_config == "auto":
            if torch.backends.mps.is_available() and IS_APPLE_SILICON:
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device_config
    
    def _get_quantization_config(self) -> Optional[Any]:
        """Create quantization config if enabled and supported."""
        quant_config = self.config['inference']['quantization']
        
        if not quant_config['enabled']:
            return None
        
        # BitsAndBytes doesn't work on Apple Silicon/MPS
        if IS_APPLE_SILICON or self.device == "mps":
            logger.warning(
                "BitsAndBytes quantization not supported on Apple Silicon. "
                "Using float16 instead for memory efficiency."
            )
            return None
        
        # Only import BitsAndBytes if we're going to use it (CUDA only)
        try:
            from transformers import BitsAndBytesConfig
            return BitsAndBytesConfig(
                load_in_4bit=quant_config['load_in_4bit'],
                bnb_4bit_compute_dtype=getattr(torch, quant_config['bnb_4bit_compute_dtype']),
                bnb_4bit_use_double_quant=quant_config['bnb_4bit_use_double_quant'],
                bnb_4bit_quant_type=quant_config['bnb_4bit_quant_type']
            )
        except ImportError:
            logger.warning("BitsAndBytes not available, using float16")
            return None
    
    def load_model(self) -> None:
        """Load model and tokenizer into memory."""
        if self.model is not None:
            logger.info(f"{self.display_name} already loaded")
            return
        
        logger.info(f"Loading {self.display_name} from {self.model_id}")
        logger.info(f"Device: {self.device}, Apple Silicon: {IS_APPLE_SILICON}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, 
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Prepare model loading arguments
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
            "low_cpu_mem_usage": True,  # Important for large models
        }
        
        # Add quantization if enabled and supported (CUDA only)
        quant_config = self._get_quantization_config()
        if quant_config:
            model_kwargs["quantization_config"] = quant_config
            model_kwargs["device_map"] = "auto"  # Required for quantization
            logger.info("Using 4-bit quantization (CUDA)")
        elif self.device == "cuda":
            model_kwargs["device_map"] = "auto"
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **model_kwargs
        )
        
        # Move to device if not using device_map (MPS or CPU)
        if "device_map" not in model_kwargs:
            logger.info(f"Moving model to {self.device}")
            self.model.to(self.device)
        
        self.model.eval()
        logger.info(f"{self.display_name} loaded successfully on {self.device}")
    
    def cleanup(self) -> None:
        """Unload model from memory."""
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            
            # Clear cache
            if self.device == "cuda":
                torch.cuda.empty_cache()
            elif self.device == "mps":
                torch.mps.empty_cache()
            
            logger.info(f"{self.display_name} unloaded from memory")
    
    def _generate_response(self, prompt: str) -> str:
        """
        Generate response from model.
        
        Args:
            prompt: Formatted prompt string
            
        Returns:
            Generated text
        """
        if self.model is None:
            self.load_model()
        
        gen_config = self.config['inference']['generation']
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=gen_config['max_new_tokens'],
                temperature=gen_config['temperature'],
                do_sample=gen_config['do_sample'],
                top_p=gen_config['top_p'],
                top_k=gen_config['top_k'],
                repetition_penalty=gen_config['repetition_penalty'],
                num_return_sequences=gen_config['num_return_sequences'],
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from the output
        if prompt in generated_text:
            generated_text = generated_text.replace(prompt, "", 1).strip()
        
        return generated_text
    
    @abstractmethod
    def format_prompt(self, contract_code: str) -> str:
        """Format prompt according to model's specific requirements."""
        pass
    
    def parse_response(self, raw_response: str) -> List[VulnerabilityReport]:
        """
        Parse LLM response into structured vulnerability reports.
        
        Args:
            raw_response: Raw text output from LLM
            
        Returns:
            List of VulnerabilityReport objects
        """
        try:
            # Extract JSON from response
            json_str = PromptTemplate.extract_json_from_response(raw_response)
            vulnerabilities_data = json.loads(json_str)
            
            # Convert to VulnerabilityReport objects
            reports = []
            for vuln in vulnerabilities_data:
                report = VulnerabilityReport(
                    vulnerability_type=vuln.get('vulnerability_type', 'unknown'),
                    line_numbers=vuln.get('line_numbers', []),
                    severity=vuln.get('severity', 'unknown'),
                    explanation=vuln.get('explanation', '')
                )
                reports.append(report)
            
            return reports
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from response: {e}")
            logger.debug(f"Raw response: {raw_response[:500]}")
            return []
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return []
    
    def analyze_contract(self, contract_code: str) -> Dict[str, Any]:
        """
        Analyze a smart contract for vulnerabilities.
        
        Args:
            contract_code: Solidity source code
            
        Returns:
            Dictionary with raw_response and parsed vulnerabilities
        """
        try:
            # Format prompt
            prompt = self.format_prompt(contract_code)
            
            # Generate response
            raw_response = self._generate_response(prompt)
            
            # Parse response
            vulnerabilities = self.parse_response(raw_response)
            
            return {
                "raw_response": raw_response,
                "vulnerabilities": [v.to_dict() for v in vulnerabilities],
                "num_vulnerabilities": len(vulnerabilities),
                "error": None
            }
        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA out of memory during inference")
            return {
                "raw_response": "",
                "vulnerabilities": [],
                "num_vulnerabilities": 0,
                "error": "out_of_memory"
            }
        except Exception as e:
            logger.error(f"Error analyzing contract: {e}")
            import traceback
            traceback.print_exc()
            return {
                "raw_response": "",
                "vulnerabilities": [],
                "num_vulnerabilities": 0,
                "error": str(e)
            }


class Qwen25CoderClient(LLMClient):
    """Client for Qwen2.5-Coder-7B-Instruct."""
    
    def format_prompt(self, contract_code: str) -> str:
        """Format prompt using Qwen's chat template."""
        messages = self.prompt_template.create_prompt(contract_code, model_type="chat")
        
        # Qwen uses its own chat template via tokenizer
        chat_prompt = f"<|im_start|>system\n{messages['system']}<|im_end|>\n<|im_start|>user\n{messages['user']}<|im_end|>\n<|im_start|>assistant\n"
        return chat_prompt


class DeepSeekCoderClient(LLMClient):
    """Client for DeepSeek-Coder-7B-Instruct-v1.5."""
    
    def format_prompt(self, contract_code: str) -> str:
        """Format prompt using DeepSeek's template."""
        messages = self.prompt_template.create_prompt(contract_code, model_type="chat")
        
        # DeepSeek format
        chat_prompt = f"### System:\n{messages['system']}\n\n### Instruction:\n{messages['user']}\n\n### Response:\n"
        return chat_prompt


class CodeLLaMAClient(LLMClient):
    """Client for CodeLLaMA-7B-Instruct."""
    
    def format_prompt(self, contract_code: str) -> str:
        """Format prompt using CodeLLaMA's template."""
        messages = self.prompt_template.create_prompt(contract_code, model_type="chat")
        
        # CodeLLaMA Instruct format
        chat_prompt = f"[INST] <<SYS>>\n{messages['system']}\n<</SYS>>\n\n{messages['user']} [/INST]"
        return chat_prompt


class MistralClient(LLMClient):
    """Client for Mistral-7B-Instruct-v0.3."""
    
    def format_prompt(self, contract_code: str) -> str:
        """Format prompt using Mistral's template."""
        messages = self.prompt_template.create_prompt(contract_code, model_type="chat")
        
        # Mistral Instruct format
        chat_prompt = f"<s>[INST] {messages['system']}\n\n{messages['user']} [/INST]"
        return chat_prompt


def create_llm_client(model_key: str, config: Dict) -> LLMClient:
    """
    Factory function to create appropriate LLM client.
    
    Args:
        model_key: Key from config (e.g., 'qwen', 'deepseek')
        config: Full configuration dictionary
        
    Returns:
        Initialized LLM client
    """
    model_config = config['models'][model_key]
    model_id = model_config['model_id']
    display_name = model_config['display_name']
    
    client_map = {
        'qwen': Qwen25CoderClient,
        'deepseek': DeepSeekCoderClient,
        'codellama': CodeLLaMAClient,
        'mistral': MistralClient
    }
    
    if model_key not in client_map:
        raise ValueError(f"Unknown model key: {model_key}")
    
    client_class = client_map[model_key]
    return client_class(model_id, display_name, config)
