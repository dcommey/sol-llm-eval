"""
Prompt engineering for LLM-based smart contract auditing.
"""

from typing import Dict


class PromptTemplate:
    """Manages prompt templates for vulnerability detection."""
    
    def __init__(self, config: Dict):
        """
        Initialize prompt template.
        
        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config['prompts']
        self.system_message = self.config['system_message'].strip()
        self.instruction_template = self.config['instruction_template'].strip()
    
    def create_prompt(self, contract_code: str, model_type: str = "chat") -> Dict[str, str]:
        """
        Create formatted prompt for a given contract.
        
        Args:
            contract_code: Solidity contract source code
            model_type: Type of model ("chat" for instruct models, "completion" for base models)
            
        Returns:
            Dictionary with 'system' and 'user' messages for chat models,
            or single 'prompt' for completion models
        """
        user_message = self.instruction_template.format(contract_code=contract_code)
        
        if model_type == "chat":
            return {
                "system": self.system_message,
                "user": user_message
            }
        else:
            # For completion models, combine system and user messages
            full_prompt = f"{self.system_message}\n\n{user_message}"
            return {"prompt": full_prompt}
    
    def format_for_model(self, messages: Dict[str, str], model_name: str) -> str:
        """
        Format messages according to model's chat template.
        
        Args:
            messages: Dictionary with system/user messages or single prompt
            model_name: Name of the model (for special formatting)
            
        Returns:
            Formatted prompt string
        """
        if "prompt" in messages:
            # Completion-style model
            return messages["prompt"]
        
        # Chat-style models - most use similar format
        # Models will apply their own chat template via tokenizer
        return messages
    
    @staticmethod
    def extract_json_from_response(response: str) -> str:
        """
        Extract JSON array from LLM response.
        
        Args:
            response: Raw LLM output
            
        Returns:
            Extracted JSON string
        """
        # Try to find JSON array in response
        import re
        
        # Look for JSON array pattern
        json_pattern = r'\[\s*(?:\{[^}]+\}\s*,?\s*)*\]'
        matches = re.findall(json_pattern, response, re.DOTALL)
        
        if matches:
            # Return the longest match (likely the complete array)
            return max(matches, key=len)
        
        # If no JSON array found, try to find just the array brackets
        start_idx = response.find('[')
        end_idx = response.rfind(']')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return response[start_idx:end_idx + 1]
        
        # Last resort: return empty array
        return "[]"
