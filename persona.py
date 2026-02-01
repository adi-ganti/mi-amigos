"""
Persona Manager - Handles LoRA adapter loading and switching
"""
import os
# Fix Windows-specific PyTorch/tokenizer issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from pathlib import Path
from typing import Optional, Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
BASE_MODEL_PATH = ROOT_DIR / "models" / "gemma-3-4b-it"
ADAPTERS_DIR = ROOT_DIR / "Adapters"

AVAILABLE_PERSONAS = []


class PersonaManager:
    def __init__(self):
        self.base_model = None
        self.tokenizer = None
        self.current_model = None
        self.current_persona = None
        self._initialized = False
    
    def get_available_personas(self) -> List[str]:
        """Get list of available personas from Adapters folder."""
        personas = []
        if ADAPTERS_DIR.exists():
            for adapter_dir in ADAPTERS_DIR.iterdir():
                if adapter_dir.is_dir():
                    if (adapter_dir / "adapter_model.safetensors").exists() or \
                       (adapter_dir / "adapter_model.bin").exists():
                        personas.append(adapter_dir.name)
        return sorted(personas)
    
    def initialize(self):
        """Load the base model and tokenizer."""
        if self._initialized:
            return
        
        print("🔧 Loading base model (this takes ~15 seconds)...")
        print(f"   Model path: {BASE_MODEL_PATH}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL_PATH))
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            str(BASE_MODEL_PATH),
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            low_cpu_mem_usage=True,
        )
        
        self._initialized = True
        print(f"✓ Base model loaded")
        print(f"  VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"  Available personas: {', '.join(self.get_available_personas())}")
    
    def load_persona(self, persona_name: str) -> bool:
        """Load a specific persona's LoRA adapter."""
        self.initialize()
        
        if persona_name == self.current_persona:
            print(f"ℹ️  {persona_name} already loaded")
            return True
        
        adapter_path = ADAPTERS_DIR / persona_name
        if not adapter_path.exists():
            print(f"❌ Adapter not found: {adapter_path}")
            return False
        
        print(f"🔄 Loading {persona_name}'s adapter...")
        
        # Unload previous adapter if exists
        if self.current_model is not None:
            del self.current_model
            torch.cuda.empty_cache()
        
        self.current_model = PeftModel.from_pretrained(
            self.base_model,
            str(adapter_path),
            device_map={"": "cuda:0"},
        )
        self.current_model.eval()
        self.current_persona = persona_name
        
        print(f"✓ Now chatting as: {persona_name}")
        return True
    
    def generate_response(
        self,
        message: str,
        context: Optional[str] = None,
        chat_history: Optional[List[Dict]] = None,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
    ) -> str:
        """Generate a response using the current persona."""
        if self.current_model is None:
            return "⚠️ No persona loaded. Please select a friend first."
        
        messages = [
            {"role": "user", "content": message}
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.current_model.device)
        
        with torch.no_grad():
            outputs = self.current_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode response - EXACTLY like test_charan_adapter.py
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "<start_of_turn>model" in response:
            response = response.split("<start_of_turn>model")[-1]
        if "<end_of_turn>" in response:
            response = response.split("<end_of_turn>")[0]
        
        response = response.strip()
        
        response = self._extract_first_response(response, user_message=message)
        
        return response
    
    def _extract_first_response(self, response: str, user_message: str = "") -> str:
        """Extract only the first complete response, handling multiple model turns and leaked training data."""
        response = response.replace("model\n", "").replace("user\n", "").strip()
        
        if "\nmodel" in response:
            response = response.split("\nmodel")[0].strip()
        
        if "\nuser" in response:
            response = response.split("\nuser")[0].strip()
        
        if user_message and response.startswith(user_message):
            response = response[len(user_message):].strip()
        
        response = self._filter_leaked_names(response)
        
        response = self._make_cohesive(response)
        
        return response
    
    def _filter_leaked_names(self, response: str) -> str:
        """Filter out leaked training data where other people's names appear."""
        lines = response.split('\n')
        filtered_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if ':' in line:
                parts = line.split(':', 1)
                potential_name = parts[0].strip()
                
                if potential_name.replace(' ', '').replace('_', '').isalpha() and len(potential_name) < 30:
                    if self.current_persona and potential_name != self.current_persona:
                        persona_first_name = self.current_persona.split()[0].split('_')[0]
                        if not potential_name.startswith(persona_first_name):
                            print(f"  Filtering leaked training data: {potential_name}")
                            continue
                    # If it IS the current persona, remove the "Name:" prefix
                    if self.current_persona and (potential_name == self.current_persona or 
                                                  potential_name.startswith(self.current_persona.split()[0].split('_')[0])):
                        line = parts[1].strip()
            
            filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)
    
    def _make_cohesive(self, response: str) -> str:
        """Make multi-line response cohesive while preserving style and content."""
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        
        if len(lines) == 0:
            return "..."
        elif len(lines) == 1:
            return lines[0]
        else:
            # Join lines intelligently to create natural flow
            result = []
            current_group = []
            
            for i, line in enumerate(lines):
                current_group.append(line)
                
                # Check if we should start a new paragraph/group
                # Keep grouping if:
                # - Line is short (< 50 chars) and doesn't end with strong punctuation
                # - Next line exists and is also short
                is_short = len(line) < 50
                ends_with_period = line.endswith('.')
                is_last = i == len(lines) - 1
                
                # Decide whether to continue grouping or flush
                if is_last or (not is_short) or (ends_with_period and len(line) > 30):
                    # Flush current group
                    if len(current_group) == 1:
                        result.append(current_group[0])
                    else:
                        # Join with spaces, adding punctuation if needed
                        joined = self._join_lines_naturally(current_group)
                        result.append(joined)
                    current_group = []
            
            # Join paragraphs with line breaks
            return '\n'.join(result)
    
    def _join_lines_naturally(self, lines: list) -> str:
        """Join multiple lines into a single cohesive sentence/paragraph."""
        if not lines:
            return ""
        if len(lines) == 1:
            return lines[0]
        
        result = []
        for i, line in enumerate(lines):
            # Add the line
            result.append(line)
            
            # Add connector between lines if needed
            if i < len(lines) - 1:
                # Check if current line ends with punctuation
                if line[-1] in '.!?':
                    # Start new sentence
                    result.append(' ')
                elif line[-1] in ',;':
                    # Already has separator
                    result.append(' ')
                else:
                    # No punctuation - add period if it makes sense, otherwise just space
                    next_line = lines[i + 1]
                    # If next line starts with capital or looks like new thought, add period
                    if next_line[0].isupper() and len(line) > 10:
                        result.append('. ')
                    else:
                        result.append(' ')
        
        return ''.join(result)
    
    def cleanup(self):
        """Clean up resources."""
        if self.current_model is not None:
            del self.current_model
        if self.base_model is not None:
            del self.base_model
        torch.cuda.empty_cache()
        print("✓ Resources cleaned up")


# Singleton instance
persona_manager = PersonaManager()


# Test
if __name__ == "__main__":
    print("Available personas:", persona_manager.get_available_personas())
    
    # Test loading
    personas = persona_manager.get_available_personas()
    if personas:
        persona_manager.load_persona(personas[0])
        response = persona_manager.generate_response("Hey, what's up?")
        print(f"\nResponse: {response}")
