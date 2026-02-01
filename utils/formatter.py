"""Convert conversations to Gemma 3 chat template format."""

import json
from typing import List, Dict
from .whatsapp_parser import Message


class Gemma3Formatter:
    """Formats conversations into Gemma 3 chat template."""
    
    USER_START = "<start_of_turn>user\n"
    USER_END = "<end_of_turn>\n"
    MODEL_START = "<start_of_turn>model\n"
    MODEL_END = "<end_of_turn>\n"
    
    def __init__(self, context_window: int = 10):
        """
        Initialize the formatter.
        
        Args:
            context_window: Number of previous messages to include as context
        """
        self.context_window = context_window
    
    def create_training_examples(
        self, 
        all_messages: List[Message], 
        target_person: str,
        aggregator = None
    ) -> List[Dict]:
        """
        Create training examples for a specific person.
        
        Each example consists of:
        - Context: Previous N messages
        - Target: The person's response
        
        Args:
            all_messages: All messages sorted by timestamp
            target_person: The person we're training a model for (canonical name)
            aggregator: PersonAggregator instance for name normalization
        
        Returns:
            List of training examples in Gemma 3 format
        """
        examples = []
        
        for i, msg in enumerate(all_messages):
            sender = aggregator.normalize_name(msg.sender) if aggregator else msg.sender
            
            if sender != target_person:
                continue
            
            if len(msg.content.strip()) < 3:
                continue
            
            context_start = max(0, i - self.context_window)
            context_messages = all_messages[context_start:i]
            
            if not context_messages:
                continue
            
            # Build the formatted example
            formatted_example = self._format_example(context_messages, msg, aggregator)
            examples.append(formatted_example)
        
        return examples
    
    def _format_example(
        self, 
        context_messages: List[Message], 
        target_message: Message,
        aggregator = None
    ) -> Dict:
        """
        Format a single training example.
        
        Format:
        <start_of_turn>user
        [Previous conversation context]
        <end_of_turn>
        <start_of_turn>model
        [Target person's response]
        <end_of_turn>
        """
        # Build context string with normalized names
        context_lines = []
        for msg in context_messages:
            sender = aggregator.normalize_name(msg.sender) if aggregator else msg.sender
            context_lines.append(f"{sender}: {msg.content}")
        
        context = "\n".join(context_lines)
        
        # Build the formatted text
        formatted_text = (
            f"{self.USER_START}"
            f"{context}\n"
            f"{self.USER_END}"
            f"{self.MODEL_START}"
            f"{target_message.content}\n"
            f"{self.MODEL_END}"
        )
        
        # Return as dictionary (ready for .jsonl export)
        return {
            "text": formatted_text,
            "metadata": {
                "sender": target_message.sender,
                "timestamp": target_message.timestamp.isoformat(),
                "chat_source": target_message.chat_source,
                "context_length": len(context_messages)
            }
        }
    
    def save_to_jsonl(self, examples: List[Dict], output_path: str):
        """
        Save training examples to .jsonl file.
        
        Args:
            examples: List of training examples
            output_path: Path to output .jsonl file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    def get_statistics(self, examples: List[Dict]) -> Dict:
        """Get statistics about the training examples."""
        if not examples:
            return {
                'total_examples': 0,
                'avg_context_length': 0,
                'total_tokens_estimate': 0,
            }
        
        context_lengths = [ex['metadata']['context_length'] for ex in examples]
        total_chars = sum(len(ex['text']) for ex in examples)
        
        return {
            'total_examples': len(examples),
            'avg_context_length': sum(context_lengths) / len(context_lengths),
            'min_context_length': min(context_lengths),
            'max_context_length': max(context_lengths),
            'total_chars': total_chars,
            'total_tokens_estimate': total_chars // 4,  # Rough estimate: ~4 chars per token
        }
