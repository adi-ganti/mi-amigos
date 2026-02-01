"""Google Chat/Hangouts JSON parser."""

import json
from datetime import datetime
from typing import List
from .whatsapp_parser import Message


class GoogleChatParser:
    """Parser for Google Chat/Hangouts JSON exports."""
    
    def __init__(self):
        """Initialize the parser."""
        pass
    
    def parse_file(self, filepath: str, chat_name: str) -> List[Message]:
        """
        Parse a Google Chat JSON file.
        
        Args:
            filepath: Path to the messages.json file
            chat_name: Name of the chat (e.g., "Stuti")
        
        Returns:
            List of Message objects
        """
        messages = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for msg_data in data.get('messages', []):
            creator = msg_data.get('creator', {})
            sender = creator.get('name', 'Unknown')
            
            content = msg_data.get('text', '').strip()
            
            if not content:
                continue
            
            date_str = msg_data.get('created_date', '')
            try:
                date_str = date_str.split(', ', 1)[1] if ', ' in date_str else date_str
                date_str = date_str.replace(' UTC', '').strip()
                
                timestamp = datetime.strptime(date_str, "%d %B %Y at %H:%M:%S")
            except (ValueError, IndexError):
                timestamp = datetime(1970, 1, 1)
            
            messages.append(Message(
                timestamp=timestamp,
                sender=sender,
                content=content,
                chat_source=chat_name
            ))
        
        return messages
    
    def merge_consecutive_messages(self, messages: List[Message]) -> List[Message]:
        """
        Merge consecutive messages from the same sender.
        
        Same logic as WhatsApp parser.
        """
        if not messages:
            return []
        
        merged = []
        current = messages[0]
        
        for next_msg in messages[1:]:
            # Same sender and within 2 minutes
            time_diff = (next_msg.timestamp - current.timestamp).total_seconds()
            
            if next_msg.sender == current.sender and time_diff < 120:
                # Merge messages
                current.content += "\n" + next_msg.content
            else:
                # Save current and start new
                merged.append(current)
                current = next_msg
        
        # Add the last message
        merged.append(current)
        
        return merged
