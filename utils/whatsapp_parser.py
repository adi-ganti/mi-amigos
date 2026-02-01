"""WhatsApp chat parser for Indian date format."""

import re
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class Message:
    """Represents a single chat message."""
    timestamp: datetime
    sender: str
    content: str
    chat_source: str  # Which chat file this came from
    
    def __repr__(self):
        return f"Message(sender='{self.sender}', content='{self.content[:30]}...')"


class WhatsAppParser:
    """Parser for WhatsApp chat exports (Indian format: DD/MM/YY, HH:MM:SS)."""
    
    MESSAGE_PATTERN = re.compile(
        r'\[(\d{2}/\d{2}/\d{2}),\s*(\d{2}:\d{2}:\d{2})\]\s*([^:]+):\s*(.+)',
        re.DOTALL
    )
    
    # System message patterns
    SYSTEM_PATTERNS = [
        r'‎Messages and calls are end-to-end encrypted',
        r'‎<Media omitted>',
        r'‎<media omitted>',
        r'‎image omitted',
        r'‎video omitted',
        r'‎audio omitted',
        r'‎document omitted',
        r'‎sticker omitted',
        r'‎GIF omitted',
        r'‎Contact card omitted',
        r'deleted this message',
        r'‎Missed voice call',
        r'‎Missed video call',
        r'changed the subject to',
        r'added .+',
        r'removed .+',
        r'You created group',
        r'created group',
        r'\bleft\b',
        r'changed this group\'s icon',
        r'^https?://',
    ]
    
    def __init__(self, system_patterns: Optional[List[str]] = None):
        """
        Initialize the parser.
        
        Args:
            system_patterns: Additional regex patterns to filter as system messages
        """
        if system_patterns:
            self.SYSTEM_PATTERNS.extend(system_patterns)
        self.system_regex = re.compile('|'.join(self.SYSTEM_PATTERNS), re.IGNORECASE)
    
    def is_system_message(self, content: str) -> bool:
        """Check if a message is a system/media message."""
        content = content.strip()
        
        if not content or len(content) < 2:
            return True
        
        if self.system_regex.search(content):
            return True
        
        if re.match(r'^[A-Z]{3}\d{8}_\d+\.(jpg|png|pdf|mp4|jpeg)$', content, re.IGNORECASE):
            return True
            
        return False
    
    def parse_file(self, filepath: str, chat_name: str) -> List[Message]:
        """
        Parse a WhatsApp chat file.
        
        Args:
            filepath: Path to the _chat.txt file
            chat_name: Name of the chat (e.g., "Charan", "Darlings")
        
        Returns:
            List of Message objects
        """
        messages = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        current_message = None
        
        for line in lines:
            match = self.MESSAGE_PATTERN.match(line)
            
            if match:
                if current_message:
                    messages.append(current_message)
                
                date_str, time_str, sender, msg_content = match.groups()
                
                # Parse timestamp (DD/MM/YY, HH:MM:SS)
                try:
                    timestamp = datetime.strptime(
                        f"{date_str} {time_str}", 
                        "%d/%m/%y %H:%M:%S"
                    )
                except ValueError:
                    continue
                
                sender = sender.strip()
                msg_content = msg_content.strip()
                
                current_message = Message(
                    timestamp=timestamp,
                    sender=sender,
                    content=msg_content,
                    chat_source=chat_name
                )
            
            elif current_message and line.strip():
                current_message.content += "\n" + line.strip()
        
        if current_message:
            messages.append(current_message)
        
        messages = [
            msg for msg in messages 
            if not self.is_system_message(msg.content)
        ]
        
        return messages
    
    def merge_consecutive_messages(self, messages: List[Message]) -> List[Message]:
        """
        Merge consecutive messages from the same sender.
        
        This simulates how people send multiple short messages in bursts.
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
