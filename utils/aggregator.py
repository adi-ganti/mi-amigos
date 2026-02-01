"""Message aggregation and person extraction logic."""

from typing import List, Dict, Set
from collections import defaultdict
from .whatsapp_parser import Message


class PersonAggregator:
    """Aggregates messages per person across all chats."""
    
    def __init__(self, name_mapping: Dict[str, List[str]] = None):
        """
        Initialize the aggregator.
        
        Args:
            name_mapping: Dict mapping canonical names to list of variations
                         e.g., {"Charan": ["Charan Bp Gitam"], ...}
        """
        self.person_messages: Dict[str, List[Message]] = defaultdict(list)
        self.name_variations: Dict[str, str] = {}  # Maps variations to canonical names
        
        # Build reverse mapping: variation -> canonical
        if name_mapping:
            for canonical, variations in name_mapping.items():
                for variation in variations:
                    self.name_variations[variation] = canonical
                    self.name_variations[variation.lower()] = canonical
        
    def normalize_name(self, name: str) -> str:
        """
        Normalize a person's name using the explicit mapping.
        
        Examples:
        - "Charan Bp Gitam" -> "Charan"
        - "charan bp gitam" -> "Charan"
        - "Aditya Ganti" -> "Aditya Ganti" (unmapped, stays as-is)
        """
        name = name.strip()
        
        if name in self.name_variations:
            return self.name_variations[name]
        
        if name.lower() in self.name_variations:
            canonical = self.name_variations[name.lower()]
            self.name_variations[name] = canonical
            return canonical
        
        return name
    
    def add_messages(self, messages: List[Message]):
        """Add messages to the aggregator."""
        unmapped_names = set()
        
        for msg in messages:
            canonical_name = self.normalize_name(msg.sender)
            self.person_messages[canonical_name].append(msg)
            
            if canonical_name == msg.sender and msg.sender not in self.name_variations:
                unmapped_names.add(msg.sender)
        
        if unmapped_names:
            print(f"\n  ⚠ Unmapped names found: {sorted(unmapped_names)[:10]}")
    
    def get_all_persons(self) -> List[str]:
        """Get list of all unique persons."""
        return sorted(self.person_messages.keys())
    
    def get_messages_for_person(self, person: str) -> List[Message]:
        """Get all messages for a specific person, sorted by timestamp."""
        messages = self.person_messages.get(person, [])
        return sorted(messages, key=lambda m: m.timestamp)
    
    def get_message_count(self, person: str) -> int:
        """Get message count for a person."""
        return len(self.person_messages.get(person, []))
    
    def detect_your_name(self, known_friends: Set[str]) -> str:
        """
        Detect the user's name by finding the person who appears in all chats
        but is not in the known_friends list.
        
        Args:
            known_friends: Set of folder names (e.g., {"Charan", "Farhaan", ...})
        
        Returns:
            The detected user name
        """
        all_persons = self.get_all_persons()
        
        multi_chat_persons = []
        for person in all_persons:
            messages = self.get_messages_for_person(person)
            chat_sources = set(msg.chat_source for msg in messages)
            
            if len(chat_sources) >= 3:
                multi_chat_persons.append((person, len(chat_sources), len(messages)))
        
        multi_chat_persons.sort(key=lambda x: (x[1], x[2]), reverse=True)
        
        for person, chat_count, msg_count in multi_chat_persons:
            is_friend = False
            for friend in known_friends:
                if friend.lower() in person.lower() or person.lower() in friend.lower():
                    is_friend = True
                    break
            
            if not is_friend:
                return person
        
        if multi_chat_persons:
            return multi_chat_persons[0][0]
        
        return "Unknown"
    
    def get_statistics(self) -> Dict[str, Dict]:
        """Get statistics for all persons."""
        stats = {}
        for person in self.get_all_persons():
            messages = self.get_messages_for_person(person)
            chat_sources = set(msg.chat_source for msg in messages)
            
            stats[person] = {
                'message_count': len(messages),
                'chat_sources': list(chat_sources),
                'chat_count': len(chat_sources),
                'first_message': messages[0].timestamp if messages else None,
                'last_message': messages[-1].timestamp if messages else None,
            }
        
        return stats
