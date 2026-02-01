"""Main preprocessing script to orchestrate the entire pipeline."""

import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

from utils.whatsapp_parser import WhatsAppParser, Message
from utils.google_chat_parser import GoogleChatParser
from utils.aggregator import PersonAggregator
from utils.formatter import Gemma3Formatter
import config


class PreprocessingPipeline:
    """Main preprocessing pipeline for Digital Twin project."""
    
    def __init__(
        self, 
        chat_data_dir: str = "Chat_Data",
        output_dir: str = "Training_Data",
        context_window: int = 10
    ):
        """
        Initialize the pipeline.
        
        Args:
            chat_data_dir: Directory containing chat folders
            output_dir: Directory to save processed training data
            context_window: Number of previous messages for context
        """
        self.chat_data_dir = Path(chat_data_dir)
        self.output_dir = Path(output_dir)
        self.context_window = context_window
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize parsers
        self.whatsapp_parser = WhatsAppParser()
        self.google_chat_parser = GoogleChatParser()
        self.aggregator = PersonAggregator(name_mapping=config.NAME_MAPPING)
        self.formatter = Gemma3Formatter(context_window=context_window)
        
        # Chat organization
        self.individual_chats = [
            "Charan", "Farhaan", "Jaishish", "Rahul", 
            "Satya", "Stuti", "Sushanth", "Teja", "Urjith"
        ]
        self.group_chats = ["Darlings", "Gaylords"]
        
    def run(self):
        """Execute the full preprocessing pipeline."""
        print("=" * 60)
        print("DIGITAL TWIN PREPROCESSING PIPELINE")
        print("=" * 60)
        
        print("\n[Step 1/5] Parsing chat files...")
        all_messages = self._parse_all_chats()
        print(f"✓ Parsed {len(all_messages)} total messages from all chats")
        
        print("\n[Step 2/5] Aggregating messages by person...")
        self.aggregator.add_messages(all_messages)
        all_persons = self.aggregator.get_all_persons()
        print(f"✓ Found {len(all_persons)} unique persons")
        
        print("\n[Step 3/5] Detecting your name...")
        your_name = config.YOUR_NAME if config.YOUR_NAME else self.aggregator.detect_your_name(set(self.individual_chats))
        print(f"✓ Your name: '{your_name}'")
        
        print("\n[Step 4/5] Message statistics:")
        stats = self.aggregator.get_statistics()
        self._display_statistics(stats, your_name)
        
        print("\n[Step 5/5] Creating training datasets..."))
        self._create_training_datasets(all_messages, all_persons)
        
        print("\n" + "=" * 60)
        print("✓ PREPROCESSING COMPLETE!")
        print(f"✓ Training data saved to: {self.output_dir.absolute()}")
        print("=" * 60)
    
    def _parse_all_chats(self) -> List[Message]:
        """Parse all chat files and return combined messages."""
        all_messages = []
        all_chats = self.individual_chats + self.group_chats
        
        for chat_name in all_chats:
            chat_dir = self.chat_data_dir / chat_name
            
            whatsapp_file = chat_dir / "_chat.txt"
            if whatsapp_file.exists():
                messages = self.whatsapp_parser.parse_file(str(whatsapp_file), chat_name)
                messages = self.whatsapp_parser.merge_consecutive_messages(messages)
                all_messages.extend(messages)
                print(f"  ✓ {chat_name}: {len(messages)} messages (WhatsApp)")
            
            if chat_name == "Stuti":
                google_chat_file = chat_dir / "messages.json"
                if google_chat_file.exists():
                    messages = self.google_chat_parser.parse_file(str(google_chat_file), chat_name)
                    messages = self.google_chat_parser.merge_consecutive_messages(messages)
                    all_messages.extend(messages)
                    print(f"  ✓ {chat_name}: {len(messages)} messages (Google Chat)")
        
        return all_messages
    
    def _display_statistics(self, stats: Dict, your_name: str):
        """Display message statistics."""
        sorted_persons = sorted(
            stats.items(), 
            key=lambda x: x[1]['message_count'], 
            reverse=True
        )
        
        print(f"\n{'Person':<20} {'Messages':<10} {'Chats':<8} {'Is You?'}")
        print("-" * 55)
        
        for person, person_stats in sorted_persons[:15]:
            is_you = " ← YOU" if person == your_name else ""
            print(
                f"{person:<20} "
                f"{person_stats['message_count']:<10} "
                f"{person_stats['chat_count']:<8} "
                f"{is_you}"
            )
    
    def _create_training_datasets(self, all_messages: List[Message], all_persons: List[str]):
        """Create training datasets for each person."""
        all_messages_sorted = sorted(all_messages, key=lambda m: m.timestamp)
        
        messages_by_chat = defaultdict(list)
        for msg in all_messages_sorted:
            messages_by_chat[msg.chat_source].append(msg)
        
        training_stats = []
        
        for person in all_persons:
            person_messages = self.aggregator.get_messages_for_person(person)
            
            if len(person_messages) < 20:
                continue
            
            all_examples = []
            
            for chat_source, chat_messages in messages_by_chat.items():
                person_in_chat = any(
                    self.aggregator.normalize_name(msg.sender) == person 
                    for msg in chat_messages
                )
                if not person_in_chat:
                    continue
                
                examples = self.formatter.create_training_examples(
                    chat_messages, person, self.aggregator
                )
                all_examples.extend(examples)
            
            if all_examples:
                safe_name = person.replace(" ", "_").replace("/", "_")
                output_file = self.output_dir / f"{safe_name}.jsonl"
                
                self.formatter.save_to_jsonl(all_examples, str(output_file))
                
                stats = self.formatter.get_statistics(all_examples)
                training_stats.append((person, stats))
                
                print(f"  ✓ {person}: {stats['total_examples']} examples → {output_file.name}")
        
        print(f"\n✓ Created {len(training_stats)} training datasets")
        
        self._save_summary(training_stats)
    
    def _save_summary(self, training_stats: List[tuple]):
        """Save a summary of all training datasets."""
        summary_file = self.output_dir / "preprocessing_summary.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("DIGITAL TWIN PREPROCESSING SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            total_examples = sum(stats['total_examples'] for _, stats in training_stats)
            total_tokens = sum(stats['total_tokens_estimate'] for _, stats in training_stats)
            
            f.write(f"Total Persons: {len(training_stats)}\n")
            f.write(f"Total Training Examples: {total_examples:,}\n")
            f.write(f"Estimated Total Tokens: {total_tokens:,}\n\n")
            
            f.write("Per-Person Breakdown:\n")
            f.write("-" * 60 + "\n")
            
            for person, stats in sorted(training_stats, key=lambda x: x[1]['total_examples'], reverse=True):
                f.write(f"\n{person}:\n")
                f.write(f"  Examples: {stats['total_examples']:,}\n")
                f.write(f"  Avg Context: {stats['avg_context_length']:.1f} messages\n")
                f.write(f"  Estimated Tokens: {stats['total_tokens_estimate']:,}\n")
        
        print(f"\n✓ Summary saved to: {summary_file.name}")


if __name__ == "__main__":
    pipeline = PreprocessingPipeline()
    pipeline.run()
