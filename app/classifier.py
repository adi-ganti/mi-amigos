"""
Query Classifier - Determines if query needs RAG or direct generation
"""
import re
from typing import Tuple

MEMORY_KEYWORDS = [
    "remember", "recall", "that time", "when we", "when did", 
    "what did", "who said", "last time", "the day", "that day",
    "what happened", "tell me about", "the trip", "the party",
    "the movie", "the game", "the match", "devara", "prom",
]

TIME_REFERENCES = [
    "yesterday", "last week", "last month", "last year",
    "in january", "in february", "in march", "in april",
    "in may", "in june", "in july", "in august", "in september",
    "in october", "in november", "in december",
    "on monday", "on tuesday", "on wednesday", "on thursday",
    "on friday", "on saturday", "on sunday",
    "2024", "2023", "2022", "2021", "2020",
]

FRIEND_NAMES = [
    "charan", "stuti", "farhaan", "urjith", "satya", 
    "sushanth", "jaishish", "rahul", "teja", "subodh", "aditya"
]

GENERIC_PATTERNS = [
    r"^(hey|hi|hello|sup|wassup|what's up|yo)\b",
    r"^how are you",
    r"^what do you think",
    r"^do you like",
    r"^are you",
    r"^tell me a joke",
    r"^lol|lmao|haha",
    r"^good morning|good night|good evening",
    r"^thanks|thank you",
    r"^ok|okay|alright|sure|fine",
]


def classify_query(query: str) -> Tuple[str, float]:
    """
    Classify a query as 'specific' (needs RAG) or 'generic' (direct generation).
    
    Returns:
        Tuple of (classification, confidence)
        - classification: 'specific' or 'generic'
        - confidence: 0.0 to 1.0
    """
    query_lower = query.lower().strip()
    
    # Check for generic patterns first (high priority)
    for pattern in GENERIC_PATTERNS:
        if re.search(pattern, query_lower):
            return ("generic", 0.95)
    
    memory_score = 0
    for keyword in MEMORY_KEYWORDS:
        if keyword in query_lower:
            memory_score += 0.3
    
    for time_ref in TIME_REFERENCES:
        if time_ref in query_lower:
            memory_score += 0.25
    
    for name in FRIEND_NAMES:
        if name in query_lower:
            memory_score += 0.15
    
    if re.search(r"(what|when|where|why|how|who).*(did|was|were|said)", query_lower):
        memory_score += 0.3
    
    confidence = min(memory_score, 0.95)
    
    if memory_score >= 0.3:
        return ("specific", confidence)
    else:
        return ("generic", 0.8)


def needs_rag(query: str) -> bool:
    """Simple boolean check if query needs RAG."""
    classification, _ = classify_query(query)
    return classification == "specific"


# Test examples
if __name__ == "__main__":
    test_queries = [
        "Hey, what's up?",
        "How are you doing?",
        "Remember when we went to Devara?",
        "What did Satya say about the internship?",
        "Tell me about the party last week",
        "Lol that's funny",
        "Who said they couldn't come to prom?",
        "What do you think about movies?",
        "When did Charan join the group?",
    ]
    
    print("Query Classification Tests")
    print("=" * 60)
    for query in test_queries:
        classification, confidence = classify_query(query)
        needs = "🔍 RAG" if classification == "specific" else "💬 Direct"
        print(f"{needs} ({confidence:.0%}): {query}")
