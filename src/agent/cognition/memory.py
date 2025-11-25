# src/agent/cognition/memory.py

"""
Memory module for the GenAI DataOps Agent.

Manages short-term (conversation) and long-term (context hints) memory
for the cognition engine.

Features:
- Short-term memory: Per-user conversation history
- Long-term memory: Context hints, patterns, preferences
- Memory retrieval for context-aware routing
- Memory cleanup and expiration

Follows AGENTS.md rules: Memory is used only inside cognition.
"""

from typing import Dict, Any, List, Optional
from time import time
from collections import defaultdict, deque
from datetime import datetime, timedelta

from agent.logging_config import logger


# Memory configuration constants
MAX_CONVERSATION_HISTORY = 10  # Max messages per user in short-term memory
SHORT_TERM_TTL_SECONDS = 3600  # 1 hour TTL for conversation history
LONG_TERM_MAX_ENTRIES = 100  # Max entries in long-term memory


class Memory:
    """
    Production-grade memory manager for conversation and context.
    
    Maintains:
    - Short-term: Recent conversation history per user
    - Long-term: Context hints, patterns, user preferences
    """

    def __init__(
        self,
        max_conversation_history: int = MAX_CONVERSATION_HISTORY,
        short_term_ttl: int = SHORT_TERM_TTL_SECONDS,
        long_term_max_entries: int = LONG_TERM_MAX_ENTRIES
    ):
        """
        Initialize memory manager.
        
        Args:
            max_conversation_history: Max messages per user in short-term memory
            short_term_ttl: Time-to-live for conversation history in seconds
            long_term_max_entries: Max entries in long-term memory
        """
        self.max_conversation_history = max_conversation_history
        self.short_term_ttl = short_term_ttl
        self.long_term_max_entries = long_term_max_entries

        # Short-term memory: user_id -> deque of (timestamp, query, response)
        self._short_term: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_conversation_history))

        # Long-term memory: key -> (timestamp, value, metadata)
        self._long_term: Dict[str, Dict[str, Any]] = {}

        logger.info(
            "memory_initialized",
            max_conversation_history=max_conversation_history,
            short_term_ttl=short_term_ttl
        )

    # ------------------------------------------------------
    # Short-term Memory (Conversation History)
    # ------------------------------------------------------

    def add_conversation(
        self,
        user_id: str,
        query: str,
        response: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add conversation turn to short-term memory.
        
        Args:
            user_id: User identifier
            query: User query text
            response: Router/tool response dict
            metadata: Optional metadata (e.g., tool used, execution time)
        """
        timestamp = time()
        entry = {
            "timestamp": timestamp,
            "query": query,
            "response": response,
            "metadata": metadata or {}
        }

        self._short_term[user_id].append(entry)

        logger.debug(
            "memory_conversation_added",
            user_id=user_id,
            query_preview=query[:50],
            history_length=len(self._short_term[user_id])
        )

    def get_conversation_history(
        self,
        user_id: str,
        max_messages: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history for a user.
        
        Args:
            user_id: User identifier
            max_messages: Optional limit on number of messages to return
            
        Returns:
            List of conversation entries (most recent first)
        """
        if user_id not in self._short_term:
            return []

        # Clean expired entries
        self._clean_expired_conversations(user_id)

        history = list(self._short_term[user_id])
        
        # Reverse to get most recent first
        history.reverse()

        if max_messages:
            history = history[:max_messages]

        return history

    def get_conversation_context(
        self,
        user_id: str,
        include_responses: bool = False
    ) -> str:
        """
        Get formatted conversation context as string for LLM prompts.
        
        Args:
            user_id: User identifier
            include_responses: Whether to include response content
            
        Returns:
            Formatted context string
        """
        history = self.get_conversation_history(user_id)
        
        if not history:
            return ""

        context_parts = []
        for entry in history[-5:]:  # Last 5 messages
            context_parts.append(f"User: {entry['query']}")
            if include_responses:
                response_msg = entry['response'].get('message', '')
                context_parts.append(f"Assistant: {response_msg[:200]}")

        return "\n".join(context_parts)

    def _clean_expired_conversations(self, user_id: str) -> None:
        """Remove expired conversation entries for a user."""
        if user_id not in self._short_term:
            return

        now = time()
        expired_count = 0

        # Remove expired entries (older than TTL)
        while self._short_term[user_id]:
            entry = self._short_term[user_id][0]
            if now - entry['timestamp'] > self.short_term_ttl:
                self._short_term[user_id].popleft()
                expired_count += 1
            else:
                break

        if expired_count > 0:
            logger.debug(
                "memory_conversations_cleaned",
                user_id=user_id,
                expired_count=expired_count
            )

    # ------------------------------------------------------
    # Long-term Memory (Context Hints)
    # ------------------------------------------------------

    def store_long_term(
        self,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Store entry in long-term memory.
        
        Args:
            key: Memory key (e.g., "user_preference:user123", "pattern:revenue_queries")
            value: Value to store
            metadata: Optional metadata dict
        """
        timestamp = time()

        # Enforce max entries (remove oldest if needed)
        if len(self._long_term) >= self.long_term_max_entries:
            self._evict_oldest_long_term()

        self._long_term[key] = {
            "timestamp": timestamp,
            "value": value,
            "metadata": metadata or {}
        }

        logger.debug("memory_long_term_stored", key=key, value_type=type(value).__name__)

    def retrieve_long_term(
        self,
        key: str,
        default: Any = None
    ) -> Optional[Any]:
        """
        Retrieve entry from long-term memory.
        
        Args:
            key: Memory key
            default: Default value if key not found
            
        Returns:
            Stored value or default
        """
        if key not in self._long_term:
            return default

        entry = self._long_term[key]
        return entry["value"]

    def search_long_term(
        self,
        pattern: str
    ) -> List[Dict[str, Any]]:
        """
        Search long-term memory by key pattern.
        
        Args:
            pattern: Pattern to match in keys (substring match)
            
        Returns:
            List of matching entries with keys and values
        """
        matches = []
        pattern_lower = pattern.lower()

        for key, entry in self._long_term.items():
            if pattern_lower in key.lower():
                matches.append({
                    "key": key,
                    "value": entry["value"],
                    "timestamp": entry["timestamp"],
                    "metadata": entry["metadata"]
                })

        return matches

    def _evict_oldest_long_term(self) -> None:
        """Remove oldest entry from long-term memory."""
        if not self._long_term:
            return

        oldest_key = min(self._long_term.keys(), key=lambda k: self._long_term[k]["timestamp"])
        del self._long_term[oldest_key]

        logger.debug("memory_long_term_evicted", key=oldest_key)

    # ------------------------------------------------------
    # Memory Management
    # ------------------------------------------------------

    def clear_user_memory(self, user_id: str) -> None:
        """
        Clear all memory for a specific user.
        
        Args:
            user_id: User identifier
        """
        if user_id in self._short_term:
            del self._short_term[user_id]

        # Clear long-term entries for this user
        keys_to_remove = [k for k in self._long_term.keys() if user_id in k]
        for key in keys_to_remove:
            del self._long_term[key]

        logger.info("memory_user_cleared", user_id=user_id, long_term_keys_removed=len(keys_to_remove))

    def clear_all_memory(self) -> None:
        """Clear all memory (short-term and long-term)."""
        short_term_users = len(self._short_term)
        long_term_entries = len(self._long_term)

        self._short_term.clear()
        self._long_term.clear()

        logger.info(
            "memory_all_cleared",
            short_term_users=short_term_users,
            long_term_entries=long_term_entries
        )

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics.
        
        Returns:
            Dict with memory statistics
        """
        # Clean expired conversations for accurate stats
        for user_id in list(self._short_term.keys()):
            self._clean_expired_conversations(user_id)

        return {
            "short_term_users": len(self._short_term),
            "short_term_total_messages": sum(len(history) for history in self._short_term.values()),
            "long_term_entries": len(self._long_term),
            "max_conversation_history": self.max_conversation_history,
            "short_term_ttl_seconds": self.short_term_ttl,
            "long_term_max_entries": self.long_term_max_entries
        }