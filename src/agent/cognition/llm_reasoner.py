# src/agent/cognition/llm_reasoner.py

"""
LLM Reasoner for the GenAI DataOps Agent.

Interprets user intent, decomposes complex tasks, and provides reasoning
for the router to make intelligent tool selection decisions.

Features:
- Intent interpretation using LLM
- Task decomposition for complex queries
- Reasoning for tool selection
- Context-aware query understanding

Follows AGENTS.md rules: Used only inside cognition module.
"""

from typing import Dict, Any, List, Optional
import json
import time

from agent.config import Settings
from agent.logging_config import logger

# Optional OpenAI import (graceful degradation)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None


class LLMReasoner:
    """
    Production-grade LLM reasoner for intent interpretation and task decomposition.
    
    Uses OpenAI API to understand complex queries and provide reasoning
    for tool selection and task breakdown.
    """

    def __init__(self, settings: Settings, model: str = "gpt-3.5-turbo"):
        """
        Initialize LLM reasoner.
        
        Args:
            settings: Settings object with OpenAI API key
            model: OpenAI model to use (default: gpt-3.5-turbo)
        """
        if not OPENAI_AVAILABLE:
            logger.warning("llm_reasoner_openai_not_available")
            self.enabled = False
            return

        if not settings.openai_api_key:
            logger.warning("llm_reasoner_no_api_key")
            self.enabled = False
            return

        self.settings = settings
        self.model = model
        self.enabled = True
        
        # Initialize OpenAI client
        try:
            openai.api_key = settings.openai_api_key
            logger.info("llm_reasoner_initialized", model=model)
        except Exception as e:
            logger.error("llm_reasoner_init_failed", error=str(e))
            self.enabled = False

    def interpret_intent(
        self,
        query: str,
        available_tools: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Interpret user intent and suggest tool selection.
        
        Args:
            query: User query text
            available_tools: List of available tools (e.g., ["sql", "pandasai"])
            context: Optional context dict (e.g., previous queries, user history)
            
        Returns:
            Dict with keys: intent, suggested_tool, confidence, reasoning, decomposed_tasks
        """
        if not self.enabled:
            return {
                "intent": "unknown",
                "suggested_tool": None,
                "confidence": 0.0,
                "reasoning": "LLM reasoner not available",
                "decomposed_tasks": []
            }

        start_time = time.time()

        try:
            tools = available_tools or ["sql", "pandasai", "dataset_info"]
            
            # Build prompt for intent interpretation
            prompt = self._build_intent_prompt(query, tools, context)
            
            # Call OpenAI API
            response = self._call_llm(prompt)
            
            # Parse response
            result = self._parse_intent_response(response)
            
            execution_time = (time.time() - start_time) * 1000
            
            logger.info(
                "llm_reasoner_intent_interpreted",
                intent=result.get("intent"),
                suggested_tool=result.get("suggested_tool"),
                confidence=result.get("confidence"),
                execution_time_ms=round(execution_time, 2)
            )

            return result

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.exception(
                "llm_reasoner_intent_failed",
                error=str(e),
                execution_time_ms=round(execution_time, 2)
            )
            return {
                "intent": "unknown",
                "suggested_tool": None,
                "confidence": 0.0,
                "reasoning": f"Error: {str(e)}",
                "decomposed_tasks": []
            }

    def decompose_task(
        self,
        query: str,
        target_tool: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Decompose a complex task into smaller sub-tasks.
        
        Args:
            query: Complex user query
            target_tool: Tool to use for execution
            context: Optional context dict
            
        Returns:
            List of sub-task dicts with keys: step, description, tool, parameters
        """
        if not self.enabled:
            return [{"step": 1, "description": query, "tool": target_tool, "parameters": {}}]

        start_time = time.time()

        try:
            # Build prompt for task decomposition
            prompt = self._build_decomposition_prompt(query, target_tool, context)
            
            # Call OpenAI API
            response = self._call_llm(prompt)
            
            # Parse response
            tasks = self._parse_decomposition_response(response)
            
            execution_time = (time.time() - start_time) * 1000
            
            logger.info(
                "llm_reasoner_task_decomposed",
                original_query=query[:100],
                task_count=len(tasks),
                execution_time_ms=round(execution_time, 2)
            )

            return tasks

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.exception(
                "llm_reasoner_decomposition_failed",
                error=str(e),
                execution_time_ms=round(execution_time, 2)
            )
            # Fallback: return single task
            return [{"step": 1, "description": query, "tool": target_tool, "parameters": {}}]

    def _build_intent_prompt(
        self,
        query: str,
        tools: List[str],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Build prompt for intent interpretation."""
        context_str = ""
        if context:
            context_str = f"\nContext: {json.dumps(context, indent=2)}"

        prompt = f"""Analyze this data analytics query and determine the best tool to use.

Available tools: {', '.join(tools)}
- sql: For structured queries, metrics, aggregations
- pandasai: For data analysis, charts, visualizations, exploratory analysis
- dataset_info: For dataset metadata and schema information

Query: {query}{context_str}

Respond in JSON format:
{{
    "intent": "brief description of user intent",
    "suggested_tool": "sql|pandasai|dataset_info",
    "confidence": 0.0-1.0,
    "reasoning": "explanation of why this tool is best",
    "decomposed_tasks": ["task1", "task2"] if complex, else []
}}"""

        return prompt

    def _build_decomposition_prompt(
        self,
        query: str,
        target_tool: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Build prompt for task decomposition."""
        context_str = ""
        if context:
            context_str = f"\nContext: {json.dumps(context, indent=2)}"

        prompt = f"""Break down this complex query into smaller, executable sub-tasks.

Query: {query}
Target Tool: {target_tool}{context_str}

Respond in JSON format with array of tasks:
[
    {{
        "step": 1,
        "description": "what to do in this step",
        "tool": "{target_tool}",
        "parameters": {{"key": "value"}}
    }}
]"""

        return prompt

    def _call_llm(self, prompt: str) -> str:
        """
        Call OpenAI API with prompt.
        
        Args:
            prompt: Prompt text
            
        Returns:
            LLM response text
        """
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a data analytics assistant. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error("llm_reasoner_api_call_failed", error=str(e))
            raise

    def _parse_intent_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response for intent interpretation."""
        try:
            # Try to extract JSON from response
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()

            result = json.loads(response)
            
            # Validate structure
            return {
                "intent": result.get("intent", "unknown"),
                "suggested_tool": result.get("suggested_tool"),
                "confidence": float(result.get("confidence", 0.0)),
                "reasoning": result.get("reasoning", ""),
                "decomposed_tasks": result.get("decomposed_tasks", [])
            }

        except Exception as e:
            logger.warning("llm_reasoner_parse_failed", error=str(e), response_preview=response[:100])
            return {
                "intent": "unknown",
                "suggested_tool": None,
                "confidence": 0.0,
                "reasoning": f"Parse error: {str(e)}",
                "decomposed_tasks": []
            }

    def _parse_decomposition_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM response for task decomposition."""
        try:
            # Try to extract JSON from response
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()

            tasks = json.loads(response)
            
            # Ensure it's a list
            if not isinstance(tasks, list):
                tasks = [tasks]

            return tasks

        except Exception as e:
            logger.warning("llm_reasoner_decomposition_parse_failed", error=str(e), response_preview=response[:100])
            return []