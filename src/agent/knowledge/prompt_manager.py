# src/agent/knowledge/prompt_manager.py

"""
Prompt Manager Module - Database Agnostic

All prompts focus on ANALYSIS PROCESSES, not specific column names.
Works with ANY database schema by teaching the LLM HOW to think, not WHAT to find.
"""

from typing import Dict, List, Optional, Any
import json


class PromptManager:
    """
    Centralized prompt management - completely database agnostic.
    
    Prompts teach semantic analysis PROCESSES, not specific outcomes.
    """
    
    @staticmethod
    def build_schema_mapping_prompt(schema: Dict[str, Any], concept: str) -> str:
        """
        Build prompt for mapping concepts to database schema.
        
        Focuses on HOW to analyze, not WHAT to find.
        """
        tables_info = []
        for table_name, columns in schema.items():
            columns_with_types = [
                {"name": col["name"], "type": col.get("type", "unknown")}
                for col in columns
            ]
            tables_info.append({
                "table": table_name,
                "columns": columns_with_types
            })
        
        prompt = f"""Given this database schema and a concept, select the best table and column to represent it.

Database Schema:
{json.dumps(tables_info, indent=2)}

Concept: {concept}

ANALYSIS PROCESS (follow these steps):

1. SEMANTIC COLUMN ANALYSIS:
   - Break down the concept into its semantic components
   - For each column in each table, analyze if the column name semantically relates to the concept
   - Consider: synonyms, related terms, domain-specific terminology, abbreviations
   - Look for partial matches, word stems, and contextual relationships
   - Example PROCESS: If concept is "revenue", think: "What does revenue mean? Money earned, income, sales value. 
     Now scan column names - do any suggest monetary values, transactions, or income? Look for semantic connections."
   - Example PROCESS: If concept is "users", think: "What does users mean? People, accounts, entities. 
     Now scan column names - do any suggest entities, identities, or countable items? Look for ID patterns or entity names."
   - The goal is SEMANTIC MATCHING, not exact word matching

2. COLUMN TYPE VALIDATION:
   - After identifying semantically relevant columns, validate their types
   - For numeric concepts: Verify column type supports numeric operations (integer, numeric, decimal, float, money)
   - For counting concepts: Any column type works, but prefer ID-like columns (integer types)
   - For date concepts: Verify column type is date/timestamp related
   - Type validation ensures the column can be used for the intended operation

3. TABLE CONTEXT EVALUATION:
   - Consider what the table name suggests about its purpose
   - Tables with semantically relevant names are more likely to contain relevant columns
   - Consider the relationship between table purpose and concept domain
   - Example PROCESS: If concept is "revenue" and you see a table called "transactions", 
     evaluate if transaction tables typically contain revenue-related columns

4. MULTIPLE MATCHING STRATEGIES (try in order):
   a) Semantic relevance: Does column name conceptually relate to the concept?
   b) Type compatibility: Is column type suitable for the operation?
   c) Context fit: Does table context support this concept?
   d) Best overall match: Combine all factors

5. AGGREGATION REASONING:
   - Determine what operation the concept requires
   - SUM: For concepts that represent totals, accumulations, or additive values
   - COUNT: For concepts that represent quantities or counts of items
   - COUNT DISTINCT: For concepts that represent unique entities or distinct items
   - AVG: For concepts that represent averages or means
   - MAX/MIN: For concepts that represent extremes
   - Base selection on concept semantics, not hardcoded rules

6. DATE COLUMN DISCOVERY:
   - Scan for columns with date/timestamp types
   - Evaluate column names for temporal significance
   - Prefer columns that represent event times, creation dates, or temporal markers
   - Consider common temporal naming patterns (but don't assume they exist)

SELECTION CRITERIA:
- Choose table where concept domain best matches table purpose
- Choose column that has strongest semantic relationship to concept
- Verify column type supports required operation
- If multiple candidates exist, choose most semantically relevant
- If no good match exists, return null

Respond in JSON format only:
{{
    "table": "table_name",
    "column": "column_name",
    "date_column": "date_column_name (if temporal filtering needed)",
    "aggregation": "SUM|COUNT|COUNT DISTINCT|AVG|MAX|MIN",
    "reasoning": "explanation of semantic analysis process and why this selection was made"
}}

If semantic analysis finds no suitable match, return null."""
        
        return prompt
    
    @staticmethod
    def build_expression_suggestion_prompt(
        table: str, 
        columns: List[str],
        column_types: Optional[Dict[str, str]] = None,
        concept: str = ""
    ) -> str:
        """
        Build prompt for suggesting SQL expressions.
        
        Focuses on HOW to analyze columns semantically.
        """
        columns_info = []
        for col in columns:
            col_info = {"name": col}
            if column_types and col in column_types:
                col_info["type"] = column_types[col]
            columns_info.append(col_info)
        
        prompt = f"""Given this table and available columns, suggest a SQL expression to calculate the concept.

Table: {table}
Available columns: {json.dumps(columns_info, indent=2)}
Concept: {concept}

ANALYSIS PROCESS (follow these steps):

1. CONCEPT SEMANTIC DECOMPOSITION:
   - Break down the concept into its core meaning
   - Identify what type of value/operation the concept represents
   - Think about what kind of data would represent this concept

2. COLUMN SEMANTIC SCANNING:
   - For EACH column in the available list:
     a) Analyze the column name for semantic relationship to the concept
     b) Consider synonyms, related terms, domain terminology
     c) Look for partial matches, word components, contextual clues
     d) Evaluate type compatibility with concept requirements
   - DO NOT assume specific column names exist
   - WORK WITH WHAT IS ACTUALLY AVAILABLE

3. MATCHING STRATEGY:
   - Semantic relevance: Does column name conceptually relate to concept?
   - Type suitability: Can column type support the required operation?
   - Context appropriateness: Does column fit the concept's domain?
   - Score each column on these dimensions
   - Select the column with highest overall semantic relevance

4. AGGREGATION REASONING:
   - Analyze what operation the concept requires
   - SUM: For additive/total concepts
   - COUNT: For quantity/counting concepts  
   - COUNT DISTINCT: For unique entity concepts
   - AVG: For average/mean concepts
   - Base on concept semantics, not examples

5. EXPRESSION CONSTRUCTION:
   - Build SQL expression using selected column
   - Use appropriate aggregation function
   - Alias result with concept name

IMPORTANT:
- You are working with REAL columns that exist in THIS specific database
- Do not reference columns that don't exist
- Analyze the ACTUAL column names provided
- Find the best semantic match from what's available
- If no suitable column exists after thorough analysis, return null

Respond in JSON format:
{{
    "expression": "SQL expression using actual column names from the list",
    "column": "selected_column_name (must be from available columns)",
    "aggregation": "SUM|COUNT|COUNT DISTINCT|AVG|MAX|MIN",
    "reasoning": "explanation of semantic analysis process and column selection"
}}

If semantic analysis of available columns finds no suitable match, return null."""
        
        return prompt
    
    @staticmethod
    def build_sql_generation_prompt(
        schema: Dict[str, Any], 
        concept: str, 
        date_filter: str
    ) -> str:
        """
        Build prompt for generating simple SQL queries.
        
        Focuses on dynamic schema analysis.
        """
        tables_info = []
        for table_name, columns in schema.items():
            columns_with_types = [
                {"name": col["name"], "type": col.get("type", "unknown")}
                for col in columns
            ]
            tables_info.append({
                "table": table_name,
                "columns": columns_with_types
            })
        
        prompt = f"""Generate a safe PostgreSQL SELECT query for this concept.

Database Schema:
{json.dumps(tables_info, indent=2)}

Concept: {concept}
Date Filter: {date_filter}

GENERATION PROCESS:

1. SCHEMA ANALYSIS:
   - Examine ALL tables and columns in the provided schema
   - Identify table that best matches concept domain (semantic analysis)
   - Identify column that best represents concept (semantic matching)
   - Use ACTUAL table and column names from schema

2. SEMANTIC MATCHING:
   - Analyze concept meaning
   - Scan available columns for semantic relationships
   - Consider synonyms, related terms, domain context
   - Select best match based on semantic relevance and type compatibility

3. AGGREGATION SELECTION:
   - Determine operation based on concept semantics
   - Choose aggregation that fits concept meaning
   - Verify column type supports operation

4. SQL CONSTRUCTION:
   - Use actual table/column names from schema
   - Include date filter if provided
   - Ensure syntactically correct PostgreSQL
   - ORDER BY: Add for sorting (if query mentions "top", "sorted", "order")
   - LIMIT: Add to restrict rows (if query mentions "top N", "first N", "show N")

Rules:
- Only use SELECT statements
- Use ACTUAL table and column names from schema (do not invent)
- Include date filter if provided
- Return only SQL query, no explanations
- Ensure query is valid PostgreSQL

SQL:"""
        
        return prompt
    
    @staticmethod
    def build_complex_sql_prompt(
        schema: Dict[str, Any], 
        query: str, 
        requirements: Dict[str, Any]
    ) -> str:
        """
        Build prompt for generating complex SQL queries.
        
        Focuses on relationship discovery, not assumptions.
        """
        tables_info = []
        for table_name, columns in schema.items():
            columns_with_types = [
                {"name": col["name"], "type": col.get("type", "unknown")}
                for col in columns
            ]
            tables_info.append({
                "table": table_name,
                "columns": columns_with_types
            })
        
        requirements_str = json.dumps(requirements, indent=2)
        
        # Build metric mapping instructions if provided
        metric_mapping_section = ""
        if requirements.get("metric_mapping"):
            metric_mapping = requirements["metric_mapping"]
            
            # Format aggregation for example (handle COUNT DISTINCT specially)
            agg_func = metric_mapping.get('aggregation', 'COUNT DISTINCT')
            col_name = metric_mapping.get('column', 'user_id')
            if agg_func.upper() == 'COUNT DISTINCT':
                example_expr = f"COUNT(DISTINCT {col_name})"
            else:
                example_expr = f"{agg_func}({col_name})"
            
            metric_mapping_section = f"""
CRITICAL METRIC MAPPING:
If 'metric_mapping' is provided in Requirements, you MUST use it:
- Use the specified 'table' ({metric_mapping.get('table', 'N/A')}) as the primary table for the metric calculation
- Use the specified 'column' ({col_name}) for the aggregation
- Use the specified 'aggregation' function ({agg_func}) (e.g., COUNT DISTINCT, SUM, AVG)
- Example: If metric_mapping shows table='{metric_mapping.get('table', 'sessions')}', column='{col_name}', aggregation='{agg_func}', then use {example_expr} FROM {metric_mapping.get('table', 'sessions')}
- The metric table is the source of truth for the metric calculation - JOIN to other tables only for grouping/filtering
- IMPORTANT: Use ONLY the metric mapping provided above. DO NOT add additional metrics or columns unless explicitly requested in the query
- ANALYTICAL PROCESS: For each metric/column, ask: "Was this explicitly requested in the user query?" If no, exclude it
- ANALYTICAL PROCESS: For each JOIN, ask: "Is this table needed for the requested metric or for grouping/filtering?" If no, exclude it
- DO NOT add "active_users" or other metrics unless the query explicitly requests them
- DO NOT JOIN to tables that are not needed for the requested metric
"""
        
        prompt = f"""Generate a safe PostgreSQL SELECT query for this complex query.

Database Schema:
{json.dumps(tables_info, indent=2)}

Query: {query}

Requirements:
{requirements_str}
{metric_mapping_section}

COMPLEX QUERY GENERATION PROCESS:

1. SCHEMA RELATIONSHIP DISCOVERY:
   - Analyze ALL tables and columns in schema
   - Identify relationships by finding semantically related column names across tables
   - Look for common naming patterns that suggest relationships
   - DO NOT assume specific relationship structures exist
   - DISCOVER relationships from actual schema

2. JOIN CONSTRUCTION:
   - ONLY JOIN tables that are explicitly needed for the query
   - JOIN only when you need columns from multiple tables (for SELECT, GROUP BY, WHERE, or HAVING)
   - DO NOT add JOINs to tables that are not mentioned in the query or not needed for the result
   - Identify join keys by semantic analysis of column names across tables
   - Match columns that represent the same entity/concept
   - Use appropriate JOIN type based on query semantics
   - Join on semantically matching columns (discovered, not assumed)
   - Example: If query asks for "top users by payment amount", only JOIN users → subscriptions → payments (don't JOIN sessions unless explicitly needed)

3. FEATURE IMPLEMENTATION:
   - Window functions: Use when ranking/running totals needed
     * ROW_NUMBER() OVER (PARTITION BY col ORDER BY col): Assign sequential numbers
     * RANK() OVER (PARTITION BY col ORDER BY col): Assign ranks with gaps
     * DENSE_RANK() OVER (PARTITION BY col ORDER BY col): Assign ranks without gaps
     * Example: "top 10 by revenue" → SELECT * FROM (SELECT *, ROW_NUMBER() OVER (ORDER BY revenue DESC) as rn FROM table) WHERE rn <= 10
   - Subqueries: Use for conditional filtering or multi-step logic
     * Scalar subquery: Returns single value, used in SELECT or WHERE
       Example: SELECT col1, (SELECT MAX(col2) FROM table2) as max_val FROM table1
     * EXISTS subquery: Check if rows exist, returns boolean
       Example: SELECT * FROM users u WHERE EXISTS (SELECT 1 FROM orders o WHERE o.user_id = u.id)
     * IN subquery: Check if value exists in result set
       Example: SELECT * FROM users WHERE id IN (SELECT user_id FROM orders WHERE status = 'active')
     * Correlated subquery: References outer query, executes for each row
       Example: SELECT u.*, (SELECT COUNT(*) FROM orders o WHERE o.user_id = u.id) as order_count FROM users u
     * Derived table (subquery in FROM): Use subquery as table source
       Example: SELECT * FROM (SELECT user_id, SUM(amount) as total FROM orders GROUP BY user_id) as user_totals WHERE total > 1000
     * Common patterns:
       - "users with more than X orders" → WHERE user_id IN (SELECT user_id FROM orders GROUP BY user_id HAVING COUNT(*) > X)
       - "customers who have purchased" → WHERE EXISTS (SELECT 1 FROM orders WHERE orders.customer_id = customers.id)
       - "top customers by revenue" → FROM (SELECT customer_id, SUM(amount) as revenue FROM orders GROUP BY customer_id ORDER BY revenue DESC LIMIT 10)
   - GROUP BY: Use for categorical grouping (analyze column types)
     * TIME-BASED GROUPING PATTERNS:
       - "daily" or "by day" → DATE(column) AS date ... GROUP BY DATE(column)
       - "weekly" or "by week" → DATE_TRUNC('week', column) AS week ... GROUP BY DATE_TRUNC('week', column)
       - "bi-weekly" or "biweekly" or "every 2 weeks" → DATE_TRUNC('week', column) - INTERVAL '1 week' * (EXTRACT(WEEK FROM column)::int % 2) ... GROUP BY that expression
         OR simpler: DATE_TRUNC('week', column) + INTERVAL '1 week' * (FLOOR(EXTRACT(WEEK FROM column)::int / 2)::int) ... GROUP BY that expression
       - "monthly" or "by month" → DATE_TRUNC('month', column) AS month ... GROUP BY DATE_TRUNC('month', column)
       - "bi-monthly" or "bimonthly" or "every 2 months" → DATE_TRUNC('year', column) + INTERVAL '1 month' * (FLOOR(EXTRACT(MONTH FROM column)::int / 2) * 2) ... GROUP BY DATE_TRUNC('year', column) + INTERVAL '1 month' * (FLOOR(EXTRACT(MONTH FROM column)::int / 2) * 2)
       - "yearly" or "annually" or "by year" → DATE_TRUNC('year', column) AS year ... GROUP BY DATE_TRUNC('year', column)
       - "half-yearly" or "semi-annually" or "biannually" → DATE_TRUNC('year', column) + CASE WHEN EXTRACT(QUARTER FROM column) <= 2 THEN 'H1' ELSE 'H2' END ... GROUP BY DATE_TRUNC('year', column), CASE...
       - "quarterly" or "by quarter" → DATE_TRUNC('quarter', column) AS quarter ... GROUP BY DATE_TRUNC('quarter', column)
       - "every N days" (e.g., "every 7 days", "every 30 days") → DATE_TRUNC('day', column) - (EXTRACT(EPOCH FROM DATE_TRUNC('day', column))::int / (N * 86400)) * (N * 86400) ... GROUP BY that expression
       - "every N weeks" → DATE_TRUNC('week', column) ... GROUP BY DATE_TRUNC('week', column) (then filter/aggregate by N-week intervals if needed)
       - "Nth day" (e.g., "5th day of month") → EXTRACT(DAY FROM column) = N in WHERE clause, or GROUP BY EXTRACT(DAY FROM column)
       - "first/last day of month" → DATE_TRUNC('month', column) + INTERVAL '1 day' - INTERVAL '1 day' for last day
     
     * CATEGORICAL GROUPING:
       - If query mentions "by X" (e.g., "by country", "by device", "by category"):
         - MUST include that column in SELECT and GROUP BY
         - Example: "revenue by country" → SELECT country, SUM(amount) ... GROUP BY country
     
     * COMBINED GROUPING:
       - If query mentions BOTH time grouping AND categorical grouping:
         - Include BOTH in SELECT and GROUP BY
         - Example: "daily revenue by country" → SELECT DATE(date_col) AS date, country, SUM(amount) ... GROUP BY DATE(date_col), country
         - Example: "monthly sales by region" → SELECT DATE_TRUNC('month', date_col) AS month, region, SUM(amount) ... GROUP BY DATE_TRUNC('month', date_col), region
   - HAVING: Use for conditional aggregations
   - CASE: Use for conditional logic
   - ORDER BY: Use for sorting results
     * ASC/DESC: Specify sort direction (default is ASC)
     * Multiple columns: ORDER BY col1 DESC, col2 ASC
     * Example: "top 10 revenue" → ORDER BY revenue DESC LIMIT 10
     * Example: "sorted by date" → ORDER BY date DESC
   - LIMIT: Use to restrict number of rows returned
     * Example: "top 10", "first 5", "show 20" → LIMIT 10, LIMIT 5, LIMIT 20
     * Always use LIMIT for "top N", "first N", or "show N" queries
     * Use with ORDER BY for ranking queries
   
   - JOIN TYPES (choose appropriate type based on query semantics):
     * INNER JOIN: When you need only matching rows from both tables (default, most common)
     * LEFT JOIN: When you need all rows from left table, matching rows from right (nulls for non-matches)
     * RIGHT JOIN: When you need all rows from right table, matching rows from left (nulls for non-matches)
     * FULL JOIN: When you need all rows from both tables (nulls for non-matches)
     * Default to INNER JOIN unless query explicitly requires all rows from one side
   
   - STRING FUNCTIONS (when text manipulation needed):
     * LOWER(column): Convert text to lowercase (useful for case-insensitive comparisons)
     * UPPER(column): Convert text to uppercase
     * SUBSTRING(column, start, length): Extract substring from text
     * CONCAT(col1, col2, ...): Concatenate multiple columns/strings
     * Example: "users with lowercase email" → WHERE LOWER(email) = 'value'
   
   - MATH FUNCTIONS (when numeric operations needed):
     * ROUND(column, decimals): Round numeric values to specified decimal places
     * ABS(column): Get absolute value (remove negative sign)
     * Example: "rounded revenue" → SELECT ROUND(SUM(amount), 2) AS revenue
   
   - ADDITIONAL DATE/TIME FUNCTIONS:
     * CURRENT_DATE: Get current date (no time component)
     * NOW(): Get current timestamp (equivalent to CURRENT_TIMESTAMP)
     * CURRENT_TIMESTAMP: Get current date and time
     * INTERVAL 'N days/weeks/months/years': Date arithmetic
     * Example: "last 30 days" → WHERE date_column >= CURRENT_DATE - INTERVAL '30 days'

4. COLUMN SELECTION:
   - Analyze query requirements semantically
   - Match requirements to actual columns in schema
   - Use semantic matching, not hardcoded assumptions
   - For "daily" queries: Find date/timestamp columns in the primary table
   - For "by X" queries: Find the categorical column mentioned

Rules:
- Use ACTUAL table and column names from schema
- Discover relationships from schema structure
- Ensure all referenced columns exist
- MINIMAL QUERY CONSTRUCTION: Only include tables, columns, and JOINs that are explicitly needed for the query
  * Only SELECT columns that are requested in the query or needed for GROUP BY/ORDER BY
  * Only JOIN tables that provide columns used in SELECT, WHERE, GROUP BY, or HAVING
  * DO NOT add extra columns or JOINs "just in case" - only what's necessary
  * DO NOT add columns that are not mentioned in the query (e.g., don't add "active_users" unless explicitly requested)
  * DO NOT JOIN to tables that are not needed (e.g., don't JOIN sessions table unless query explicitly mentions sessions or session-related metrics)
  * ANALYTICAL PROCESS: For each table/column/JOIN, ask: "Is this explicitly needed for the query result?" If no, exclude it
  * ANALYTICAL PROCESS: For each SELECT column, ask: "Was this column mentioned or required by the query?" If no, exclude it
- Return only SQL query (not JSON, not markdown)
- Ensure syntactically correct PostgreSQL
- CRITICAL TIME GROUPING RULES:
  * "daily"/"by day" → MUST use DATE(column) in SELECT and GROUP BY
  * "weekly"/"by week" → MUST use DATE_TRUNC('week', column) in SELECT and GROUP BY
  * "bi-weekly"/"biweekly"/"every 2 weeks" → MUST use bi-weekly interval grouping expression
  * "monthly"/"by month" → MUST use DATE_TRUNC('month', column) in SELECT and GROUP BY
  * "bi-monthly"/"bimonthly"/"every 2 months" → MUST use bi-monthly interval grouping expression
  * "yearly"/"annually"/"by year" → MUST use DATE_TRUNC('year', column) in SELECT and GROUP BY
  * "quarterly"/"by quarter" → MUST use DATE_TRUNC('quarter', column) in SELECT and GROUP BY
  * "half-yearly"/"semi-annually" → MUST use DATE_TRUNC('year', column) + quarter logic in SELECT and GROUP BY
  * "every N days" → MUST use appropriate interval grouping expression
  * "Nth day" → MUST include day extraction in WHERE or GROUP BY
- CRITICAL: If query says "by X", include X in SELECT and GROUP BY
- Return only SQL query (not JSON, not markdown)
- Ensure syntactically correct PostgreSQL

SQL:"""
        
        return prompt
      
    @staticmethod
    def build_intent_interpretation_prompt(
        query: str, 
        tools: List[str], 
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Build prompt for intent interpretation.
        
        Focuses on semantic analysis of query intent, not hardcoded examples.
        
        Args:
            query: User query
            tools: Available tools
            context: Optional context dict
            
        Returns:
            Formatted prompt string
        """
        context_str = ""
        if context:
            context_str = f"\nContext: {json.dumps(context, indent=2)}"
        
        prompt = f"""Analyze this data analytics query and determine the best tool to use.

Available tools: {', '.join(tools)}
- sql: For structured queries, metrics, aggregations, data retrieval
- pandasai: For data analysis, charts, visualizations, exploratory analysis, pattern discovery
- dataset_info: For dataset metadata and schema information

Query: {query}{context_str}

INTENT ANALYSIS PROCESS:

1. QUERY SEMANTIC ANALYSIS:
   - Analyze the semantic meaning of the query
   - Identify what type of operation is being requested
   - Determine if query is about: data retrieval, analysis, visualization, or metadata

2. TOOL MATCHING STRATEGY:
   - SQL: Queries asking for specific metrics, aggregations, filtered data, or structured results
   - PandasAI: Queries asking for analysis, trends, patterns, visualizations, or exploratory insights
   - Dataset Info: Queries asking about schema, columns, data structure, or metadata

3. COMPLEXITY ASSESSMENT:
   - Simple queries: Single tool, straightforward operation
   - Complex queries: May require task decomposition into multiple steps
   - Identify if query needs to be broken down into sub-tasks

4. CONFIDENCE CALCULATION:
   - High confidence (0.8-1.0): Clear intent, obvious tool match
   - Medium confidence (0.5-0.8): Some ambiguity, but likely tool identified
   - Low confidence (<0.5): Unclear intent, may need clarification

Respond in JSON format:
{{
    "intent": "brief description of user intent",
    "suggested_tool": "sql|pandasai|dataset_info",
    "confidence": 0.0-1.0,
    "reasoning": "explanation of semantic analysis process and why this tool is best",
    "decomposed_tasks": ["task1", "task2"] if complex, else []
}}"""
        
        return prompt
    
    @staticmethod
    def build_task_decomposition_prompt(
        query: str, 
        tool: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Build prompt for task decomposition.
        
        Focuses on breaking down complex queries into executable steps.
        
        Args:
            query: User query
            tool: Target tool
            context: Optional context dict
            
        Returns:
            Formatted prompt string
        """
        context_str = ""
        if context:
            context_str = f"\nContext: {json.dumps(context, indent=2)}"
        
        prompt = f"""Break down this complex query into smaller, executable sub-tasks.

Query: {query}
Target Tool: {tool}{context_str}

DECOMPOSITION PROCESS:

1. QUERY ANALYSIS:
   - Identify all distinct operations requested
   - Determine dependencies between operations
   - Recognize if operations can be parallelized or must be sequential

2. STEP IDENTIFICATION:
   - Break query into logical steps
   - Each step should be independently executable
   - Steps should build upon previous results when needed

3. PARAMETER EXTRACTION:
   - Identify what parameters each step needs
   - Extract filters, aggregations, groupings from query
   - Map query requirements to tool-specific parameters

4. EXECUTION ORDER:
   - Determine optimal order of execution
   - Consider data dependencies
   - Minimize redundant operations

Respond in JSON format with array of tasks:
[
    {{
        "step": 1,
        "description": "what to do in this step",
        "tool": "{tool}",
        "parameters": {{"key": "value"}}
    }}
]"""
        
        return prompt
    
    @staticmethod
    def build_query_complexity_prompt(query: str, schema: Dict[str, Any]) -> str:
        """
        Build prompt for analyzing query complexity.
        
        Focuses on semantic analysis of query requirements, not hardcoded patterns.
        
        Args:
            query: User query
            schema: Database schema
            
        Returns:
            Formatted prompt string
        """
        tables_info = []
        for table_name, columns in schema.items():
            columns_with_types = [
                {"name": col["name"], "type": col.get("type", "unknown")}
                for col in columns
            ]
            tables_info.append({
                "table": table_name,
                "columns": columns_with_types
            })
        
        prompt = f"""Classify this SQL query request by type and complexity.

Query: {query}

Database Schema:
{json.dumps(tables_info, indent=2)}

COMPLEXITY ANALYSIS PROCESS:

1. QUERY SEMANTIC ANALYSIS:
   - Analyze the semantic meaning of the query
   - Identify what operations are being requested
   - Determine if query involves: single table, multiple tables, aggregations, time series, rankings

2. SQL FEATURE REQUIREMENTS:
   - JOINs: Does query mention multiple entities/tables that need to be combined?
   - GROUP BY: Does query mention grouping by categories, dates, or dimensions?
   - Window Functions: Does query mention rankings, running totals, or relative comparisons?
   - Subqueries: Does query have conditional logic like "with more than", "that have", "which have"?
   - CASE Statements: Does query have conditional logic like "if", "when", "depending on"?

3. COMPLEXITY ASSESSMENT:
   - Low: Simple SELECT with single aggregation, single table
   - Medium: Multiple tables (JOINs), grouping, or time series
   - High: Window functions, subqueries, complex conditional logic, or multiple advanced features

4. TYPE CLASSIFICATION:
   - simple: Basic SELECT with aggregation, single table
   - multi_table: Requires JOINs across multiple tables
   - time_series: Requires GROUP BY date/time (daily, weekly, monthly patterns)
   - ranking: Requires window functions (ROW_NUMBER, RANK, etc.) for top/bottom/rank queries
   - conditional: Requires CASE statements for conditional logic
   - subquery: Requires nested queries for conditional filtering

Respond in JSON:
{{
    "query_type": "simple|multi_table|time_series|ranking|conditional|subquery",
    "complexity": "low|medium|high",
    "requires_joins": true/false,
    "requires_grouping": true/false,
    "requires_window_functions": true/false,
    "requires_subqueries": true/false,
    "requires_case_statements": true/false,
    "reasoning": "brief explanation of semantic analysis process"
}}"""
        
        return prompt

    @staticmethod
    def build_query_relevance_prompt(
        query: str,
        available_tables: List[str],
        available_metrics: List[str],
        schema_info: Optional[str] = None
    ) -> str:
        """
        Build prompt for validating query relevance to database.
        
        Focuses on semantic analysis of query intent, not hardcoded examples.
        
        Args:
            query: User query
            available_tables: List of available table names
            available_metrics: List of available metric names
            schema_info: Optional additional schema context
            
        Returns:
            Formatted prompt string
        """
        tables_str = ', '.join(available_tables[:10])
        metrics_str = ', '.join(available_metrics[:10])
        
        schema_context = ""
        if schema_info:
            schema_context = f"\n{schema_info}"
        
        prompt = f"""Determine if this query can be answered from the database.

Available tables: {tables_str}
Available metrics: {metrics_str}{schema_context}

Query: {query}

CRITICAL DECISION RULE: If the query asks about ANY of the following, it is ALWAYS RELEVANT:
- Sessions, session duration, average duration, session data
- Users, active users, user data
- Payments, revenue, transactions, amounts
- Subscriptions, plans
- Any metric calculation (average, count, sum, total)
- Any aggregation or analysis of database data

RELEVANCE ANALYSIS PROCESS:

1. SEMANTIC ENTITY MATCHING:
   - Analyze the query for entities, concepts, or domain terms
   - Compare query entities against available tables and metrics
   - Use semantic matching (synonyms, related terms, domain terminology)
   - Consider: "sales" might match "revenue", "customers" might match "users", "transactions" might match "payments"
   - Look for conceptual relationships, not just exact word matches
   - If query mentions "session", "duration", "average", "users", "payments", "revenue" → RELEVANT

2. DOMAIN CLASSIFICATION:
   - Determine the domain/subject matter of the query
   - Check if query domain aligns with database domain
   - Database-relevant domains: business data, analytics, metrics, KPIs, transactions, user data, operational data, session data
   - Irrelevant domains: general knowledge, personal questions, unrelated topics (weather, movies, news, etc.)

3. INTENT ANALYSIS:
   - Analyze what the query is asking for
   - Database-relevant intents: data retrieval, metrics calculation, trend analysis, aggregations, filtering, averages, counts, sums
   - Irrelevant intents: general questions, personal information, unrelated domain queries
   - If query asks "What's the average X?" or "How many X?" → RELEVANT (these are database queries)

4. MATCHING STRATEGY:
   - If query mentions entities/concepts that semantically match tables or metrics → RELEVANT
   - If query is about business data, analytics, or database entities → RELEVANT
   - If query is about unrelated topics (general knowledge, personal, external domains) → IRRELEVANT
   - If query cannot be answered from structured database data → IRRELEVANT

5. DECISION CRITERIA:
   - RELEVANT: Query can be answered using database tables, metrics, or business data
   - IRRELEVANT: Query is about topics outside database scope (general knowledge, personal, unrelated domains)
   - DEFAULT TO RELEVANT when uncertain

CRITICAL EXAMPLES OF RELEVANT QUERIES (MUST RETURN is_relevant: true):
- "What's the average session duration?" → RELEVANT (session data, average calculation)
- "Average session duration" → RELEVANT (session metric, average)
- "What's the average duration?" → RELEVANT (likely about sessions or similar metric)
- "How many active users?" → RELEVANT (user data, count)
- "Show me revenue by country" → RELEVANT (revenue data, aggregation)
- "What's the total payment amount?" → RELEVANT (payment data, sum)
- Any query containing: "session", "duration", "average", "users", "payments", "revenue", "subscriptions" → RELEVANT

CRITICAL EXAMPLES OF IRRELEVANT QUERIES (MUST RETURN is_relevant: false):
- "What is your name?" → IRRELEVANT (personal question)
- "What's the weather?" → IRRELEVANT (external domain)
- "Tell me a joke" → IRRELEVANT (general knowledge)

IMPORTANT RULES:
- "What's the average session duration?" is ALWAYS RELEVANT - it's asking for a database metric
- Queries about "sessions", "duration", "average" are database-related
- If query contains database-related terms (sessions, users, payments, revenue, duration, average, count, sum) → RELEVANT
- Focus on semantic analysis, not exact word matching
- Consider synonyms and related terms
- Base decision on whether query CAN be answered from database, not whether it's easy to answer
- If uncertain, err on the side of RELEVANT (let query processing handle edge cases)
- "session duration", "average duration", "session data" are ALWAYS database-related

Respond in JSON format only:
{{
    "is_relevant": true/false,
    "reason": "brief explanation of semantic analysis process and why this query is or isn't relevant to the database"
}}"""
        
        return prompt

    @staticmethod
    def build_nl_to_sql_prompt(query: str, schema_context: str = "") -> str:
        """
        Build prompt for converting natural language to SQL.
        
        Args:
            query: Natural language query
            schema_context: Optional schema information
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""Convert this natural language query to a safe PostgreSQL SELECT query.
Only return the SQL query, nothing else.

Query: {query}{schema_context}

Rules:
- Only use SELECT statements
- Use COUNT(*) for counting rows
- Use COUNT(DISTINCT column_name) for counting unique values (NOT COUNT DISTINCT(column_name))
- Use AVG(column_name) for averages
- Use SUM(column_name) for totals
- Use WHERE column_name = 'value' for filtering
- Use actual table and column names from the schema above
- Return only the SQL query
- Do not include markdown code blocks
- Do not include explanations

SQL:"""
        
        return prompt        