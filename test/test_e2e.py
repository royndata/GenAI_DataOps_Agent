# test/test_e2e.py

"""
End-to-end integration test: Slack → Router → Tool → Response
Tests the complete flow with PostgreSQL beam dataset.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.config import Settings
from agent.logging_config import configure_logging, logger
from agent.knowledge.database import Database
from agent.tools.sql_tool import SQLTool
from agent.tools.pandasai_tool import PandasAITool
from agent.cognition.router import Router
from agent.ingestion.input_guardrails import InputGuardrails
from agent.output.output_guardrails import OutputGuardrails
from agent.output.output_formatter import OutputFormatter


def test_e2e_beam_dataset():
    """
    End-to-end test simulating Slack message → Router → Tool → Response
    """
    print("\n" + "="*60)
    print("  E2E INTEGRATION TEST: Beam Dataset")
    print("="*60 + "\n")

    # Step 1: Initialize components
    print("[1/6] Initializing components...")
    settings = Settings()
    configure_logging()
    
    db = Database(settings)
    sql_tool = SQLTool(db=db)
    pandas_tool = PandasAITool(settings=settings)
    router = Router(sql_tool, pandas_tool, database=db, settings=settings)
    
    input_guardrails = InputGuardrails(semantic_loader=router.semantic_loader)
    output_guardrails = OutputGuardrails()
    output_formatter = OutputFormatter()
    
    print("✅ Components initialized\n")

    # Step 2: Test database connection
    print("[2/6] Testing database connection...")
    if not db.test_connection():
        print("❌ Database connection failed!")
        return False
    print("✅ Database connected\n")

    # Step 3: Verify beam dataset exists
    print("[3/6] Verifying beam dataset...")
    # Replace 'beam_table' with your actual table name
    beam_table = "sessions"  # or "beam_data", "beam_sales", etc.
    try:
        result = sql_tool.run_safe_query(f"SELECT COUNT(*) as count FROM {beam_table} LIMIT 1")
        row_count = result.get("rows", [])[0][0] if result.get("rows") else 0
        print(f"✅ Beam dataset found: {row_count:,} rows")
    except Exception as e:
        print(f"❌ Beam dataset not found: {e}")
        print("   Please verify table name in test file")
        return False
    print()

    # Step 4: Simulate Slack message (input validation)
    print("[4/6] Simulating Slack message with input guardrails...")
    test_queries = [
        f"SELECT COUNT(*) FROM {beam_table}",
        f"Show me total rows in {beam_table}",
        f"What are the columns in {beam_table}?",
    ]
    
    for query in test_queries:
        print(f"\n   Testing query: '{query}'")
        
        # Input guardrails validation
        is_valid, error_msg, _ = input_guardrails.validate(query)
        if not is_valid:
            print(f"   ⚠️  Input guardrails blocked: {error_msg}")
            continue
        print("   ✅ Input guardrails passed")

    # Step 5: Router → Tool execution
    print("\n[5/6] Testing Router → Tool execution...")
    test_query = f"SELECT COUNT(*) as total FROM {beam_table}"
    print(f"   Query: '{test_query}'")
    
    # Simulate user_id
    user_id = "test_user_123"
    
    # Route query
    response = router.route(test_query, user_id=user_id)
    
    print(f"   Router response keys: {list(response.keys())}")
    print(f"   Success: {response.get('success', False)}")
    print(f"   Message preview: {response.get('message', '')[:100]}...")
    
    if not response.get("success"):
        print(f"   ❌ Router failed: {response.get('error', 'Unknown error')}")
        return False
    print("   ✅ Router executed successfully")

    # Step 6: Output validation & formatting
    print("\n[6/6] Testing output guardrails & formatting...")
    
    # Output guardrails validation
    output_valid, output_error, sanitized = output_guardrails.validate_router_response(response)
    if not output_valid:
        print(f"   ❌ Output guardrails failed: {output_error}")
        return False
    print("   ✅ Output guardrails passed")
    
    # Format response
    formatted = output_formatter.format_router_response(sanitized)
    print(f"   ✅ Formatted message length: {len(formatted)} chars")
    print(f"\n   Formatted output preview:\n   {formatted[:200]}...")

    # Final summary
    print("\n" + "="*60)
    print("  ✅ E2E TEST COMPLETED SUCCESSFULLY")
    print("="*60 + "\n")
    
    return True


if __name__ == "__main__":
    try:
        success = test_e2e_beam_dataset()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ E2E test failed with exception:")
        import traceback
        traceback.print_exc()
        sys.exit(1)