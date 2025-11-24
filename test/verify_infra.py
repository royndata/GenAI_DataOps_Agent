# ----------------------------------------------------------
# tests/verify_infra.py
#
# One-shot Infra Smoke Test:
# - Validates Settings/.env loading
# - Validates Database connection
# - Validates SQLTool safe SELECT
# - Validates dataset_loader
# ----------------------------------------------------------

import traceback

from agent.config import Settings
from agent.knowledge.database import Database
from agent.tools.sql_tool import SQLTool
from agent.knowledge.dataset_loader import load_dataset


def test_settings():
    print("\n[1] Testing Settings / .env loading...")
    env = Settings()
    # Print only non-sensitive env fields
    print(f"Loaded settings:")
    print(f"  DB_HOST = {env.db_host}")
    print(f"  DB_NAME = {env.db_name}")
    print(f"  OPENAI_API_KEY present = {bool(env.openai_api_key)}")
    return env


def test_database_connection(env):
    print("\n[2] Testing Database connection...")
    db = Database(env)
    ok = db.test_connection()
    print("Database connection OK!\n")
    return db


def test_sqltool(db):
    print("[3] Testing SQLTool...")
    sql = SQLTool(db)
    rows = sql.run_safe_query("SELECT 1 as test")
    print(f"SQLTool returned: {rows}\n")
    return True


def test_dataset_loader():
    print("[4] Testing dataset_loader...")
    try:
        # Adjust filename to your dataset name if needed
        filename = None

        # Try discovering a dataset automatically
        import os
        datasets_path = "datasets"
        if os.path.exists(datasets_path):
            for f in os.listdir(datasets_path):
                if f.endswith(".csv"):
                    filename = f
                    break

        if not filename:
            print("⚠️  No dataset found in /datasets. Skipping dataset test.\n")
            return False

        print(f"Loading dataset: {filename}")
        df = load_dataset(filename)
        print(df.head())
        print("Dataset loaded successfully!\n")
        return True

    except Exception as e:
        print("Dataset loader failed!")
        print(traceback.format_exc())
        return False


def main():
    print("\n========================================")
    print("     INFRASTRUCTURE SMOKE TEST")
    print("========================================")

    try:
        env = test_settings()
        db = test_database_connection(env)
        test_sqltool(db)
        test_dataset_loader()

        print("\n========================================")
        print("     ALL INFRA TESTS COMPLETED")
        print("========================================")

    except Exception as e:
        print("\n❌ Infra test failed!")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
