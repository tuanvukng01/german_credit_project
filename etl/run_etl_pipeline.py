import sqlite3
import csv
import os

# Paths
db_path = "./data/german_credit.db"
sql_path = "./sql/feature_engineering.sql"
csv_path = "./data/processed/credit_features.csv"

def run_etl():
    # Step 1: Ensure output folder exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Step 2: Read the SQL view definition from file
    with open(sql_path, "r") as f:
        create_view_sql = f.read()

    # Step 3: Connect to SQLite and create the view + export it
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Create or replace the view
        cursor.executescript(create_view_sql)
        print("✅ View `v_credit_features` created successfully.")

        # Query the view
        cursor.execute("SELECT * FROM v_credit_features")
        rows = cursor.fetchall()

        # Export to CSV
        with open(csv_path, "w", encoding="utf-8", newline="\n") as f:
            # Write header
            f.write(",".join([desc[0] for desc in cursor.description]) + "\n")

            # Write rows
            for row in rows:
                f.write(",".join(map(str, row)) + "\n")

        print(f"✅ Exported {len(rows)} rows to {csv_path}")

    except sqlite3.OperationalError as e:
        print(f"❌ SQLite Error: {e}")

    finally:
        conn.close()

if __name__ == "__main__":
    run_etl()