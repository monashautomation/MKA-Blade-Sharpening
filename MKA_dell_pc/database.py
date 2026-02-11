import sqlite3

DB_PATH = "blade_database.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS blades (
    bade_id TEXT PRIMARY KEY,
    depth1 REAL,
    depth2 REAL,
    thickness REAL,
    middle_depth REAL,
    corner_depth REAL,
    length REAL,
    reach_angle REAL,
    depth_of_cut_from_angle REAL,
    v_blade INTEGER,      -- store as 0/1
    tip_angle REAL
);
"""

def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(SCHEMA)
    conn.commit()
    conn.close()
    print(f"Created: {DB_PATH}")

if __name__ == "__main__":
    main()
