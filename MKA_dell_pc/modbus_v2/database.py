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

def add_blade_data(blade_data): 
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT OR REPLACE INTO blades VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", blade_data)
    conn.commit()
    conn.close()

if __name__ == "__main__":
    print("Blade Database ")
    print("-------------")
    print("Enter :")
    print("1. Create Database")
    print("2. Add Blade Data")
    print("3. retrieve data")
    print("4. read data")
    print("5. exit")
    while True:
        inp = input("Enter : ")
        if (inp=="1"):
            main()
        elif (inp=="2"):
            bade_id = input("Enter blade ID: ")
            depth1 = float(input("Enter depth 1: "))
            depth2 = float(input("Enter depth 2: "))
            thickness = float(input("Enter thickness: "))
            middle_depth = float(input("Enter middle depth: "))
            corner_depth = float(input("Enter corner depth: "))
            length = float(input("Enter length: "))
            reach_angle = float(input("Enter reach angle: "))
            depth_of_cut_from_angle = float(input("Enter depth of cut from angle: "))
            v_blade = int(input("Is it a V-blade? (1 for Yes, 0 for No): "))
            tip_angle = float(input("Enter tip angle: "))
            blade_data = (bade_id, depth1, depth2, thickness, middle_depth
                            , corner_depth, length, reach_angle, depth_of_cut_from_angle, v_blade, tip_angle)
            add_blade_data(blade_data)
        elif (inp=="3"):
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute("SELECT * FROM blades")
            rows = cur.fetchall()
            for row in rows:
                print(row)
            conn.close()
        elif (inp=="4"):   
            bade_id = input("Enter blade ID to read: ")
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute("SELECT * FROM blades WHERE bade_id=?", (bade_id,))
            row = cur.fetchone()
            if row:
                print(row)
            else:
                print(f"No data found for blade ID: {bade_id}")
            conn.close()      
        elif (inp=="5"):
            print("Exiting...")
            break

# >>> import sqlite3
# >>> 
# >>> conn = sqlite3.connect("blade_database.sqlite")
# >>> cur = conn.cursor()
# >>> cur.execute("""INSERT INTO blades (bade_id, length, reach_angle, depth_of_cut_from_angle) VALUES (?, ?, ?, ?) """, ("A129", 360, 30.17, 1.76)) 
# >>> conn.commit()