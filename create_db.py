import sqlite3

# Connect to the SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('database.db')

# Create a cursor object to execute SQL commands
cursor = conn.cursor()

# Create the profiles table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS profiles (
        id TEXT PRIMARY KEY,
        image TEXT NOT NULL,
        data TEXT NOT NULL
    )
''')

# Commit the changes and close the connection
conn.commit()
conn.close()

print('Database and profiles table created successfully.')  