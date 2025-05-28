"""
This script is the only approved way to access sensitive data (e.g., ames_sensitive_data.csv).
It logs every access event for auditing and compliance. Only whitelisted users may access the data.
"""
import csv
import sqlite3
from datetime import datetime
import getpass
import os
import sys

CSV_FILE = '../data/sensitive/ames_sensitive_data.csv'  # Adjust path as needed
LOG_DB = '../data/sensitive/data_access_log.db'
ALLOWED_USERS = {'kdhenderson', 'analyst1', 'data_steward'}  # Add authorized usernames here

# Logging function
def log_data_access(user, fields, purpose, action, logfile=LOG_DB):
    conn = sqlite3.connect(logfile)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS data_access_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            fields_accessed TEXT NOT NULL,
            purpose TEXT,
            action TEXT
        )
    ''')
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute('''
        INSERT INTO data_access_log (user, timestamp, fields_accessed, purpose, action)
        VALUES (?, ?, ?, ?, ?)
    ''', (user, timestamp, ', '.join(fields), purpose, action))
    conn.commit()
    conn.close()

# Permission check
def check_permissions(user):
    if user not in ALLOWED_USERS:
        print(f"Access denied: {user} is not authorized.")
        sys.exit(1)
    if not os.path.exists(CSV_FILE):
        print(f"ERROR: {CSV_FILE} does not exist or you do not have permission to access it.")
        sys.exit(1)

if __name__ == '__main__':
    user = getpass.getuser()
    check_permissions(user)
    fields = ['Address', 'Sale Date']  # List the sensitive fields being accessed
    purpose = input("Enter the purpose for accessing this data: ")
    action = 'Read'
    log_data_access(user, fields, purpose, action)
    print("Access granted. Reading data...")
    with open(CSV_FILE, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            print(row)  # Replace or remove in production 