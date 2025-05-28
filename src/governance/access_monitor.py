"""
This script allows the Data Steward to monitor and review the data access log.
Usage:
    python monitor_access_log.py [--user username] [--date YYYY-MM-DD]
If no arguments are given, all access events are shown.
"""
import sqlite3
import sys
from datetime import datetime

# File paths
LOG_DB = '../data/sensitive/data_access_log.db'

def print_log(user=None, date=None):
    conn = sqlite3.connect(LOG_DB)
    cursor = conn.cursor()
    query = 'SELECT user, timestamp, fields_accessed, purpose, action FROM data_access_log WHERE 1=1'
    params = []
    if user:
        query += ' AND user = ?'
        params.append(user)
    if date:
        query += ' AND date(timestamp) = ?'
        params.append(date)
    query += ' ORDER BY timestamp DESC'
    for row in cursor.execute(query, params):
        print(f"User: {row[0]}, Time: {row[1]}, Fields: {row[2]}, Purpose: {row[3]}, Action: {row[4]}")
    conn.close()

if __name__ == '__main__':
    user = None
    date = None
    args = sys.argv[1:]
    if '--user' in args:
        idx = args.index('--user')
        if idx + 1 < len(args):
            user = args[idx + 1]
    if '--date' in args:
        idx = args.index('--date')
        if idx + 1 < len(args):
            date = args[idx + 1]
    print_log(user, date) 