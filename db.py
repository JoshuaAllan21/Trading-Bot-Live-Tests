import sqlite3
conn = sqlite3.connect("paper_trading.db")
c = conn.cursor()
c.execute("UPDATE account SET cash_balance = 100000.0 WHERE id = 1")
conn.commit()
conn.close()
