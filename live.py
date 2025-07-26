import sqlite3
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from tensorflow.keras.models import load_model
from apscheduler.schedulers.blocking import BlockingScheduler
from simulation import TechnicalIndicators

DB_PATH = "paper_trading.db"
STOCK_LIST = ['HSBA.L', 'BT-A.L', 'RR.L', 'AMZN']
SELL_CHECK_INTERVAL_MIN = 5  # Sell check every X mins

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS account (
            id INTEGER PRIMARY KEY,
            cash_balance REAL,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS holdings (
            symbol TEXT PRIMARY KEY,
            shares INTEGER,
            avg_buy_price REAL,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            trade_type TEXT,
            shares INTEGER,
            price REAL,
            trade_time DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)
    # New pending order table
    c.execute("""
        CREATE TABLE IF NOT EXISTS pending_orders (
            symbol TEXT PRIMARY KEY,
            limit_price REAL,
            shares INTEGER,
            order_time DATETIME,
            expires_at DATETIME
        );
    """)
    c.execute("SELECT COUNT(*) FROM account WHERE id=1")
    if c.fetchone()[0] == 0:
        c.execute("INSERT INTO account (id, cash_balance) VALUES (1, ?)", (float(10000.0),))
    conn.commit()
    conn.close()

def get_balance():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT cash_balance FROM account WHERE id=1")
    row = c.fetchone()
    conn.close()
    if not row or row[0] is None:
        return 0.0
    return float(row[0])

def update_balance(new_balance):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE account SET cash_balance = ?, updated_at = ? WHERE id=1", (float(new_balance), datetime.now()))
    conn.commit()
    conn.close()

def get_holdings(symbol=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if symbol:
        c.execute("SELECT shares, avg_buy_price FROM holdings WHERE symbol=?", (symbol,))
        row = c.fetchone()
        conn.close()
        if row:
            return int(row[0]), float(row[1])
        return 0, 0.0
    else:
        c.execute("SELECT symbol FROM holdings")
        rows = c.fetchall()
        conn.close()
        return [row[0] for row in rows]

def update_holdings(symbol, shares, avg_buy_price):
    shares = int(shares)
    avg_buy_price = float(avg_buy_price)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if shares == 0:
        c.execute("DELETE FROM holdings WHERE symbol=?", (symbol,))
    else:
        c.execute("""
            INSERT INTO holdings (symbol, shares, avg_buy_price, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET shares=excluded.shares, avg_buy_price=excluded.avg_buy_price, updated_at=excluded.updated_at
        """, (symbol, shares, avg_buy_price, datetime.now()))
    conn.commit()
    conn.close()

def add_trade(symbol, trade_type, shares, price):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO trades (symbol, trade_type, shares, price) VALUES (?, ?, ?, ?)",
        (symbol, trade_type, int(shares), float(price))
    )
    conn.commit()
    conn.close()

def add_pending_order(symbol, limit_price, shares, expires_at):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT OR REPLACE INTO pending_orders (symbol, limit_price, shares, order_time, expires_at)
        VALUES (?, ?, ?, ?, ?)
    """, (symbol, float(limit_price), int(shares), datetime.now(), expires_at))
    conn.commit()
    conn.close()

def remove_pending_order(symbol):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM pending_orders WHERE symbol=?", (symbol,))
    conn.commit()
    conn.close()

def get_pending_orders():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT symbol, limit_price, shares, order_time, expires_at FROM pending_orders")
    rows = c.fetchall()
    conn.close()
    return rows

def print_balance():
    bal = get_balance()
    holdings = get_holdings()
    print(f"\n==== Account summary at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ====")
    print(f"Cash balance: ${bal:,.2f}")
    if holdings:
        print("Open positions:")
        for symbol in holdings:
            shares, avg = get_holdings(symbol)
            print(f"  {symbol}: {shares} shares @ ${avg:.2f}")
    else:
        print("No open positions.")
    print("============================================\n")

def fetch_latest_daily(symbol):
    data = yf.download(symbol, period="60d", interval="1d", auto_adjust=False)
    if data.empty or len(data) < 30:
        return None
    data = TechnicalIndicators.compute_RSI(data)
    data = TechnicalIndicators.compute_MACD(data)
    data = TechnicalIndicators.compute_BollingerBands(data)
    data['MA7'] = data['Close'].rolling(window=7).mean()
    data['MA21'] = data['Close'].rolling(window=21).mean()
    data = data.dropna()
    if len(data) < 21:
        return None
    return data.tail(21)

def fetch_latest_hourly(symbol):
    end = datetime.now()
    start = end - timedelta(days=90)
    data = yf.download(symbol, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), interval="1h", auto_adjust=False)
    print(f"{symbol}: Downloaded {len(data)} hourly bars from {start.date()} to {end.date()}")
    if data.empty or len(data) < 168:
        print(f"Not enough hourly data for {symbol}. Needed 168, got {len(data)}.")
        return None
    data['MA4'] = data['Close'].rolling(window=4).mean()
    data = data.dropna()
    if len(data) < 168:
        return None
    return data.tail(168)

def place_pending_buy(symbol, model_trend, model_low):
    # Only place new buy if not already held or already pending
    shares, _ = get_holdings(symbol)
    pending_symbols = [row[0] for row in get_pending_orders()]
    if shares > 0 or symbol in pending_symbols:
        print(f"[{symbol}] Already held or pending. Skipping buy.")
        return
    daily = fetch_latest_daily(symbol)
    if daily is None:
        print(f"[{symbol}] Not enough daily data.")
        return
    hourly = fetch_latest_hourly(symbol)
    if hourly is None:
        print(f"[{symbol}] Not enough hourly data.")
        return
    X_daily = np.expand_dims(daily.values, axis=0)
    trend_pred = model_trend.predict(X_daily, verbose=0)[0]
    predicted_class = np.argmax(trend_pred)
    print(f"[{symbol}] Trend prediction: {trend_pred}, class: {predicted_class}")
    if predicted_class != 2:
        print(f"[{symbol}] No bullish signal today.")
        return
    X_hourly = np.expand_dims(hourly.values, axis=0)
    low_pred = model_low.predict(X_hourly, verbose=0)[0][0]
    latest_open = float(hourly['Open'].iloc[-1])
    predicted_low_price = float(latest_open * (1 + low_pred))
    limit_order_price = float(predicted_low_price * 1.02)
    cash = float(get_balance())
    risk_amount = cash * 0.10
    buy_shares = max(int(risk_amount / limit_order_price), 1)
    expires_at = (datetime.now() + timedelta(days=1)).replace(hour=16, minute=30, second=0, microsecond=0)  # "Market close"
    print(f"[{symbol}] Placing pending BUY order: shares={buy_shares}, limit={limit_order_price:.2f}, expires={expires_at}")
    add_pending_order(symbol, limit_order_price, buy_shares, expires_at)

def check_and_fill_pending_orders():
    pending = get_pending_orders()
    for symbol, limit_price, shares, order_time, expires_at in pending:
        # Check if order expired (end of day)
        expires_at_dt = pd.to_datetime(expires_at)
        now = datetime.now()
        df = yf.download(symbol, start=order_time, end=now, interval="1m", progress=False)
        filled = False
        if not df.empty:
            # Did price touch our limit? (use 'Low')
            if (df['Low'] <= float(limit_price)).any():
                fill_price = float(limit_price)
                filled = True
                print(f"[{symbol}] Pending BUY FILLED at limit: {fill_price:.2f}")
            elif now >= expires_at_dt:
                # Order expired; buy at close price
                fill_price = float(df['Close'].iloc[-1])
                filled = True
                print(f"[{symbol}] Pending BUY expired, BOUGHT at close: {fill_price:.2f}")
        elif now >= expires_at_dt:
            # No intraday data, fallback: buy at current close price
            close_now = yf.download(symbol, period="1d", interval="1m")['Close'].iloc[-1]
            fill_price = float(close_now)
            filled = True
            print(f"[{symbol}] Pending BUY expired (no intraday), BOUGHT at close: {fill_price:.2f}")

        if filled:
            cash = float(get_balance())
            cost = float(fill_price) * int(shares)
            if cash >= cost:
                cash -= cost
                update_balance(cash)
                add_trade(symbol, 'BUY', int(shares), float(fill_price))
                update_holdings(symbol, int(shares), float(fill_price))
                print(f"[{symbol}] Bought {shares} shares at {fill_price:.2f}. New cash: {cash:.2f}")
            else:
                print(f"[{symbol}] Not enough cash for pending order (wanted {cost:.2f}, had {cash:.2f}). SKIPPED.")
            remove_pending_order(symbol)

def live_check_sell(symbol, stop_loss_pct=0.95, take_profit_pct=1.1):
    shares, avg_buy_price = get_holdings(symbol)
    if shares == 0:
        return
    data = yf.download(symbol, period="2d", interval="1h", auto_adjust=False)
    if data.empty:
        print(f"[{symbol}] No current data for sell check.")
        return
    current_price = float(data['Close'].iloc[-1])
    highest_price = float(max(data['High'].iloc[-24:].max(), avg_buy_price))
    stop_price = highest_price * stop_loss_pct
    take_profit_price = avg_buy_price * take_profit_pct
    cash = float(get_balance())
    print(f"[{symbol}] Checking sell: current {current_price:.2f}, stop {stop_price:.2f}, TP {take_profit_price:.2f}, avg buy {avg_buy_price:.2f}")
    if current_price <= stop_price or current_price >= take_profit_price:
        proceeds = float(shares * current_price)
        cash += proceeds
        update_balance(float(cash))
        add_trade(symbol, 'SELL', shares, float(current_price))
        update_holdings(symbol, 0, 0.0)
        print(f"[{symbol}] Sold {shares} shares at {current_price:.2f}. New cash balance: {cash:.2f}")

def force_sell_all_holdings():
    print(f"\n[{datetime.now()}] [WEEKLY RESET] Force-selling all positions at market price...")
    held = get_holdings()
    for symbol in held:
        shares, avg_buy_price = get_holdings(symbol)
        if shares > 0:
            data = yf.download(symbol, period="1d", interval="1m", auto_adjust=False)
            if not data.empty:
                market_price = float(data['Close'].iloc[-1])
                cash = float(get_balance()) + float(shares * market_price)
                update_balance(float(cash))
                add_trade(symbol, 'SELL', shares, float(market_price))
                update_holdings(symbol, 0, 0.0)
                print(f"[{symbol}] Force-sold {shares} shares at {market_price:.2f}.")
            else:
                print(f"[{symbol}] WARNING: Could not fetch market price for force-sell.")

def cancel_all_pending_orders():
    pending = get_pending_orders()
    for symbol, *_ in pending:
        print(f"[{symbol}] Cancelling pending order.")
        remove_pending_order(symbol)

def sell_check_all():
    check_and_fill_pending_orders()
    print(f"\n[{datetime.now()}] Running sell checks...")
    for symbol in get_holdings():
        live_check_sell(symbol)
    print_balance()

def weekly_reset_and_buy():
    force_sell_all_holdings()
    cancel_all_pending_orders()
    print_balance()
    print(f"\n[{datetime.now()}] Weekly buy scan running...")
    for symbol in STOCK_LIST:
        place_pending_buy(symbol, model_trend, model_low)
    print_balance()

if __name__ == "__main__":
    print("Setting up database and models...")
    init_db()
    global model_trend, model_low
    model_trend = load_model("models/trend_model.keras", compile=False)
    model_low = load_model("models/low_price_model.keras", compile=False)
    print("Starting scheduled live paper trading...\n")

    scheduler = BlockingScheduler()
    scheduler.add_job(sell_check_all, 'interval', minutes=SELL_CHECK_INTERVAL_MIN)
    scheduler.add_job(weekly_reset_and_buy, 'interval', weeks=1)
    weekly_reset_and_buy()
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("Scheduler stopped.")


