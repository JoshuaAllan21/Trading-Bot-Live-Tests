import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import torch
from datetime import datetime
from tensorflow.keras.models import load_model
import joblib
# --- Technical Indicators ---
class TechnicalIndicators:
    @staticmethod
    def compute_RSI(data, window=14):
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        RS = gain / loss
        data['RSI'] = 100 - (100 / (1 + RS))
        return data

    @staticmethod
    def compute_MACD(data, fast=12, slow=26, signal=9):
        data['EMA_fast'] = data['Close'].ewm(span=fast, adjust=False).mean()
        data['EMA_slow'] = data['Close'].ewm(span=slow, adjust=False).mean()
        data['MACD'] = data['EMA_fast'] - data['EMA_slow']
        data['MACD_signal'] = data['MACD'].ewm(span=signal, adjust=False).mean()
        data.drop(['EMA_fast', 'EMA_slow'], axis=1, inplace=True)
        return data

    @staticmethod
    def compute_BollingerBands(data, window=20, num_std=2):
        rolling_mean = data['Close'].rolling(window=window).mean()
        rolling_std = data['Close'].rolling(window=window).std()
        data['BB_upper'] = rolling_mean + (rolling_std * num_std)
        data['BB_lower'] = rolling_mean - (rolling_std * num_std)
        return data

# --- Data Handler ---
class DataHandler:
    def __init__(self, stock_symbol):
        self.stock_symbol = stock_symbol
        self.daily_data = None
        self.hourly_data = None

    def download_data(self, start, end):
        self.daily_data = yf.download(self.stock_symbol, start=start, end=end, interval="1d", auto_adjust=False)
        self.hourly_data = yf.download(self.stock_symbol, start=start, end=end, interval="1h", auto_adjust=False)
        if self.daily_data.empty or self.hourly_data.empty:
            raise ValueError(f"No data found for {self.stock_symbol}")
        self.daily_data = TechnicalIndicators.compute_RSI(self.daily_data.copy())
        self.daily_data = TechnicalIndicators.compute_MACD(self.daily_data.copy())
        self.daily_data = TechnicalIndicators.compute_BollingerBands(self.daily_data.copy())
        self.daily_data['MA7'] = self.daily_data['Close'].rolling(window=7).mean()
        self.daily_data['MA21'] = self.daily_data['Close'].rolling(window=21).mean()
        self.daily_data.dropna(inplace=True)
        self.hourly_data['MA4'] = self.hourly_data['Close'].rolling(window=4).mean()
        self.hourly_data.dropna(inplace=True)
        self.hourly_data.index = self.hourly_data.index.tz_localize(None)

    def get_valid_simulation_dates(self, seq_length_daily=21, seq_length_hourly=168, next_days_hourly_needed=168):
        valid_dates = []
        for sim_date in self.daily_data.index:
            daily_window = self.daily_data[self.daily_data.index < sim_date].tail(seq_length_daily)
            hourly_past_window = self.hourly_data[self.hourly_data.index < sim_date].tail(seq_length_hourly)
            future_candles = self.hourly_data[self.hourly_data.index > sim_date]
            if len(daily_window) == seq_length_daily and len(hourly_past_window) == seq_length_hourly and len(future_candles) >= next_days_hourly_needed:
                valid_dates.append(sim_date.strftime('%Y-%m-%d'))
        return valid_dates

# --- Generate Simulation Dates ---
def generate_simulation_dates(stock_symbols, start_date, end_date):
    simulation_dates_per_stock = {}
    for symbol in stock_symbols:
        print(f"\nProcessing {symbol}...")
        handler = DataHandler(symbol)
        handler.download_data(start_date, end_date)
        valid_dates = handler.get_valid_simulation_dates()
        simulation_dates_per_stock[symbol] = valid_dates
        print(f"{symbol} - {len(valid_dates)} valid dates.")
    return simulation_dates_per_stock

# --- Simulation Logic ---
def run_market_simulation(stock_symbol, start_date, end_date, starting_balance, valid_dates):
    try:
        cash_balance = starting_balance
        initial_balance = starting_balance
        wins = 0
        losses = 0
        balance_history = []
        date_history = []
        price_history = []
        portfolio_value_history = []

        model_trend = load_model("models/trend_model.keras", compile=False)
        model_low = load_model("models/low_price_model.keras", compile=False)

        handler = DataHandler(stock_symbol)
        handler.download_data(start_date, end_date)

        current_holdings = 0

        for date_str in valid_dates:
            print(f"\nProcessing {stock_symbol} on {date_str}...")
            sim_date = pd.to_datetime(date_str)
            past_daily = handler.daily_data[handler.daily_data.index < sim_date].tail(21)

            if past_daily.empty or past_daily.shape[0] < 21:
                print(" - Not enough daily data. Skipping.")
                continue

            X_daily = np.expand_dims(past_daily.values, axis=0)
            prediction = model_trend.predict(X_daily, verbose=0)[0]
            predicted_class = np.argmax(prediction)
            print(f" - Trend prediction: {prediction}, class: {predicted_class}")

            if predicted_class != 2:
                print(" - Not a bullish trend. Skipping.")
                continue

            past_hourly = handler.hourly_data[handler.hourly_data.index < sim_date].tail(168)
            if len(past_hourly) < 168:
                print(f" - Not enough hourly data. Skipping.")
                continue

            X_hourly = np.expand_dims(past_hourly.values, axis=0)
            predicted_low_pct_change = model_low.predict(X_hourly, verbose=0)[0][0]
            print(f" - Predicted low % change: {predicted_low_pct_change:.4f}")

            future_candles = handler.hourly_data[handler.hourly_data.index > sim_date]
            if future_candles.empty:
                print(" - No future candles available. Skipping.")
                continue

            next_day_candles = future_candles.head(24)
            if len(next_day_candles) < 5:
                print(" - Not enough next-day candles. Skipping.")
                continue

            next_open = next_day_candles['Open'].iloc[0].item()
            predicted_low_price = next_open * (1 + predicted_low_pct_change)
            limit_order_price = predicted_low_price * 1.02
            print(f" - Next open: {next_open:.2f}, predicted low price: {predicted_low_price:.2f}, limit order price: {limit_order_price:.2f}")

            actual_low = next_day_candles['Low'].min().item()
            if actual_low <= limit_order_price:
                buy_price = min(limit_order_price, actual_low)
            else:
                buy_price = actual_low
            print(f" - Actual low: {actual_low:.2f}, Buy price: {buy_price:.2f}")

            risk_amount = cash_balance * 0.10
            shares = max(int(risk_amount / buy_price), 1)
            buy_cost = shares * buy_price
            print(f" - Risk: {risk_amount:.2f}, Shares: {shares}, Cost: {buy_cost:.2f}, Cash before: {cash_balance:.2f}")

            if buy_cost > cash_balance:
                print(" - Insufficient funds. Skipping.")
                continue

            cash_balance -= buy_cost
            current_holdings += shares
            date_history.append(sim_date.strftime('%Y-%m-%d'))
            price_history.append(buy_price)
            print(f" - Bought {shares} shares at {buy_price:.2f}. Holdings: {current_holdings}, Cash after: {cash_balance:.2f}")

            highest_price = buy_price
            sold = False

            hold_candles = future_candles.head(24 * 7)
            for i in range(0, len(hold_candles), 24):
                day_candles = hold_candles.iloc[i:i+24]
                if day_candles.empty:
                    continue

                day_high = day_candles['High'].max().item()
                day_low = day_candles['Low'].min().item()

                if day_high > highest_price:
                    highest_price = day_high

                stop_price = highest_price * 0.95
                print(f"   - Hold Day {i//24+1}: High={day_high:.2f}, Low={day_low:.2f}, Stop={stop_price:.2f}")

                if day_low <= stop_price:
                    sell_price = stop_price
                    cash_balance += sell_price * current_holdings
                    print(f"   - Stop triggered. Sold at {sell_price:.2f}. Cash now: {cash_balance:.2f}")
                    current_holdings = 0
                    date_history.append(day_candles.index[-1].strftime('%Y-%m-%d'))
                    price_history.append(sell_price)
                    sold = True
                    if sell_price > buy_price:
                        wins += 1
                        print("   - Win")
                    else:
                        losses += 1
                        print("   - Loss")
                    break

            if not sold:
                final_close = hold_candles['Close'].iloc[-1].item()
                cash_balance += final_close * current_holdings
                print(f"   - Held to end. Sold at close: {final_close:.2f}. Cash now: {cash_balance:.2f}")
                current_holdings = 0
                date_history.append(hold_candles.index[-1].strftime('%Y-%m-%d'))
                price_history.append(final_close)
                if final_close > buy_price:
                    wins += 1
                    print("   - Win")
                else:
                    losses += 1
                    print("   - Loss")

            portfolio_value = cash_balance + (current_holdings * next_open)
            balance_history.append(portfolio_value)
            print(f" - Portfolio value: {portfolio_value:.2f}")

        print(f"\nSimulation Complete for {stock_symbol}. Wins: {wins}, Losses: {losses}, Final Balance: {cash_balance:.2f}")
        return {
            "wins": wins,
            "losses": losses,
            "final_balance": cash_balance,
            "return_pct": (cash_balance - initial_balance) / initial_balance,
            "dates": date_history,
            "history": balance_history,
            "share_prices": price_history
        }

    except Exception as e:
        print(f"ERROR during simulation of {stock_symbol}: {e}")
        return {"error": str(e)}

# --- Get Top Stocks ---
def get_top_simulated_stocks(stock_list, start_date, end_date, starting_balance, top_n=5):
    stock_results = []

    for symbol in stock_list:
        try:
            sim_dates = generate_simulation_dates([symbol], start_date, end_date)
            if not sim_dates[symbol]:
                continue
            result = run_market_simulation(symbol, start_date, end_date, starting_balance, sim_dates[symbol])
            if "error" not in result:
                stock_results.append({
                    "stock": symbol,
                    "return": result['return_pct']
                })
        except Exception as e:
            print(f"Skipping {symbol} due to error: {e}")

    stock_results.sort(key=lambda x: x['return'], reverse=True)
    return stock_results[:top_n]

def run_market_simulation_multi(stock_list, start_date, end_date, starting_balance):
    if not stock_list:
        return {"error": "No stocks provided."}

    balance_per_stock = starting_balance / len(stock_list)

    total_wins = 0
    total_losses = 0
    total_final_balance = 0
    results = []

    for sym in stock_list:
        print(f"\nRunning simulation for {sym}...")
        sim_dates = generate_simulation_dates([sym], start_date, end_date).get(sym, [])
        if not sim_dates:
            print(f"  → No valid dates for {sym}, skipping.")
            continue

        res = run_market_simulation(sym, start_date, end_date, balance_per_stock, sim_dates)
        if "error" in res:
            print(f"  → Error for {sym}: {res['error']}, skipping.")
            continue

        results.append(res)
        total_wins += res["wins"]
        total_losses += res["losses"]
        total_final_balance += res["final_balance"]

    if not results:
        return {"error": "No valid simulations."}

    max_hist_len = max(len(r["history"]) for r in results)

    combined_history = [
        sum(
            r["history"][i] if i < len(r["history"]) else r["history"][-1]
            for r in results
        )
        for i in range(max_hist_len)
    ]

    max_sp_len = max(len(r["share_prices"]) for r in results)

    combined_share_prices = [
        sum(
            r["share_prices"][i] if i < len(r["share_prices"]) else r["share_prices"][-1]
            for r in results
        )
        for i in range(max_sp_len)
    ]

    base_dates = results[0]["dates"]

    return {
        "wins": total_wins,
        "losses": total_losses,
        "final_balance": total_final_balance,
        "return_pct": (total_final_balance - starting_balance) / starting_balance,
        "dates": base_dates,
        "history": combined_history,
        "share_prices": combined_share_prices
    }




# --- Main Runner ---
if __name__ == "__main__":
    # Single‐stock setup
    single_list = ['AAPL']
    # Multi‐stock setup (two tickers)
    multi_list = ['AAPL', 'MSFT', 'TSLA']

    start_date = '2024-08-01'
    end_date = '2025-01-01'
    starting_balance = 100_000

    # 1) Generate dates & run single
    print("Generating simulation dates for single stock...")
    sim_dates_single = generate_simulation_dates(single_list, start_date, end_date)
    dates_single = sim_dates_single.get('AAPL', [])

    print("\nRunning single-stock simulation (original NN)…")
    result_single = run_market_simulation('AAPL', start_date, end_date, starting_balance, dates_single)
    if "error" in result_single:
        print("  Single-stock simulation error:", result_single["error"])
    else:
        print(
            f"  Single → dates: {len(result_single['dates'])}, "
            f"history: {len(result_single['history'])}, "
            f"share_prices: {len(result_single['share_prices'])}"
        )

    # 2) Run multi
    print("\nRunning multi-stock simulation (original NN)…")
    result_multi = run_market_simulation_multi(multi_list, start_date, end_date, starting_balance)
    if "error" in result_multi:
        print("  Multi-stock simulation error:", result_multi["error"])
    else:
        print(
            f"  Multi  → dates: {len(result_multi['dates'])}, "
            f"history: {len(result_multi['history'])}, "
            f"share_prices: {len(result_multi['share_prices'])}"
        )
