import numpy as np
import pandas as pd
import yfinance as yf
import math
import matplotlib.pyplot as plt

# Load your top 50 NIFTY50 stocks list (must contain a 'Ticker' column)
df = pd.read_csv(r'C:\Users\HP\OneDrive\Desktop\NLP\Projects\top_50_indian_stocks.csv')
ticker_list = df['Ticker'].values.tolist()

# User Input
portfolio_size = int(input("💰 Enter the amount you want to invest: "))
top_n = int(input("📊 How many top stocks by market cap do you want to include? (e.g., 10): "))

# Fetch latest market data
def fetch_market_data(tickers):
    data = yf.download(tickers, period='1d', group_by='ticker', auto_adjust=False, threads=True)
    all_data = []

    for ticker in tickers:
        try:
            latest_price = data[ticker]['Close'].iloc[-1]
            info = yf.Ticker(ticker).info
            market_cap = info.get('marketCap', np.nan)
            pe_ratio = info.get('trailingPE', np.nan)
            pb_ratio = info.get('priceToBook', np.nan)
            beta = info.get('beta', np.nan)
            dividend_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0

            all_data.append({
                "Ticker": ticker,
                "Latest Price": latest_price,
                "Market Cap": market_cap,
                "P/E": pe_ratio,
                "P/B": pb_ratio,
                "Beta": beta,
                "Dividend Yield (%)": round(dividend_yield, 2)
            })

        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            continue

    return pd.DataFrame(all_data)

# Main data collection
df1 = fetch_market_data(ticker_list)
df1 = df1.sort_values(by="Market Cap", ascending=False).head(top_n).reset_index(drop=True)

# Equal Weight Portfolio Calculation
position_size = portfolio_size / top_n
df1['Shares to Buy'] = df1['Latest Price'].apply(lambda price: math.floor(position_size / price))

# Save to CSV
df1.to_csv("recommended_portfolio.csv", index=False)
print("\n✅ Portfolio saved to 'recommended_portfolio.csv'\n")
print(df1)

# Plot price history for a chosen stock
def plot_stock_history():
    ticker = input("\n📈 Enter the ticker of a stock to plot its 6-month history (e.g., INFY.NS): ").upper()
    try:
        hist = yf.Ticker(ticker).history(period="6mo")
        if hist.empty:
            print("❌ No historical data found.")
            return
        hist['Close'].plot(title=f"{ticker} - Last 6 Months", figsize=(10, 4))
        plt.xlabel("Date")
        plt.ylabel("Closing Price")
        plt.grid()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("⚠️ Could not plot the chart:", e)

plot_history = input("\nDo you want to plot historical price of a stock? (y/n): ").lower()
if plot_history == 'y':
    plot_stock_history()
