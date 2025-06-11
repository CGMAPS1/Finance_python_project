import streamlit as st
import pandas as pd
import yfinance as yf
import math
import matplotlib.pyplot as plt

st.set_page_config(page_title="NIFTY50 Portfolio Analyzer", layout="centered")

st.title("📊 NIFTY50 Stock Portfolio Analyzer")

uploaded_file = st.file_uploader("📥 Upload the Top 50 NIFTY Stocks CSV (must have 'Ticker' column)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    tickers = df['Ticker'].tolist()
    portfolio_size = st.number_input("💰 Enter the amount you want to invest:", min_value=1000, step=1000)
    top_n = st.slider("📈 Select top N stocks by market cap:", 1, len(tickers), 10)

    if st.button("Analyze Portfolio"):
        with st.spinner("Fetching data from Yahoo Finance..."):
            def fetch_market_data(tickers):
                data = yf.download(tickers, period='1d', group_by='ticker', auto_adjust=False, threads=True)
                all_data = []
                for ticker in tickers:
                    try:
                        latest_price = data[ticker]['Close'].iloc[-1]
                        info = yf.Ticker(ticker).info
                        all_data.append({
                            "Ticker": ticker,
                            "Latest Price": latest_price,
                            "Market Cap": info.get('marketCap', None),
                            "P/E": info.get('trailingPE', None),
                            "P/B": info.get('priceToBook', None),
                            "Beta": info.get('beta', None),
                            "Dividend Yield (%)": round(info.get('dividendYield', 0) * 100, 2) if info.get('dividendYield') else 0
                        })
                    except Exception as e:
                        st.warning(f"Error fetching {ticker}: {e}")
                        continue
                return pd.DataFrame(all_data)

            result_df = fetch_market_data(tickers)
            result_df = result_df.sort_values(by="Market Cap", ascending=False).head(top_n).reset_index(drop=True)
            position_size = portfolio_size / top_n
            result_df['Shares to Buy'] = result_df['Latest Price'].apply(lambda price: math.floor(position_size / price))

            st.success("✅ Portfolio Generated")
            st.dataframe(result_df, use_container_width=True)

            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("📤 Download Portfolio CSV", csv, "recommended_portfolio.csv", "text/csv")
            
            # Plot historical data
            st.subheader("📈 Visualize 6-Month History")
            selected_ticker = st.selectbox("Select a stock to view price history", result_df['Ticker'].tolist())
            if selected_ticker:
                hist = yf.Ticker(selected_ticker).history(period="6mo")
                if not hist.empty:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(hist.index, hist['Close'])
                    ax.set_title(f"{selected_ticker} - Last 6 Months")
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Closing Price")
                    ax.grid(True)
                    st.pyplot(fig)
                else:
                    st.warning("No historical data found for the selected ticker.")

else:
    st.info("Please upload a CSV file to proceed.")
    