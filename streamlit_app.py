import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pypfopt import risk_models, expected_returns, BlackLittermanModel
import matplotlib.pyplot as plt

st.set_page_config(page_title="Black-Litterman Portfolio Optimizer", layout="wide")
DATA_CACHE = "historical_data.csv"

@st.cache_data(show_spinner=False)
def fetch_data(tickers: list[str]) -> pd.DataFrame:
    """
    Fetch max-available historical price data from Yahoo Finance for given tickers.
    """
    all_data = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="max")[['Close']]
            hist = hist.rename(columns={'Close': ticker})
            hist.index.name = 'Date'
            all_data.append(hist)
        except Exception as e:
            st.error(f"Error fetching {ticker}: {e}")
    if not all_data:
        return pd.DataFrame()
    # Merge on Date
    df = pd.concat(all_data, axis=1).dropna()
    # Save to CSV
    df.to_csv(DATA_CACHE)
    return df

@st.cache_data(show_spinner=False)
def load_cached_data() -> pd.DataFrame:
    """
    Load previously fetched data from CSV cache.
    """
    try:
        return pd.read_csv(DATA_CACHE, index_col='Date', parse_dates=True)
    except FileNotFoundError:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def run_black_litterman(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    Calculate Black-Litterman posterior returns, covariance, and optimal weights.
    """
    # Calculate historical returns
    returns = df.pct_change().dropna()
    # Prior: equilibrium returns
    mu = expected_returns.mean_historical_return(df)
    Sigma = risk_models.sample_cov(df)

    # Define views: use equilibrium returns as neutral views with moderate confidence
    Q = mu.values
    view_confidences = np.array([0.5] * len(mu))
    # Identity matrix for P implies each view is on a single asset
    bl = BlackLittermanModel(Sigma, pi=mu, absolute_views=Q, view_confidences=view_confidences)
    # Obtain posterior estimates
    ret_bl = bl.bl_returns()
    cov_bl = bl.bl_cov()
    weights = bl.optimize()
    return weights, ret_bl, cov_bl

# -- Streamlit UI --
def main():
    st.title("Black-Litterman Portfolio Optimizer")
    st.markdown("Enter comma-separated ticker symbols (e.g. AAPL, MSFT) to optimize your portfolio.")

    tickers_input = st.text_input("Ticker Symbols:")
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]

    if st.button("Fetch & Optimize"):
        if not tickers:
            st.warning("Please input at least one ticker symbol.")
            return

        with st.spinner("Fetching data from Yahoo Finance..."):
            df = fetch_data(tickers)
        if df.empty:
            st.error("No data fetched. Check ticker symbols and try again.")
            return
        st.success(f"Fetched data for {len(tickers)} symbols from {df.index.min().date()} to {df.index.max().date()}")

        with st.spinner("Running Black-Litterman model..."):
            weights, ret_bl, cov_bl = run_black_litterman(df)
        st.success("Optimization complete!")

        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Optimal Weights")
            st.bar_chart(weights)
            fig_w, ax_w = plt.subplots()
            weights.plot.pie(autopct='%.1f%%', ax=ax_w)
            ax_w.set_ylabel('')
            st.pyplot(fig_w)
        with col2:
            st.subheader("Posterior Expected Returns")
            st.dataframe(ret_bl.to_frame("Expected Return"))
            st.subheader("Posterior Covariance")
            st.dataframe(cov_bl)

if __name__ == "__main__":
    main()
