import streamlit as st
import yfinance as yf
import pandas as pd
from pypfopt import risk_models, expected_returns, BlackLittermanModel
import matplotlib.pyplot as plt

st.set_page_config(page_title="Black-Litterman Portfolio Optimizer", layout="wide")
DATA_CACHE = "historical_data.csv"

@st.cache_data(show_spinner=False)
def fetch_data(tickers: list[str]) -> pd.DataFrame:
    """
    Fetch max-available historical price data from Yahoo Finance for given tickers.
    Returns empty DataFrame on failure.
    """
    all_data = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="max")[['Close']]
            if hist.empty:
                st.warning(f"No data returned for {ticker}.")
                continue
            hist = hist.rename(columns={'Close': ticker})
            hist.index.name = 'Date'
            all_data.append(hist)
        except Exception as e:
            st.error(f"Error fetching {ticker}: {e}")
    if not all_data:
        return pd.DataFrame()
    try:
        df = pd.concat(all_data, axis=1).dropna()
        df.to_csv(DATA_CACHE)
        return df
    except Exception as e:
        st.error(f"Error processing data: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_cached_data() -> pd.DataFrame:
    """
    Load previously fetched data from CSV cache.
    Returns empty DataFrame if cache missing or corrupt.
    """
    try:
        df = pd.read_csv(DATA_CACHE, index_col='Date', parse_dates=True)
        return df
    except FileNotFoundError:
        st.info("No cached data found.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading cache: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def run_black_litterman(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.DataFrame] | None:
    """
    Calculate Black-Litterman posterior returns, covariance, and optimal weights.
    Returns None on error.
    """
    try:
        returns = df.pct_change().dropna()
        if returns.empty:
            st.error("Insufficient price movement data to calculate returns.")
            return None
        mu = expected_returns.mean_historical_return(df)
        Sigma = risk_models.sample_cov(df)

        # Define neutral views
        absolute_views = mu
        view_confidences = pd.Series(0.5, index=mu.index)

        bl = BlackLittermanModel(cov_matrix=Sigma, pi=mu,
                                  absolute_views=absolute_views,
                                  view_confidences=view_confidences)
        ret_bl = bl.bl_returns()
        cov_bl = bl.bl_cov()
        weights = bl.optimize()
        return weights, ret_bl, cov_bl
    except Exception as e:
        st.error(f"Error running Black-Litterman model: {e}")
        return None

# -- Streamlit UI --
@st.cache_resource

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
        if df.empty or df.shape[0] < 2:
            st.error("Insufficient data fetched. Please check ticker symbols and try again.")
            return

        st.success(f"Fetched data for {len(tickers)} symbols from {df.index.min().date()} to {df.index.max().date()}")

        with st.spinner("Running Black-Litterman model..."):
            result = run_black_litterman(df)
        if result is None:
            return
        weights, ret_bl, cov_bl = result
        st.success("Optimization complete!")

        try:
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
        except Exception as e:
            st.error(f"Error displaying results: {e}")

if __name__ == "__main__":
    main()
