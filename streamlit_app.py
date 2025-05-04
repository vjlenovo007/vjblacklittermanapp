import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import date
from pypfopt import risk_models, expected_returns, BlackLittermanModel, EfficientFrontier, plotting
import matplotlib.pyplot as plt

st.set_page_config(page_title="Black-Litterman Portfolio Optimizer", layout="wide")
DATA_CACHE = "historical_data.csv"

@st.cache_data(show_spinner=False)
def fetch_data(tickers: list[str], start_date: date, end_date: date) -> pd.DataFrame:
    """
    Fetch historical price data from Yahoo Finance for given tickers and date range.
    Returns empty DataFrame on failure.
    """
    all_data = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)[['Close']]
            if hist.empty:
                st.warning(f"No data returned for {ticker} between {start_date} and {end_date}.")
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
def run_black_litterman(df: pd.DataFrame, allow_short: bool) -> tuple[pd.Series, pd.Series, pd.DataFrame] | None:
    """
    Calculate Black-Litterman posterior returns, covariance, and optimal weights.
    If allow_short is False, enforce no-short constraints via EfficientFrontier.
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

        # Black-Litterman model
        bl = BlackLittermanModel(cov_matrix=Sigma, pi=mu,
                                  absolute_views=absolute_views,
                                  view_confidences=view_confidences)
        ret_bl = bl.bl_returns()
        cov_bl = bl.bl_cov()

        # Optimization
        if allow_short:
            raw_weights = bl.optimize()
            weights = pd.Series(raw_weights)
        else:
            # No shorting: use EfficientFrontier on posterior estimates
            ef_post = EfficientFrontier(ret_bl, cov_bl)
            ef_post.max_sharpe()
            weights = pd.Series(ef_post.clean_weights())
        return weights, ret_bl, cov_bl
    except Exception as e:
        st.error(f"Error running Black-Litterman model: {e}")
        return None

# -- Streamlit UI --
def main():
    st.title("Black-Litterman Portfolio Optimizer")
    st.markdown("Select a date range, enter comma-separated ticker symbols (e.g. AAPL, MSFT), and choose constraints to optimize your portfolio.")

    # Date inputs
    col_dates = st.columns(2)
    with col_dates[0]:
        start_date = st.date_input("Start Date", value=date.today().replace(year=date.today().year-1))
    with col_dates[1]:
        end_date = st.date_input("End Date", value=date.today())
    if start_date >= end_date:
        st.error("Start Date must be before End Date.")
        return

    # Ticker input and constraints
    tickers_input = st.text_input("Ticker Symbols:")
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    allow_short = st.checkbox("Allow short positions", value=False)

    if st.button("Fetch & Optimize"):
        if not tickers:
            st.warning("Please input at least one ticker symbol.")
            return

        with st.spinner("Fetching data from Yahoo Finance..."):
            df = fetch_data(tickers, start_date, end_date)
        if df.empty or df.shape[0] < 2:
            st.error("Insufficient data fetched. Please check ticker symbols and date range.")
            return

        st.success(f"Fetched data for {len(tickers)} symbols from {df.index.min().date()} to {df.index.max().date()}")

        with st.spinner("Running Black-Litterman model..."):
            result = run_black_litterman(df, allow_short)
        if result is None:
            return
        weights, ret_bl, cov_bl = result
        st.success("Optimization complete!")

        # Display Black-Litterman results
        try:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Optimal Weights")
                st.bar_chart(weights)
                fig_w, ax_w = plt.subplots()
                pos = weights.clip(lower=0)
                if pos.sum() > 0:
                    norm = pos / pos.sum()
                    norm.plot.pie(autopct='%.1f%%', ax=ax_w)
                    ax_w.set_ylabel('')
                    ax_w.set_title('Portfolio Weight Distribution')
                    st.pyplot(fig_w)
                else:
                    st.info("Pie chart skipped due to negative or zero weights.")
            with col2:
                st.subheader("Posterior Expected Returns")
                st.dataframe(ret_bl.to_frame("Expected Return"))
                fig_r, ax_r = plt.subplots()
                ret_bl.plot.bar(ax=ax_r)
                ax_r.set_title('Posterior Expected Returns')
                st.pyplot(fig_r)
                st.subheader("Posterior Covariance")
                st.dataframe(cov_bl)
        except Exception as e:
            st.error(f"Error displaying BL results: {e}")

        # Efficient Frontier
        try:
            mu_hist = expected_returns.mean_historical_return(df)
            Sigma_hist = risk_models.sample_cov(df)
            ef = EfficientFrontier(mu_hist, Sigma_hist)
            fig_ef, ax_ef = plt.subplots()
            plotting.plot_efficient_frontier(ef, ax=ax_ef, show_assets=False)
            ax_ef.set_title('Efficient Frontier')
            st.subheader("Efficient Frontier")
            st.pyplot(fig_ef)
        except Exception as e:
            st.error(f"Error plotting efficient frontier: {e}")

if __name__ == "__main__":
    main()
