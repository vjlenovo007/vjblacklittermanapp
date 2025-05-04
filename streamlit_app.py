import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import date
from pypfopt import risk_models, expected_returns, BlackLittermanModel, EfficientFrontier, plotting
import matplotlib.pyplot as plt

# -- Streamlit Page Config --
st.set_page_config(
    page_title="Black-Litterman Portfolio Optimizer",
    layout="wide",
    page_icon=":chart_with_upwards_trend:"
)
# Apply a professional matplotlib style
plt.style.use('ggplot')

@st.cache_data(show_spinner=False)
def fetch_data(tickers: list[str], start_date: date, end_date: date) -> pd.DataFrame:
    """
    Fetch historical price data from Yahoo Finance for given tickers and date range.
    Returns empty DataFrame on failure.
    """
    all_data = []
    for ticker in tickers:
        try:
            hist = yf.Ticker(ticker).history(start=start_date, end=end_date)[['Close']]
            if hist.empty:
                st.warning(f"No data for {ticker}.")
                continue
            hist = hist.rename(columns={'Close': ticker})
            hist.index.name = 'Date'
            all_data.append(hist)
        except Exception as e:
            st.error(f"Fetching error for {ticker}: {e}")
    if not all_data:
        return pd.DataFrame()
    df = pd.concat(all_data, axis=1).dropna()
    return df

@st.cache_data(show_spinner=False)
def run_black_litterman(df: pd.DataFrame, allow_short: bool):
    """
    Compute BL posterior, covariance, and optimal weights.
    """
    returns = df.pct_change().dropna()
    mu = expected_returns.mean_historical_return(df)
    Sigma = risk_models.sample_cov(df)

    # Neutral views
    bl = BlackLittermanModel(cov_matrix=Sigma, pi=mu,
                              absolute_views=mu,
                              view_confidences=pd.Series(0.5, index=mu.index))
    ret_bl = bl.bl_returns()
    cov_bl = bl.bl_cov()
    if allow_short:
        raw_w = bl.optimize()
    else:
        ef_post = EfficientFrontier(ret_bl, cov_bl)
        ef_post.max_sharpe()
        raw_w = ef_post.clean_weights()
    weights = pd.Series(raw_w)
    return weights, ret_bl, cov_bl

# -- Sidebar Inputs --
st.sidebar.header("🔧 Configuration")
with st.sidebar.form(key='input_form'):
    start_date = st.date_input("Start Date", value=date.today().replace(year=date.today().year-1))
    end_date   = st.date_input("End Date", value=date.today())
    tickers    = st.text_input("Tickers (comma-separated)")
    allow_short = st.checkbox("Allow Short Positions")
    submit = st.form_submit_button(label='Run Optimization')

if submit:
    tickers_list = [t.strip().upper() for t in tickers.split(',') if t.strip()]
    if not tickers_list or start_date >= end_date:
        st.sidebar.error("Enter valid dates and at least one ticker.")
    else:
        df = fetch_data(tickers_list, start_date, end_date)
        if df.empty:
            st.error("No data fetched.")
        else:
            # Run model
            weights, ret_bl, cov_bl = run_black_litterman(df, allow_short)

            # Compute frontier and key portfolios
            mu_hist = expected_returns.mean_historical_return(df)
            Sigma_hist = risk_models.sample_cov(df)
            ef = EfficientFrontier(mu_hist, Sigma_hist)
            fig_ef, ax_ef = plt.subplots()
            plotting.plot_efficient_frontier(ef, ax=ax_ef, show_assets=False)
            # Max Sharpe
            ef_max = EfficientFrontier(mu_hist, Sigma_hist)
            ef_max.max_sharpe()
            r_max, v_max, _ = ef_max.portfolio_performance()
            ax_ef.scatter(v_max, r_max, marker='*', s=200, label='Max Sharpe')
            # Min Vol
            ef_min = EfficientFrontier(mu_hist, Sigma_hist)
            ef_min.min_volatility()
            r_min, v_min, _ = ef_min.portfolio_performance()
            ax_ef.scatter(v_min, r_min, marker='o', s=100, label='Min Volatility')
            ax_ef.set_title('Efficient Frontier')
            ax_ef.legend()

            # Main display
            st.header("📊 Optimization Results")
            tabs = st.tabs(["Weights & Metrics", "Efficient Frontier"])

            with tabs[0]:
    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("Optimal Weights")
        st.bar_chart(weights)
        fig_w, ax_w = plt.subplots()
        pos = weights.clip(lower=0)
        if pos.sum() > 0:
            norm = pos / pos.sum()
            colors = plt.get_cmap('tab10').colors
            norm.plot.pie(autopct='%.1f%%', ax=ax_w, colors=colors)
            ax_w.set_ylabel('')
            ax_w.set_title('Weight Distribution')
            st.pyplot(fig_w)
    with col2:
        st.subheader("Key Portfolio Metrics")
        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("Max Sharpe Return", f"{r_max:.2%}")
        metric_col2.metric("Volatility", f"{v_max:.2%}")
        metric_col1.metric("Min Vol Return", f"{r_min:.2%}")
        metric_col2.metric("Volatility", f"{v_min:.2%}")

    st.subheader("Posterior Expected Returns & Covariance")
    st.dataframe(ret_bl.to_frame('Expected Return'))
    st.dataframe(cov_bl)

with tabs[1]:
    st.subheader("Efficient Frontier")
    try:
        st.pyplot(fig_ef)
    except Exception as e:
        st.error(f"Error displaying Efficient Frontier: {e}")
