import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import date
from pypfopt import risk_models, expected_returns, BlackLittermanModel, EfficientFrontier, plotting
import matplotlib.pyplot as plt

# -- Page Config --
st.set_page_config(
    page_title="Black-Litterman Portfolio Optimizer",
    layout="wide",
    page_icon=":chart_with_upwards_trend:"
)
plt.style.use('ggplot')

# -- Data Fetching --
@st.cache_data(show_spinner=False)
def fetch_data(tickers: list[str], start_date: date, end_date: date) -> pd.DataFrame:
    """Fetch historical price data and cache results."""
    all_data = []
    for ticker in tickers:
        hist = yf.Ticker(ticker).history(start=start_date, end=end_date)[['Close']]
        if hist.empty:
            st.warning(f"No data for {ticker}.")
            continue
        hist = hist.rename(columns={'Close': ticker})
        hist.index.name = 'Date'
        all_data.append(hist)
    if not all_data:
        return pd.DataFrame()
    return pd.concat(all_data, axis=1).dropna()

# -- Model Computation --
@st.cache_data(show_spinner=False)
def run_black_litterman(df: pd.DataFrame, allow_short: bool, custom_views: dict) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Compute BL posterior returns, covariance, and weights."""
    returns = df.pct_change().dropna()
    mu = expected_returns.mean_historical_return(df)
    Sigma = risk_models.sample_cov(df)

    # Determine absolute views and confidences
    if custom_views:
        abs_views = pd.Series(custom_views)
        conf = pd.Series(0.5, index=abs_views.index)
    else:
        abs_views = mu
        conf = pd.Series(0.5, index=mu.index)

    bl = BlackLittermanModel(cov_matrix=Sigma, pi=mu,
                              absolute_views=abs_views,
                              view_confidences=conf)
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
# Date range selection
start_date = st.sidebar.date_input(
    "Start Date", date.today().replace(year=date.today().year-1)
)
end_date = st.sidebar.date_input(
    "End Date", date.today()
)
# Ticker input
tickers_input = st.sidebar.text_input(
    "Tickers (comma-separated)"
)
# Short positions toggle
allow_short = st.sidebar.checkbox(
    "Allow Short Positions"
)
# Custom views toggle and inputs
use_custom = st.sidebar.checkbox(
    "Customize Expected Returns (Opinion)"
)
custom_views = {}
# Prepare ticker list for custom views
tickers_list_tmp = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
if use_custom and tickers_list_tmp:
    st.sidebar.markdown("---")
    st.sidebar.write("### Enter expected returns (%) for each ticker")
    for t in tickers_list_tmp:
        val = st.sidebar.number_input(
            f"{t}",
            min_value=-100.0,
            max_value=100.0,
            value=0.0,
            step=0.01,
            format="%.2f"
        )
        custom_views[t] = val / 100
# Run button
submit = st.sidebar.button("Run Optimization")

# -- Main --
if submit:
    # Validate inputs
    tickers_list = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    if not tickers_list or start_date >= end_date:
        st.error("Please provide valid tickers and date range.")
    else:
        df = fetch_data(tickers_list, start_date, end_date)
        if df.empty:
            st.error("No data fetched for the given tickers/date range.")
        else:
            # Compute model
            weights, ret_bl, cov_bl = run_black_litterman(df, allow_short, custom_views)

            # Compute frontier
            mu_hist = expected_returns.mean_historical_return(df)
            Sigma_hist = risk_models.sample_cov(df)
            ef = EfficientFrontier(mu_hist, Sigma_hist)
            fig_ef, ax_ef = plt.subplots()
            plotting.plot_efficient_frontier(ef, ax=ax_ef, show_assets=False)
            # Max Sharpe point
            ef_max = EfficientFrontier(mu_hist, Sigma_hist); ef_max.max_sharpe()
            r_max, v_max, _ = ef_max.portfolio_performance()
            ax_ef.scatter(v_max, r_max, marker='*', s=200, label='Max Sharpe')
            # Min Vol point
            ef_min = EfficientFrontier(mu_hist, Sigma_hist); ef_min.min_volatility()
            r_min, v_min, _ = ef_min.portfolio_performance()
            ax_ef.scatter(v_min, r_min, marker='o', s=100, label='Min Volatility')
            ax_ef.set_title('Efficient Frontier')
            ax_ef.legend()

            # Display results
            st.header("📊 Optimization Results")
            tabs = st.tabs(["Overview", "Efficient Frontier"])

            with tabs[0]:
                st.subheader("Price Trends")
                st.line_chart(df)

                st.subheader("Optimal Weights")
                st.bar_chart(pd.Series(weights))
                st.subheader("Posterior Expected Returns")
                st.dataframe(ret_bl.to_frame('Expected Return'))
                st.subheader("Posterior Covariance")
                st.dataframe(cov_bl)

            with tabs[1]:
                st.pyplot(fig_ef)
else:
    st.info("Configure parameters in the sidebar and click 'Run Optimization'.")
