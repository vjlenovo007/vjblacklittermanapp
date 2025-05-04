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
    # Validate inputs    tickers_list = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
