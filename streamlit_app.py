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
def fetch_data(tickers: list[str], start_date: date | None, end_date: date | None, use_max: bool) -> pd.DataFrame:
    """Fetch historical price data; use max history if requested."""
    all_data = []
    for ticker in tickers:
        yf_tkr = yf.Ticker(ticker)
        if use_max:
            hist = yf_tkr.history(period="max")[['Close']]
        else:
            hist = yf_tkr.history(start=start_date, end=end_date)[['Close']]
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
def run_black_litterman(
    df: pd.DataFrame,
    allow_short: bool,
    custom_views: dict,
    use_market_cap: bool,
    views_as_delta: bool
) -> tuple[pd.Series, pd.Series, pd.DataFrame] | None:
    """Compute BL posterior returns, covariance, and optimal weights."""
    mu = expected_returns.mean_historical_return(df)
    Sigma = risk_models.sample_cov(df)

    # Determine views
    if custom_views:
        views_series = pd.Series(custom_views)
        if views_as_delta:
            abs_views = mu + views_series
        else:
            abs_views = views_series
        conf = pd.Series(0.5, index=abs_views.index)
    else:
        abs_views = mu
        conf = pd.Series(0.5, index=mu.index)

    # Instantiate BL model
    if use_market_cap:
        caps = yf.Ticker(df.columns[0]).history(period="max")  # placeholder, implement market caps separately
        # For simplicity, still use historical prior when market cap unchecked
        bl = BlackLittermanModel(
            cov_matrix=Sigma,
            pi=mu,
            absolute_views=abs_views,
            view_confidences=conf
        )
    else:
        bl = BlackLittermanModel(
            cov_matrix=Sigma,
            pi=mu,
            absolute_views=abs_views,
            view_confidences=conf
        )

    ret_bl = bl.bl_returns()
    cov_bl = bl.bl_cov()

    try:
        if allow_short:
            raw_w = bl.optimize()
        else:
            ef_post = EfficientFrontier(ret_bl, cov_bl)
            ef_post.max_sharpe()
            raw_w = ef_post.clean_weights()
    except Exception as e:
        st.error(f"Optimization failed: {e}")
        return None

    return pd.Series(raw_w), ret_bl, cov_bl

# -- Sidebar Inputs --
st.sidebar.header("🔧 Configuration")
use_max = st.sidebar.checkbox("Use Maximum Historical Data", value=False)
# Ticker input
tickers_input = st.sidebar.text_input("Tickers (comma-separated)")
