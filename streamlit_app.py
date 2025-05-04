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
    Caches results to avoid repeated API calls for the same inputs.
    Returns empty DataFrame on failure.
    """
    all_data = []
    for ticker in tickers:
        try:
            hist = yf.Ticker(ticker).history(start=start_date, end=end_date)[['Close']]
            if hist.empty:
                st.warning(f"No data for {ticker} between {start_date} and {end_date}.")
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
    use_custom = st.checkbox("Customize Views (Opinion)")
    custom_views = {}
    if use_custom:
        st.markdown("---")
        st.write("### Enter your expected returns for each ticker:")
        tickers_list_tmp = [t.strip().upper() for t in tickers.split(',') if t.strip()]
        for tkr in tickers_list_tmp:
            val = st.number_input(f"Expected return for {tkr} (%)", min_value=-100.0, max_value=100.0, value=0.0, format="%.2f")
            custom_views[tkr] = val / 100
    submit = st.form_submit_button(label='Run Optimization')
