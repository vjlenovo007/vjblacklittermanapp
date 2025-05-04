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
def fetch_data(tickers: list[str], start_date: date, end_date: date, use_max: bool) -> pd.DataFrame:
    """Fetch historical price data and cache results. Use max history if requested."""
    all_data = []
    for ticker in tickers:
        try:
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
        except Exception as e:
            st.error(f"Fetching error for {ticker}: {e}")
    if not all_data:
        return pd.DataFrame()
    return pd.concat(all_data, axis=1).dropna()

@st.cache_data(show_spinner=False)
def fetch_market_caps(tickers: list[str]) -> pd.Series:
    """Fetch current market capitalizations for given tickers."""
    caps = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            caps[ticker] = info.get('marketCap', 0) or 0
        except Exception:
            caps[ticker] = 0
    return pd.Series(caps)

# -- Model Computation --
@st.cache_data(show_spinner=False)
def run_black_litterman(df: pd.DataFrame, allow_short: bool, custom_views: dict, use_market_cap: bool) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Compute BL posterior returns, covariance, and weights using either market caps or historical priors."""
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

    # Choose prior: market-implied or historical
    if use_market_cap:
        caps = fetch_market_caps(df.columns.tolist())
        w_mkt = caps / caps.sum()
        # Set a risk aversion parameter delta
        delta = 2.5
        bl = BlackLittermanModel(
            cov_matrix=Sigma,
            market_weights=w_mkt,
            risk_aversion=delta,
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

    # Optimization
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
use_max = st.sidebar.checkbox("Use Maximum Historical Data", value=False)
# Date range selection (disabled when using max)
if use_max:
    st.sidebar.info("Using full available history for each ticker")
    start_date = None
    end_date = None
else:
    start_date = st.sidebar.date_input(
        "Start Date", date.today().replace(year=date.today().year-1)
    )
    end_date = st.sidebar.date_input(
        "End Date", date.today()
    )
# Ticker input
tickers_input = st.sidebar.text_input("Tickers (comma-separated)")
allow_short = st.sidebar.checkbox("Allow Short Positions")
use_custom = st.sidebar.checkbox("Customize Expected Returns (Opinion)")
use_market_cap = st.sidebar.checkbox("Use Market-Cap Prior", value=False)
custom_views = {}
ticker_list_tmp = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
if use_custom and ticker_list_tmp:
    st.sidebar.markdown("---")
    st.sidebar.write("### Enter views (% expected return)")
    for t in ticker_list_tmp:
        val = st.sidebar.number_input(f"{t}", min_value=-100.0, max_value=100.0, value=0.0, step=0.01)
        custom_views[t] = val / 100
submit = st.sidebar.button("Run Optimization")

# -- Main --
if submit:
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    if not tickers or start_date >= end_date:
        st.error("Provide valid tickers and date range.")
    else:
        df = fetch_data(tickers, start_date, end_date, use_max)
        if df.empty:
            st.error("No data fetched.")
        else:
            # Show data range for each ticker
            st.subheader("📅 Data Availability Ranges")
            ranges = {}
            for tkr in tickers:
                try:
                    hist_full = yf.Ticker(tkr).history(period="max")[['Close']]
                    if not hist_full.empty:
                        start = hist_full.index.min().date()
                        end = hist_full.index.max().date()
                        ranges[tkr] = {'Start Date': start, 'End Date': end}
                except Exception as e:
                    ranges[tkr] = {'Start Date': None, 'End Date': None}
            ranges_df = pd.DataFrame.from_dict(ranges, orient='index')
            st.dataframe(ranges_df)

            weights, ret_bl, cov_bl = run_black_litterman(df, allow_short, custom_views, use_market_cap)
            # Compute frontier
            mu_hist = expected_returns.mean_historical_return(df)
            Sigma_hist = risk_models.sample_cov(df)
            ef = EfficientFrontier(mu_hist, Sigma_hist)
            fig_ef, ax_ef = plt.subplots()
            plotting.plot_efficient_frontier(ef, ax=ax_ef, show_assets=False)
            ef_max = EfficientFrontier(mu_hist, Sigma_hist); ef_max.max_sharpe()
            r_max, v_max, _ = ef_max.portfolio_performance()
            ax_ef.scatter(v_max, r_max, marker='*', s=200, label='Max Sharpe')
            ef_min = EfficientFrontier(mu_hist, Sigma_hist); ef_min.min_volatility()
            r_min, v_min, _ = ef_min.portfolio_performance()
            ax_ef.scatter(v_min, r_min, marker='o', s=100, label='Min Volatility')
            ax_ef.set_title('Efficient Frontier')
            ax_ef.legend()

            st.header("📊 Optimization Results")
            tabs = st.tabs(["Overview", "Efficient Frontier"])
            with tabs[0]:
                st.subheader("Price Trends")
                st.line_chart(df)
                st.subheader("Optimal Weights")
                st.bar_chart(weights)
                st.subheader("Posterior Expected Returns")
                st.dataframe(ret_bl.to_frame('Expected Return'))
                st.subheader("Posterior Covariance")
                st.dataframe(cov_bl)
            with tabs[1]:
                st.pyplot(fig_ef)
else:
    st.info("Configure options and click 'Run Optimization'.")
