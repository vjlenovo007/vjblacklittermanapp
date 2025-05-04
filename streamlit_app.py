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
# Date range selection when not using max history
if not use_max:
    start_date = st.sidebar.date_input("Start Date", date.today().replace(year=date.today().year-1),
                                      min_value=date(1900,1,1), max_value=date.today())
    end_date   = st.sidebar.date_input("End Date", date.today(),
                                      min_value=start_date, max_value=date.today())
else:
    start_date = None
    end_date   = None
# Ticker input
tickers_input = st.sidebar.text_input("Tickers (comma-separated)")
allow_short = st.sidebar.checkbox("Allow Short Positions", value=False)
use_custom = st.sidebar.checkbox("Customize Expected Returns (Opinion)", value=False)
views_as_delta = False
custom_views = {}
if use_custom:
    views_as_delta = st.sidebar.checkbox(
        "Treat views as delta on historical means", value=False,
        help="If checked, custom views will be added to the historical mean returns"
    )
    tickers_list_tmp = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    if tickers_list_tmp:
        st.sidebar.markdown("---")
        st.sidebar.write("### Custom Views (% expected return)")
        for tkr in tickers_list_tmp:
            val = st.sidebar.number_input(
                f"{tkr}", min_value=-100.0, max_value=100.0,
                value=0.0, step=0.01, format="%.2f"
            )
            custom_views[tkr] = val / 100
submit = st.sidebar.button("Run Optimization")

# -- Main --
if submit:
    tickers_list = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    if not tickers_list:
        st.error("Enter at least one ticker symbol.")
    else:
        df = fetch_data(tickers_list, None, None, use_max)
        if df.empty:
            st.error("No data fetched for the given tickers.")
        else:
            # Show data availability ranges for each ticker (full history)
            st.subheader("📅 Data Availability Ranges")
            full_ranges = {}
            for tkr in tickers_list:
                try:
                    full_hist = yf.Ticker(tkr).history(period="max")[['Close']]
                    start_full = full_hist.index.min().date() if not full_hist.empty else None
                    end_full = full_hist.index.max().date() if not full_hist.empty else None
                    full_ranges[tkr] = {'Full Start Date': start_full, 'Full End Date': end_full}
                except Exception:
                    full_ranges[tkr] = {'Full Start Date': None, 'Full End Date': None}
            full_ranges_df = pd.DataFrame.from_dict(full_ranges, orient='index')
            st.dataframe(full_ranges_df)

            # Also show fetched data range
            fetched_ranges = {tkr: (df.index.min().date(), df.index.max().date()) for tkr in df.columns}
            fetched_ranges_df = pd.DataFrame.from_dict(fetched_ranges, orient='index', columns=['Fetched Start Date','Fetched End Date'])
            st.subheader("🗓️ Fetched Data Range")
            st.dataframe(fetched_ranges_df)

            result = run_black_litterman(
                df,
                allow_short,
                custom_views,
                False,
                views_as_delta
            )
            if result is None:
                st.error("Optimization could not be completed.")
                st.stop()
            weights, ret_bl, cov_bl = result

            # Efficient Frontier
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

            # Display
            st.header("📊 Results")
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
