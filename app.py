import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, timedelta
from scipy import stats

# -- Page Configuration --
st.set_page_config(page_title="Portfolio Analytics Pro", layout="wide")
st.title("📈 Portfolio Risk & Diversification Analyzer")

with st.sidebar.expander("📖 About & Methodology", expanded=False):
    st.markdown("""
    **What this app does:**
    This tool analyzes the risk, return, and diversification benefits of a 2-5 asset portfolio compared to the S&P 500.
    
    **Key Assumptions:**
    * **Annualization:** Uses **252 trading days** for all return and volatility metrics.
    * **Returns:** Uses **simple arithmetic returns** via daily percentage changes.
    * **Data Source:** Real-time data via **Yahoo Finance (yfinance)**.
    * **Benchmark:** The S&P 500 (^GSPC) is used for performance comparison.
    """)

# =================================================================
# 2.1 USER INPUTS & DATA RETRIEVAL
# =================================================================
st.sidebar.header("1. Global Settings")

# 2.1.1: Ticker entry (2-5)
ticker_input = st.sidebar.text_input("Enter 2-5 Tickers (e.g., AAPL, MSFT, TSLA)", value="AAPL, MSFT, GOOGL").upper()
tickers = [t.strip() for t in ticker_input.split(",") if t.strip()]

# 2.1.2: Date selection (Back to 1970, Min 1 year)
default_start = date.today() - timedelta(days=365)
start_date = st.sidebar.date_input("Start Date", value=default_start, min_value=date(1970, 1, 1))
end_date = st.sidebar.date_input("End Date", value=date.today(), min_value=date(1970, 1, 1))

# Sidebar Validations
if not (2 <= len(tickers) <= 5):
    st.sidebar.error("Please enter between 2 and 5 tickers.")
    st.stop()
if (end_date - start_date).days < 365:
    st.sidebar.error("Minimum range of 1 year required.")
    st.stop()

# 2.1.3 & 2.1.5: Data Retrieval & Caching
@st.cache_data(show_spinner="Downloading Market Data...", ttl=3600)
def get_data(ticker_list, start, end):
    all_symbols = ticker_list + ["^GSPC"]
    try:
        data = yf.download(all_symbols, start=start, end=end, progress=False)
        if data.empty: return None, "No data found."
        prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
        return prices, None
    except Exception as e: return None, str(e)

prices_raw, error = get_data(tickers, start_date, end_date)

if error:
    st.error(f"Error: {error}"); st.stop()

# Ensure all tickers exist in the download
missing = [t for t in tickers if t not in prices_raw.columns or prices_raw[t].isnull().all()]
if missing:
    st.error(f"Insufficient data for: {', '.join(missing)}"); st.stop()

# 2.1.4: Handling Partial Data (Truncation)
df_clean = prices_raw.dropna()
returns_df = df_clean.pct_change().dropna()

# --- Tab Layout (Requirement 2.0) ---
tab1, tab2, tab3 = st.tabs(["📊 Performance", "🛡️ Risk & Stats", "🤝 Diversification"])

# =================================================================
# 2.2 PRICE AND RETURN ANALYSIS
# =================================================================
with tab1:
    st.header("Price & Return Analysis")
    
    # 2.2.1: Price Chart (Handling the S&P 500 Scale issue)
    st.subheader("Asset Price Comparison")
    chart_mode = st.radio("Chart View:", ["Raw Price", "Normalized (Base 100)"], horizontal=True, key="view_mode")
    
    selected_chart_tickers = st.multiselect(
        "Select tickers for chart visibility:",
        options=list(df_clean.columns),
        default=list(df_clean.columns),
        key="main_price_select"
    )

    if selected_chart_tickers:
        if chart_mode == "Normalized (Base 100)":
            plot_df = (df_clean[selected_chart_tickers] / df_clean[selected_chart_tickers].iloc[0]) * 100
            y_label = "Index (Start = 100)"
        else:
            plot_df = df_clean[selected_chart_tickers]
            y_label = "Price (USD)"
        
        fig_p = px.line(plot_df, labels={"value": y_label, "Date": ""})
        fig_p.update_layout(template="plotly_white", hovermode="x unified")
        st.plotly_chart(fig_p, use_container_width=True)

    # 2.2.3: Summary Statistics Table
    st.subheader("Summary Statistics (Annualized)")
    stats_list = []
    for col in df_clean.columns:
        r = returns_df[col]
        stats_list.append({
            "Ticker": "S&P 500" if col == "^GSPC" else col,
            "Ann. Mean Return": f"{r.mean() * 252:.2%}",
            "Ann. Volatility": f"{r.std() * np.sqrt(252):.2%}",
            "Skewness": round(r.skew(), 2),
            "Kurtosis": round(r.kurtosis(), 2),
            "Min Daily Return": f"{r.min():.2%}",
            "Max Daily Return": f"{r.max():.2%}"
        })
    st.table(pd.DataFrame(stats_list))

    # 2.2.4: Cumulative Wealth Index ($10k)
    st.subheader("Cumulative Wealth Index ($10,000 Initial)")
    wealth_df = returns_df.copy()
    wealth_df['Equal-Weight Portfolio'] = returns_df[tickers].mean(axis=1)
    wealth_index = 10000 * (1 + wealth_df).cumprod()
    fig_w = px.line(wealth_index, labels={"value": "Portfolio Value ($)", "Date": ""})
    st.plotly_chart(fig_w, use_container_width=True)

# =================================================================
# 2.3 RISK AND DISTRIBUTION ANALYSIS
# =================================================================
with tab2:
    st.header("Risk Profile")
    
    # 2.3.1: Rolling Volatility
    vol_win = st.slider("Rolling Volatility Window (Days)", 20, 126, 60, key="vol_window_slider")
    roll_vol = returns_df[tickers].rolling(vol_win).std() * np.sqrt(252)
    st.plotly_chart(px.line(roll_vol, title=f"Rolling {vol_win}-Day Ann. Volatility", template="plotly_white"), use_container_width=True)

    st.divider()

    # Asset Selection for Distribution
    d_stock = st.selectbox("Select Asset for Statistical Analysis:", tickers, key="dist_ticker")
    d_data = returns_df[d_stock]
    
    dist_tabs = st.tabs(["Distribution Plot", "Q-Q Plot"])
    
    with dist_tabs[0]:
        # 2.3.2: Distribution Plot + Normal Fit
        mu, sigma = stats.norm.fit(d_data)
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=d_data, nbinsx=60, histnorm='probability density', name="Actual Returns"))
        x_range = np.linspace(d_data.min(), d_data.max(), 100)
        fig_hist.add_trace(go.Scatter(x=x_range, y=stats.norm.pdf(x_range, mu, sigma), name="Normal Fit", line=dict(color='red')))
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # 2.3.4: Normality Test (Jarque-Bera)
        jb_s, jb_p = stats.jarque_bera(d_data)
        st.markdown(f"**Jarque-Bera Statistic:** {jb_s:.2f} | **p-value:** {jb_p:.4f}")
        if jb_p < 0.05:
            st.warning("🚨 Rejects normality at the 5% level (p < 0.05)")
        else:
            st.success("✅ Fails to reject normality (p >= 0.05)")

    with dist_tabs[1]:
        # 2.3.3: Q-Q Plot
        (osm, osr), (slope, intercept, r) = stats.probplot(d_data, dist="norm")
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(x=osm, y=osr, mode='markers', name='Quantiles'))
        fig_qq.add_trace(go.Scatter(x=osm, y=slope*osm + intercept, mode='lines', name='Normal Ref'))
        fig_qq.update_layout(xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles")
        st.plotly_chart(fig_qq, use_container_width=True)

    # 2.3.5: Box Plot
    st.subheader("Daily Return Distribution Comparison")
    st.plotly_chart(px.box(returns_df[tickers], template="plotly_white"), use_container_width=True)

# =================================================================
# 2.4 CORRELATION AND DIVERSIFICATION
# =================================================================
with tab3:
    st.header("Diversification & Portfolio")
    
    # 2.4.1: Heatmap
    corr_m = returns_df[tickers].corr()
    st.plotly_chart(px.imshow(corr_m, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1), use_container_width=True)

    c1, c2 = st.columns(2)
    # 2.4.2 & 2.4.3: Pairwise analysis
    pair = st.multiselect("Select Two Stocks for Pairwise Analysis:", tickers, default=tickers[:2], max_selections=2, key="pair_select")
    
    if len(pair) == 2:
        with c1:
            st.plotly_chart(px.scatter(returns_df, x=pair[0], y=pair[1], trendline="ols", title=f"Scatter: {pair[0]} vs {pair[1]}"), use_container_width=True)
        with c2:
            r_corr_win = st.number_input("Rolling Correlation Window:", 20, 252, 60, key="rc_input")
            r_corr = returns_df[pair[0]].rolling(r_corr_win).corr(returns_df[pair[1]])
            st.plotly_chart(px.line(r_corr, title=f"Rolling {r_corr_win}-Day Correlation"), use_container_width=True)

    st.divider()

    # 2.4.4: Two-Asset Portfolio Explorer
    st.subheader("Two-Asset Portfolio Explorer")
    if len(pair) == 2:
        s1, s2 = pair[0], pair[1]
        w1 = st.slider(f"Weight on {s1} (%)", 0, 100, 50, key="p_slider") / 100.0
        
        # Portfolio Math
        r1, r2 = returns_df[s1].mean()*252, returns_df[s2].mean()*252
        v1, v2 = returns_df[s1].std()*np.sqrt(252), returns_df[s2].std()*np.sqrt(252)
        rho = returns_df[s1].corr(returns_df[s2])
        
        curr_ret = (w1 * r1) + ((1-w1) * r2)
        curr_vol = np.sqrt((w1**2 * v1**2) + ((1-w1)**2 * v2**2) + (2 * w1 * (1-w1) * v1 * v2 * rho))
        
        # Curve generation
        weights = np.linspace(0, 1, 101)
        v_curve = [np.sqrt((w**2 * v1**2) + ((1-w)**2 * v2**2) + (2 * w * (1-w) * v1 * v2 * rho)) for w in weights]
        
        m_a, m_b = st.columns(2)
        m_a.metric(f"Portfolio Return", f"{curr_ret:.2%}")
        m_b.metric(f"Portfolio Volatility", f"{curr_vol:.2%}")
        
        fig_ef = go.Figure()
        fig_ef.add_trace(go.Scatter(x=weights*100, y=v_curve, name="Volatility Curve"))
        fig_ef.add_trace(go.Scatter(x=[w1*100], y=[curr_vol], mode='markers', marker=dict(size=14, color='red'), name='Current Mix'))
        fig_ef.update_layout(xaxis_title=f"Weight in {s1} (%)", yaxis_title="Ann. Volatility", template="plotly_white")
        st.plotly_chart(fig_ef, use_container_width=True)
        
        st.info(f"**The Diversification Effect:** Combining these two assets creates a portfolio with an annualized volatility of **{curr_vol:.2%}**. "
                "Notice how the curve 'dips' to the left—this shows that you can reduce risk without necessarily sacrificing all your returns, "
                "an effect that is most powerful when correlation (currently **{rho:.2f}**) is low.")