import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.database.db_manager import DBManager
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

# Page Config
st.set_page_config(page_title="Alpha Intelligence Hedge Fund", layout="wide", page_icon="🏦")

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3e4461;
    }
    h1, h2, h3 {
        color: #00ffcc !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🏦 Alpha Intelligence Hedge Fund")
st.subheader("Institutional-Grade Portfolio & Performance Analytics")

# Initialize DB
db = DBManager()

if not db.db_url:
    st.error("DATABASE_URL not found. Please check your .env file.")
    st.stop()

# Fetch Data
@st.cache_data(ttl=600)
def load_data():
    return db.get_full_portfolio_data()

data = load_data()

if not data:
    st.warning("No data found in the database. Run a market scan to populate the portfolio.")
    st.stop()

# --- TOP METRICS ---
col1, col2, col3, col4 = st.columns(4)

equity_curve = pd.DataFrame(data['equity_curve'])
holdings = pd.DataFrame(data['holdings'])
trades = pd.DataFrame(data['recent_trades'])

if not equity_curve.empty:
    current_equity = equity_curve['equity'].iloc[-1]
    prev_equity = equity_curve['equity'].iloc[-2] if len(equity_curve) > 1 else 100000.0
    change = ((current_equity - prev_equity) / prev_equity) * 100
    
    col1.metric("Total AUM", f"${current_equity:,.0f}", f"{change:+.2f}%")
    
    # Simple Alpha Calculation vs Benchmark (SPY)
    latest_spy = equity_curve['spy'].iloc[-1] or 1.0
    first_spy = equity_curve['spy'].iloc[0] or 1.0
    spy_return = ((latest_spy - first_spy) / first_spy) * 100
    fund_return = ((current_equity - 100000.0) / 100000.0) * 100
    alpha = fund_return - spy_return
    
    col2.metric("Fund Return (ITD)", f"{fund_return:.2f}%")
    col3.metric("Benchmark (SPY)", f"{spy_return:.2f}%")
    col4.metric("Alpha", f"{alpha:+.2f}%", delta_color="normal")

# --- PERFORMANCE CHART ---
st.write("### 📈 Equity Curve vs Benchmark")
if not equity_curve.empty:
    # Normalize performance for comparison
    equity_curve['Fund (Indexed)'] = (equity_curve['equity'] / 100000.0) * 100
    first_spy = equity_curve['spy'].iloc[0] if equity_curve['spy'].iloc[0] else 1.0
    equity_curve['Benchmark (Indexed)'] = (equity_curve['spy'] / first_spy) * 100
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity_curve['date'], y=equity_curve['Fund (Indexed)'], 
                           name='Alpha Intelligence Fund', line=dict(color='#00ffcc', width=3)))
    fig.add_trace(go.Scatter(x=equity_curve['date'], y=equity_curve['Benchmark (Indexed)'], 
                           name='S&P 500 (SPY)', line=dict(color='#ff3366', dash='dash')))
    
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0, r=0, t=0, b=0),
                     legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)

# --- PORTFOLIO COMPOSITION ---
c1, c2 = st.columns([2, 1])

with c1:
    st.write("### 💼 Current Positions")
    if not holdings.empty:
        holdings['Market Value'] = holdings['qty'] * holdings['current_price']
        holdings['Gain/Loss %'] = ((holdings['current_price'] - holdings['avg_price']) / holdings['avg_price']) * 100
        
        # Display as styled dataframe
        st.dataframe(holdings[['ticker', 'sector', 'qty', 'avg_price', 'current_price', 'Market Value', 'Gain/Loss %']].style.format({
            'avg_price': '${:.2f}', 'current_price': '${:.2f}', 'Market Value': '${:,.0f}', 'Gain/Loss %': '{:+.2f}%'
        }), height=300, use_container_width=True)
    else:
        st.info("No active positions.")

with c2:
    st.write("### 🏗️ Sector Allocation")
    if not holdings.empty:
        sector_dist = holdings.groupby('sector')['Market Value'].sum().reset_index()
        fig_pie = px.pie(sector_dist, values='Market Value', names='sector', 
                        color_discrete_sequence=px.colors.sequential.Teal_r)
        fig_pie.update_layout(template="plotly_dark", height=300, showlegend=False, 
                             margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)

# --- TRADE LEDGER ---
st.write("### 📜 Recent Executive Trades")
if not trades.empty:
    st.table(trades[['ticker', 'action', 'qty', 'price', 'date']].head(10))
else:
    st.info("No recent trades recorded.")

st.sidebar.image("https://img.icons8.com/wired/128/00ffcc/bank.png", width=100)
st.sidebar.title("Fund Control Panel")
if st.sidebar.button("Force Data Refresh"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.write("---")
st.sidebar.write("System Status: **LIVE** 🟢")
st.sidebar.write(f"Last Scan: {datetime.now().strftime('%H:%M:%S')}")
