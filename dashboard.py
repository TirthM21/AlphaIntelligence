import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.database.db_manager import DBManager
from datetime import datetime
import os
import json
from dotenv import load_dotenv

load_dotenv()

# Page Config
st.set_page_config(page_title="Alpha Intelligence | Command Center", layout="wide", page_icon="📡")

# Custom CSS for Premium Tooling
st.markdown("""
    <style>
    .main { background-color: #0d1117; color: #c9d1d9; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 8px; border: 1px solid #30363d; }
    h1, h2, h3 { color: #58a6ff !important; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #161b22; border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px; }
    .stTabs [aria-selected="true"] { background-color: #1f2428; border-bottom: 2px solid #58a6ff; }
    .status-hold { color: #3fb950; font-weight: bold; }
    .status-sell { color: #f85149; font-weight: bold; }
    .status-reduce { color: #d29922; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# Initialize Components
db = DBManager()

@st.cache_data(ttl=300)
def get_universe_signals():
    if db.db_url:
        return db.get_latest_recommendations(limit=250)
    return []

@st.cache_data(ttl=300)
def get_portfolio_data():
    if db.db_url:
        return db.get_full_portfolio_data()
    return {}

@st.cache_data(ttl=300)
def load_reports():
    reports = {}
    if os.path.exists("data/reports/latest_allocation_plan.csv"):
        reports['allocation'] = pd.read_csv("data/reports/latest_allocation_plan.csv")
    if os.path.exists("data/reports/latest_rebalance_actions.txt"):
        with open("data/reports/latest_rebalance_actions.txt", "r") as f:
            reports['rebalance'] = f.read()
    return reports

st.title("📡 Alpha Intelligence Market Engine")
st.write(f"Universe Analysis System | {datetime.now().strftime('%Y-%m-%d %H:%M')}")

tabs = st.tabs(["💎 Market Universe", "💼 Portfolio Analytics", "🧪 Custom Audit"])

# --- TAB 1: MARKET UNIVERSE ---
with tabs[0]:
    st.subheader("Latest Buy Signals: Full Market Scan")
    universe_data = get_universe_signals()
    
    if universe_data:
        df_unv = pd.DataFrame(universe_data)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Opportunities Found", len(df_unv))
        m2.metric("Top Setup", df_unv['ticker'].iloc[0])
        avg_score = df_unv['score'].mean()
        m3.metric("Avg Setup Quality", f"{avg_score:.1f}")

        st.dataframe(df_unv[['ticker', 'score', 'price', 'date']].style.format({
            'score': '{:.1f}', 'price': '${:.2f}', 'date': lambda x: x.strftime('%Y-%m-%d %H:%M')
        }), use_container_width=True, height=600)
    else:
        st.warning("No market signals found in Database. Run `python run_optimized_scan.py` to populate.")

# --- TAB 2: PORTFOLIO MANAGEMENT ---
with tabs[1]:
    st.subheader("Institutional Portfolio Tracker")
    p_data = get_portfolio_data()
    reports = load_reports()

    if not p_data or not p_data.get('holdings'):
        st.info("No active holdings found. Running a scan will automatically populate elite positions.")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("### 🏠 Current Holdings")
            df_hold = pd.DataFrame(p_data['holdings'])
            df_hold['Market Value'] = df_hold['qty'] * df_hold['current_price']
            df_hold['ROI %'] = ((df_hold['current_price'] - df_hold['avg_price']) / df_hold['avg_price']) * 100
            
            st.dataframe(df_hold[['ticker', 'sector', 'qty', 'avg_price', 'current_price', 'Market Value', 'ROI %']].style.format({
                'avg_price': '${:.2f}', 'current_price': '${:.2f}', 'Market Value': '${:,.0f}', 'ROI %': '{:+.2f}%'
            }), use_container_width=True)

        with col2:
            st.write("### ⚖️ Rebalance Actions")
            if 'rebalance' in reports:
                st.code(reports['rebalance'], language='text')
            else:
                st.write("No active rebalance signals.")

        st.write("---")
        st.write("### 📈 Allocation Plan (Next Trades)")
        if 'allocation' in reports:
            st.dataframe(reports['allocation'].style.format({
                'Current_Price': '${:.2f}', 'Est_Cost': '${:,.0f}'
            }), use_container_width=True)
        else:
            st.write("Run a scan to generate new capital allocation suggestions.")

# --- TAB 3: CUSTOM AUDIT ---
with tabs[2]:
    st.subheader("Minervini Protocol Audit (Personal Portfolio)")
    uploaded_file = st.file_uploader("Upload portfolio.json for Audit", type=['json'])
    
    if uploaded_file or os.path.exists("data/portfolio_evaluation.json"):
        if uploaded_file:
            user_port = json.load(uploaded_file)
            with open("portfolio.json", "w") as f:
                json.dump(user_port, f)
            if st.button("🚀 Execute Audit"):
                import subprocess
                subprocess.run(["python", "evaluate_portfolio.py"])
                st.cache_data.clear()
                st.rerun()

        if os.path.exists("data/portfolio_evaluation.json"):
            with open("data/portfolio_evaluation.json", "r") as f:
                aud_res = json.load(f)
            
            df_aud = pd.DataFrame(aud_res)
            st.write(f"Evaluating `{len(df_aud)}` personal positions...")
            
            st.dataframe(df_aud[['ticker', 'gain_loss_pct', 'minervini_score', 'phase_name', 'is_minervini_compliant', 'reasons']].style.format({
                'gain_loss_pct': '{:+.2f}%', 'minervini_score': '{:.1f}'
            }), use_container_width=True)

st.sidebar.title("Engine Status")
st.sidebar.write(f"Connected to DB: `{'Yes' if db.db_url else 'No'}`")
st.sidebar.write(f"System Time: {datetime.now().strftime('%H:%M:%S')}")

if st.sidebar.button("Refresh All Data"):
    st.cache_data.clear()
    st.rerun()
