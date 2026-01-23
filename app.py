import streamlit as st
from datetime import date, timedelta
import yfinance as yf
from plotly import graph_objs as go
import pandas as pd
import numpy as np
from openai import OpenAI
import google.generativeai as genai
import json
import re

# --- 1. 頁面設定 ---
st.set_page_config(page_title="Alpha-Sim: 2026 AI 戰情室", layout="wide", page_icon="🛡️")

# 自定義 CSS
st.markdown("""
    <style>
    .stButton>button {width: 100%; border-radius: 8px; height: 3.5em; font-weight: bold; font-size: 1.2em;}
    .report-box {background-color: #1e1e1e; border-left: 6px solid #ff4b4b; padding: 20px; border-radius: 10px; margin-bottom: 20px;}
    .metric-card {background-color: #f0f2f6; padding: 15px; border-radius: 8px; text-align: center;}
    /* 讓 Plotly 圖表背景更融合 */
    .js-plotly-plot .plotly .main-svg {background: rgba(0,0,0,0) !important;}
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ Alpha-Sim: 雙引擎量化決策系統 (2026 Ver.)")
st.caption("架構：Perplexity (情報搜集) ➡ Gemini 3.0/2.5 (參數決策) ➡ Monte Carlo (風險模擬)")

# --- 2. 側邊欄 ---
with st.sidebar:
    st.header("🔑 啟動金鑰")
    pplx_api_key = st.text_input("Perplexity API Key", type="password")
    google_api_key = st.text_input("Google AI Studio Key", type="password")
    
    st.divider()
    
    st.header("🎯 標的設定")
    # 預設給個 4938.TW (和碩) 當範例
    selected_stock = st.text_input("股票代碼 (Ticker)", "4938.TW") 
    st.caption("💡 台股請務必加上 .TW (例如: 00733.TW)")
    
    st.subheader("模擬參數")
    simulations = st.slider("模擬路徑數 (N)", 500, 3000, 1000)
    days_to_predict = st.slider("預測天數 (T)", 30, 365, 90)

# --- 3. 核心函數 ---

@st.cache_data(ttl=3600)
def get_market_data(ticker):
    """抓取並清洗歷史數據"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="2y")
        
        # 備用方案
        if data.empty:
            data = yf.download(ticker, period="2y", interval="1d", progress=False)
        
        if data.empty:
            return None
        
        # 清洗 MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        # 移除時區
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)

        # 確保有 Close
        if 'Close' not in data.columns:
            if 'Adj Close' in data.columns:
                data['Close'] = data['Adj Close']
            else:
                return None
                
        return data.ffill()
        
    except Exception as e:
        return None

def get_perplexity_intel(ticker, api_key):
    """引擎 A: 搜尋情報"""
    if not api_key: return None
    client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")
    today_str = date.today().strftime("%Y-%m-%d")
    
    prompt = f"""
    Search for the latest financial news, institutional reports, and supply chain rumors for {ticker} as of today ({today_str}).
    Focus on:
    1. Revenue/Earnings outlook for 2026.
    2. Key themes (e.g., AI Server, iPhone, EV).
    3. Institutional sentiment (Target prices).
    Keep it concise.
    """
    try:
        response = client.chat.completions.create(
            model="sonar-pro", 
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Perplexity Error: {str(e)}"

def clean_json_string(json_str):
    """清洗 JSON 字串，移除 Markdown 標記"""
    try:
        cleaned = re.sub(r"```json\s*", "", json_str, flags=re.IGNORECASE)
        cleaned = re.sub(r"```", "", cleaned)
        return cleaned.strip()
    except:
        return json_str

def get_gemini_decision(ticker, api_key, intel_text):
    """引擎 B: 決策與參數設定 (修復幻覺 + 未來模型)"""
    if not api_key: return None
    
    genai.configure(api_key=api_key)
    
    # 🔥 關鍵修復：加入 CRITICAL CONTEXT RULES 防止台美股混淆
    prompt = f"""
    You are an aggressive Quantitative Portfolio Manager in 2026. Target: {ticker}
    Intel: {intel_text}
    
    CRITICAL CONTEXT RULES:
    1. If the ticker ends with ".TW" (e.g., 2330.TW, 00733.TW), it is a **TAIWAN STOCK**. 
    2. DO NOT confuse it with US stocks that have similar ticker symbols (e.g., do not confuse '00733.TW' with 'TW' Tradeweb Markets).
    3. Analyze the stock based on the TAIWAN market context (TWD currency, Taiwan supply chain).

    Task:
    Determine 'Expected Annual Return' (drift) and 'Volatility Multiplier'.
    
    CRITICAL STRATEGY RULES:
    1. If Verdict is BUY, 'expected_return' MUST be significant (e.g., > 0.20 for 20% annual).
    2. If Verdict is SELL, 'expected_return' MUST be negative (e.g., -0.15).
    3. 'volatility_multiplier': If the stock is an AI/Tech stock, this should be > 1.2.
    
    Output JSON ONLY:
    {{
        "verdict": "BUY/SELL/HOLD",
        "reasoning": "Short summary...",
        "expected_return": 0.25,
        "volatility_multiplier": 1.2
    }}
    """
    
    # 🔥 2026 頂規模型清單 (含向下兼容)
    models = [
        "gemini-3-flash-preview",       # 夢幻首選
        "gemini-2.5-flash-preview",     # 穩定次選
        "gemini-flash-latest",          # 速度保底
        "gemini-2.0-flash"              # 最後防線
    ]
    
    last_error = ""
    
    for model_name in models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                prompt, 
                generation_config={"response_mime_type": "application/json"}
            )
            
            cleaned_text = clean_json_string(response.text)
            result = json.loads(cleaned_text)
            result['model_used'] = model_name 
            return result
            
        except Exception as e:
            last_error = str(e)
            continue
            
    return {
        "verdict": "NEUTRAL",
        "reasoning": f"AI Analysis failed. Last error: {last_error}",
        "expected_return": 0.05,
        "volatility_multiplier": 1.0,
        "model_used": "Fallback"
    }

def run_monte_carlo(data, days, sims, expected_return, vol_mult):
    """執行 GBM 模擬"""
    # 計算歷史數據
    log_returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
    hist_vol = log_returns.std() * (252 ** 0.5)
    
    # 調整參數
    adj_vol = hist_vol * vol_mult
    # 確保波動率有最低門檻 (避免大牛股波動太小導致區間太窄)
    adj_vol = max(adj_vol, 0.20) 
    
    daily_vol = adj_vol / (252 ** 0.5)
    daily_drift = (expected_return / 252) - (0.5 * daily_vol ** 2)
    
    last_price = data['Close'].iloc[-1]
    Z = np.random.normal(0, 1, (days, sims))
    daily_returns = np.exp(daily_drift + daily_vol * Z)
    
    price_paths = np.zeros_like(daily_returns)
    price_paths[0] = last_price
    
    for t in range(1, days):
        price_paths[t] = price_paths[t-1] * daily_returns[t]
        
    return price_paths

# --- 4. 主程式邏輯 ---

data = get_market_data(selected_stock)

if data is None:
    st.error(f"❌ 無法獲取 {selected_stock} 的數據。")
    st.warning("請確認代碼是否正確 (台股請加 .TW)。Yahoo Finance 可能暫時不穩，請稍後再試。")
else:
    last_price = data['Close'].iloc[-1]
    
    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("當前股價", f"{last_price:.2f}")
    col_info2.metric("資料日期", str(data.index[-1].date()))
    col_info3.markdown(f"**分析標的**: {selected_stock}")
    
    st.divider()

    if st.button("🚀 啟動雙引擎戰情室 (Start Analysis)", type="primary"):
        if not pplx_api_key or not google_api_key:
            st.error("❌ 請輸入 API Keys")
        else:
            with st.status("正在進行深度運算...", expanded=True) as status:
                
                # Step 1
                st.write("🌍 Perplexity: 掃描 2026 市場情報...")
                intel = get_perplexity_intel(selected_stock, pplx_api_key)
                
                # Step 2
                st.write("🧠 Gemini: 正在切換 3.0/2.5 未來模型...")
                ai_decision = get_gemini_decision(selected_stock, google_api_key, intel)
                
                ai_return = ai_decision.get("expected_return", 0.05)
                ai_vol = ai_decision.get("volatility_multiplier", 1.0)
                verdict = ai_decision.get("verdict", "HOLD")
                model_used = ai_decision.get("model_used", "Unknown")
                
                # Step 3
                st.write(f"📊 Monte Carlo: 執行 {simulations} 次價格模擬...")
                paths = run_monte_carlo(data, days_to_predict, simulations, ai_return, ai_vol)
                
                status.update(label="分析完成！", state="complete", expanded=False)

            # --- 結果呈現區 ---
            
            # 計算目標價
            final_prices = paths[-1]
            bull_price = np.percentile(final_prices, 95)
            base_price = np.percentile(final_prices, 50)
            bear_price = np.percentile(final_prices, 5)

            # 顯示目標價 Metrics
            st.subheader(f"📅 {days_to_predict} 天後價格預測")
            m1, m2, m3 = st.columns(3)
            m1.metric("樂觀情境 (Bull)", f"{bull_price:.2f}", f"{(bull_price/last_price - 1)*100:.1f}%")
            m2.metric("中位數 (Base)", f"{base_price:.2f}", f"{(base_price/last_price - 1)*100:.1f}%")
            m3.metric("悲觀情境 (Bear)", f"{bear_price:.2f}", f"{(bear_price/last_price - 1)*100:.1f}%")

            col_chart, col_report = st.columns([2, 1])
            
            with col_chart:
                future_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=days_to_predict)
                p5 = np.percentile(paths, 5, axis=1)
                p50 = np.percentile(paths, 50, axis=1)
                p95 = np.percentile(paths, 95, axis=1)
                
                fig = go.Figure()
                # 歷史走勢
                fig.add_trace(go.Scatter(x=data.index[-90:], y=data['Close'][-90:], name='歷史股價', line=dict(color='white', width=1.5)))
                # 預測區間
                fig.add_trace(go.Scatter(x=future_dates, y=p95, mode='lines', line=dict(width=0), showlegend=False))
                fig.add_trace(go.Scatter(x=future_dates, y=p5, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 255, 100, 0.1)' if verdict=="BUY" else 'rgba(255, 50, 50, 0.1)', name='90% 機率區間'))
                # 中位數線
                fig.add_trace(go.Scatter(x=future_dates, y=p50, mode='lines', name='預測路徑', line=dict(color='#00d2ff', width=3)))
                
                fig.update_layout(
                    template="plotly_dark",
                    height=500,
                    title="未來價格路徑模擬 (Monte Carlo)",
                    xaxis_title="日期",
                    yaxis_title="股價",
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)
                
            with col_report:
                st.subheader("📝 CIO 決策報告")
                
                # 根據 Verdict 變色
                color_map = {"BUY": "#28a745", "SELL": "#dc3545", "HOLD": "#ffc107", "NEUTRAL": "gray"}
                v_color = color_map.get(verdict, "gray")
                
                st.markdown(f"""
                <div style="border: 2px solid {v_color}; padding: 20px; border-radius: 12px; text-align: center; margin-bottom: 20px; background-color: {v_color}20;">
                    <h1 style="color: {v_color}; margin:0; font-size: 3em;">{verdict}</h1>
                    <p style="margin-top: 5px; color: #ccc;">Strategy: {model_used}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### 💡 核心觀點")
                st.info(ai_decision.get("reasoning"))
                
                st.markdown("### ⚙️ 模型參數")
                st.markdown(f"""
                - **預期年化報酬 (Drift):** `{ai_return*100:+.1f}%`
                - **波動率加權 (Vol Mult):** `{ai_vol}x`
                """)
                
                with st.expander("查看原始情報 (Perplexity)"):
                    st.write(intel)
    
    else:
        st.info("👈 請輸入 API Keys 並點擊按鈕，開始 2026 AI 戰情分析！")
