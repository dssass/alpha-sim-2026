import streamlit as st
from datetime import date, timedelta
import yfinance as yf
from plotly import graph_objs as go
import pandas as pd
import numpy as np
from openai import OpenAI
import google.generativeai as genai
import json

# --- 1. 頁面設定 ---
st.set_page_config(page_title="Alpha-Sim: 2026 AI 戰情室", layout="wide", page_icon="🛡️")

# 自定義 CSS (美化介面)
st.markdown("""
    <style>
    .stButton>button {width: 100%; border-radius: 8px; height: 3.5em; font-weight: bold; font-size: 1.2em;}
    .report-box {background-color: #f0f2f6; border-left: 6px solid #ff4b4b; padding: 20px; border-radius: 10px; margin-bottom: 20px;}
    .metric-container {background-color: #ffffff; padding: 10px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);}
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ Alpha-Sim: 雙引擎量化決策系統 (2026 版)")
st.caption("架構：Perplexity (最新情報) ➡ Gemini (參數決策) ➡ Monte Carlo (風險模擬)")

# --- 2. 核心邏輯函數 ---

@st.cache_data(ttl=3600)
def get_market_data(ticker):
    """抓取歷史股價數據"""
    try:
        data = yf.download(ticker, period="2y", interval="1d", auto_adjust=True, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        if data.empty: return None
        data = data.ffill()
        return data
    except Exception as e:
        return None

def get_perplexity_intel(ticker, api_key):
    """引擎 A: 使用 Perplexity 搜尋最新全球情報"""
    if not api_key: return "API Key 缺失。"
    
    client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")
    today_str = date.today().strftime("%Y-%m-%d")
    
    prompt = f"""
    Search for the latest financial news, institutional reports, and supply chain rumors for {ticker} as of today ({today_str}).
    Focus on:
    1. Revenue/Earnings outlook for 2026.
    2. Key themes (e.g., AI Server, iPhone 17, Tesla).
    3. Institutional sentiment (Foreign investors buying/selling?).
    Keep it concise, factual, and forward-looking.
    """
    
    try:
        response = client.chat.completions.create(
            model="sonar-pro", 
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Perplexity 錯誤: {str(e)}"

def get_gemini_decision(ticker, api_key, intel_text):
    """引擎 B: 使用 Gemini 進行 Agentic 參數判斷"""
    if not api_key: return None
    
    genai.configure(api_key=api_key)
    
    prompt = f"""
    You are a Quantitative Portfolio Manager in year 2026. 
    Target Stock: {ticker}
    Latest Intel:
    {intel_text}
    
    Task:
    1. Analyze the sentiment based on 2026 market conditions.
    2. Determine the 'Expected Annual Return' (drift) for a Monte Carlo simulation.
       - Range: -0.30 to +0.50.
    3. Determine the 'Volatility Multiplier'.
       - 1.0 = Normal, >1.0 = High Risk.
    
    Output strictly in JSON format:
    {{
        "verdict": "BUY" or "HOLD" or "SELL",
        "reasoning": "summary...",
        "expected_return": 0.15,
        "volatility_multiplier": 1.1
    }}
    """
    
    models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
    
    for model_name in models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                prompt, 
                generation_config={"response_mime_type": "application/json"}
            )
            return json.loads(response.text)
        except:
            continue
    return None

def run_monte_carlo(data, days, sims, expected_return, vol_mult):
    """蒙地卡羅數學模擬 (GBM 模型)"""
    log_returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
    hist_vol = log_returns.std() * (252 ** 0.5)
    adj_vol = hist_vol * vol_mult
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

# --- 3. 側邊欄設定 (方案 B 修改處) ---
with st.sidebar:
    st.header("🔑 啟動金鑰")
    
    # 從 st.secrets 讀取 (若本地端無 secrets.toml，則回傳空字串)
    saved_pplx = st.secrets.get("perplexity_api_key", "")
    saved_google = st.secrets.get("google_api_key", "")
    
    pplx_api_key = st.text_input("Perplexity API Key", value=saved_pplx, type="password")
    google_api_key = st.text_input("Google AI Studio Key", value=saved_google, type="password")
    
    if not (saved_pplx and saved_google):
        st.warning("⚠️ 尚未偵測到 Secrets。請在 Streamlit Cloud 後台設定或於本地建立 .streamlit/secrets.toml")
    else:
        st.success("✅ Secrets 已自動載入")

    st.divider()
    
    st.header("🎯 標的設定")
    selected_stock = st.text_input("股票代碼 (e.g. 4938.TW)", "4938.TW") 
    
    st.subheader("模擬參數")
    simulations = st.slider("模擬路徑數 (N)", 500, 3000, 1000)
    days_to_predict = st.slider("預測天數 (T)", 30, 365, 90)

# --- 4. 主程式 UI 邏輯 ---

data = get_market_data(selected_stock)

if data is None:
    st.error(f"❌ 無法獲取 {selected_stock} 數據。請確認代碼 (台股需加 .TW)。")
else:
    last_price = data['Close'].iloc[-1]
    
    # 頂部儀表板
    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("當前股價", f"{last_price:.2f}")
    col_info2.metric("資料日期", str(data.index[-1].date()))
    col_info3.markdown(f"**分析標的**: {selected_stock}")
    
    st.divider()

    if st.button("🚀 啟動 AI 雙引擎深度分析", type="primary"):
        if not pplx_api_key or not google_api_key:
            st.error("❌ 請提供雙引擎 API Keys 才能進行大師級分析。")
        else:
            with st.status("正在進行深度量化運算...", expanded=True) as status:
                st.write("🌐 搜尋全球即時財經情報 (Perplexity)...")
                intel = get_perplexity_intel(selected_stock, pplx_api_key)
                
                st.write("🧠 解讀情緒並推算數學模型參數 (Gemini)...")
                ai_decision = get_gemini_decision(selected_stock, google_api_key, intel)
                
                if ai_decision:
                    ai_return = ai_decision.get("expected_return", 0.05)
                    ai_vol = ai_decision.get("volatility_multiplier", 1.0)
                    verdict = ai_decision.get("verdict", "HOLD")
                    reasoning = ai_decision.get("reasoning", "No data")
                    
                    st.write(f"📊 執行 {simulations} 次蒙地卡羅機率路徑模擬...")
                    paths = run_monte_carlo(data, days_to_predict, simulations, ai_return, ai_vol)
                    
                    status.update(label="分析完成！", state="complete", expanded=False)

                    # 呈現結果
                    col_chart, col_report = st.columns([2, 1])
                    
                    with col_chart:
                        st.subheader("🔮 價格未來走勢模擬")
                        future_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=days_to_predict)
                        p5 = np.percentile(paths, 5, axis=1)
                        p50 = np.percentile(paths, 50, axis=1)
                        p95 = np.percentile(paths, 95, axis=1)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=data.index[-90:], y=data['Close'][-90:], name='歷史股價', line=dict(color='gray', width=1.5)))
                        fig.add_trace(go.Scatter(x=future_dates, y=p95, mode='lines', line=dict(width=0), showlegend=False))
                        fig.add_trace(go.Scatter(x=future_dates, y=p5, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 100, 255, 0.1)', name='90% 信賴區間'))
                        fig.add_trace(go.Scatter(x=future_dates, y=p50, mode='lines', name='AI 中位數預測', line=dict(color='#0064ff', width=3)))
                        fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=20, b=20))
                        st.plotly_chart(fig, use_container_width=True)
                        
                        m1, m2, m3 = st.columns(3)
                        m1.metric("悲觀情境 (P5)", f"{p5[-1]:.2f}", delta=f"{(p5[-1]/last_price-1)*100:.1f}%", delta_color="inverse")
                        m2.metric("預測中位 (P50)", f"{p50[-1]:.2f}", delta=f"{(p50[-1]/last_price-1)*100:.1f}%")
                        m3.metric("樂觀情境 (P95)", f"{p95[-1]:.2f}", delta=f"{(p95[-1]/last_price-1)*100:.1f}%")

                    with col_report:
                        st.subheader("📝 CIO 決策報告")
                        color_map = {"BUY": "#28a745", "SELL": "#dc3545", "HOLD": "#ffc107"}
                        v_color = color_map.get(verdict, "gray")
                        st.markdown(f"""
                        <div style="border: 2px solid {v_color}; padding: 15px; border-radius: 10px; text-align: center; background-color: {v_color}10;">
                            <h2 style="color: {v_color}; margin:0;">{verdict}</h2>
                            <small>AI-Powered Quant Analysis</small>
                        </div>
                        """, unsafe_allow_html=True)
                        st.info(f"**AI 判斷邏輯：**\n\n{reasoning}")
                        st.write(f"📈 預期年化報酬：{ai_return*100:+.1f}%")
                        st.write(f"📉 波動調整係數：{ai_vol}x")
                        with st.expander("查看原始搜集情報"):
                            st.write(intel)
                else:
                    st.error("AI 決策引擎發生錯誤，請稍後再試。")

