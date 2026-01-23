import streamlit as st
from datetime import date, timedelta
import yfinance as yf
from plotly import graph_objs as go
import pandas as pd
import numpy as np
from openai import OpenAI
import google.generativeai as genai
import json
import extra_streamlit_components as stx  # <-- 瀏覽器儲存套件

# --- 1. 頁面設定 ---
st.set_page_config(page_title="Alpha-Sim: 2026 AI 戰情室", layout="wide", page_icon="🛡️")

# 自定義 CSS
st.markdown("""
    <style>
    .stButton>button {width: 100%; border-radius: 8px; height: 3.5em; font-weight: bold; font-size: 1.2em;}
    .report-box {background-color: #f0f2f6; border-left: 6px solid #ff4b4b; padding: 20px; border-radius: 10px; margin-bottom: 20px;}
    .metric-container {background-color: #ffffff; padding: 10px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);}
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ Alpha-Sim: 雙引擎量化決策系統 (2026 Cookie 版)")
st.caption("架構：Perplexity ➡ Gemini ➡ Monte Carlo | 金鑰儲存：Local Browser Cookie")

# --- 2. 核心邏輯函數 ---

@st.cache_data(ttl=3600)
def get_market_data(ticker):
    try:
        data = yf.download(ticker, period="2y", interval="1d", auto_adjust=True, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        if data.empty: return None
        data = data.ffill()
        return data
    except:
        return None

def get_perplexity_intel(ticker, api_key):
    if not api_key: return "API Key 缺失。"
    client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")
    today_str = date.today().strftime("%Y-%m-%d")
    prompt = f"Search for the latest financial news and 2026 outlook for {ticker} as of today ({today_str}). Focus on AI servers, supply chain, and institutional sentiment. Keep it concise."
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
    if not api_key: return None
    genai.configure(api_key=api_key)
    prompt = f"""
    You are a Quant PM in 2026. Target: {ticker}. Intel: {intel_text}
    Task: Determine 'expected_return' (-0.3 to 0.5) and 'volatility_multiplier' (0.5 to 2.0).
    Output JSON: {{"verdict": "BUY/HOLD/SELL", "reasoning": "...", "expected_return": 0.1, "volatility_multiplier": 1.0}}
    """
    models = ["gemini-1.5-flash", "gemini-pro"]
    for model_name in models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            return json.loads(response.text)
        except:
            continue
    return None

def run_monte_carlo(data, days, sims, expected_return, vol_mult):
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

# --- 3. Cookie 管理器初始化 ---
# 關鍵：移除 @st.cache_resource 裝飾器
cookie_manager = stx.CookieManager()

# --- 4. 側邊欄：金鑰管理 (Cookie 邏輯) ---
with st.sidebar:
    st.header("🔑 金鑰管理")
    st.caption("金鑰將儲存於您的瀏覽器中，不會上傳至伺服器。")

    # 讀取已存的 Cookie
    cookie_pplx = cookie_manager.get("pplx_key")
    cookie_google = cookie_manager.get("google_key")

    # 輸入框
    pplx_input = st.text_input("Perplexity Key", value=cookie_pplx if cookie_pplx else "", type="password")
    google_input = st.text_input("Google AI Key", value=cookie_google if cookie_google else "", type="password")

    col_btn1, col_btn2 = st.columns(2)
    if col_btn1.button("記住我"):
        # 設定 Cookie 效期
        cookie_manager.set("pplx_key", pplx_input, key="set_pplx")
        cookie_manager.set("google_key", google_input, key="set_google")
        st.success("✅ 已記憶，請手動重新整理網頁")
    
    if col_btn2.button("清除記憶"):
        cookie_manager.delete("pplx_key")
        cookie_manager.delete("google_key")
        st.rerun()

    st.divider()
    st.header("🎯 標的設定")
    selected_stock = st.text_input("股票代碼", "4938.TW") 
    simulations = st.slider("模擬路徑數", 500, 2000, 1000)
    days_to_predict = st.slider("預測天數", 30, 180, 90)

# --- 5. 主程式頁面 ---

data = get_market_data(selected_stock)

if data is None:
    st.error(f"❌ 無法獲取 {selected_stock} 數據。")
else:
    last_price = data['Close'].iloc[-1]
    st.subheader(f"📊 標的分析：{selected_stock} | 當前股價：{last_price:.2f}")
    
    if st.button("🚀 執行 AI 深度量化預測", type="primary"):
        if not pplx_input or not google_input:
            st.error("❌ 請輸入 API Keys。")
        else:
            with st.status("運算中...", expanded=True) as status:
                st.write("🌐 搜尋最新情報...")
                intel = get_perplexity_intel(selected_stock, pplx_input)
                
                st.write("🧠 AI 模型分析...")
                ai_decision = get_gemini_decision(selected_stock, google_input, intel)
                
                if ai_decision:
                    ai_return = ai_decision.get("expected_return", 0.05)
                    ai_vol = ai_decision.get("volatility_multiplier", 1.0)
                    verdict = ai_decision.get("verdict", "HOLD")
                    reasoning = ai_decision.get("reasoning", "No data")
                    
                    st.write("📊 執行蒙地卡羅模擬...")
                    paths = run_monte_carlo(data, days_to_predict, simulations, ai_return, ai_vol)
                    
                    status.update(label="分析完成！", state="complete")

                    # 圖表與報告
                    col_chart, col_report = st.columns([2, 1])
                    with col_chart:
                        future_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=days_to_predict)
                        p5, p50, p95 = np.percentile(paths, [5, 50, 95], axis=1)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=data.index[-90:], y=data['Close'][-90:], name='歷史', line=dict(color='gray')))
                        fig.add_trace(go.Scatter(x=future_dates, y=p95, mode='lines', line=dict(width=0), showlegend=False))
                        fig.add_trace(go.Scatter(x=future_dates, y=p5, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 100, 255, 0.1)', name='90% 區間'))
                        fig.add_trace(go.Scatter(x=future_dates, y=p50, mode='lines', name='AI 中位數', line=dict(color='#0064ff', width=3)))
                        st.plotly_chart(fig, use_container_width=True)

                    with col_report:
                        st.markdown(f"### 綜合評等: {verdict}")
                        st.write(f"**分析邏輯：**\n{reasoning}")
                        st.write(f"預期報酬: {ai_return*100:+.1f}% | 波動調整: {ai_vol}x")
                        with st.expander("情報原文"):
                            st.write(intel)
                else:
                    st.error("AI 決策引擎發生錯誤，請稍後再試。")
