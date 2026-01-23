import streamlit as st
from datetime import date, timedelta
import yfinance as yf
from plotly import graph_objs as go
import pandas as pd
import numpy as np
from openai import OpenAI
import google.generativeai as genai
import json
import extra_streamlit_components as stx

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

st.title("🛡️ Alpha-Sim: 雙引擎量化決策系統 (2026 旗艦版)")
st.caption("架構：Perplexity (Sonar-Pro) ➡ Gemini (2026 旗艦清單) ➡ Monte Carlo")

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
    """引擎 A: 搜尋 2026 最新全球情報"""
    if not api_key: return "API Key 缺失。"
    
    client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")
    today_str = date.today().strftime("%Y-%m-%d")
    
    prompt = f"""
    Search for the latest financial news, institutional reports, and supply chain rumors for {ticker} as of today ({today_str}).
    Focus on:
    1. Revenue/Earnings outlook for 2026.
    2. Key themes (e.g., AI Server, iPhone 17, Tesla).
    3. Institutional sentiment.
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
        return f"Perplexity Error: {str(e)}"

def get_gemini_decision(ticker, api_key, intel_text):
    """引擎 B: 2026 旗艦決策與參數設定 (Agentic)"""
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
    
    # --- 恢復 2026 旗艦模型清單 ---
    models = [
        "gemini-3-flash-preview",
        "gemini-2.5-flash-preview",
        "gemini-flash-latest",
        "gemini-2.0-flash"
    ]
    
    for model_name in models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                prompt, 
                generation_config={"response_mime_type": "application/json"}
            )
            result = json.loads(response.text)
            result['model_used'] = model_name 
            return result
        except:
            continue
            
    return None

def run_monte_carlo(data, days, sims, expected_return, vol_mult):
    """執行 2026 量化風險模擬 (GBM)"""
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

# --- 3. Cookie 管理器 (修正黃色警告) ---
# 直接初始化，不使用快取裝飾器
cookie_manager = stx.CookieManager()

# --- 4. 側邊欄：設定 ---
with st.sidebar:
    st.header("🔑 金鑰管理")
    st.caption("2026 模式：金鑰將加密儲存於瀏覽器 Cookie 中。")
    
    # 從 Cookie 讀取
    saved_pplx = cookie_manager.get("pplx_key")
    saved_google = cookie_manager.get("google_key")
    
    pplx_api_key = st.text_input(
        "Perplexity API Key", 
        type="password",
        value=saved_pplx if saved_pplx else ""
    )
    google_api_key = st.text_input(
        "Google AI Studio Key", 
        type="password",
        value=saved_google if saved_google else ""
    )
    
    col_btn1, col_btn2 = st.columns(2)
    if col_btn1.button("記住我 (Save)"):
        cookie_manager.set("pplx_key", pplx_api_key, key="save_pplx")
        cookie_manager.set("google_key", google_api_key, key="save_google")
        st.success("✅ 記憶成功，請重新整理頁面。")
    
    if col_btn2.button("清除記憶 (Clear)"):
        cookie_manager.delete("pplx_key")
        cookie_manager.delete("google_key")
        st.rerun()
    
    st.divider()
    
    st.header("🎯 標的設定")
    selected_stock = st.text_input("股票代碼", "4938.TW") 
    simulations = st.slider("模擬路徑數", 500, 3000, 1000)
    days_to_predict = st.slider("預測天數", 30, 365, 90)

# --- 5. 主程式執行 ---

data = get_market_data(selected_stock)

if data is None:
    st.error(f"❌ 無法獲取 {selected_stock} 數據。")
else:
    last_price = data['Close'].iloc[-1]
    
    st.subheader(f"📊 分析標的：{selected_stock} | 當前參考價：{last_price:.2f}")
    
    if st.button("🚀 啟動 2026 雙引擎量化決策", type="primary"):
        if not pplx_api_key or not google_api_key:
            st.error("❌ 請提供 API Keys 以啟動 2026 旗艦引擎。")
        else:
            with st.status("正在連線 2026 數據中心...", expanded=True) as status:
                
                st.write("🌐 Perplexity (Sonar-Pro): 正在掃描全球情報...")
                intel = get_perplexity_intel(selected_stock, pplx_api_key)
                
                st.write("🧠 Gemini (2026 旗艦清單): 正在推算模型參數...")
                ai_decision = get_gemini_decision(selected_stock, google_api_key, intel)
                
                if ai_decision:
                    ai_return = ai_decision.get("expected_return", 0.05)
                    ai_vol = ai_decision.get("volatility_multiplier", 1.0)
                    verdict = ai_decision.get("verdict", "HOLD")
                    reasoning = ai_decision.get("reasoning", "No data")
                    model_used = ai_decision.get("model_used", "Unknown")
                    
                    st.write(f"📊 Monte Carlo: 正在執行 {simulations} 次平行宇宙模擬...")
                    paths = run_monte_carlo(data, days_to_predict, simulations, ai_return, ai_vol)
                    
                    status.update(label="2026 分析完成！", state="complete", expanded=False)

                    # 結果展現
                    col_chart, col_report = st.columns([2, 1])
                    
                    with col_chart:
                        st.subheader("🔮 預測路徑模擬")
                        future_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=days_to_predict)
                        p5, p50, p95 = np.percentile(paths, [5, 50, 95], axis=1)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=data.index[-90:], y=data['Close'][-90:], name='歷史', line=dict(color='gray')))
                        fig.add_trace(go.Scatter(x=future_dates, y=p95, mode='lines', line=dict(width=0), showlegend=False))
                        fig.add_trace(go.Scatter(x=future_dates, y=p5, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 100, 255, 0.1)', name='90% 區間'))
                        fig.add_trace(go.Scatter(x=future_dates, y=p50, mode='lines', name='AI 預測中位數', line=dict(color='#0064ff', width=3)))
                        st.plotly_chart(fig, use_container_width=True)
                        
                        m1, m2, m3 = st.columns(3)
                        m1.metric("悲觀 (P5)", f"{p5[-1]:.2f}", delta=f"{(p5[-1]/last_price-1)*100:.1f}%", delta_color="inverse")
                        m2.metric("中位 (P50)", f"{p50[-1]:.2f}", delta=f"{(p50[-1]/last_price-1)*100:.1f}%")
                        m3.metric("樂觀 (P95)", f"{p95[-1]:.2f}", delta=f"{(p95[-1]/last_price-1)*100:.1f}%")

                    with col_report:
                        st.subheader("📝 CIO 決策報告")
                        st.markdown(f"""
                        <div style="border: 2px solid #0064ff; padding: 15px; border-radius: 10px; text-align: center; background-color: #0064ff10;">
                            <h2 style="color: #0064ff; margin:0;">{verdict}</h2>
                            <small>Engine: {model_used}</small>
                        </div>
                        """, unsafe_allow_html=True)
                        st.info(f"**AI 分析邏輯：**\n\n{reasoning}")
                        st.write(f"📈 預期年報酬：{ai_return*100:+.1f}%")
                        st.write(f"📉 波動系數：{ai_vol}x")
                        with st.expander("情報原文"):
                            st.write(intel)
                else:
                    st.error("2026 旗艦引擎回傳異常，請檢查金鑰權限。")
