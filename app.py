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
    .report-box {background-color: #f0f2f6; border-left: 6px solid #ff4b4b; padding: 20px; border-radius: 10px; margin-bottom: 20px;}
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
    selected_stock = st.text_input("股票代碼 (Ticker)", "4938.TW") 
    
    st.subheader("模擬參數")
    simulations = st.slider("模擬路徑數 (N)", 500, 3000, 1000)
    days_to_predict = st.slider("預測天數 (T)", 30, 365, 90)

# --- 3. 核心函數 ---

@st.cache_data(ttl=3600)
def get_market_data(ticker):
    """
    抓取並清洗歷史數據 
    (修復版: 改用 Ticker.history 增強雲端環境穩定性)
    """
    try:
        # 方法 1: 使用 Ticker 物件 (通常在 Streamlit Cloud 較穩定)
        stock = yf.Ticker(ticker)
        data = stock.history(period="2y")
        
        # 如果方法 1 失敗 (空的)，嘗試方法 2: 一般下載
        if data.empty:
            data = yf.download(ticker, period="2y", interval="1d", progress=False)
        
        if data.empty:
            return None
        
        # --- 資料清洗與防呆 ---
        
        # 1. 處理 MultiIndex (移除多餘層級，yfinance 常見坑)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        # 2. 移除時區資訊 (Plotly 繪圖時如果有时区會報錯)
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)

        # 3. 確保有 Close 欄位
        if 'Close' not in data.columns:
            if 'Adj Close' in data.columns:
                data['Close'] = data['Adj Close']
            else:
                return None # 資料結構嚴重錯誤
                
        # 簡單補值
        return data.ffill()
        
    except Exception as e:
        # 在開發模式下可以 print(e) 來看錯誤，但這裡回傳 None 讓主程式處理
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
    3. Institutional sentiment.
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
    """🛠️ 關鍵修復：移除 Markdown 標記，防止 JSON 解析失敗"""
    try:
        cleaned = re.sub(r"```json\s*", "", json_str, flags=re.IGNORECASE)
        cleaned = re.sub(r"```", "", cleaned)
        return cleaned.strip()
    except:
        return json_str

def get_gemini_decision(ticker, api_key, intel_text):
    """引擎 B: 決策與參數設定"""
    if not api_key: return None
    
    genai.configure(api_key=api_key)
    
    prompt = f"""
    You are a Quantitative Portfolio Manager in 2026. Target: {ticker}
    Intel: {intel_text}
    
    Task:
    Determine 'Expected Annual Return' (drift) and 'Volatility Multiplier' based on the intel.
    
    Output JSON ONLY:
    {{
        "verdict": "BUY/SELL/HOLD",
        "reasoning": "Short summary...",
        "expected_return": 0.15,
        "volatility_multiplier": 1.1
    }}
    """
    
    # ✅ 嚴格鎖定模型清單 (2026 配置)
    models = [
        "gemini-3-flash-preview",       
        "gemini-2.5-flash-preview",     
        "gemini-flash-latest",          
        "gemini-2.0-flash"              
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
            print(f"⚠️ {model_name} 失敗，切換下一順位...")
            continue
            
    return {
        "verdict": "NEUTRAL",
        "reasoning": f"AI Analysis failed. Last error: {last_error}",
        "expected_return": 0.05,
        "volatility_multiplier": 1.0,
        "model_used": "None"
    }

def run_monte_carlo(data, days, sims, expected_return, vol_mult):
    """執行 GBM 模擬"""
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

# --- 4. 主程式邏輯 ---

data = get_market_data(selected_stock)

if data is None:
    # 顯示更詳細的錯誤指引
    st.error(f"❌ 無法獲取 {selected_stock} 的數據。")
    st.warning("""
    可能的解決方案：
    1. 確認代碼是否正確 (台股請加 .TW，如 4938.TW)。
    2. 如果你是剛部署，Yahoo Finance 可能暫時封鎖了雲端 IP，請等待 5-10 分鐘後再試。
    3. 嘗試輸入美股代號 (如 NVDA) 測試系統是否正常。
    """)
else:
    last_price = data['Close'].iloc[-1]
    
    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("當前股價", f"{last_price:.2f}")
    col_info2.metric("資料日期", str(data.index[-1].date()))
    col_info3.markdown(f"**分析標的**: {selected_stock}")
    
    st.divider()

    # 按鈕邏輯
    if st.button("🚀 啟動雙引擎戰情室 (Start Analysis)", type="primary"):
        if not pplx_api_key or not google_api_key:
            st.error("❌ 請輸入 API Keys")
        else:
            with st.status("正在進行深度運算...", expanded=True) as status:
                
                # Step 1: 搜集情報
                st.write("🌍 Perplexity (Sonar-Pro): 掃描 2026 市場情報...")
                intel = get_perplexity_intel(selected_stock, pplx_api_key)
                
                # Step 2: AI 決策
                st.write("🧠 Gemini (Auto-Switch): 正在切換模型進行決策...")
                ai_decision = get_gemini_decision(selected_stock, google_api_key, intel)
                
                ai_return = ai_decision.get("expected_return", 0.05)
                ai_vol = ai_decision.get("volatility_multiplier", 1.0)
                verdict = ai_decision.get("verdict", "HOLD")
                model_used = ai_decision.get("model_used", "Unknown")
                
                st.success(f"決策模型鎖定: {model_used}")
                
                # Step 3: 模擬
                st.write(f"📊 Monte Carlo: 執行 {simulations} 次模擬...")
                paths = run_monte_carlo(data, days_to_predict, simulations, ai_return, ai_vol)
                
                status.update(label="分析完成！", state="complete", expanded=False)

            # 結果呈現
            col_chart, col_report = st.columns([2, 1])
            
            with col_chart:
                st.subheader("🔮 價格路徑模擬")
                future_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=days_to_predict)
                p5 = np.percentile(paths, 5, axis=1)
                p50 = np.percentile(paths, 50, axis=1)
                p95 = np.percentile(paths, 95, axis=1)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data.index[-60:], y=data['Close'][-60:], name='歷史', line=dict(color='gray')))
                fig.add_trace(go.Scatter(x=future_dates, y=p95, mode='lines', line=dict(width=0), showlegend=False))
                fig.add_trace(go.Scatter(x=future_dates, y=p5, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 100, 255, 0.1)', name='90% 區間'))
                fig.add_trace(go.Scatter(x=future_dates, y=p50, mode='lines', name='預測中位數', line=dict(color='#0064ff', width=2)))
                st.plotly_chart(fig, use_container_width=True)
                
            with col_report:
                st.subheader("📝 CIO 決策報告")
                color_map = {"BUY": "#28a745", "SELL": "#dc3545", "HOLD": "#ffc107", "NEUTRAL": "gray"}
                v_color = color_map.get(verdict, "gray")
                
                st.markdown(f"""
                <div style="border: 2px solid {v_color}; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 20px; background-color: {v_color}10;">
                    <h2 style="color: {v_color}; margin:0;">{verdict}</h2>
                    <small>Model: {model_used}</small>
                </div>
                """, unsafe_allow_html=True)
                
                st.write(ai_decision.get("reasoning"))
                st.markdown("---")
                st.markdown(f"- 預期趨勢: **{ai_return*100:+.1f}%**")
                st.markdown(f"- 波動係數: **{ai_vol}x**")
                
                with st.expander("查看原始情報"):
                    st.write(intel)
    
    else:
        st.info("👈 請輸入 API Keys 並點擊按鈕，開始 2026 AI 戰情分析！")
